"""Closed-loop learned-policy evaluation in the existing physical scene.

Only ordinary scene reset sets initial state. Every rollout command goes through
Simulation.step; neither the expert nor a trajectory planner is invoked here.
Evaluation artifacts are kept separate from demonstration datasets.
"""
from __future__ import annotations

import copy
from contextlib import nullcontext
from datetime import datetime
import json
import math
from pathlib import Path
import random
import shutil
from tempfile import TemporaryDirectory
import time

import mujoco
import numpy as np
import yaml

from mujoco_tools.paths import CONFIG_DIR, OUTPUTS_DIR
from .action_replay_checks import ReplayContactMonitor
from .execution_guard import ExecutionSafetyError
from .scene import build_scene, load_config
from .simulation import Simulation

POLICY_KEYS = ('observation.state', 'observation.images.head', 'observation.images.wrist')


def _json_finite(value):
    """Keep failed numerical-run reports readable without nonstandard JSON."""
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def select_layouts(config, position=None, positions_config=None):
    """Copy fixed layouts without mutating the source config or randomizing it."""
    if position is None or str(position) == 'config':
        return [('config', copy.deepcopy(config))]
    manifest_path = Path(positions_config or CONFIG_DIR / 'insert_plug_positions.yaml')
    manifest = yaml.safe_load(manifest_path.read_text(encoding='utf-8'))
    rows = manifest.get('positions', [])
    names = [row['name'] for row in rows]
    if len(set(names)) != len(names) or not names:
        raise ValueError('Position manifest must contain unique named positions')
    requested = str(position)
    if requested != 'all':
        requested = requested if requested.startswith('pos') else f'pos{requested}'
        if requested not in names:
            raise ValueError(f'Unknown position {position!r}; expected one of {names} or all')
        rows = [row for row in rows if row['name'] == requested]
    result = []
    for row in rows:
        cfg = copy.deepcopy(config)
        for key in ('source_xy', 'target_xy'):
            values = np.asarray(row[key], dtype=float)
            if values.shape != (2,) or not np.isfinite(values).all():
                raise ValueError(f'{row["name"]}/{key} must contain two finite coordinates')
            cfg['board'][key] = values.tolist()
        result.append((row['name'], cfg))
    return result


def validate_contract(config, policy):
    """Fail before motion if the task/checkpoint contract does not match."""
    if not math.isclose(float(config['task']['control_hz']), 10.0, rel_tol=0, abs_tol=1e-10):
        raise ValueError('This insert_plug policy contract requires 10 Hz control (100 ms held actions)')
    if not math.isclose(float(config['physics']['timestep']), .001, rel_tol=0, abs_tol=1e-12):
        raise ValueError('This insert_plug policy contract requires a 1 ms physics timestep')
    height, width = int(config['cameras']['height']), int(config['cameras']['width'])
    expected = {
        'observation.state': {'type': 'STATE', 'shape': [8]},
        'observation.images.head': {'type': 'VISUAL', 'shape': [3, height, width]},
        'observation.images.wrist': {'type': 'VISUAL', 'shape': [3, height, width]},
    }
    if policy.input_features != expected:
        raise ValueError(f'Checkpoint observation contract differs from scene: expected {expected}, '
                         f'got {policy.input_features}')
    if policy.output_features != {'action': {'type': 'ACTION', 'shape': [8]}}:
        raise ValueError('insert_plug requires eight absolute targets: seven joint radians and full aperture metres')


class PhysicalSuccessMonitor:
    """Evaluate every physics tick on a copy, preserving the live solver state."""

    def __init__(self, sim, required_seconds=.25):
        self.sim = sim
        self.snapshot = copy.copy(sim)
        self.snapshot.data = mujoco.MjData(sim.model)
        self.required_seconds = required_seconds
        self.first_success_time = None
        self.continuous_success_seconds = 0.0
        self.stable = False
        self.samples = 0

    def metrics(self):
        mujoco.mj_copyData(self.snapshot.data, self.sim.model, self.sim.data)
        mujoco.mj_forward(self.sim.model, self.snapshot.data)
        return self.snapshot.metrics()

    def sample(self):
        metrics = self.metrics()
        self.samples += 1
        if metrics['success']:
            if self.first_success_time is None:
                self.first_success_time = float(self.sim.data.time)
            self.continuous_success_seconds = float(self.sim.data.time) - self.first_success_time
        else:
            self.first_success_time = None
            self.continuous_success_seconds = 0.0
        self.stable = bool(metrics['success'] and
                           self.continuous_success_seconds + 1e-10 >= self.required_seconds)
        return metrics


def _seed_policy(seed):
    random.seed(seed)
    np.random.seed(seed)
    # Loaded only for real rollouts; simulator-only tests need no torch install.
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_episode(sim, policy, *, duration=60.0, video_path=None, viewer=False,
                realtime=False, stop_on_contact=True, quiet=False):
    """Run one already-created scene and return metrics, including failures.

    Setup/contract errors are raised. Inference, invalid actions and physics
    failures during rollout become explicit failed results; targets are never
    silently clipped or replaced. Contact violations stop at the physics tick
    by default, which may leave the last 100 ms interval partially executed.
    """
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('duration must be finite and positive')
    interval_count = duration / sim.dt
    if not math.isclose(interval_count, round(interval_count), rel_tol=0, abs_tol=1e-8):
        raise ValueError('duration must be an exact multiple of the 100 ms control interval')
    validate_contract(sim.config, policy)
    policy.reset()
    monitor = ReplayContactMonitor(sim)
    monitor.phase = 'policy_rollout'
    stability = PhysicalSuccessMonitor(sim)
    previous_callback = sim.safety_callback
    previous_depth = sim.depth_enabled
    # The RGB/state checkpoint has no depth input. Rendering no unused depth
    # changes neither RGB calibration nor any scene/physics parameter.
    sim.depth_enabled = False

    def physics_check():
        monitor()
        if stop_on_contact and monitor.unsafe_contact_count:
            row = monitor.last_violation
            raise ExecutionSafetyError(f'{row["reason"]} at t={row["time_s"]:.3f}s: '
                                       f'{row["geom1"]} / {row["geom2"]}')
        stability.sample()

    sim.safety_callback = physics_check
    writer = None
    video_frames = 0
    started = time.monotonic()
    initial_sim_time = float(sim.data.time)
    completed_steps = attempted_steps = 0
    selected_actions = 0
    selection_wall_times = []
    stop_reason = 'timeout'
    error = None
    last_action = None
    final_metrics = None
    viewer_context = nullcontext(None)
    if viewer:
        import mujoco.viewer as mj_viewer
        viewer_context = mj_viewer.launch_passive(sim.model, sim.data)
    try:
        if video_path is not None:
            import imageio.v2 as imageio
            video_path = Path(video_path)
            if video_path.exists():
                raise FileExistsError(video_path)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(video_path), fps=sim.control_hz, codec='libx264',
                                       quality=8, macro_block_size=1, ffmpeg_log_level='error')
        with viewer_context as handle:
            sim.viewer = handle
            if handle is not None:
                with handle.lock():
                    handle.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    handle.cam.fixedcamid = sim.model.camera('third_person').id
            try:
                for index in range(round(interval_count)):
                    if handle is not None and not handle.is_running():
                        stop_reason = 'viewer_closed'
                        break
                    observations = sim.observe()
                    if writer is not None:
                        writer.append_data(np.concatenate([
                            observations['observation.images.head'],
                            observations['observation.images.wrist']], axis=1))
                        video_frames += 1
                    policy_observation = {key: observations[key] for key in POLICY_KEYS}
                    selection_start = time.monotonic()
                    action = policy.select_action(policy_observation)
                    selection_wall_times.append(time.monotonic() - selection_start)
                    selected_actions += 1
                    # Rejected NaN/Inf targets are still reviewable in JSON;
                    # this representation never changes the actual command.
                    last_action = _json_finite(np.asarray(action).tolist())
                    attempted_steps += 1
                    sim.step(action)
                    completed_steps += 1
                    if stability.stable:
                        stop_reason = ('success' if not monitor.unsafe_contact_count else
                                       'task_success_with_contact_violations')
                        break
                    if not quiet and completed_steps % 100 == 0:
                        print(json.dumps(dict(event='rollout_progress',
                                              simulation_seconds=round(float(sim.data.time), 3),
                                              actions=completed_steps,
                                              unsafe_contacts=monitor.unsafe_contact_count)), flush=True)
                    if realtime:
                        remaining = (float(sim.data.time) - initial_sim_time) - (time.monotonic() - started)
                        if remaining > 0:
                            time.sleep(remaining)
            except ExecutionSafetyError as exc:
                stop_reason = 'contact_violation'
                error = f'{type(exc).__name__}: {exc}'
            except Exception as exc:
                stop_reason = 'runtime_error'
                error = f'{type(exc).__name__}: {exc}'
            except KeyboardInterrupt:
                stop_reason = 'interrupted'
                error = 'KeyboardInterrupt'
            finally:
                # The final view is refreshed on the snapshot even when a
                # contact exception interrupted mj_step mid-control interval.
                try:
                    final_metrics = stability.metrics()
                    if writer is not None:
                        final_observation = sim.observe(stability.snapshot.data)
                        writer.append_data(np.concatenate([
                            final_observation['observation.images.head'],
                            final_observation['observation.images.wrist']], axis=1))
                        video_frames += 1
                except Exception as exc:
                    stop_reason = 'runtime_error'
                    error = error or f'Final observation failed: {type(exc).__name__}: {exc}'
    finally:
        sim.viewer = None
        sim.safety_callback = previous_callback
        sim.depth_enabled = previous_depth
        if writer is not None:
            writer.close()
    contacts = monitor.summary()
    task_success = bool(stability.stable and final_metrics and final_metrics['success'] and error is None)
    clean = contacts['unsafe_contact_count'] == 0
    times = np.asarray(selection_wall_times)
    elapsed = float(sim.data.time) - initial_sim_time
    return _json_finite(dict(
        success=bool(task_success and clean), task_success=task_success, contact_clean=clean,
        stop_reason=stop_reason, runtime_error=error,
        requested_duration_s=duration, simulation_seconds=float(sim.data.time) - initial_sim_time,
        wall_seconds=time.monotonic() - started,
        control_hz=sim.control_hz, action_hold_s=sim.dt, physics_timestep_s=sim.model.opt.timestep,
        completed_control_steps=completed_steps, attempted_control_steps=attempted_steps,
        selected_actions=selected_actions, physics_checks=monitor.checks,
        partial_final_control_interval=not math.isclose(elapsed / sim.dt, round(elapsed / sim.dt),
                                                       rel_tol=0, abs_tol=1e-8),
        policy_selection_wall_seconds=dict(count=len(times), mean=float(times.mean()) if len(times) else None,
                                           max=float(times.max()) if len(times) else None,
                                           p95=float(np.quantile(times, .95)) if len(times) else None),
        final_metrics=final_metrics,
        continuous_success_seconds=stability.continuous_success_seconds,
        success_required_seconds=stability.required_seconds,
        success_physics_samples=stability.samples,
        last_action=last_action, contacts=contacts,
        video=str(video_path.name) if video_path is not None else None,
        video_frames=video_frames, video_layout='head_left_wrist_right' if video_path else None,
        execution='LeRobot policy closed loop; absolute joint/aperture targets held 100 ms; no expert or planner',
    ))


def run_rollouts(*, checkpoint, config=None, output=None, position=None, positions_config=None,
                 episodes=1, seed=0, duration=60.0, device='auto', n_action_steps=1,
                 viewer=False, realtime=False, video=True, stop_on_contact=True, quiet=False):
    """Load one checkpoint, evaluate fixed layouts, and save reviewable results."""
    from mujoco_tools.lerobot_policy import LeRobotPolicy

    if isinstance(episodes, bool) or int(episodes) != episodes or episodes < 1:
        raise ValueError('episodes must be a positive integer (per selected position)')
    if isinstance(seed, bool) or int(seed) != seed or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    if not np.isfinite(duration) or duration <= 0 or not math.isclose(duration * 10, round(duration * 10), abs_tol=1e-8):
        raise ValueError('duration must be positive and a multiple of 0.1 seconds')
    cfg = load_config(config)
    layouts = select_layouts(cfg, position, positions_config)
    destination = (Path(output).expanduser().resolve() if output is not None else
                   OUTPUTS_DIR / 'insert_plug' / 'rollouts' / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f'))
    if destination.exists():
        raise FileExistsError(f'Rollout output already exists; choose a new directory: {destination}')
    policy = LeRobotPolicy(checkpoint, device=device, n_action_steps=n_action_steps)
    for _, layout_cfg in layouts:
        validate_contract(layout_cfg, policy)
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    interrupted = False
    with TemporaryDirectory(prefix='insert-plug-policy-') as temporary:
        for layout_name, layout_cfg in layouts:
            scene = build_scene(layout_cfg, Path(temporary) / f'{layout_name}.xml')
            for repeat in range(int(episodes)):
                episode_index = len(rows)
                episode_seed = (int(seed) + episode_index) % 2**32
                episode_dir = destination / f'episode_{episode_index:06d}'
                episode_dir.mkdir()
                shutil.copy2(scene, episode_dir / 'scene.xml')
                _seed_policy(episode_seed)
                sim = Simulation(scene, layout_cfg, render=True)
                try:
                    if not quiet:
                        print(json.dumps(dict(event='rollout_start', position=layout_name,
                                              episode=episode_index, seed=episode_seed)), flush=True)
                    result = run_episode(sim, policy, duration=duration,
                                         video_path=episode_dir / 'rollout.mp4' if video else None,
                                         viewer=viewer, realtime=realtime,
                                         stop_on_contact=stop_on_contact, quiet=quiet)
                finally:
                    sim.close()
                result.update(episode_index=episode_index, seed=episode_seed, position=layout_name,
                              repetition=repeat, config=layout_cfg, policy=policy.metadata(),
                              stop_on_contact=stop_on_contact, scene='scene.xml')
                report_path = episode_dir / 'result.json'
                report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                                       encoding='utf-8')
                rows.append(dict(episode_index=episode_index, position=layout_name,
                                 success=result['success'], task_success=result['task_success'],
                                 contact_clean=result['contact_clean'], stop_reason=result['stop_reason'],
                                 simulation_seconds=result['simulation_seconds'],
                                 unsafe_contact_count=result['contacts']['unsafe_contact_count'],
                                 report=str(report_path.relative_to(destination)),
                                 video=str((episode_dir / 'rollout.mp4').relative_to(destination)) if video else None))
                if not quiet:
                    print(json.dumps(dict(event='rollout_end', **rows[-1])), flush=True)
                summary = dict(output=str(destination), checkpoint=policy.metadata()['checkpoint'],
                               policy=policy.metadata(), episodes_requested=len(layouts) * int(episodes),
                               episodes_completed=len(rows), successes=sum(row['success'] for row in rows),
                               success_rate=sum(row['success'] for row in rows) / len(rows),
                               control_hz=10, physics_hz=1000, stop_on_contact=stop_on_contact,
                               results=rows)
                summary['complete'] = len(rows) == summary['episodes_requested']
                summary['success'] = bool(summary['complete'] and summary['successes'] == len(rows))
                (destination / 'summary.json').write_text(
                    json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
                if result['stop_reason'] in {'interrupted', 'viewer_closed'}:
                    interrupted = True
                    break
            if interrupted:
                break
    return summary
