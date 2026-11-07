#!/usr/bin/env python3
"""Physically execute saved 10 Hz targets, with no expert or state playback.

Each target is held for 100 ms: one interval for aligned 10 Hz control, or five
intervals for legacy 50 Hz control. The last legacy hold is truncated at the
original rollout end. Only the recorded scene's normal reset initializes the
world; all subsequent motion goes through step().
Reports belong under outputs/_checks and source datasets are opened read-only.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))


def load_episode(path: Path, action_dtype='float32') -> dict:
    """Read only small control/state arrays, never decode dataset RGB/depth."""
    with h5py.File(path, 'r') as handle:
        metadata = json.loads(handle.attrs['metadata_json'])
        if not handle.attrs.get('complete') or not handle.attrs.get('success'):
            raise ValueError('Expected a complete successful source episode')
        arrays = {key: handle[key][:] for key in (
            'action', 'timestamp', 'simulation/qpos', 'simulation/qvel',
            'observations/joint_position', 'privileged/peg_pose',
        )}
        arrays['phases'] = handle['phase'].asstr()[:].tolist()
    config = yaml.safe_load(path.with_suffix('.yaml').read_text())
    if config != metadata['config']:
        raise ValueError('YAML and source metadata differ')
    model_path = path.with_suffix('.xml')
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata['model_sha256']:
        raise ValueError('Source XML hash mismatch')
    sample_hz, control_hz, timestep = (float(metadata[key]) for key in
                                     ('sample_hz', 'control_hz', 'physics_timestep'))
    if not np.isfinite([sample_hz, control_hz, timestep]).all() or min(sample_hz, control_hz, timestep) <= 0:
        raise ValueError('Source frequencies and timestep must be finite and positive')
    if sample_hz != 10 or timestep != .001:
        raise ValueError('This check requires 10 Hz saved actions and 1 kHz physics')
    repeats = control_hz / sample_hz
    physics_steps = 1 / (control_hz * timestep)
    if not (math.isclose(repeats, round(repeats), rel_tol=0, abs_tol=1e-10) and repeats >= 1
            and math.isclose(physics_steps, round(physics_steps), rel_tol=0, abs_tol=1e-8)):
        raise ValueError('Saved action periods must contain whole control and physics intervals')
    repeats = round(repeats)
    if (config['task']['control_hz'] != control_hz
            or config['recording']['sample_hz'] != sample_hz
            or config['physics']['timestep'] != timestep):
        raise ValueError('YAML rates and source metadata differ')
    action_timing = ('active_control_target_at_observation; held_until_next_observation' if repeats == 1
                     else 'active_control_target_at_observation; may change before next observation')
    if metadata.get('action_timing') != action_timing:
        raise ValueError('Unexpected source action convention')
    if repeats == 1 and metadata.get('action_hold_duration_s') != 1 / sample_hz:
        raise ValueError('Aligned source action hold duration must be 100 ms')
    if metadata.get('sampling_timing') != 'nearest_physics_tick' or metadata.get('sample_origin_s') != 0:
        raise ValueError('Expected physics-tick samples anchored at control time zero')
    times = arrays['timestamp']
    if not len(times) or not np.allclose(times, np.arange(len(times)) / sample_hz, rtol=0, atol=1e-7):
        raise ValueError('Expected uninterrupted 10 Hz samples starting at zero')
    actions = arrays['action'].astype(action_dtype).astype(np.float64)
    if actions.shape != (len(times), 8) or not np.isfinite(actions).all():
        raise ValueError('Invalid action array')
    original_steps = metadata['execution_result']['steps']
    full_intervals = len(actions) * repeats
    if not isinstance(original_steps, int) or not (full_intervals - repeats < original_steps <= full_intervals):
        raise ValueError('Recorded execution duration disagrees with sampled actions')
    return dict(path=path, model_path=model_path, config=config, metadata=metadata,
                actions=actions, arrays=arrays, control_steps_per_action=repeats,
                action_hold_s=1 / sample_hz)


def replay_episode(path, *, action_dtype='float32', video=None):
    import mujoco
    from examples.insert_plug.action_replay_checks import ReplayContactMonitor
    from examples.insert_plug.simulation import Simulation

    started = time.monotonic()
    source = load_episode(Path(path), action_dtype)
    meta, arrays, actions = source['metadata'], source['arrays'], source['actions']
    if meta['mujoco_version'] != mujoco.__version__:
        raise ValueError(f'MuJoCo version differs: {mujoco.__version__} vs recorded {meta["mujoco_version"]}')
    sim = Simulation(source['model_path'], source['config'], render=video is not None)
    writer = None
    try:
        initial_errors = {
            'qpos_max_abs': float(np.max(np.abs(sim.data.qpos - arrays['simulation/qpos'][0]))),
            'qvel_max_abs': float(np.max(np.abs(sim.data.qvel - arrays['simulation/qvel'][0]))),
            'time_abs_s': float(abs(sim.data.time - arrays['timestamp'][0])),
        }
        if max(initial_errors.values()) > 1e-12:
            raise ValueError(f'Normal scene reset differs from recorded initial state: {initial_errors}')
        monitor = ReplayContactMonitor(sim)
        repeats = source['control_steps_per_action']
        if not (math.isclose(sim.dt * repeats, source['action_hold_s'], rel_tol=0, abs_tol=1e-10)
                and sim.control_hz == meta['control_hz']
                and math.isclose(sim.model.opt.timestep, meta['physics_timestep'], rel_tol=0, abs_tol=1e-12)):
            raise ValueError('Compiled scene control/physics timing differs from source metadata')
        original_steps = int(meta['execution_result']['steps'])
        total_steps = original_steps
        stable_ticks = []
        snapshot = copy.copy(sim)
        snapshot.data = mujoco.MjData(sim.model)
        first_stable_tick = round(total_steps * sim.dt / sim.model.opt.timestep) - 250

        def physics_check():
            monitor()
            tick = round(sim.data.time / sim.model.opt.timestep)
            if tick >= first_stable_tick:
                # Inspect a separate state: never refresh or alter the live
                # solver just to compute a success metric at an interior tick.
                mujoco.mj_copyData(snapshot.data, sim.model, sim.data)
                mujoco.mj_forward(sim.model, snapshot.data)
                stable_ticks.append(bool(snapshot.metrics()['success']))

        sim.safety_callback = physics_check
        if video is not None:
            import imageio.v2 as imageio
            video = Path(video)
            video.parent.mkdir(parents=True, exist_ok=True)
            if video.exists():
                raise FileExistsError(video)
            writer = imageio.get_writer(str(video), fps=meta['sample_hz'], codec='libx264',
                                       quality=8, macro_block_size=1, ffmpeg_log_level='error')

        metrics_history, phase_ends = [], []
        max_joint_deviation = max_peg_deviation = 0.
        contacts_seen = False
        extracted = False
        bilateral_run = max_bilateral_run = 0
        lost_while_carrying = 0
        duration_at_original_end = None
        failure = None
        completed_steps = 0

        def endpoint_report():
            # Include both endpoints of the final 250 ms at 1 kHz (251 states).
            stable = len(stable_ticks) == 251 and all(stable_ticks)
            legacy_count = max(1, int(round(.25 / sim.dt)))
            legacy_stable = len(metrics_history) >= legacy_count and all(metrics_history[-legacy_count:])
            return dict(time_s=float(sim.data.time), metrics=sim.metrics(),
                        stable_for_final_250ms=stable,
                        final_stability_physics_samples=len(stable_ticks),
                        legacy_expert_stability_check=legacy_stable,
                        final_success=bool(stable and sim.metrics()['success']))

        for index, action in enumerate(actions):
            max_joint_deviation = max(max_joint_deviation, float(np.max(np.abs(
                sim.data.qpos[sim.arm_qadr] - arrays['observations/joint_position'][index]))))
            max_peg_deviation = max(max_peg_deviation, float(np.linalg.norm(
                sim.data.xpos[sim.peg_id] - arrays['privileged/peg_pose'][index, :3])))
            if writer is not None:
                images = sim.render_images()
                writer.append_data(np.concatenate([images['head'], images['wrist']], axis=1))
            for _ in range(repeats):
                if completed_steps == total_steps:
                    break
                try:
                    sim.step(action)
                except (RuntimeError, ValueError) as error:
                    failure = f'{type(error).__name__}: {error}'
                    break
                completed_steps += 1
                metrics = sim.metrics()
                metrics_history.append(bool(metrics['success']))
                sides = sim.gripper_contacts()
                contacts_seen = contacts_seen or len(sides) == 2
                bilateral_run = bilateral_run + 1 if len(sides) == 2 else 0
                max_bilateral_run = max(max_bilateral_run, bilateral_run)
                if sim.site('peg_tip')[2] > max(sim.site('source_hole')[2], sim.site('target_hole')[2]) + .015:
                    extracted = True
                # A recorded phase is a diagnostic label only; it never affects
                # commands, collision permissions, or progression.
                if arrays['phases'][index] in ('extract', 'transfer', 'align', 'shallow_insert', 'insertion_check') and len(sides) != 2:
                    lost_while_carrying += 1
                if completed_steps == original_steps:
                    duration_at_original_end = endpoint_report()
            if failure:
                break
            if index == len(actions) - 1 or arrays['phases'][index + 1] != arrays['phases'][index]:
                phase_ends.append(dict(source_phase=arrays['phases'][index],
                                       replay_time_s=float(sim.data.time),
                                       gripper_contacts=sorted(sim.gripper_contacts()),
                                       metrics=sim.metrics()))
        final = endpoint_report()
        contact_report = monitor.summary()
        physical_success = bool(final['final_success'] and failure is None and completed_steps == total_steps)
        clean = contact_report['unsafe_contact_count'] == 0
        return dict(
            source_hdf5=str(Path(path).resolve().relative_to(ROOT)),
            position=Path(path).parent.parent.name, episode=Path(path).stem,
            source_success=True, action_dtype=action_dtype, frames=len(actions),
            control_hz=sim.control_hz, physics_hz=1 / sim.model.opt.timestep,
            action_hz=float(meta['sample_hz']), control_steps_per_action=repeats,
            source_action_timing=meta['action_timing'], action_hold_s=source['action_hold_s'],
            execution='zero_order_hold; no planner, feedback correction, interpolation or state restoration',
            initial_state_errors=initial_errors,
            source_simulation_duration_s=original_steps * sim.dt,
            replay_simulation_duration_s=float(sim.data.time),
            last_action_extension_s=0.,
            last_action_hold_s=(total_steps - (len(actions) - 1) * repeats) * sim.dt,
            expected_control_steps=total_steps, completed_control_steps=completed_steps,
            task_success=physical_success, contact_clean=clean,
            success=bool(physical_success and clean), runtime_error=failure,
            bilateral_contact_seen=contacts_seen,
            longest_bilateral_contact_s=max_bilateral_run * sim.dt,
            extraction_clearance_reached=extracted,
            missing_bilateral_steps_in_source_carry_phases=lost_while_carrying,
            max_joint_deviation_from_source_rad=max_joint_deviation,
            max_peg_deviation_from_source_m=max_peg_deviation,
            at_original_end=duration_at_original_end, final=final,
            contacts=contact_report, source_phase_end_diagnostics=phase_ends,
            video=str(video) if video is not None else None,
            wall_seconds=time.monotonic() - started,
        )
    finally:
        if writer is not None:
            writer.close()
        sim.close()


def summarize(results):
    groups = {}
    for position in sorted({r['position'] for r in results}, key=lambda name: int(name[3:])):
        rows = [r for r in results if r['position'] == position]
        reasons = Counter()
        for row in rows:
            reasons.update(row['contacts']['violations_by_reason'])
        groups[position] = dict(episodes=len(rows), passed=sum(r['success'] for r in rows),
                                task_successes=sum(r['task_success'] for r in rows),
                                contact_clean=sum(r['contact_clean'] for r in rows),
                                contact_reasons=dict(reasons))
    return dict(episodes=len(results), passed=sum(r['success'] for r in results),
                task_successes=sum(r['task_success'] for r in results),
                contact_clean=sum(r['contact_clean'] for r in results), groups=groups)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'outputs/datasets/insert_plug')
    parser.add_argument('--episode', type=Path, action='append', help='Check selected source HDF5 files only')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--action-dtype', choices=('float32', 'float64'), default='float32',
                        help='float32 matches exported LeRobot actions; float64 diagnoses source precision')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/_checks' /
                        f'insert_plug_action_replay_{datetime.now():%Y%m%d_%H%M%S}' / 'report.json')
    parser.add_argument('--video', type=Path, help='Optional head+wrist video; requires one --episode and --workers 1')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('workers must be positive')
    if args.video and (not args.episode or len(args.episode) != 1 or args.workers != 1):
        parser.error('--video requires exactly one --episode and --workers 1')
    episodes = args.episode or [p for index in range(10)
                               for p in sorted((args.root / f'pos{index}').glob('episode_*/*.h5'))]
    if not episodes or (not args.episode and len(episodes) != 100):
        parser.error('Expected 100 source episodes, or explicit --episode paths')
    if args.output.exists():
        parser.error('Output report already exists; choose a new path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(check='10hz_saved_action_physics_replay',
                  note='Saved targets are held for 100 ms. This is not learned-policy evaluation.',
                  contact_time_convention='Post-step timestamp; contact geometry is evaluated at the preceding 1 ms integration start.',
                  phase_diagnostic_convention='Ends of held 10 Hz source labels, not original expert stage boundaries.',
                  action_dtype=args.action_dtype, complete=False, errors=[], results=[])

    def save():
        report['summary'] = summarize(report['results'])
        temporary = args.output.with_suffix('.json.partial')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.output)

    def completed(result):
        report['results'].append(result)
        save()
        print(json.dumps({k: result[k] for k in ('position', 'episode', 'success', 'task_success',
                                               'contact_clean', 'wall_seconds')}), flush=True)

    if args.workers == 1:
        for path in episodes:
            completed(replay_episode(path, action_dtype=args.action_dtype, video=args.video))
    else:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=multiprocessing.get_context('spawn')) as executor:
            futures = {executor.submit(replay_episode, path, action_dtype=args.action_dtype): path for path in episodes}
            for future in as_completed(futures):
                try:
                    completed(future.result())
                except Exception as error:
                    report['errors'].append(dict(episode=str(futures[future]), error=f'{type(error).__name__}: {error}'))
                    save()
    report['results'].sort(key=lambda row: row['episode'])
    report['complete'] = len(report['results']) == len(episodes) and not report['errors']
    save()
    print(json.dumps(dict(report=str(args.output), complete=report['complete'], **report['summary']),
                     ensure_ascii=False), flush=True)
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
