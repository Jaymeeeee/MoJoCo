"""Ten paired closed-loop rollouts, gated on physically verified baseline lifts.

This is a separate experiment from the initial-chunk translation diagnostic.
Only the screw and its source holder are translated; actions are absolute
joint targets, never sums of joint angles or overlapping predicted chunks.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np

from mujoco_tools.paths import WORKSPACE_ROOT
from .translation_probe import prepare_scenes, sha256, translation_conditions, write_json


def rng_state(torch):
    return dict(cpu=torch.get_rng_state().cpu().numpy().copy(),
                cuda=np.asarray([x.cpu().numpy().copy() for x in torch.cuda.get_rng_state_all()],
                                dtype=np.uint8))


def run_condition(adapter, scene, directory, seed, steps):
    import imageio.v2 as imageio
    import mujoco
    from ..environment.simulation import Simulation
    from .diagnose_lift import LiftTrace
    from .grasp_translation_metrics import GraspLiftMonitor
    from .rollout import _seed_policy, validate_contract

    directory.mkdir(parents=True, exist_ok=False)
    sim = Simulation(scene['xml'], scene['config'], render=True)
    validate_contract(scene['config'], adapter)
    adapter.reset()
    _seed_policy(seed)
    initial_rng = rng_state(adapter._torch)
    monitor = GraspLiftMonitor()
    snapshot = copy.copy(sim)
    snapshot.data = mujoco.MjData(sim.model)
    trace = LiftTrace(sim, adapter)
    source_rim_z = float(sim.site('source_hole')[2])
    rows, chunk_rng = [], []
    active_index = -1

    def sample():
        mujoco.mj_copyData(snapshot.data, sim.model, sim.data)
        mujoco.mj_forward(sim.model, snapshot.data)
        tcp = snapshot.site('tcp')
        rotation = snapshot.data.site('tcp').xmat.copy().reshape(3, 3)
        tip = snapshot.site('peg_tip')
        dual = set(snapshot.gripper_contacts()) == {'left', 'right'}
        stamp = float(snapshot.data.time)
        monitor.sample(stamp, tcp, rotation, tip, source_rim_z, dual, active_index)
        rows.append((stamp, tcp, rotation, tip, dual, active_index))

    def predict(batch, **kwargs):
        before = rng_state(adapter._torch)
        # Diffusion select_action forwards noise=None explicitly. Preserve its
        # exact call, including kwargs, without changing the older ACT trace.
        result = trace.original_predict(batch, **kwargs)
        trace.chunks.append(dict(time=float(sim.data.time), qpos=sim.data.qpos.copy(),
                                 normalized=result.detach().cpu().numpy().copy()))
        after = rng_state(adapter._torch)
        chunk_rng.append((before, after))
        return result

    trace.install()
    adapter.policy.predict_action_chunk = predict
    sim.safety_callback = sample
    writer = imageio.get_writer(str(directory / 'rollout.mp4'), fps=sim.control_hz,
                               codec='libx264', quality=8, macro_block_size=1,
                               ffmpeg_log_level='error')
    started = time.perf_counter()
    failure = None
    video_frames = 0
    observation_chunks = {}
    try:
        reset_observation = sim.observe()
        for key, value in scene['observation'].items():
            np.testing.assert_array_equal(reset_observation[key], value)
        sample()
        for active_index in range(steps):
            observation = sim.observe()
            writer.append_data(np.concatenate([observation['observation.images.head'],
                                               observation['observation.images.wrist']], axis=1))
            video_frames += 1
            if active_index % adapter.n_action_steps == 0:
                for key, value in observation.items():
                    observation_chunks.setdefault(key, []).append(np.asarray(value).copy())
            action = adapter.select_action(observation)
            sim.step(action)
        observation = sim.observe()
        writer.append_data(np.concatenate([observation['observation.images.head'],
                                           observation['observation.images.wrist']], axis=1))
        video_frames += 1
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
    finally:
        final_rng = rng_state(adapter._torch)
        trace.uninstall()
        sim.safety_callback = None
        writer.close()

    try:
        arrays = trace.arrays() if trace.rows else {}
        arrays.update(physics_time=np.asarray([r[0] for r in rows]),
                      physics_tcp=np.asarray([r[1] for r in rows]),
                      physics_tcp_rotation=np.asarray([r[2] for r in rows]),
                      physics_peg_tip=np.asarray([r[3] for r in rows]),
                      physics_dual_contact=np.asarray([r[4] for r in rows], dtype=bool),
                      physics_action_index=np.asarray([r[5] for r in rows]),
                      source_rim_z=np.asarray(source_rim_z), delta=scene['delta'],
                      source_xy=np.asarray(scene['config']['board']['source_xy']),
                      initial_qpos=scene['state']['qpos'], initial_qvel=scene['state']['qvel'],
                      initial_ctrl=scene['state']['ctrl'], initial_tcp=scene['state']['tcp'],
                      rng_start=initial_rng['cpu'], rng_end=final_rng['cpu'],
                      cuda_rng_start=initial_rng['cuda'], cuda_rng_end=final_rng['cuda'])
        for label, offset in (('start', 0), ('end', 1)):
            for device in ('cpu', 'cuda'):
                arrays[f'chunk_rng_{device}_{label}'] = np.asarray([r[offset][device] for r in chunk_rng])
        np.savez_compressed(directory / 'trace.npz', **arrays)
        np.savez_compressed(directory / 'chunk_observations.npz', **{
            key: np.asarray(values) for key, values in observation_chunks.items()})
        prefix_error = None
        if trace.rows:
            prefix = arrays['chunk_action'][:, :adapter.n_action_steps].reshape(-1, 8)[:len(arrays['action'])]
            prefix_error = float(np.max(np.abs(prefix - arrays['action'])))
            if prefix_error != 0:
                failure = failure or f'Captured chunk prefix differs from executed actions: {prefix_error}'
        result = dict(seed=seed, delta_m=scene['delta'].tolist(),
                      source_xy=scene['config']['board']['source_xy'],
                      target_xy=scene['config']['board']['target_xy'],
                      within_training_source_bounds=scene['within_training_source_bounds'],
                      requested_control_steps=steps, completed_control_steps=len(trace.rows),
                      simulation_seconds=float(sim.data.time), wall_seconds=time.perf_counter()-started,
                      runtime_error=failure, qualification=monitor.report(),
                      actual_model_forward_calls=len(trace.chunks),
                      captured_chunk_length=(int(arrays['chunk_action'].shape[1]) if trace.rows else None),
                      executed_prefix_length=adapter.n_action_steps,
                      chunk_prefix_vs_executed_action_max_abs=prefix_error,
                      video_frames=video_frames, policy=adapter.metadata(),
                      trace='trace.npz', video='rollout.mp4', chunk_observations='chunk_observations.npz')
        write_json(directory / 'result.json', result)
        print(json.dumps(dict(event='episode_complete', model=adapter.policy_type,
                              condition=directory.name, **{k: result[k] for k in
                              ('simulation_seconds', 'runtime_error', 'qualification')})), flush=True)
        return result, arrays
    finally:
        sim.close()


def compare(base, shifted, base_result, shifted_result, delta):
    from .grasp_translation_metrics import paired_grasp_metrics
    for key in ('rng_start', 'rng_end', 'cuda_rng_start', 'cuda_rng_end',
                'chunk_rng_cpu_start', 'chunk_rng_cpu_end', 'chunk_rng_cuda_start', 'chunk_rng_cuda_end'):
        np.testing.assert_array_equal(base[key], shifted[key], err_msg=key)
    for key in ('initial_qvel', 'initial_ctrl', 'initial_tcp', 'after_time'):
        np.testing.assert_array_equal(base[key], shifted[key], err_msg=key)
    base_event = base_result['qualification']['grasp_event']
    shifted_contact_event = shifted_result['qualification']['grasp_event']
    shifted_event = (shifted_contact_event if shifted_result['qualification']['passed'] else None)
    metrics = paired_grasp_metrics(base['target_tcp'], shifted['target_tcp'],
                                  base['after_tcp'], shifted['after_tcp'], delta,
                                  base_event, shifted_event)
    i = base_event['confirmed_action_index']
    source = shifted['source_xy']
    metrics.update(full_rng_stream_paired=True,
                   same_initial_robot_state_verified=True,
                   shifted_lift_verified=shifted_result['qualification']['passed'],
                   shifted_stable_contact_detected=shifted_contact_event is not None,
                   independent_grasps_lift_qualified=shifted_event is not None,
                   baseline_grasp_action_index=i,
                   baseline_grasp_control_endpoint_time_s=float(base['after_time'][i]),
                   matched_time_shifted_target_xy_error_mm=float(np.linalg.norm(shifted['target_tcp'][i, :2]-source)*1000),
                   matched_time_shifted_actual_xy_error_mm=float(np.linalg.norm(shifted['after_tcp'][i, :2]-source)*1000),
                   baseline_target_xy_error_mm=float(np.linalg.norm(base['target_tcp'][i, :2]-base['source_xy'])*1000),
                   baseline_actual_xy_error_mm=float(np.linalg.norm(base['after_tcp'][i, :2]-base['source_xy'])*1000))
    if shifted_event:
        j = shifted_event['confirmed_action_index']
        metrics.update(independent_grasp_shifted_actual_xy_error_mm=float(
            np.linalg.norm(shifted['after_tcp'][j, :2]-source)*1000))
    return metrics


def run(args):
    import torch
    from mujoco_tools.lerobot_policy import LeRobotPolicy, resolve_checkpoint

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = {m: resolve_checkpoint(p) for m, p in
                   (('act', args.act_checkpoint), ('diffusion', args.diffusion_checkpoint))}
    source_paths = [Path(__file__), Path(__file__).with_name('grasp_translation_metrics.py'),
                    Path(__file__).with_name('diagnose_lift.py'),
                    Path(__file__).with_name('translation_probe.py'),
                    Path(__file__).with_name('rollout.py'),
                    WORKSPACE_ROOT/'src/mujoco_tools/lerobot_policy.py',
                    WORKSPACE_ROOT/'src/mujoco_tools/robotiq.py',
                    WORKSPACE_ROOT/'examples/insert_plug/environment/simulation.py',
                    WORKSPACE_ROOT/'examples/insert_plug/environment/scene.py',
                    WORKSPACE_ROOT/'third_party/lerobot/src/lerobot/policies/act/modeling_act.py',
                    WORKSPACE_ROOT/'third_party/lerobot/src/lerobot/policies/diffusion/modeling_diffusion.py',
                    WORKSPACE_ROOT/'config/insert_plug.yaml',
                    WORKSPACE_ROOT/'config/insert_plug_positions.yaml']
    # Store exact code/config/checkpoint provenance before any model outcomes.
    protected = {str(p): sha256(p) for p in source_paths if p.is_file()}
    protected.update({str(p): sha256(p) for ckpt in checkpoints.values()
                      for p in ckpt.iterdir() if p.is_file()})
    old_summary = WORKSPACE_ROOT/'outputs/insert_plug/translation_probe/act_diffusion_insert_plug_v1_rgbd_initial_open_loop_20261110_145234/summary.json'
    if old_summary.is_file():
        protected[str(old_summary)] = sha256(old_summary)
    protocol = dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        position='pos2', selection='Chosen before shifted rollouts: both policies previously 10/10 successful at pos2; seed20 successful for both.',
        prior_evidence={model: str(WORKSPACE_ROOT/f'outputs/test/{model}_insert_plug_v1_rgbd_ind_20261110_100724/pos2/episode_000000/result.json') for model in checkpoints},
        seed=20, distance_m=.03, control_hz=10, physics_hz=1000, n_action_steps=32,
        horizon=64, control_steps=128, duration_s=12.8, max_episodes=10,
        baseline_gate='Run both baselines first; require physical lift in each before any shifted rollout.',
        grasp_definition='Left and right finger contact continuously for >=200 ms; peg-tip position relative to TCP drifts <=0.2 mm over that window (relative to final sample).',
        lift_definition='Same grasp criterion plus peg tip at least 5 mm above source rim throughout >=200 ms.',
        grasp_endpoint='Confirmed first stable grasp in the continuous-contact episode leading to verified lift; otherwise stable contact alone is explicitly unqualified.',
        action_semantics='Absolute 7 joint targets + aperture. Execute 32 returned targets before replanning. Concatenate only executed prefixes; no summing joint angles.',
        comparison='Signed projection of shifted-minus-baseline TCP onto source translation at baseline grasp control interval; report full time curves and independently timed grasps too.',
        randomness='Reset all policy/processor queues and full Python/NumPy/Torch CPU/CUDA seed20 for every condition; verify full per-replan RNG states.',
        scene_control='Translate screw and source holder together; target holder, initial robot, cameras and all other settings fixed.',
        limitations=['One shared layout and one seed: case study, not a success-rate estimate.',
                     'Closed-loop response includes later state feedback; it is not a frozen-state encoder ablation.',
                     'No verified lift within 12.8 s means window-limited non-success, not proven eventual failure.',
                     '+X and +Y source locations lie outside the training source-coordinate bounding box; inside the box is not strict interpolation.',
                     '30 mm is the expected displacement at equivalent grasp phases; during transfer the target holder remains fixed.',
                     'Small TCP response does not uniquely establish an internal spatial-representation deficit.'],
        protected_sha256=protected)
    write_json(output/'protocol.json', protocol)
    scenes = prepare_scenes(output, ['pos2'], .03)['pos2']
    adapters = {model: LeRobotPolicy(checkpoint, device='cuda', n_action_steps=32)
                for model, checkpoint in checkpoints.items()}
    episodes = {model: {} for model in adapters}
    arrays = {model: {} for model in adapters}
    summary = dict(protocol='protocol.json', status='running_baselines', episodes=episodes, pairs={})
    write_json(output/'summary.json', summary)
    for model, adapter in adapters.items():
        result, raw = run_condition(adapter, scenes['baseline'], output/'episodes'/model/'baseline', 20, 128)
        episodes[model]['baseline'] = result
        arrays[model]['baseline'] = raw
        write_json(output/'summary.json', summary)
    gate = all(episodes[m]['baseline']['qualification']['passed'] and
               episodes[m]['baseline']['runtime_error'] is None for m in episodes)
    summary['baseline_gate_passed'] = gate
    if gate:
        summary['status'] = 'running_translations'
        for model, adapter in adapters.items():
            summary['pairs'][model] = {}
            for direction, delta in translation_conditions(.03)[1:]:
                result, raw = run_condition(adapter, scenes[direction], output/'episodes'/model/direction, 20, 128)
                episodes[model][direction] = result
                if result['runtime_error'] is not None:
                    summary['status'] = 'runtime_error'
                    write_json(output/'summary.json', summary)
                    raise RuntimeError(result['runtime_error'])
                summary['pairs'][model][direction] = compare(arrays[model]['baseline'], raw,
                    episodes[model]['baseline'], result, delta)
                write_json(output/'summary.json', summary)
        summary['status'] = 'complete'
    else:
        summary['status'] = 'baseline_gate_failed_no_translated_rollouts'
    summary['protected_hashes_unchanged'] = all(sha256(Path(p)) == h for p, h in protected.items())
    if not summary['protected_hashes_unchanged']:
        raise RuntimeError('Protected file changed')
    write_json(output/'summary.json', summary)
    print(json.dumps(dict(event='finished', status=summary['status'], baseline_gate_passed=gate)), flush=True)
    return 0 if gate else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--act-checkpoint', required=True, type=Path)
    parser.add_argument('--diffusion-checkpoint', required=True, type=Path)
    args = parser.parse_args()
    from examples.insert_plug.cli import _configure_gl
    _configure_gl('egl')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
