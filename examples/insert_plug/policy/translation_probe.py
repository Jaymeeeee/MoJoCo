"""Paired source-assembly translation probe for trained RGB-D ACT/Diffusion.

Move the screw AND its source holder, keeping the target holder, robot reset,
cameras, appearance and policy noise fixed. Predict a complete initial chunk,
map absolute joint targets through FK, and optionally execute that chunk open
loop. Neither FK targets nor aperture crossings establish successful grasping.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np

from mujoco_tools.paths import WORKSPACE_ROOT


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def translation_conditions(distance):
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError('distance must be positive and finite')
    return [('baseline', np.zeros(3))] + [
        (f'{axis}_{sign}', np.eye(3)[index] * distance * multiplier)
        for index, axis in enumerate(('x', 'y'))
        for sign, multiplier in (('plus', 1), ('minus', -1))]


def shifted_config(base, delta):
    """Scene builder positions both the source holder and free screw here."""
    result = copy.deepcopy(base)
    delta = np.asarray(delta, dtype=float)
    if delta.shape != (3,) or not np.isfinite(delta).all() or delta[2] != 0:
        raise ValueError('This probe requires a finite XY translation')
    result['board']['source_xy'] = (np.asarray(base['board']['source_xy']) + delta[:2]).tolist()
    return result


def predict_chunk(adapter, observation, seed):
    """Reset all queues and pair the entire RNG stream, including DDPM steps."""
    torch = adapter._torch
    adapter.reset()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    start_rng = torch.get_rng_state().cpu().numpy().copy()
    start_cuda = [value.cpu().numpy().copy() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    start = time.perf_counter()
    with torch.inference_mode():
        prepared = adapter.preprocessor(adapter._prepare_observation(observation))
        batch = {key: prepared[key] for key in adapter.input_features}
        if adapter.policy_type == 'diffusion':
            if adapter.n_obs_steps != 1:
                raise ValueError('Initial-state probe currently supports n_obs_steps=1 only')
            batch['observation.state'] = batch['observation.state'].unsqueeze(1)
        normalized = adapter.policy.predict_action_chunk(batch)
        expected = (1, adapter.chunk_size, 8)
        if tuple(normalized.shape) != expected or not bool(torch.isfinite(normalized).all()):
            raise ValueError(f'Expected finite full chunk {expected}, got {normalized.shape}')
        normalized_array = normalized[0].detach().cpu().float().numpy().copy()
        physical = adapter.postprocessor(normalized.reshape(-1, 8).clone())
        physical_array = physical.detach().cpu().float().numpy().copy()
    if physical_array.shape != expected[1:] or not np.isfinite(physical_array).all():
        raise ValueError('Invalid decoded action chunk')
    end_rng = torch.get_rng_state().cpu().numpy().copy()
    end_cuda = [value.cpu().numpy().copy() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    return dict(actions=physical_array, normalized=normalized_array, rng_start=start_rng, rng_end=end_rng,
                cuda_rng_start=np.asarray(start_cuda, dtype=np.uint8), cuda_rng_end=np.asarray(end_cuda, dtype=np.uint8),
                seconds=time.perf_counter() - start)


def forward_kinematics(sim, actions):
    import mujoco
    data = mujoco.MjData(sim.model)
    data.qpos[:] = sim.data.qpos
    positions, rotations = [], []
    for action in actions:
        data.qpos[sim.arm_qadr] = action[:7]
        mujoco.mj_kinematics(sim.model, data)
        positions.append(data.site('tcp').xpos.copy())
        rotations.append(data.site('tcp').xmat.copy().reshape(3, 3))
    return np.asarray(positions), np.asarray(rotations)


def replay_chunk(sim, actions):
    """Execute absolute targets at unchanged 10 Hz / 1 kHz, without replanning."""
    sim.reset()
    tcp, qpos, aperture, peg, contacts = [], [], [], [], []
    failure = None
    for index, action in enumerate(actions):
        try:
            sim.step(action)
        except (ValueError, RuntimeError) as error:
            failure = dict(query=index, error=f'{type(error).__name__}: {error}')
            break
        tcp.append(sim.site('tcp'))
        qpos.append(sim.data.qpos.copy())
        aperture.append(sim.gripper_width())
        peg.append(sim.site('peg_head'))
        contacts.append(set(sim.gripper_contacts()) == {'left', 'right'})
    return dict(tcp=np.asarray(tcp).reshape(-1, 3), qpos=np.asarray(qpos), aperture=np.asarray(aperture),
                peg=np.asarray(peg).reshape(-1, 3), dual_contact=np.asarray(contacts, dtype=bool), failure=failure)


def _preview(path, observations):
    from PIL import Image, ImageDraw
    width, height = 256, 256
    canvas = Image.new('RGB', (5 * width, 2 * height + 58), 'white')
    draw = ImageDraw.Draw(canvas)
    for column, (name, observation) in enumerate(observations.items()):
        draw.text((column * width + 8, 8), name, fill='black')
        for row, camera in enumerate(('head', 'wrist')):
            canvas.paste(Image.fromarray(observation[f'observation.images.{camera}']), (column * width, 30 + row * (height + 14)))
    canvas.save(path)


def prepare_scenes(output, positions, distance):
    import mujoco
    import yaml
    from ..environment.scene import build_scene, load_config
    from ..environment.simulation import Simulation
    from ..environment.layouts import head_visibility

    base_config = load_config()
    manifest = yaml.safe_load((WORKSPACE_ROOT / 'config/insert_plug_positions.yaml').read_text())
    selected = [row for row in manifest['positions'] if row['name'] in positions]
    if len(selected) != len(positions):
        raise ValueError('Unknown or duplicate position name')
    scenes = {}
    training_sources = np.asarray([row['source_xy'] for row in manifest['positions']])
    for position in selected:
        name = position['name']
        directory = output / 'scenes' / name
        directory.mkdir(parents=True)
        config = copy.deepcopy(base_config)
        config['board'].update({key: position[key] for key in ('source_xy', 'target_xy')})
        scenes[name] = {}
        initial = None
        observations = {}
        for direction, delta in translation_conditions(distance):
            shifted = shifted_config(config, delta)
            scene_path = build_scene(shifted, directory / f'{direction}.xml')
            sim = Simulation(scene_path, shifted, render=True)
            try:
                observation = sim.observe()
                state = dict(qpos=sim.data.qpos.copy(), qvel=sim.data.qvel.copy(), ctrl=sim.data.ctrl.copy(),
                             tcp=sim.site('tcp'), peg=sim.site('peg_head'), source=sim.site('source_hole'),
                             target=sim.site('target_hole'), camera_pos=sim.data.cam_xpos.copy(),
                             camera_mat=sim.data.cam_xmat.copy())
                if initial is None:
                    initial = state
                peg_adr = sim.model.jnt_qposadr[sim.model.joint('peg_free').id]
                expected_qpos = initial['qpos'].copy()
                expected_qpos[peg_adr:peg_adr + 3] += delta
                np.testing.assert_allclose(state['qpos'], expected_qpos, rtol=0, atol=1e-12)
                for key in ('qvel', 'ctrl', 'tcp', 'target', 'camera_pos', 'camera_mat'):
                    np.testing.assert_array_equal(state[key], initial[key])
                for key in ('peg', 'source'):
                    np.testing.assert_allclose(state[key] - initial[key], delta, rtol=0, atol=1e-12)
                if direction == 'baseline':
                    repeat = sim.observe()
                    for key in observation:
                        np.testing.assert_array_equal(observation[key], repeat[key])
                for camera in ('head', 'wrist'):
                    depth = observation[f'observation.depth.{camera}']
                    assert depth.dtype == np.float32 and np.isfinite(depth).all() and np.all(depth > 0)
                visibility = head_visibility(sim)
                assert visibility['objects']['peg_visual']['inside'], (name, direction, visibility)
                path = directory / f'{direction}.npz'
                np.savez_compressed(path, **observation, **state, delta=delta)
                write_json(directory / f'{direction}.json', dict(
                    config=shifted, delta_m=delta.tolist(), source_xy=shifted['board']['source_xy'],
                    target_xy=shifted['board']['target_xy'], visibility=visibility,
                    calibration=sim.camera_calibration(), observation_sha256=sha256(path),
                    invariants_verified=True, repeat_render_bitwise_equal=True if direction == 'baseline' else None))
                source = np.asarray(shifted['board']['source_xy'])
                inside_bounds = bool(np.all(source >= training_sources.min(axis=0) - 1e-12)
                                     and np.all(source <= training_sources.max(axis=0) + 1e-12))
                scenes[name][direction] = dict(config=shifted, xml=scene_path, observation=observation, state=state, delta=delta,
                                               within_training_source_bounds=inside_bounds)
                observations[direction] = observation
            finally:
                sim.close()
        _preview(directory / 'paired_views.png', observations)
        print(json.dumps(dict(event='scenes_ready', position=name)), flush=True)
    return scenes


def run_probe(args):
    import torch
    from mujoco_tools.lerobot_policy import LeRobotPolicy, resolve_checkpoint
    from ..environment.simulation import Simulation
    from .translation_metrics import trajectory_metrics

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = {name: resolve_checkpoint(path) for name, path in (('act', args.act_checkpoint), ('diffusion', args.diffusion_checkpoint)) if path}
    protected = {str(path): sha256(path) for checkpoint in checkpoints.values() for path in checkpoint.iterdir() if path.is_file()}
    protected.update({str(WORKSPACE_ROOT / path): sha256(WORKSPACE_ROOT / path) for path in ('config/insert_plug.yaml', 'config/insert_plug_positions.yaml')})
    runtime_sources = [Path(__file__), Path(__file__).with_name('translation_metrics.py'),
                       WORKSPACE_ROOT / 'src/mujoco_tools/lerobot_policy.py',
                       WORKSPACE_ROOT / 'examples/insert_plug/environment/simulation.py',
                       WORKSPACE_ROOT / 'examples/insert_plug/environment/scene.py',
                       WORKSPACE_ROOT / 'examples/insert_plug/environment/objects.py',
                       *[(WORKSPACE_ROOT / f'third_party/lerobot/src/lerobot/policies/{kind}/modeling_{kind}.py')
                         for kind in ('act', 'diffusion')]]
    protected.update({str(path): sha256(path) for path in runtime_sources})
    report = dict(experiment=dict(
        created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        intervention='Translate screw and source holder together; target holder and robot initial state fixed.',
        distance_m=args.distance, positions=args.positions, diffusion_seeds=args.seeds, act_seeds=[args.seeds[0]],
        condition='peg_and_holder', control_hz=10, full_chunk_steps=64, initial_observation_only=True,
        action_semantics='7 absolute joint targets (rad) + full aperture (m); never sum absolute joint angles',
        motion_measure='FK target difference between matched scenes; sum of successive Cartesian deltas telescopes to terminal minus initial TCP',
        noise_pairing='Reset policy and entire torch CPU/CUDA RNG before each condition, including all DDPM denoising noise',
        closure_proxy_width_m=args.closure_width, closure_proxy='First two consecutive commanded apertures <= threshold; not verified physical grasp',
        descriptive_tracking_criterion=dict(gain_interval=[.8, 1.2], maximum_orthogonal_m=.005,
                                            meaning='Descriptive 30mm tracking tolerance, not a statistical significance test'),
        replay_full_chunk=args.replay, replay_description='Open-loop execution of all 64 original targets; no policy replanning, no corrections or clipping',
        backend=os.environ.get('MUJOCO_GL'), device=args.device,
        limitations=[
            'This tests translation response of the initial action chunk, not closed-loop task success or a complete spatial representation diagnosis.',
            'Both screw and source holder move; their visual contributions cannot be separated by this intervention.',
            'Early queries need not shift by the full 30 mm, and the final query may already be in extraction.',
            'Aperture crossings identify intended closure, not actual grasp. Missing closure endpoints remain null.',
            'Seeds repeat model sampling at the same scene; they are not independent object layouts.',
            'Some shifted source locations may lie outside training coordinate bounds; report these separately.',
        ]), models={}, rows=[], protected_before=protected)
    write_json(output / 'experiment.json', report['experiment'])
    scenes = prepare_scenes(output, args.positions, args.distance)
    for model_name, checkpoint in checkpoints.items():
        adapter = LeRobotPolicy(checkpoint, device=args.device, n_action_steps=64)
        if adapter.policy_type != model_name:
            raise ValueError(f'{model_name} argument points to a {adapter.policy_type} checkpoint')
        if not adapter.requires_depth or adapter.chunk_size != 64 or adapter.temporal_ensemble_coeff is not None:
            raise ValueError('Expected a full 64-step RGB-D checkpoint without temporal ensembling')
        seeds = args.seeds if model_name == 'diffusion' else args.seeds[:1]
        report['models'][model_name] = dict(policy=adapter.metadata(), seeds=seeds, zero_shift_checks=[], queue_equivalence_checks=[])
        for position, conditions in scenes.items():
            simulations = {name: Simulation(entry['xml'], entry['config'], render=False) for name, entry in conditions.items()}
            try:
                for seed in seeds:
                    base = None
                    for direction, entry in conditions.items():
                        sim = simulations[direction]
                        sim.reset()
                        prediction = predict_chunk(adapter, entry['observation'], seed)
                        prediction['tcp'], prediction['rotation'] = forward_kinematics(sim, prediction['actions'])
                        if direction == 'baseline' and seed == seeds[0]:
                            duplicate = predict_chunk(adapter, entry['observation'], seed)
                            np.testing.assert_array_equal(duplicate['actions'], prediction['actions'])
                            np.testing.assert_array_equal(duplicate['rng_end'], prediction['rng_end'])
                            report['models'][model_name]['zero_shift_checks'].append(dict(position=position, seed=seed, max_action_difference=0.0))
                            if position == args.positions[0]:
                                adapter.reset()
                                torch.manual_seed(seed)
                                if torch.cuda.is_available():
                                    torch.cuda.manual_seed_all(seed)
                                queued = np.stack([adapter.select_action(entry['observation']) for _ in range(64)])
                                np.testing.assert_array_equal(queued, prediction['actions'])
                                report['models'][model_name]['queue_equivalence_checks'].append(dict(position=position, seed=seed, all_64_actions_bitwise_equal=True))
                        prediction['replay'] = replay_chunk(sim, prediction['actions']) if args.replay else None
                        directory = output / 'data' / model_name / position / f'seed_{seed:06d}'
                        directory.mkdir(parents=True, exist_ok=True)
                        raw_path = directory / f'{direction}.npz'
                        raw = {key: value for key, value in prediction.items() if key not in ('replay', 'seconds')}
                        if args.replay:
                            raw.update({f'realized_{key}': value for key, value in prediction['replay'].items() if key != 'failure'})
                        np.savez_compressed(raw_path, **raw)
                        write_json(directory / f'{direction}.json', dict(
                            inference_seconds=prediction['seconds'], archive_sha256=sha256(raw_path),
                            replay_failure=prediction['replay']['failure'] if args.replay else None))
                        if direction == 'baseline':
                            base = prediction
                        else:
                            for key in ('rng_start', 'rng_end', 'cuda_rng_start', 'cuda_rng_end'):
                                np.testing.assert_array_equal(base[key], prediction[key])
                            metrics = trajectory_metrics(base['tcp'], prediction['tcp'], base['actions'], prediction['actions'],
                                entry['delta'], entry['state']['tcp'], conditions['baseline']['state']['peg'], entry['state']['peg'],
                                control_hz=10., closure_width_m=args.closure_width)
                            paired_path = directory / f'{direction}_paired.npz'
                            paired = dict(baseline_tcp=base['tcp'], shifted_tcp=prediction['tcp'], delta=entry['delta'],
                                          baseline_actions=base['actions'], shifted_actions=prediction['actions'], initial_tcp=entry['state']['tcp'])
                            realized_metrics = None
                            if args.replay and len(base['replay']['tcp']) == 64 and len(prediction['replay']['tcp']) == 64:
                                realized_metrics = trajectory_metrics(base['replay']['tcp'], prediction['replay']['tcp'], base['actions'], prediction['actions'],
                                    entry['delta'], entry['state']['tcp'], conditions['baseline']['state']['peg'], entry['state']['peg'],
                                    control_hz=10., closure_width_m=args.closure_width)
                                realized_metrics['target_semantics'] = 'Measured TCP after each 100 ms physics execution interval; not FK command targets'
                                for endpoint in realized_metrics['endpoints'].values():
                                    if endpoint is not None:
                                        endpoint['base_observation_time_s'] = (endpoint['base_query'] + 1) / 10.
                                        endpoint['shifted_observation_time_s'] = (endpoint['shifted_query'] + 1) / 10.
                                paired.update(baseline_realized_tcp=base['replay']['tcp'], shifted_realized_tcp=prediction['replay']['tcp'])
                            np.savez_compressed(paired_path, **paired)
                            report['rows'].append(dict(model=model_name, position=position, seed=seed, condition='peg_and_holder',
                                direction=direction, metrics=metrics, realized_metrics=realized_metrics,
                                within_training_source_bounds=entry['within_training_source_bounds'],
                                source_xy=entry['config']['board']['source_xy'], target_xy=entry['config']['board']['target_xy'],
                                paired_rng_equal=True, archive=str(paired_path.relative_to(output))))
                        print(json.dumps(dict(event='prediction', model=model_name, position=position, seed=seed, direction=direction,
                                              seconds=round(prediction['seconds'], 3), pairs=len(report['rows']))), flush=True)
                    write_json(output / 'progress.json', dict(model=model_name, position=position, seed=seed, pairs=len(report['rows'])))
            finally:
                for sim in simulations.values():
                    sim.close()
            write_json(output / 'summary.partial.json', report)
        del adapter
    report['protected_after'] = {path: sha256(path) for path in protected}
    assert report['protected_after'] == report['protected_before'], 'Protected checkpoint/config changed'
    report['protected_unchanged'] = True
    report['complete'] = True  # Diagnostic finished; not a claim of grasp/task success.
    write_json(output / 'summary.json', report)
    if args.plot:
        from .plot_translation_probe import plot_results
        report['plots'] = plot_results(output)
        write_json(output / 'summary.json', report)
    print(json.dumps(dict(event='complete', output=str(output), pairs=len(report['rows']))), flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--act-checkpoint', type=Path)
    parser.add_argument('--diffusion-checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--positions', nargs='+', default=[f'pos{i}' for i in range(10)])
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(5)))
    parser.add_argument('--distance', type=float, default=.03)
    parser.add_argument('--closure-width', type=float, default=.026)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='cpu')
    parser.add_argument('--backend', choices=('osmesa', 'egl'), default='osmesa')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--replay', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, default=False,
                        help='Requires matplotlib in the inference environment; separate plotting CLI also supported')
    args = parser.parse_args(argv)
    if not args.act_checkpoint and not args.diffusion_checkpoint:
        parser.error('At least one trained checkpoint is required')
    if args.threads < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error('threads must be positive; seeds must be unique integers in [0, 2**32)')
    if not np.isfinite(args.closure_width) or not 0 < args.closure_width < .085:
        parser.error('closure width must be between 0 and .085 metres')
    os.environ['MUJOCO_GL'] = args.backend
    run_probe(args)


if __name__ == '__main__':
    main()
