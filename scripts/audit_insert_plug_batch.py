#!/usr/bin/env python3
"""Read-only audit of the fixed-grid insert_plug RGB-D collection.

Checks the project HDF5 recordings and their associated head-camera videos;
it does not modify data, infer task success from pixels, or upload anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import h5py
import imageio.v2 as imageio
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.insert_plug.dataset import validate_episode


REQUIRED_PHASES = {
    'settle_initial', 'approach', 'descend_to_grasp', 'grasp', 'extract',
    'transfer', 'align', 'shallow_insert', 'insertion_check', 'release',
    'retract', 'settle_final',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def grid_steps(xy, base, spacing):
    xy = np.asarray(xy, dtype=float)
    require(xy.shape == (2,) and np.isfinite(xy).all(), 'Holder XY must be a finite pair')
    steps = (xy - np.asarray(base, dtype=float)) / spacing
    require(np.allclose(steps, np.rint(steps), atol=1e-8, rtol=0), 'Holder is not on the declared 2 cm grid')
    return np.rint(steps).astype(int).tolist()


def audit_episode(folder, sample_hz=10.0, control_hz=10.0):
    """Return compact evidence for one complete, successful episode."""
    folder = Path(folder)
    stem = folder.name
    require(bool(re.fullmatch(r'episode_\d{6}', stem)), 'Invalid episode directory name')
    expected = {f'{stem}.{suffix}' for suffix in ('h5', 'yaml', 'xml', 'mp4')}
    entries = list(folder.iterdir())
    require({p.name for p in entries} == expected, 'Episode must contain exactly same-stem HDF5/YAML/XML/MP4')
    require(all(p.is_file() and not p.is_symlink() for p in entries), 'Episode components must be regular files')
    path = folder / f'{stem}.h5'
    report = validate_episode(path)
    require(report['valid'], '; '.join(report['errors']))
    require(report['success'], 'Recording reports task failure')
    metadata = report['metadata']
    require(metadata.get('task') == 'insert_plug', 'Wrong task metadata')
    require(metadata.get('sample_hz') == sample_hz, 'Wrong recording frequency')
    require(metadata.get('control_hz') == control_hz and metadata.get('physics_timestep') == .001,
            f'Expected {control_hz:g} Hz control and 1 kHz physics')
    require(metadata.get('sampling_timing') == 'nearest_physics_tick', 'Missing real physics-tick sampling convention')
    aligned = math.isclose(control_hz, sample_hz, rel_tol=0, abs_tol=1e-10)
    action_timing = ('active_control_target_at_observation; held_until_next_observation' if aligned
                     else 'active_control_target_at_observation; may change before next observation')
    require(metadata.get('action_timing') == action_timing, 'Missing expected action convention')
    if aligned:
        require(metadata.get('sample_origin_s') == 0, 'Aligned actions must start at control time zero')
        require(metadata.get('action_hold_duration_s') == 1 / control_hz,
                'Aligned action hold duration must match the full control interval')
    require(metadata.get('depth_enabled') is True, 'RGB-D recording is required')
    require(metadata.get('video_camera') == 'head' and metadata.get('video_file') == f'{stem}.mp4',
            'Head-video metadata does not match the episode')
    require(metadata.get('model_path') == f'{stem}.xml', 'Model path must resolve to the local same-stem XML')
    require(metadata.get('model_sha256') == hashlib.sha256((folder / metadata['model_path']).read_bytes()).hexdigest(),
            'Model hash does not match the episode XML')
    config = yaml.safe_load((folder / f'{stem}.yaml').read_text(encoding='utf8'))
    require(config == metadata.get('config'), 'YAML configuration and HDF5 metadata disagree')
    require(config['recording']['sample_hz'] == sample_hz, 'YAML recording rate disagrees')
    require(config['task']['control_hz'] == control_hz, 'YAML control rate disagrees')
    require(config['physics']['timestep'] == metadata['physics_timestep'], 'YAML physics timestep disagrees')
    require(config['cameras']['depth']['enabled'] is True, 'YAML depth is disabled')
    execution = metadata.get('execution_result', {})
    require(execution.get('success') is True and execution.get('failure_reason') == '', 'Execution result is not successful')
    require(execution.get('model') == f'{stem}.xml', 'Execution XML reference is invalid')
    require(REQUIRED_PHASES <= {x.get('phase') for x in execution.get('phases', [])}, 'Execution stages are missing')
    plans = execution.get('planning', [])
    require(bool(plans) and all(p.get('valid') is True for p in plans), 'Missing or unsuccessful planning checks')
    checks = execution.get('execution_checks', {})
    require(checks.get('unsafe_contact_count') == 0 and checks.get('last_violation') is None,
            'Execution contains an unsafe-contact violation')
    control_steps = execution.get('steps', 0)
    require(isinstance(control_steps, int) and control_steps > 0, 'Missing executed control-step count')
    physics_steps = control_steps * round(1 / (metadata['control_hz'] * metadata['physics_timestep']))
    require(checks.get('checks') == physics_steps, 'Contact guard did not check every physics step')
    metrics = metadata.get('final_metrics', {})
    require(metrics.get('success') is True and metrics.get('stable_for_final_250ms') is True,
            'Final seating/stability checks did not succeed')
    frames = report['steps']
    duration = control_steps / metadata['control_hz']
    require(frames == math.ceil(duration * sample_hz - 1e-9), 'Recorded frame count does not cover the full rollout')
    if aligned:
        require(frames == control_steps, 'Aligned collection must record every control action exactly once')
    require(execution.get('recorded_frames') == frames, 'Execution frame count disagrees with HDF5')
    video_path = folder / f'{stem}.mp4'
    with h5py.File(path, 'r') as handle:
        if aligned:
            require(np.allclose(handle['timestamp'][:], np.arange(frames) / control_hz,
                                atol=1e-7, rtol=0),
                    'Aligned action timestamps must cover successive complete control intervals')
        height, width = int(config['cameras']['height']), int(config['cameras']['width'])
        for name in ('head', 'wrist'):
            require(handle[f'observations/images/{name}'].shape == (frames, height, width, 3), 'RGB shape mismatch')
            require(handle[f'observations/depth/{name}'].shape == (frames, height, width), 'Depth/RGB shape mismatch')
        for side in ('source', 'target'):
            xy = config['board'][f'{side}_xy']
            require(np.allclose(handle[f'privileged/{side}_pose'][:, :2], xy, atol=1e-10, rtol=0),
                    'Recorded holder poses disagree with fixed YAML positions')
        with imageio.get_reader(video_path) as reader:
            video = reader.get_meta_data()
            require(video.get('fps') == sample_hz, 'MP4 frame rate is wrong')
            require(video.get('size') == (width, height), 'MP4 is not the native-size single-camera image')
            require(reader.count_frames() == frames, 'MP4 frame count differs from HDF5')
            require(abs(video.get('duration', 0) - frames / sample_hz) <= .02, 'MP4 duration differs from frame count')
            errors = []
            for index in sorted({0, frames - 1}):
                rgb = handle['observations/images/head'][index]
                error = float(np.mean(np.abs(reader.get_data(index).astype(float) - rgb)))
                require(error < 12, 'MP4 frames do not match the recorded head camera')
                errors.append(error)
    return dict(
        episode=stem, frames=frames, simulation_duration_s=duration,
        control_hz=control_hz, control_steps=control_steps, physics_steps=physics_steps,
        action_hold_s=1 / control_hz if aligned else None,
        video_duration_s=frames / sample_hz, bytes=sum(p.stat().st_size for p in entries),
        source_xy=config['board']['source_xy'], target_xy=config['board']['target_xy'],
        max_video_rgb_mean_absolute_error=max(errors),
        radial_error_m=float(metrics['radial_error_m']),
        unsafe_contacts=int(checks['unsafe_contact_count']),
    )


def audit_batch(root, *, positions=10, episodes_per_position=10, sample_hz=10.0, control_hz=None,
                base_source=(.45, -.07), base_target=(.45, .07), spacing=.02,
                manifest=None):
    """Audit only pos0..posN; unrelated historical/LeRobot directories are ignored."""
    root = Path(root)
    require(positions > 0 and episodes_per_position > 0, 'Counts must be positive')
    manifest_positions = {}
    if manifest is not None:
        manifest = yaml.safe_load(Path(manifest).read_text(encoding='utf8'))
        manifest_control_hz = manifest.get('control_hz')
        if manifest_control_hz is not None:
            require(control_hz is None or control_hz == manifest_control_hz,
                    'Manifest control rate differs from the explicitly requested rate')
            control_hz = float(manifest_control_hz)
        require(manifest.get('recording_hz', sample_hz) == sample_hz,
                'Manifest recording rate differs from the requested rate')
        spacing = float(manifest['grid_spacing_m'])
        base_source, base_target = manifest['base_source_xy'], manifest['base_target_xy']
        manifest_positions = {row['name']: row for row in manifest['positions']}
        require(set(manifest_positions) == {f'pos{i}' for i in range(positions)}, 'Manifest position names differ from requested batch')
    control_hz = 10.0 if control_hz is None else float(control_hz)
    require(np.isfinite([sample_hz, control_hz, spacing]).all() and min(sample_hz, control_hz, spacing) > 0,
            'Sample/control rates and grid spacing must be finite and positive')
    report = dict(valid=False, root=str(root.resolve()), expected_episodes=positions * episodes_per_position,
                  sample_hz=sample_hz, control_hz=control_hz, grid_spacing_m=spacing, groups=[], errors=[])
    seen = {'source': set(), 'target': set()}
    for index in range(positions):
        name = f'pos{index}'
        folder = root / name
        group = dict(position=name, valid=False, episodes=[], errors=[])
        report['groups'].append(group)
        try:
            entries = sorted(folder.iterdir())
            require(len(entries) == episodes_per_position and all(p.is_dir() and re.fullmatch(r'episode_\d{6}', p.name) for p in entries),
                    'Position must contain exactly the requested number of episode directories')
            for episode in entries:
                try:
                    group['episodes'].append(audit_episode(episode, sample_hz, control_hz))
                except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
                    group['errors'].append(f'{episode.name}: {error}')
            require(not group['errors'], 'One or more episodes failed audit')
            for side, base in (('source', base_source), ('target', base_target)):
                xy = group['episodes'][0][f'{side}_xy']
                require(all(np.allclose(x[f'{side}_xy'], xy, rtol=0, atol=1e-10) for x in group['episodes']),
                        'Holder positions change within a position group')
                steps = grid_steps(xy, base, spacing)
                require(steps != [0, 0], 'Holder has not moved from its original position')
                key = tuple(steps)
                require(key not in seen[side], f'{side} holder position was reused across groups')
                seen[side].add(key)
                group[f'{side}_xy'], group[f'{side}_steps'] = xy, steps
                if manifest_positions:
                    require(np.allclose(xy, manifest_positions[name][f'{side}_xy'], atol=1e-10, rtol=0),
                            'Holder position differs from the selected layout manifest')
            if manifest_positions:
                # Recompute containment from the actual recorded XML, rather
                # than trusting a pre-collection visibility flag in a manifest.
                import mujoco
                from examples.insert_plug.layouts import head_visibility
                from examples.insert_plug.simulation import Simulation
                first = entries[0]
                config = yaml.safe_load((first / f'{first.name}.yaml').read_text(encoding='utf8'))
                sim = Simulation(first / f'{first.name}.xml', config, render=False)
                try:
                    margin = float(manifest.get('required_edge_margin_px', 5.0))
                    initial = head_visibility(sim, margin)
                    with h5py.File(first / f'{first.name}.h5', 'r') as handle:
                        sim.data.qpos[:] = handle['simulation/qpos'][-1]
                        sim.data.qvel[:] = handle['simulation/qvel'][-1]
                        sim.data.time = float(handle['timestamp'][-1])
                    mujoco.mj_forward(sim.model, sim.data)
                    final = head_visibility(sim, margin)
                    require(initial['all_inside'] and final['all_inside'], 'Recorded scene is outside the head camera view')
                    group['head_frustum_checked'] = True
                    group['minimum_head_edge_margin_px'] = min(
                        item['edge_margin_px'] for view in (initial, final) for item in view['objects'].values())
                finally:
                    sim.close()
            group['valid'] = True
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            group['errors'].append(str(error))
        if not group['valid']:
            report['errors'].append(f'{name}: ' + '; '.join(group['errors']))
    episodes = [episode for group in report['groups'] for episode in group['episodes']]
    report.update(episodes=len(episodes), total_frames=sum(x['frames'] for x in episodes),
                  total_bytes=sum(x['bytes'] for x in episodes), valid=not report['errors'])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'outputs/datasets/insert_plug')
    parser.add_argument('--positions-config', type=Path)
    parser.add_argument('--positions', type=int, default=10)
    parser.add_argument('--episodes-per-position', type=int, default=10)
    parser.add_argument('--sample-hz', type=float, default=10.0)
    parser.add_argument('--control-hz', type=float,
                        help='Expected expert rate; manifest control_hz or 10 Hz by default; use 50 for legacy collections')
    args = parser.parse_args(argv)
    report = audit_batch(args.root, positions=args.positions, episodes_per_position=args.episodes_per_position,
                         sample_hz=args.sample_hz, control_hz=args.control_hz, manifest=args.positions_config)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
