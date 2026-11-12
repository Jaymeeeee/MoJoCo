"""Freeze unseen in-range layouts after auditing training data and expert physics.

Selection uses geometry and a seeded ordering only. Policy outcomes are never
inputs. Coordinate bounds do not establish membership of the training 4-D hull.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import yaml

from examples.insert_plug.environment.layouts import PositionsDumper
from examples.insert_plug.environment.scene import load_config
from mujoco_tools.paths import WORKSPACE_ROOT


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def display_path(path):
    path = Path(path).resolve()
    return str(path.relative_to(WORKSPACE_ROOT)) if path.is_relative_to(WORKSPACE_ROOT) else str(path)


def pair_array(rows):
    pairs = np.asarray([list(row['source_xy']) + list(row['target_xy']) for row in rows], dtype=float)
    if pairs.ndim != 2 or pairs.shape[1] != 4 or not np.isfinite(pairs).all():
        raise ValueError('Layouts must contain finite source/target XY pairs')
    return pairs


def audit_training_data(dataset_root, training_manifest, checkpoint_train_configs,
                        *, expected_positions=10, episodes_per_position=10):
    """Use every raw episode config as evidence; cross-check manifest/checkpoints."""
    dataset_root = Path(dataset_root).resolve()
    manifest_path = Path(training_manifest).resolve()
    manifest = yaml.safe_load(manifest_path.read_text())
    names = [f'pos{i}' for i in range(expected_positions)]
    manifest_rows = {row['name']: row for row in manifest['positions']}
    if set(manifest_rows) != set(names):
        raise ValueError('Training manifest position names do not match the expected training set')
    files = sorted(dataset_root.glob('pos*/episode_*/*.yaml'))
    counts = Counter(path.parent.parent.name for path in files)
    if counts != Counter({name: episodes_per_position for name in names}):
        raise ValueError(f'Unexpected raw training episode counts: {dict(counts)}')
    records, groups = [], {}
    for path in files:
        board = yaml.safe_load(path.read_text())['board']
        row = dict(source_xy=list(board['source_xy']), target_xy=list(board['target_xy']))
        pair = pair_array([row])[0]
        name = path.parent.parent.name
        if not np.allclose(pair, pair_array([manifest_rows[name]])[0], atol=1e-10, rtol=0):
            raise ValueError(f'Raw episode disagrees with training manifest: {path}')
        groups.setdefault(name, row)
        records.append(dict(path=display_path(path), sha256=sha256(path),
                            training_position=name, **row))
    training_rows = [dict(training_position=name, episodes=counts[name], **groups[name]) for name in names]
    pairs = pair_array(training_rows)
    if len(np.unique(pairs, axis=0)) != expected_positions:
        raise ValueError('Training layout pairs must be distinct')
    checkpoint_records = []
    for path in checkpoint_train_configs:
        path = Path(path).resolve()
        data = json.loads(path.read_text())['dataset']
        root = Path(data['root']).resolve()
        if root != dataset_root / 'lerobot':
            raise ValueError(f'Checkpoint uses a different dataset: {path}: {root}')
        if data.get('episodes') is not None or data.get('exclude_episodes'):
            raise ValueError(f'Checkpoint episode filtering needs a separate source audit: {path}')
        checkpoint_records.append(dict(path=display_path(path), sha256=sha256(path),
                                       dataset_root=str(root), episodes=data.get('episodes'),
                                       exclude_episodes=data.get('exclude_episodes')))
    if len(checkpoint_records) != 2:
        raise ValueError('Provide ACT and Diffusion checkpoint train_config.json files')
    metadata_path = dataset_root / 'lerobot/meta/info.json'
    metadata = json.loads(metadata_path.read_text())
    if metadata['total_episodes'] != len(files):
        raise ValueError('LeRobot episode count differs from raw training episodes')
    evidence = dict(
        dataset_root=str(dataset_root), total_raw_episodes=len(files), unique_training_pairs=len(pairs),
        episodes_per_position=dict(counts), training_positions=training_rows,
        training_manifest=dict(path=display_path(manifest_path), sha256=sha256(manifest_path)),
        dataset_info=dict(path=display_path(metadata_path), sha256=sha256(metadata_path),
                          total_episodes=metadata['total_episodes'], fps=metadata['fps']),
        checkpoint_train_configs=checkpoint_records, raw_episode_configs=records,
        raw_inventory_sha256=hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
    )
    return pairs, evidence


def select_candidates(training_pairs, *, seed=20261109, count=10, grid_spacing_m=.02,
                      minimum_point_distance_m=.01, minimum_pair_distance_m=.02):
    """Seeded interstitial candidates, unique source/target and novel versus all data."""
    training = np.asarray(training_pairs, dtype=float)
    if training.ndim != 2 or training.shape[1] != 4 or not len(training) or not np.isfinite(training).all():
        raise ValueError('Training coordinates must be a nonempty finite N by 4 array')
    if count < 1 or not np.isfinite(grid_spacing_m) or grid_spacing_m <= 0:
        raise ValueError('Count and grid spacing must be positive')
    if any(not np.isfinite(v) or v <= 0 for v in (minimum_point_distance_m, minimum_pair_distance_m)):
        raise ValueError('Novelty distance thresholds must be finite and positive')
    bounds = np.stack((training.min(axis=0), training.max(axis=0)), axis=1)
    axes = [np.round(np.arange(low + grid_spacing_m / 2, high, grid_spacing_m), 10)
            for low, high in bounds]
    sources = [(float(x), float(y)) for x in axes[0] for y in axes[1]]
    targets = [(float(x), float(y)) for x in axes[2] for y in axes[3]]
    pool = [(source, target) for source in sources for target in targets]
    chosen, used_source, used_target = [], set(), set()
    for index in np.random.default_rng(seed).permutation(len(pool)):
        source, target = pool[index]
        if source in used_source or target in used_target:
            continue
        source_distance = float(np.linalg.norm(training[:, :2] - source, axis=1).min())
        target_distance = float(np.linalg.norm(training[:, 2:] - target, axis=1).min())
        pair_distance = float(np.linalg.norm(training - np.array(source + target), axis=1).min())
        if (min(source_distance, target_distance) < minimum_point_distance_m - 1e-10
                or pair_distance < minimum_pair_distance_m - 1e-10):
            continue
        chosen.append(dict(name=f'pos{len(chosen)}', heldout_layout_id=f'generalization_v1_pos{len(chosen)}',
                           split='heldout_coordinate_range', source_xy=list(source), target_xy=list(target),
                           candidate_pool_index=int(index),
                           novelty=dict(min_training_source_xy_distance_m=source_distance,
                                        min_training_target_xy_distance_m=target_distance,
                                        min_training_pair_4d_distance_m=pair_distance)))
        used_source.add(source)
        used_target.add(target)
        if len(chosen) == count:
            return chosen, bounds.tolist(), dict(source_points=len(sources), target_points=len(targets),
                                                 pairs=len(pool))
    raise ValueError(f'Only {len(chosen)} distinct unseen layouts satisfy the requested {count}')


def write_json(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=WORKSPACE_ROOT/'config/insert_plug.yaml')
    parser.add_argument('--training-manifest', type=Path, default=WORKSPACE_ROOT/'config/insert_plug_positions.yaml')
    parser.add_argument('--dataset-root', type=Path, default=WORKSPACE_ROOT/'outputs/datasets/insert_plug_v1')
    parser.add_argument('--checkpoint-train-config', action='append', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=WORKSPACE_ROOT/'config/insert_plug_generalization_v1.yaml')
    parser.add_argument('--output', type=Path, default=WORKSPACE_ROOT/'outputs/_checks/insert_plug_generalization_v1')
    parser.add_argument('--seed', type=int, default=20261109)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    output, manifest_path = args.output.resolve(), args.manifest.resolve()
    checks_root = (WORKSPACE_ROOT/'outputs/_checks').resolve()
    if output == checks_root or not output.is_relative_to(checks_root):
        parser.error('--output must be a fresh directory under outputs/_checks')
    if args.workers < 1:
        parser.error('--workers must be positive')
    if output.exists() or manifest_path.exists():
        raise FileExistsError('Evidence directory and manifest must both be new')
    training, evidence = audit_training_data(args.dataset_root, args.training_manifest,
                                             args.checkpoint_train_config)
    candidates, bounds, pool = select_candidates(training, seed=args.seed)
    base = load_config(args.config)
    base_sha = sha256(args.config)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'training_evidence.json', evidence)
    (output/'base_config.yaml').write_bytes(args.config.read_bytes())
    plan = dict(selection_seed=args.seed, selection_method='seeded permutation of interstitial grid pairs; greedy unique source and target',
                selection_uses_policy_outcomes=False, coordinate_order=['source_x', 'source_y', 'target_x', 'target_y'],
                training_coordinate_bounds_m=bounds, candidate_pool=pool, grid_spacing_m=.02,
                grid_offset_from_training_min_m=.01, minimum_training_point_distance_m=.01,
                minimum_training_pair_4d_distance_m=.02, positions=candidates)
    write_json(output/'selection_plan.json', plan)
    print(json.dumps(dict(event='selection_frozen_before_physics', output=str(output), positions=candidates)), flush=True)
    from examples.insert_plug.diagnostics.validate_expert import run_position
    started = time.monotonic()
    if args.workers == 1:
        results = [run_position(base, row, output) for row in candidates]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as executor:
            futures = [executor.submit(run_position, base, row, output) for row in candidates]
            results = [future.result() for future in as_completed(futures)]
    results.sort(key=lambda row: int(row['position'].removeprefix('pos')))
    summary = dict(format_version=1, wall_seconds=time.monotonic()-started, base_config_sha256=base_sha,
                   rendered=False, randomized=False, selection_uses_policy_outcomes=False,
                   positions=len(results), successes=sum(row['validated_success'] for row in results),
                   all_successful=all(row['validated_success'] for row in results),
                   visibility_passed=all(row['visibility_passed'] for row in results),
                   required_edge_margin_px=5., occlusion_checked=False,
                   collision_checks_enabled=any(row['collision_checks_enabled'] for row in results),
                   contact_clean=None if not any(row['collision_checks_enabled'] for row in results)
                   else all(row['contact_clean'] for row in results), results=results)
    write_json(output/'summary.json', summary)
    if not summary['all_successful']:
        print('Expert validation failed; evidence retained and no final manifest written.', flush=True)
        return 1
    if sha256(args.config) != base_sha:
        raise RuntimeError('Base configuration changed during validation; refusing to freeze')
    rows = []
    for candidate, result in zip(candidates, results, strict=True):
        visibility = {stage: dict(all_inside=result['visibility'][stage]['all_inside'],
                                 min_edge_margin_px=min(obj['edge_margin_px'] for obj in result['visibility'][stage]['objects'].values()))
                      for stage in ('initial', 'final')}
        result_path = output/candidate['name']/'result.json'
        trace_path = output/candidate['name']/'trace.npz'
        rows.append(dict(**candidate, visibility=visibility,
                         physics_check=dict(success=result['validated_success'], duration_seconds=result['simulated_seconds'],
                                            control_steps=result['result']['steps'],
                                            radial_error_m=result['result']['metrics']['radial_error_m'],
                                            insertion_depth_m=result['result']['metrics']['insertion_depth_m'],
                                            collision_checks_enabled=result['collision_checks_enabled'], contact_clean=result['contact_clean']),
                         validation_report=dict(path=display_path(result_path), sha256=sha256(result_path)),
                         validation_trace=dict(path=display_path(trace_path), sha256=sha256(trace_path))))
    manifest = dict(format_version=1, split='heldout_coordinate_range', layout_set_id='insert_plug_generalization_v1',
                    description='Previously unseen source and target XY, each within training coordinate bounds; no claim of 4-D convex-hull interpolation.',
                    names_are_local_to_manifest=True, training_position_ids=[f'pos{i}' for i in range(10)],
                    training_position_id_note='These IDs belong to the original training manifest, not the held-out pos0..pos9 rows below.',
                    base_config=display_path(args.config), base_config_sha256=base_sha,
                    base_source_xy=base['board']['source_xy'], base_target_xy=base['board']['target_xy'],
                    selection_seed=args.seed, selection_method=plan['selection_method'], selection_uses_policy_outcomes=False,
                    coordinate_order=plan['coordinate_order'], training_coordinate_bounds_m=bounds,
                    grid_spacing_m=.02, grid_offset_from_training_min_m=.01,
                    minimum_training_point_distance_m=.01, minimum_training_pair_4d_distance_m=.02,
                    required_edge_margin_px=5., occlusion_checked=False,
                    visibility_method='all compiled visual mesh vertices projected into head optical camera',
                    control_hz=base['task']['control_hz'], recording_hz=base['recording']['sample_hz'],
                    action_hold_seconds=1./base['task']['control_hz'], episodes_per_position=10,
                    selection_complete=True,
                    training_evidence=dict(path=display_path(output/'training_evidence.json'), sha256=sha256(output/'training_evidence.json')),
                    selection_plan=dict(path=display_path(output/'selection_plan.json'), sha256=sha256(output/'selection_plan.json')),
                    expert_validation=dict(path=display_path(output/'summary.json'), sha256=sha256(output/'summary.json')),
                    positions=rows)
    with manifest_path.open('x', encoding='utf8') as stream:
        yaml.dump(manifest, stream, Dumper=PositionsDumper, sort_keys=False)
    write_json(output/'frozen_manifest.json', dict(path=display_path(manifest_path), sha256=sha256(manifest_path)))
    print(json.dumps(dict(event='validated_manifest_frozen', manifest=str(manifest_path), successes=summary['successes'])), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
