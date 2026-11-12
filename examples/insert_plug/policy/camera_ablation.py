"""Paired camera masking on observations saved during an ACT execution.

The robot state and other camera stay fixed in every comparison. This measures
action-target sensitivity on a reference trajectory, not masked closed-loop
task success. Diagnostic predictions never advance the deployed policy queue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .output_naming import experiment_output_path, experiment_timestamp


def build_conditions(repeats):
    if type(repeats) is not int or repeats < 1:
        raise ValueError('patch repeats must be a positive integer')
    conditions = [dict(index=0, name='baseline', camera=None, scheme='baseline', repeat=None)]
    for scheme in ('full_mean', 'full_black', 'patch_mean'):
        for repeat in range(repeats if scheme == 'patch_mean' else 1):
            for camera in ('head', 'wrist'):
                conditions.append(dict(index=len(conditions), name=f'{camera}_{scheme}_{repeat}',
                                       camera=camera, scheme=scheme,
                                       repeat=repeat if scheme == 'patch_mean' else None))
    return conditions


def action_metrics(physical_delta, normalized_delta):
    physical = np.asarray(physical_delta, dtype=np.float64)
    normalized = np.asarray(normalized_delta, dtype=np.float64)
    if physical.shape != normalized.shape or physical.ndim < 2 or physical.shape[-1] != 8:
        raise ValueError('deltas must have matching (..., queries, 8) shapes')
    if not np.isfinite(physical).all() or not np.isfinite(normalized).all():
        raise ValueError('action differences must be finite')
    arm = np.rad2deg(physical[..., :7])
    gripper = physical[..., 7] * 1000
    return dict(q0_normalized_rms=np.sqrt(np.mean(normalized[..., 0, :]**2, axis=-1)),
                chunk_normalized_rms=np.sqrt(np.mean(normalized**2, axis=(-2, -1))),
                q0_arm_rms_deg=np.sqrt(np.mean(arm[..., 0, :]**2, axis=-1)),
                q0_arm_max_deg=np.max(np.abs(arm[..., 0, :]), axis=-1),
                q0_gripper_abs_mm=np.abs(gripper[..., 0]),
                chunk_arm_rms_deg=np.sqrt(np.mean(arm**2, axis=(-2, -1))),
                chunk_gripper_rms_mm=np.sqrt(np.mean(gripper**2, axis=-1)))


def isolated_ensemble_delta(delta, step, chunk_size, coeff):
    """Perturb only the newest prediction, preserving identical prior history."""
    from .visualize_attention import temporal_contributions
    weight = 1. if coeff is None else temporal_contributions(step, chunk_size, coeff)[-1]['weight']
    return np.asarray(delta) * weight


def _describe(values):
    return dict(mean=float(np.mean(values)), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)), max=float(np.max(values)))


def summarize_metrics(metrics, conditions, time_s, joint_threshold_deg=1., gripper_threshold_mm=1.):
    """Descriptive statistics; mask repeats are averaged within each time step."""
    time_s = np.asarray(time_s)
    result = []
    for scheme in ('full_mean', 'full_black', 'patch_mean'):
        for camera in ('head', 'wrist'):
            columns = [c['index']-1 for c in conditions if c['camera'] == camera and c['scheme'] == scheme]
            if not columns:
                continue
            series = {key: np.asarray(values)[:, columns].mean(axis=1) for key, values in metrics.items()}
            obvious = ((series['q0_arm_max_deg'] > joint_threshold_deg) |
                       (series['q0_gripper_abs_mm'] > gripper_threshold_mm))
            bins = []
            for start in np.arange(0., float(time_s[-1])+.1, 5.):
                selection = (time_s >= start) & (time_s < start+5)
                if selection.any():
                    bins.append(dict(start_s=float(start), end_s=float(start+5), samples=int(selection.sum()),
                                     metrics={key: _describe(values[selection]) for key, values in series.items()}))
            result.append(dict(camera=camera, scheme=scheme, samples=len(time_s), mask_repeats=len(columns),
                               metrics={key: _describe(values) for key, values in series.items()},
                               obvious_change_fraction=float(obvious.mean()), time_bins=bins))
    return result


def _load_source(directory):
    directory = Path(directory).expanduser().resolve()
    report = json.loads((directory / 'report.json').read_text())
    frames = report['attention']['frames']
    chunks = report['attention']['chunks']
    if not frames or len(frames) != len(chunks) or report['policy']['n_action_steps'] != 1:
        raise ValueError('Source must contain one saved prediction per control step (n_action_steps=1)')
    hz = float(report['rollout']['control_hz'])
    for index, (frame, chunk) in enumerate(zip(frames, chunks, strict=True)):
        if (frame['frame'] != index or frame['chunk_index'] != index or frame['query_index'] != 0
                or chunk['index'] != index or chunk['source_step'] != index
                or frame['source_age_s'] != 0 or not np.isclose(frame['time_s'], index/hz)
                or frame.get('execution_status') != 'completed'):
            raise ValueError('Source must contain contiguous completed steps with current-image query 0')
        path = (directory / chunk['weights']).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError('Source prediction archive is missing or outside source directory')
        if set(chunk['cameras']) != {'observation.images.head', 'observation.images.wrist'}:
            raise ValueError('Expected head and wrist source cameras')
    return directory, report


def _save_mask_preview(path, observation, mask, time_s):
    from PIL import Image, ImageDraw
    from mujoco_tools.policy_ablation import mask_camera
    from .visualize_attention import _font
    canvas = Image.new('RGB', (560, 914), '#141922')
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 10), f'Masking examples | t = {time_s:.1f}s', font=_font(21), fill='white')
    for column, camera in enumerate(('head', 'wrist')):
        key = f'observation.images.{camera}'
        for row, (label, selection) in enumerate((('Original', None), ('Full view / mean', np.ones(mask.shape, bool)),
                                                 (f'Random {mask.mean():.0%} blocks / mean', mask))):
            rgb = observation[key] if selection is None else mask_camera(observation, key, selection)[key]
            x, y = 16+272*column, 48+284*row
            draw.text((x, y), f'{camera}: {label}', font=_font(13), fill='#c5d4e5')
            canvas.paste(Image.fromarray(rgb), (x, y+22))
    canvas.save(path)


def run_ablation(*, source_run, output=None, device='cpu', seed=20261108, patch_repeats=3,
                 patch_fraction=.5, drop_probability=.5, torch_threads=2,
                 joint_threshold_deg=1., gripper_threshold_mm=1., max_frames=None):
    started_at = experiment_timestamp()
    import torch
    from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
    from mujoco_tools.lerobot_policy import LeRobotPolicy
    from mujoco_tools.paths import OUTPUTS_DIR
    from mujoco_tools.policy_ablation import mask_camera, predict_action_chunks, random_patch_mask

    conditions = build_conditions(patch_repeats)
    if not 0 < patch_fraction < 1 or not 0 < drop_probability < 1:
        raise ValueError('patch_fraction and drop_probability must be between zero and one')
    if torch_threads < 1 or not 0 <= seed < 2**32:
        raise ValueError('torch_threads must be positive and seed in [0, 2**32)')
    if any(not math.isfinite(v) or v <= 0 for v in (joint_threshold_deg, gripper_threshold_mm)):
        raise ValueError('descriptive change thresholds must be finite and positive')
    source_dir, source = _load_source(source_run)
    frames, chunks = source['attention']['frames'], source['attention']['chunks']
    if max_frames is not None:
        if type(max_frames) is not int or not 1 <= max_frames <= len(frames):
            raise ValueError('max_frames must be within the saved trajectory')
        frames, chunks = frames[:max_frames], chunks[:max_frames]
    checkpoint = Path(source['policy']['checkpoint'])
    config_sha = hashlib.sha256((checkpoint / 'config.json').read_bytes()).hexdigest()
    if config_sha != source['policy']['config_sha256']:
        raise ValueError('Checkpoint configuration differs from the recorded policy')
    saved_config = json.loads((checkpoint / 'config.json').read_text())
    destination = experiment_output_path(
        output, parent=OUTPUTS_DIR / 'insert_plug' / 'camera_ablation', checkpoint=checkpoint,
        saved=saved_config, config=source.get('config', {}),
        kind='ensemble' if source['policy']['temporal_ensemble_coeff'] is not None else 'stateless',
        started_at=started_at)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite ablation results: {destination}')
    if saved_config['normalization_mapping']['ACTION'] != 'MEAN_STD':
        raise ValueError('This diagnostic requires affine MEAN_STD action normalization')
    torch.set_num_threads(torch_threads)
    adapter = LeRobotPolicy(checkpoint, device=device, n_action_steps=1,
                            temporal_ensemble_coeff=source['policy']['temporal_ensemble_coeff'])
    coefficient = adapter.temporal_ensemble_coeff
    baseline_ensemble = ACTTemporalEnsembler(coefficient, adapter.chunk_size) if coefficient is not None else None
    shadow_ensemble = ACTTemporalEnsembler(coefficient, adapter.chunk_size) if coefficient is not None else None
    seeds = np.random.SeedSequence(seed).spawn(2)
    mask_rng, drop_rng = (np.random.default_rng(s) for s in seeds)
    dropout = drop_rng.random(len(frames)) < drop_probability
    all_normalized, all_physical, all_masks = [], [], []
    baseline_actions, shadow_actions, shadow_normalized_deltas = [], [], []
    source_hashes = []
    full_mean_indices = [next(c['index'] for c in conditions if c['scheme'] == 'full_mean' and c['camera'] == camera)
                         for camera in ('head', 'wrist')]
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'data').mkdir()
    (destination / 'previews').mkdir()
    started = time.monotonic()
    for step, chunk in enumerate(chunks):
        source_path = source_dir / chunk['weights']
        source_hashes.append(hashlib.sha256(source_path.read_bytes()).hexdigest())
        with np.load(source_path) as archive:
            observation = {'observation.state': archive['source_state'].copy()}
            observation.update({key: archive[f'image_{index}'].copy() for index, key in enumerate(chunk['cameras'])})
        shape = observation['observation.images.head'].shape[:2]
        if observation['observation.images.wrist'].shape[:2] != shape:
            raise ValueError('Paired masks require matching camera resolution')
        patch_masks = [random_patch_mask(shape, patch_fraction, mask_rng) for _ in range(patch_repeats)]
        all_masks.append(np.stack([mask[::shape[0]//8, ::shape[1]//8] for mask in patch_masks]))
        observations = [observation]
        for condition in conditions[1:]:
            mask = patch_masks[condition['repeat']] if condition['scheme'] == 'patch_mean' else np.ones(shape, bool)
            observations.append(mask_camera(observation, f"observation.images.{condition['camera']}", mask,
                                            fill='black' if condition['scheme'] == 'full_black' else 'mean'))
        predictions = predict_action_chunks(adapter, observations)
        all_normalized.append(predictions.normalized)
        all_physical.append(predictions.physical)
        with torch.inference_mode():
            baseline_norm = torch.from_numpy(predictions.normalized[0:1].copy())
            indices = full_mean_indices if dropout[step] else [0, 0]
            shadow_norm = torch.from_numpy(predictions.normalized[indices].copy())
            baseline_action_norm = baseline_ensemble.update(baseline_norm) if baseline_ensemble else baseline_norm[:, 0]
            shadow_action_norm = shadow_ensemble.update(shadow_norm) if shadow_ensemble else shadow_norm[:, 0]
            shadow_normalized_deltas.append((shadow_action_norm-baseline_action_norm).numpy().copy())
            baseline_actions.append(adapter.postprocessor(baseline_action_norm.clone())[0].cpu().numpy().copy())
            shadow_actions.append(adapter.postprocessor(shadow_action_norm.clone()).cpu().numpy().copy())
        if step in {0, min(100, len(frames)-1)}:
            _save_mask_preview(destination / 'previews' / f'masks_{step:04d}.png', observation, patch_masks[0], frames[step]['time_s'])
        if (step+1) % 25 == 0 or step+1 == len(frames):
            print(json.dumps(dict(event='ablation_progress', frames=step+1, total=len(frames),
                                  predictions=(step+1)*len(conditions), wall_s=round(time.monotonic()-started, 1))), flush=True)
    normalized, physical = np.stack(all_normalized), np.stack(all_physical)
    baseline_actions, shadow_actions = np.stack(baseline_actions), np.stack(shadow_actions)
    reference_actions = np.array([frame['action'] for frame in frames])
    residual = baseline_actions-reference_actions
    tolerance = np.array([5e-5]*7+[5e-6])
    baseline_match = bool(np.all(np.abs(residual) <= tolerance))
    delta_physical, delta_normalized = physical[:, 1:]-physical[:, :1], normalized[:, 1:]-normalized[:, :1]
    metrics = action_metrics(delta_physical, delta_normalized)
    latest_weights = np.array([float(isolated_ensemble_delta(1., step, adapter.chunk_size, coefficient)) for step in range(len(frames))])
    for key in ('arm_rms_deg', 'arm_max_deg', 'gripper_abs_mm', 'normalized_rms'):
        metrics[f'isolated_ensemble_{key}'] = metrics[f'q0_{key}'] * latest_weights[:, None]
    intermittent_delta = shadow_actions-baseline_actions[:, None]
    intermittent_metrics = action_metrics(intermittent_delta[:, :, None, :], np.stack(shadow_normalized_deltas)[:, :, None, :])
    intermittent_arrays = {f'intermittent_ensemble_{key}': intermittent_metrics[f'q0_{key}']
                           for key in ('arm_rms_deg', 'arm_max_deg', 'gripper_abs_mm', 'normalized_rms')}
    times = np.array([frame['time_s'] for frame in frames])
    np.savez_compressed(destination / 'data' / 'predictions.npz', normalized=normalized, physical=physical,
                        baseline_ensemble_actions=baseline_actions, intermittent_ensemble_actions=shadow_actions,
                        reference_actions=reference_actions, patch_masks=np.stack(all_masks), dropout_active=dropout,
                        condition_names=np.array([c['name'] for c in conditions]))
    np.savez_compressed(destination / 'data' / 'metrics.npz', time_s=times, **metrics, **intermittent_arrays,
                        latest_prediction_weight=latest_weights, dropout_active=dropout)
    grouped = summarize_metrics(metrics, conditions, times, joint_threshold_deg, gripper_threshold_mm)
    comparisons = []
    for scheme in ('full_mean', 'full_black', 'patch_mean'):
        means = {camera: next(g for g in grouped if g['camera'] == camera and g['scheme'] == scheme)['metrics']['q0_normalized_rms']['mean']
                 for camera in ('head', 'wrist')}
        comparisons.append(dict(scheme=scheme, **means,
                                head_over_wrist=means['head']/means['wrist'] if means['wrist'] else None))
    intermittent_summary = []
    for index, camera in enumerate(('head', 'wrist')):
        intermittent_summary.append(dict(camera=camera,
            metrics={key: _describe(values[:, index]) for key, values in intermittent_arrays.items()},
            obvious_change_fraction=float(((intermittent_arrays['intermittent_ensemble_arm_max_deg'][:, index] > joint_threshold_deg) |
                                           (intermittent_arrays['intermittent_ensemble_gripper_abs_mm'][:, index] > gripper_threshold_mm)).mean())))
    report = dict(
        started_at=started_at,
        source=dict(directory=str(source_dir), report_sha256=hashlib.sha256((source_dir/'report.json').read_bytes()).hexdigest(),
                    position=source['position'], frames=len(frames), duration_s=len(frames)/source['rollout']['control_hz'],
                    original_task_success=source['rollout']['task_success'], original_stop_reason=source['rollout']['stop_reason'],
                    observation_archive_sha256=source_hashes),
        policy=adapter.metadata(), conditions=conditions,
        protocol=dict(seed=seed, patch_fraction=patch_fraction, patch_repeats=patch_repeats, mask_grid=[8, 8],
                      drop_probability=drop_probability, actual_dropped_steps=int(dropout.sum()),
                      fill_mean='Per-channel mean of the unmasked source image, rounded to uint8; global color remains.',
                      fill_black='All RGB channels zero; a severe missing-camera stress test.',
                      pairing='Identical state, other camera, patch geometry and dropout schedule for head/wrist.',
                      primary='Independent one-observation interventions at every recorded control step.',
                      isolated_ensemble='Only newest prediction changed; all previous predictions held identical.',
                      intermittent_ensemble='Separate head/wrist histories with the paired random full-mean dropout schedule; fixed recorded robot states.',
                      units='Seven joint targets in degrees; full gripper aperture in mm; normalized RMS uses checkpoint MEAN_STD action units.',
                      thresholds=dict(max_joint_deg=joint_threshold_deg, gripper_mm=gripper_threshold_mm,
                                      meaning='Descriptive engineering thresholds, not statistical significance or physical failure.'),
                      aggregation='Average patch repeats within each frame, then descriptive statistics across the trajectory; no independent-sample p-values.'),
        validation=dict(baseline_reconstruction_matches=baseline_match, baseline_max_abs_error=np.abs(residual).max(axis=0).tolist(),
                        baseline_tolerance=tolerance.tolist(), checkpoint_config_unchanged=hashlib.sha256((checkpoint/'config.json').read_bytes()).hexdigest()==config_sha,
                        diagnostic_policy_select_action_calls=adapter.calls),
        group_summary=grouped, paired_camera_comparison=comparisons, intermittent_summary=intermittent_summary,
        limitations=['Single recorded position/trajectory; no all-layout reliance or task-success claim.',
                     'Masked observations can be outside the training distribution; compare mask/fill schemes.',
                     'Action-target changes are not measured arm motion or masked closed-loop task success.',
                     'Intermittent histories use fixed reference states; they do not include feedback from executing masked actions.'],
        artifacts=dict(predictions='data/predictions.npz', metrics='data/metrics.npz', previews='previews/'),
        wall_seconds=time.monotonic()-started)
    (destination/'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(destination), baseline_match=baseline_match, comparison=comparisons), ensure_ascii=False), flush=True)
    if not baseline_match:
        raise RuntimeError('Unmasked baseline does not reproduce the saved executing policy; inspect report before interpreting ablation')
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True, help='Attention run with one current observation per executed step')
    parser.add_argument('--output')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=20261108)
    parser.add_argument('--patch-repeats', type=int, default=3)
    parser.add_argument('--patch-fraction', type=float, default=.5)
    parser.add_argument('--drop-probability', type=float, default=.5)
    parser.add_argument('--torch-threads', type=int, default=2)
    parser.add_argument('--joint-threshold-deg', type=float, default=1.)
    parser.add_argument('--gripper-threshold-mm', type=float, default=1.)
    parser.add_argument('--max-frames', type=int, help='Optional prefix for a short diagnostic')
    run_ablation(**vars(parser.parse_args(argv)))


if __name__ == '__main__':
    main()
