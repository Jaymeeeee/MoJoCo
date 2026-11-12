"""Visualize Diffusion image-encoder spatial pooling in a real fixed-layout rollout.

This convolutional Diffusion policy has no action-query Transformer attention.
Maps show independently normalized SpatialSoftmax keypoints, equally averaged;
they are not action-specific attribution or cross-camera importance scores.
Cached actions keep their prediction's source image. Actual execution remains
unchanged and is checked by replaying every saved observation without tracing.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mujoco_tools.paths import CONFIG_DIR, OUTPUTS_DIR, WORKSPACE_ROOT
from .output_naming import experiment_output_path, experiment_timestamp
from .visualize_attention import _colors, _font, _overlay


def keypoint_pixels(points, grid_shape, image_size):
    """Map normalized feature coordinates to image pixel-center coordinates.

SpatialSoftmax spans the first/last feature-grid centers, not the RGB borders.
The mapping matches bilinear upsampling's half-cell convention.
"""
    height, width = grid_shape
    image_width, image_height = image_size
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Keypoints must be finite (K, 2) normalized coordinates')
    if min(height, width, image_width, image_height) < 1:
        raise ValueError('Grid and image dimensions must be positive')
    cells = (points + 1) / 2 * np.array([width - 1, height - 1])
    return (cells + .5) * np.array([image_width / width, image_height / height]) - .5


def _points(image, points, grid_shape):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    pixels = keypoint_pixels(points, grid_shape, result.size)
    for index, (x, y) in enumerate(pixels):
        color = tuple(_colors(np.array(index / max(1, len(pixels) - 1))).astype(np.uint8))
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline='white', width=1)
    return result


def render_frame(live_images, source_images, capture, action_offset, *, time_s,
                 source_time_s, chunk_index, color_max):
    keys = tuple(capture.camera_weights)
    if len(keys) != 2:
        raise ValueError('The task visualization requires two camera views')
    canvas = Image.new('RGB', (688, 1008), (20, 25, 34))
    draw = ImageDraw.Draw(canvas)
    white, muted, accent = '#edf1f7', '#b5c1d0', '#65d8cc'
    draw.text((16, 12), 'Diffusion | encoder spatial pooling', font=_font(25), fill=white)
    draw.text((16, 49), f't = {time_s:.1f}s  |  prediction {chunk_index}  |  cached action {action_offset}',
              font=_font(17), fill=accent)
    for column, key in enumerate(keys):
        x = 16 + 336 * column
        name = key.rsplit('.', 1)[-1]
        draw.text((x, 83), f'LIVE / {name}', font=_font(17), fill=white)
        canvas.paste(Image.fromarray(live_images[key]).resize((320, 320)), (x, 109))
        draw.text((x, 444), f'PREDICTION IMAGE / {name}', font=_font(16), fill=accent)
        mean = capture.mean_weights[key]
        blended = Image.fromarray(_overlay(source_images[key], mean, color_max)).resize((320, 320))
        canvas.paste(_points(blended, capture.keypoints[key], capture.grid_shapes[key]), (x, 472))
        h, w = capture.grid_shapes[key]
        draw.text((x, 802), f'{h}x{w} | mean of {capture.keypoint_counts[key]} keypoints', font=_font(15), fill=white)
        draw.text((x, 824), f'Clipped mean-map cells: {int((mean > color_max).sum())}', font=_font(14), fill=muted)
    draw.text((16, 854), f'Source image: {source_time_s:.1f}s | age: {time_s-source_time_s:.1f}s',
              font=_font(16), fill=accent)
    gradient = np.repeat(_colors(np.linspace(0, 1, 180))[None], 12, axis=0).astype(np.uint8)
    canvas.paste(Image.fromarray(gradient), (16, 889))
    draw.text((208, 882), f'Fixed mean probability: 0 to {color_max:g}', font=_font(14), fill=muted)
    draw.text((16, 916), 'Dots: learned feature expectations, not labeled object landmarks.', font=_font(14), fill=muted)
    draw.text((16, 941), 'Encoder pooling only; not action-specific or causal attribution.', font=_font(14), fill=muted)
    draw.text((16, 966), 'Each camera/keypoint is normalized separately; do not compare total mass.',
              font=_font(13), fill=muted)
    return np.asarray(canvas)


def render_keypoint_atlas(source_image, weights, keypoints, *, name, grid_shape):
    """Display all keypoints with one fixed probability scale [0, 1]."""
    count = len(weights)
    rows = math.ceil(count / 4)
    canvas = Image.new('RGB', (688, 80 + rows * 188), (20, 25, 34))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 9), f'{name} | all {count} learned spatial keypoints', font=_font(22), fill='white')
    draw.text((12, 43), 'Probability scale fixed at 0..1 for every map; raw arrays archived.',
              font=_font(14), fill='#b5c1d0')
    for index, weight in enumerate(weights):
        x, y = 12 + (index % 4) * 168, 80 + (index // 4) * 188
        draw.text((x, y), f'kp {index:02d}  max {weight.max():.3f}', font=_font(13), fill='white')
        tile = Image.fromarray(_overlay(source_image, weight, 1.)).resize((160, 160))
        canvas.paste(_points(tile, keypoints[index:index + 1], grid_shape), (x, y + 22))
    return canvas


class DiffusionAttentionPolicy:
    """Read-only recorder at the adapter boundary; return the original action."""

    def __init__(self, adapter, trace, output, writer, *, control_hz=10, color_max=.05):
        if type(adapter.n_action_steps) is not int or adapter.n_action_steps < 1:
            raise ValueError('n_action_steps must be a positive integer')
        if not math.isfinite(color_max) or color_max <= 0:
            raise ValueError('color_max must be finite and positive')
        self.adapter, self.trace, self.writer = adapter, trace, writer
        self.output = Path(output)
        self.control_hz, self.color_max = control_hz, color_max
        self.frames, self.chunks, self.observations, self.actions = [], [], [], []
        self.capture_count = trace.capture_count
        self.source_images = None
        self.source_step = 0
        for folder in ('chunks', 'previews', 'keypoints'):
            (self.output / folder).mkdir(exist_ok=True)

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def reset(self):
        if self.frames:
            raise RuntimeError('Create a fresh Diffusion recorder for each rollout')
        self.adapter.reset()
        self.trace.clear()
        self.capture_count = self.trace.capture_count
        self.source_images = None
        self.source_step = 0

    def select_action(self, observation):
        saved_observation = {key: np.array(value, copy=True) for key, value in observation.items()}
        action = self.adapter.select_action(observation)
        capture = self.trace.latest
        if capture is None:
            raise RuntimeError('Diffusion returned an action without capturing visual spatial pooling')
        step = len(self.frames)
        expected_new = step % self.adapter.n_action_steps == 0
        actual_new = self.trace.capture_count != self.capture_count
        if actual_new != expected_new:
            raise RuntimeError('Diffusion spatial capture does not match the action queue timing')
        if actual_new:
            if self.trace.capture_count != self.capture_count + 1:
                raise RuntimeError('Expected exactly one visual encoding per new Diffusion action chunk')
            self.capture_count = self.trace.capture_count
            self.source_step = step
            self.source_images = {key: saved_observation[key].copy() for key in capture.camera_weights}
            chunk_index = len(self.chunks)
            filename = f'chunks/chunk_{chunk_index:04d}.npz'
            arrays = dict(source_state=saved_observation['observation.state'], source_step=np.array(step))
            atlases = {}
            for camera_index, key in enumerate(capture.camera_weights):
                arrays[f'image_{camera_index}'] = self.source_images[key]
                arrays[f'attention_{camera_index}'] = capture.camera_weights[key]
                arrays[f'mean_{camera_index}'] = capture.mean_weights[key]
                arrays[f'keypoints_{camera_index}'] = capture.keypoints[key]
                atlas_name = f'keypoints/chunk_{chunk_index:04d}_{key.rsplit(".", 1)[-1]}.png'
                render_keypoint_atlas(self.source_images[key], capture.camera_weights[key], capture.keypoints[key],
                                      name=key, grid_shape=capture.grid_shapes[key]).save(self.output / atlas_name)
                atlases[key] = atlas_name
            np.savez_compressed(self.output / filename, **arrays)
            self.chunks.append(dict(index=chunk_index, source_step=step, source_time_s=step / self.control_hz,
                                    cameras=list(capture.camera_weights), grid_shapes=capture.grid_shapes,
                                    keypoint_counts=capture.keypoint_counts, weights=filename, atlases=atlases,
                                    keypoint_reconstruction_max_abs_error=capture.keypoint_reconstruction_max_abs_error))
        if self.source_images is None:
            raise RuntimeError('Missing source images for cached Diffusion action')
        offset = step - self.source_step
        time_s, source_time_s = step / self.control_hz, self.source_step / self.control_hz
        frame = render_frame(observation, self.source_images, capture, offset, time_s=time_s,
                             source_time_s=source_time_s, chunk_index=len(self.chunks) - 1,
                             color_max=self.color_max)
        self.writer.append_data(frame)
        if actual_new:
            Image.fromarray(frame).save(self.output / 'previews' / f'chunk_{len(self.chunks)-1:04d}.png')
        if not self.frames:
            Image.fromarray(frame).save(self.output / 'preview.png')
        self.observations.append(saved_observation)
        self.actions.append(np.array(action, copy=True))
        self.frames.append(dict(frame=step, step=step, time_s=time_s, chunk_index=len(self.chunks)-1,
                                action_offset=offset, source_step=self.source_step, source_time_s=source_time_s,
                                source_age_s=time_s-source_time_s, new_prediction=actual_new,
                                saturated_cells={key: int((mean > self.color_max).sum())
                                                 for key, mean in capture.mean_weights.items()},
                                action=np.asarray(action).tolist(), execution_status='not_attempted'))
        return action

    def save_trace(self):
        if not self.observations:
            return None
        keys = tuple(self.chunks[0]['cameras'])
        arrays = dict(states=np.stack([row['observation.state'] for row in self.observations]),
                      actions=np.stack(self.actions), time_s=np.array([row['time_s'] for row in self.frames]),
                      source_steps=np.array([row['source_step'] for row in self.frames]),
                      action_offsets=np.array([row['action_offset'] for row in self.frames]))
        for index, key in enumerate(keys):
            arrays[f'images_{index}'] = np.stack([row[key] for row in self.observations])
        np.savez_compressed(self.output / 'trace.npz', **arrays)
        return dict(path='trace.npz', cameras=list(keys), frames=len(self.frames))


def _fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return dict(sha256=digest.hexdigest(), bytes=Path(path).stat().st_size)


def _rng_snapshot(torch):
    return dict(cpu=torch.get_rng_state().clone(),
                cuda=[state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def _rng_equal(torch, first, second):
    return bool(torch.equal(first['cpu'], second['cpu']) and len(first['cuda']) == len(second['cuda'])
                and all(torch.equal(a, b) for a, b in zip(first['cuda'], second['cuda'], strict=True)))


def verify_actions(adapter, observations, actions, *, seed, expected_initial_rng, expected_final_rng,
                   seed_function, torch):
    """Replay fixed observations without any hooks; never run counterfactual physics."""
    adapter.reset()
    seed_function(seed)
    initial_rng = _rng_snapshot(torch)
    mismatches, maximum = [], 0.
    for step, (observation, expected) in enumerate(zip(observations, actions, strict=True)):
        actual = adapter.select_action({key: value.copy() for key, value in observation.items()})
        if not np.array_equal(actual, expected):
            mismatches.append(step)
        maximum = max(maximum, float(np.max(np.abs(np.asarray(actual, dtype=float)-expected))))
    final_rng = _rng_snapshot(torch)
    initial_equal = _rng_equal(torch, initial_rng, expected_initial_rng)
    final_equal = _rng_equal(torch, final_rng, expected_final_rng)
    return dict(enabled=True, passed=not mismatches and initial_equal and final_equal,
                actions_checked=len(actions), actions_bitwise_equal=not mismatches,
                mismatched_steps=mismatches, max_absolute_action_difference=maximum,
                initial_torch_rng_equal=initial_equal, final_torch_rng_equal=final_equal,
                method='Reset adapter; reseed after model loading; replay all fixed observations with hooks removed.'), final_rng


def _validate_reference(path, *, config, checkpoint, seed, position, duration, n_action_steps, files):
    if path is None:
        return None
    path = Path(path).expanduser().resolve()
    reference = json.loads(path.read_text())
    expected = dict(config=config, seed=seed, position=position, requested_duration_s=duration,
                    control_hz=10, action_hold_s=.1, physics_timestep_s=.001,
                    stop_on_contact=False, contact_diagnostics=False)
    for key, value in expected.items():
        if reference.get(key) != value:
            raise ValueError(f'Reference report differs in {key}')
    policy = reference['policy']
    if (Path(policy['checkpoint']).resolve() != checkpoint or policy.get('policy_type') != 'diffusion'
            or policy.get('n_action_steps') != n_action_steps
            or policy.get('config_sha256') != files['config.json']['sha256']):
        raise ValueError('Reference checkpoint or execution settings differ')
    batch_config = path.parents[2] / 'eval_config.json'
    if batch_config.exists():
        recorded = json.loads(batch_config.read_text())
        if recorded.get('checkpoint_files') != files:
            raise ValueError('Reference batch checkpoint file hashes differ')
    return dict(path=str(path), fingerprint=_fingerprint(path), result=reference,
                batch_config=str(batch_config) if batch_config.exists() else None)


def _reference_comparison(reference, result):
    if reference is None:
        return None
    previous = reference['result']
    keys = ('success', 'task_success', 'stop_reason', 'completed_control_steps', 'runtime_error')
    fields = {key: previous.get(key) == result.get(key) for key in keys}
    fields['simulation_seconds'] = math.isclose(previous['simulation_seconds'], result['simulation_seconds'], abs_tol=1e-9)
    metric_errors, metric_equal = {}, True
    for key, value in previous.get('final_metrics', {}).items():
        current = (result.get('final_metrics') or {}).get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and isinstance(current, (int, float)):
            metric_errors[key] = abs(float(current)-value)
            metric_equal &= math.isclose(float(current), value, rel_tol=0, abs_tol=1e-8)
        elif isinstance(value, list) and value and all(isinstance(x, (float, int)) for x in value):
            equal = isinstance(current, list) and np.shape(current) == np.shape(value)
            metric_errors[key] = float(np.max(np.abs(np.array(current)-value))) if equal else None
            metric_equal &= equal and np.allclose(current, value, rtol=0, atol=1e-8)
        else:
            metric_equal &= current == value
    return dict(reference=reference['path'], reference_fingerprint=reference['fingerprint'],
                fields_match=fields, final_metrics_match=bool(metric_equal),
                final_metric_absolute_errors=metric_errors,
                same_recorded_outcome=bool(all(fields.values()) and metric_equal))


def _audit_video(path, expected_frames):
    import imageio.v2 as imageio
    with imageio.get_reader(str(path)) as reader:
        metadata = reader.get_meta_data()
        frames = sum(1 for _ in reader)
    return dict(path=Path(path).name, decoded_frames=frames, expected_frames=expected_frames,
                fps=metadata.get('fps'), full_decode=True,
                valid=frames == expected_frames and math.isclose(metadata.get('fps', 0), 10.))


def run_visualization(*, checkpoint, output=None, position='0', config=None,
                      positions_config=None, duration=60., n_action_steps=32, device='cuda', seed=0,
                      color_max=.05, torch_threads=1, reference_report=None, verify_actions_enabled=True):
    started_at = experiment_timestamp()
    # main() sets MUJOCO_GL before importing any simulator or ML execution code.
    import imageio.v2 as imageio
    import torch
    from mujoco_tools.diffusion_attention import DiffusionAttentionTrace
    from mujoco_tools.lerobot_policy import LeRobotPolicy, resolve_checkpoint
    from .rollout import _seed_policy, run_episode, select_layouts, validate_contract
    from ..environment.scene import build_scene, load_config
    from ..environment.simulation import Simulation

    if not math.isfinite(duration) or duration <= 0 or not math.isclose(duration*10, round(duration*10), abs_tol=1e-8):
        raise ValueError('duration must be positive and a multiple of 0.1 seconds')
    if type(seed) is not int or not 0 <= seed < 2**32 or torch_threads < 1:
        raise ValueError('Invalid seed or torch_threads')
    if not math.isfinite(color_max) or color_max <= 0:
        raise ValueError('color_max must be finite and positive')
    checkpoint = resolve_checkpoint(checkpoint)
    saved = json.loads((checkpoint / 'config.json').read_text())
    if saved.get('type') != 'diffusion':
        raise ValueError('A Diffusion checkpoint is required')
    scene_config = Path(config or CONFIG_DIR / 'insert_plug.yaml').expanduser().resolve()
    positions_config = Path(positions_config or CONFIG_DIR / 'insert_plug_positions.yaml').expanduser().resolve()
    layouts = select_layouts(load_config(scene_config), position, positions_config)
    if len(layouts) != 1:
        raise ValueError('Choose exactly one fixed position')
    name, cfg = layouts[0]
    destination = experiment_output_path(
        output, parent=OUTPUTS_DIR / 'insert_plug' / 'attention', checkpoint=checkpoint,
        saved=saved, config=cfg, kind=None, started_at=started_at)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite visualization: {destination}')
    files = {str(path.relative_to(checkpoint)): _fingerprint(path)
             for path in sorted(checkpoint.rglob('*')) if path.is_file()}
    source_files = {str(path): _fingerprint(path) for path in (scene_config, positions_config)}
    reference = _validate_reference(reference_report, config=cfg, checkpoint=checkpoint, seed=seed,
                                    position=name, duration=duration, n_action_steps=n_action_steps, files=files)
    torch.set_num_threads(torch_threads)
    adapter = LeRobotPolicy(checkpoint, device=device, n_action_steps=n_action_steps)
    validate_contract(cfg, adapter)
    # Match run_rollouts: initialize the model before seeding each episode.
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'configs').mkdir()
    (destination / 'configs/scene_config.yaml').write_bytes(scene_config.read_bytes())
    (destination / 'configs/positions_config.yaml').write_bytes(positions_config.read_bytes())
    scene = build_scene(cfg, destination / 'scene.xml')
    _seed_policy(seed)
    sim = Simulation(scene, cfg, render=True)
    initial_rng = _rng_snapshot(torch)
    try:
        with DiffusionAttentionTrace(adapter.policy.diffusion) as trace:
            with imageio.get_writer(str(destination / 'attention.mp4'), fps=sim.control_hz,
                                    codec='libx264', quality=8, macro_block_size=1,
                                    ffmpeg_log_level='error') as writer:
                recorder = DiffusionAttentionPolicy(adapter, trace, destination, writer,
                                                    control_hz=sim.control_hz, color_max=color_max)
                result = run_episode(sim, recorder, duration=duration, video_path=destination / 'rollout.mp4')
                final_rng = _rng_snapshot(torch)
                real_policy_metadata = deepcopy(adapter.metadata())
    finally:
        sim.close()
    for row in recorder.frames:
        index = row['step']
        row['execution_status'] = ('completed' if index < result['completed_control_steps'] else
                                   'attempted_not_completed' if index < result['attempted_control_steps'] else
                                   'not_attempted')
    archived_trace = recorder.save_trace()
    verification = dict(enabled=verify_actions_enabled, passed=None)
    replay_rng = None
    if verify_actions_enabled:
        try:
            verification, replay_rng = verify_actions(
                adapter, recorder.observations, recorder.actions, seed=seed,
                expected_initial_rng=initial_rng, expected_final_rng=final_rng,
                seed_function=_seed_policy, torch=torch)
        except Exception as exc:
            verification = dict(enabled=True, passed=False, error=f'{type(exc).__name__}: {exc}')
    rng_arrays = {}
    for prefix, state in [('initial', initial_rng), ('executed_final', final_rng), ('replay_final', replay_rng)]:
        if state is None:
            continue
        rng_arrays[f'{prefix}_cpu'] = state['cpu'].cpu().numpy()
        for index, array in enumerate(state['cuda']):
            rng_arrays[f'{prefix}_cuda_{index}'] = array.cpu().numpy()
    np.savez_compressed(destination / 'rng_states.npz', **rng_arrays)
    unchanged = all(_fingerprint(checkpoint / name) == fingerprint for name, fingerprint in files.items())
    sources_unchanged = all(_fingerprint(name) == fingerprint for name, fingerprint in source_files.items())
    videos = []
    video_error = None
    try:
        videos = [_audit_video(destination / 'attention.mp4', len(recorder.frames)),
                  _audit_video(destination / 'rollout.mp4', result['video_frames'])]
    except Exception as exc:
        video_error = f'{type(exc).__name__}: {exc}'
    valid = bool(result['runtime_error'] is None and recorder.frames and unchanged and sources_unchanged
                 and video_error is None and all(row['valid'] for row in videos)
                 and (not verify_actions_enabled or verification['passed']))
    report = dict(
        started_at=started_at, position=name, seed=seed, config=cfg, policy=real_policy_metadata, rollout=result,
        checkpoint_files=files, source_files=source_files,
        attention=dict(method='encoder_spatial_softmax', reduction='unweighted_mean_over_keypoints',
                       native_grid='measured from actual pool', color_scale=[0, color_max],
                       keypoint_atlas_color_scale=[0, 1], display_interpolation='bilinear',
                       display_saturation='probabilities above color_max clipped only for display',
                       interpretation='Visual encoder pooling; not Transformer attention, action-specific attribution, or causal pixel importance.',
                       camera_comparison='Each camera/keypoint has its own unit-sum probability map; total mass cannot rank cameras.',
                       keypoint_mapping='Feature-grid cell centers mapped to RGB pixel centers, with half-cell padding.',
                       timing='Only actual new action chunks encode images. Cached actions retain their original prediction image.',
                       frame_timing='Observations before each selected 100 ms control interval; final physical view is in rollout.mp4.',
                       chunks=recorder.chunks, frames=recorder.frames),
        trace=archived_trace, comparison=_reference_comparison(reference, result),
        validation=dict(valid=valid, action_replay=verification, checkpoint_files_unchanged=unchanged,
                        source_files_unchanged=sources_unchanged, videos=videos, video_error=video_error),
        video='attention.mp4', physical_video='rollout.mp4', preview='preview.png', scene='scene.xml', rng='rng_states.npz')
    (destination / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(destination), frames=len(recorder.frames), chunks=len(recorder.chunks),
                          task_success=result['task_success'], stop_reason=result['stop_reason'], validation_passed=valid)), flush=True)
    if not valid:
        raise RuntimeError(f'Diffusion visualization validation failed; see {destination / "report.json"}')
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', help='New directory; existing outputs are preserved')
    parser.add_argument('--position', default='0')
    parser.add_argument('--config')
    parser.add_argument('--positions-config')
    parser.add_argument('--duration', type=float, default=60.)
    parser.add_argument('--n-action-steps', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--backend', choices=('egl', 'osmesa', 'glfw'), default='egl')
    parser.add_argument('--color-max', type=float, default=.05)
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--reference-report', help='Require the reference rollout config, seed and checkpoint to match')
    parser.add_argument('--verify-actions', dest='verify_actions_enabled', action=argparse.BooleanOptionalAction,
                        default=True, help='Replay all fixed observations without hooks and require bitwise action/RNG agreement')
    args = vars(parser.parse_args(argv))
    os.environ['MUJOCO_GL'] = args.pop('backend')
    run_visualization(**args)


if __name__ == '__main__':
    main()
