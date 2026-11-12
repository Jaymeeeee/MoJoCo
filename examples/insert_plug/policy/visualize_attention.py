"""Visualize ACT action-query attention during a real insert-plug rollout.

Queued actions show their original prediction image. With temporal ensembling,
every control step shows the latest prediction's attention and the separate
blend weights of the predictions contributing to the executed action.
Run with ``python -m examples.insert_plug.policy.visualize_attention --help``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mujoco_tools.paths import OUTPUTS_DIR
from .output_naming import experiment_output_path, experiment_timestamp


def _font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def _colors(values):
    """A fixed blue/cyan/yellow/red scale, shared by cameras and time steps."""
    palette = np.array([[32, 70, 190], [20, 175, 220], [255, 220, 70], [240, 55, 35]])
    levels = np.linspace(0, 1, len(palette))
    return np.stack([np.interp(values, levels, palette[:, c]) for c in range(3)], axis=-1)


def _overlay(rgb, weights, color_max):
    height, width = rgb.shape[:2]
    # This interpolation is only for display. The NPZ retains the native grid.
    resized = np.asarray(Image.fromarray(weights.astype(np.float32)).resize(
        (width, height), Image.Resampling.BILINEAR))
    levels = np.clip(resized / color_max, 0, 1)
    alpha = (.75 * np.sqrt(levels))[..., None]
    return np.round(rgb * (1 - alpha) + _colors(levels) * alpha).astype(np.uint8)


def temporal_contributions(step, chunk_size, coefficient, control_hz=10):
    """Identify overlapping predictions, matching LeRobot's oldest-first weights.

    These are action blend weights, not weights of an attention map aligned to
    the current image. Older queries refer to different source observations.
    """
    if type(step) is not int or step < 0 or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('step must be nonnegative and chunk_size positive integers')
    if not math.isfinite(coefficient) or not math.isfinite(control_hz) or control_hz <= 0:
        raise ValueError('coefficient must be finite and control_hz finite and positive')
    sources = list(range(max(0, step-chunk_size+1), step+1))
    logits = -coefficient * np.arange(len(sources), dtype=np.float64)
    weights = np.exp(logits-logits.max())
    weights /= weights.sum()
    return [dict(chunk_index=source, source_step=source, source_time_s=source/control_hz,
                 query_index=step-source, source_age_s=(step-source)/control_hz, weight=float(weight))
            for source, weight in zip(sources, weights, strict=True)]


def render_frame(live_images, source_images, capture, query_index, *, time_s,
                 source_time_s, chunk_index, color_max, ensemble=None):
    """Render a diagnostic frame; absolute attention scale never auto-rescales."""
    keys = tuple(capture.camera_weights)
    if len(keys) != 2:
        raise ValueError('insert_plug attention visualization requires two cameras')
    canvas = Image.new('RGB', (688, 1056 if ensemble is not None else 912), (20, 25, 34))
    draw = ImageDraw.Draw(canvas)
    white, muted, accent = '#edf1f7', '#b5c1d0', '#65d8cc'
    draw.text((16, 12), 'ACT | action-to-image attention', font=_font(25), fill=white)
    draw.text((16, 49), f't = {time_s:.1f}s   chunk {chunk_index}   action query {query_index}',
              font=_font(18), fill=accent)
    for column, key in enumerate(keys):
        x = 16 + 336 * column
        name = key.rsplit('.', 1)[-1]
        draw.text((x, 83), f'LIVE / {name}', font=_font(17), fill=white)
        canvas.paste(Image.fromarray(live_images[key]).resize((320, 320)), (x, 109))
        label = 'CURRENT q0 ATTENTION' if ensemble is not None else 'SOURCE + ATTENTION'
        draw.text((x, 444), f'{label} / {name}', font=_font(16), fill=accent)
        weights = capture.camera_weights[key][query_index]
        blended = _overlay(source_images[key], weights, color_max)
        canvas.paste(Image.fromarray(blended).resize((320, 320)), (x, 472))
        h, w = capture.grid_shapes[key]
        draw.text((x, 801), f'{h}x{w} grid | token mass {weights.sum():.1%}',
                  font=_font(15), fill=white)
    draw.text((16, 831), f'Prediction image: {source_time_s:.1f}s  |  age: {time_s-source_time_s:.1f}s',
              font=_font(16), fill=accent)
    gradient = np.repeat(_colors(np.linspace(0, 1, 180))[None], 12, axis=0).astype(np.uint8)
    canvas.paste(Image.fromarray(gradient), (16, 864))
    draw.text((208, 858), f'0 to {color_max:g} weight / token (fixed; clipped)', font=_font(14), fill=muted)
    draw.text((16, 886), 'Final decoder, mean of heads. Attention is not causal pixel attribution.',
              font=_font(13), fill=muted)
    if ensemble is not None:
        contributors = ensemble['contributions']
        draw.text((16, 914), f'Executed action: blend of {len(contributors)} overlapping predictions',
                  font=_font(17), fill=white)
        draw.text((16, 942), f"Latest: {ensemble['current_prediction_weight']:.1%}  |  "
                  f"history: {ensemble['historical_prediction_weight']:.1%}  |  "
                  f"m = {ensemble['coefficient']:g}", font=_font(16), fill=accent)
        max_weight = max(item['weight'] for item in contributors)
        bar_width = 656 / len(contributors)
        for index, item in enumerate(contributors):
            x = 16 + index*bar_width
            height = 36 * item['weight'] / max_weight
            draw.rectangle((x, 1008-height, x+max(1, bar_width-2), 1008),
                           fill='#ffdc46' if index == len(contributors)-1 else '#659ccb')
        draw.text((16, 1020), 'Action blend weights: oldest -> newest (yellow). Heatmap shows newest only.',
                  font=_font(13), fill=muted)
    return np.asarray(canvas)


class AttentionPolicy:
    """Observe the adapter without replacing actions or changing queue timing."""

    def __init__(self, adapter, trace, output, writer, *, control_hz=10, color_max=.05):
        self.adapter = adapter
        self.trace = trace
        self.output = Path(output)
        self.writer = writer
        self.control_hz = control_hz
        self.color_max = color_max
        self.frames = []
        self.chunks = []
        self.source_images = None
        self.capture_count = trace.capture_count
        self.query_index = 0
        self.source_step = 0
        self.temporal_ensemble_coeff = getattr(adapter, 'temporal_ensemble_coeff', None)
        if self.temporal_ensemble_coeff is not None and adapter.n_action_steps != 1:
            raise ValueError('Temporal ensembling requires n_action_steps=1')
        (self.output / 'chunks').mkdir(exist_ok=True)
        (self.output / 'previews').mkdir(exist_ok=True)

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def reset(self):
        if self.frames:
            raise RuntimeError('Create a fresh attention recorder for each rollout')
        self.adapter.reset()
        self.trace.clear()
        self.capture_count = self.trace.capture_count
        self.source_images = None
        self.query_index = 0

    def select_action(self, observation):
        action = self.adapter.select_action(observation)
        capture = self.trace.latest
        if capture is None:
            raise RuntimeError('ACT returned an action without capturing decoder attention')
        step = len(self.frames)
        new_chunk = self.trace.capture_count != self.capture_count
        if new_chunk:
            if self.trace.capture_count != self.capture_count + 1:
                raise RuntimeError('Expected one ACT inference per new action chunk')
            if self.source_images is not None and self.query_index != self.adapter.n_action_steps:
                raise RuntimeError('ACT predicted a new chunk before its action queue was consumed')
            self.capture_count = self.trace.capture_count
            self.query_index = 0
            self.source_step = step
            self.source_images = {key: observation[key].copy() for key in capture.camera_weights}
            chunk_index = len(self.chunks)
            filename = f'chunks/chunk_{chunk_index:04d}.npz'
            arrays = dict(token_weights=capture.token_weights,
                          source_state=np.array(observation['observation.state'], copy=True))
            for camera_index, key in enumerate(capture.camera_weights):
                arrays[f'image_{camera_index}'] = self.source_images[key]
                arrays[f'attention_{camera_index}'] = capture.camera_weights[key]
            np.savez_compressed(self.output / filename, **arrays)
            self.chunks.append(dict(index=chunk_index, source_step=step,
                                    source_time_s=step / self.control_hz,
                                    cameras=list(capture.camera_weights),
                                    grid_shapes=capture.grid_shapes,
                                    decoder_layer_index=capture.decoder_layer_index,
                                    special_tokens=list(capture.special_token_weights),
                                    weights=filename))
        elif self.source_images is None or self.query_index >= self.adapter.n_action_steps:
            raise RuntimeError('Attention capture does not match the ACT action queue')
        query = self.query_index
        if query >= capture.token_weights.shape[0]:
            raise RuntimeError('Executed action query exceeds the captured chunk')
        time_s = step / self.control_hz
        source_time_s = self.source_step / self.control_hz
        ensemble = None
        if self.temporal_ensemble_coeff is not None:
            if not new_chunk or query != 0 or self.source_step != step:
                raise RuntimeError('Temporal ensembling requires a new prediction at every control step')
            contributions = temporal_contributions(step, capture.token_weights.shape[0],
                                                  self.temporal_ensemble_coeff, self.control_hz)
            ensemble = dict(coefficient=self.temporal_ensemble_coeff, contributions=contributions,
                            current_prediction_weight=contributions[-1]['weight'],
                            historical_prediction_weight=1-contributions[-1]['weight'])
        frame = render_frame(observation, self.source_images, capture, query,
                             time_s=time_s, source_time_s=source_time_s,
                             chunk_index=len(self.chunks) - 1, color_max=self.color_max,
                             ensemble=ensemble)
        self.writer.append_data(frame)
        if new_chunk and (self.temporal_ensemble_coeff is None or step % round(self.control_hz) == 0):
            Image.fromarray(frame).save(self.output / 'previews' / f'chunk_{len(self.chunks)-1:04d}.png')
        if not self.frames:
            Image.fromarray(frame).save(self.output / 'preview.png')
        self.frames.append(dict(
            frame=step, time_s=time_s, chunk_index=len(self.chunks) - 1, query_index=query,
            source_time_s=source_time_s, source_age_s=(step-self.source_step) / self.control_hz,
            temporal_ensemble=ensemble,
            camera_mass={key: float(weights[query].sum()) for key, weights in capture.camera_weights.items()},
            special_token_mass={key: float(weights[query]) for key, weights in capture.special_token_weights.items()},
            saturated_cells={key: int((weights[query] > self.color_max).sum())
                             for key, weights in capture.camera_weights.items()},
            action=np.asarray(action).tolist()))
        self.query_index += 1
        return action


def run_visualization(*, checkpoint, output=None, position='0', config=None,
                      duration=20., n_action_steps=None, device='cpu', seed=1000,
                      color_max=.05, torch_threads=2, temporal_ensemble_coeff=None):
    started_at = experiment_timestamp()
    # Import rendering only after main() selects the requested GL backend.
    import imageio.v2 as imageio
    import torch
    from mujoco_tools.act_attention import ACTAttentionTrace
    from mujoco_tools.lerobot_policy import LeRobotPolicy, resolve_checkpoint
    from .rollout import _seed_policy, run_episode, select_layouts, validate_contract
    from ..environment.scene import build_scene, load_config
    from ..environment.simulation import Simulation

    if not math.isfinite(duration) or duration <= 0 or not math.isclose(duration*10, round(duration*10), abs_tol=1e-8):
        raise ValueError('duration must be positive and an exact multiple of 0.1s')
    if not math.isfinite(color_max) or color_max <= 0:
        raise ValueError('color_max must be finite and positive')
    if torch_threads < 1:
        raise ValueError('torch_threads must be positive')
    checkpoint = resolve_checkpoint(checkpoint)
    saved = json.loads((checkpoint / 'config.json').read_text())
    coefficient = (saved.get('temporal_ensemble_coeff') if temporal_ensemble_coeff is None
                   else temporal_ensemble_coeff)
    if n_action_steps is None:
        n_action_steps = 1 if coefficient is not None else saved['n_action_steps']
    if coefficient is not None and n_action_steps != 1:
        raise ValueError('Temporal ensembling requires n_action_steps=1')
    layouts = select_layouts(load_config(config), position)
    if len(layouts) != 1:
        raise ValueError('Choose one position for each attention visualization')
    name, cfg = layouts[0]
    destination = experiment_output_path(
        output, parent=OUTPUTS_DIR / 'insert_plug' / 'attention', checkpoint=checkpoint,
        saved=saved, config=cfg, kind='ensemble' if coefficient is not None else 'chunked',
        started_at=started_at)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite visualization: {destination}')
    torch.set_num_threads(torch_threads)
    _seed_policy(seed)
    adapter = LeRobotPolicy(checkpoint, device=device, n_action_steps=n_action_steps,
                            temporal_ensemble_coeff=temporal_ensemble_coeff)
    coefficient = adapter.temporal_ensemble_coeff
    validate_contract(cfg, adapter)
    destination.mkdir(parents=True, exist_ok=False)
    scene = build_scene(cfg, destination / 'scene.xml')
    sim = Simulation(scene, cfg, render=True)
    try:
        with ACTAttentionTrace(adapter.policy.model) as trace:
            with imageio.get_writer(str(destination / 'attention.mp4'), fps=sim.control_hz,
                                    codec='libx264', quality=8, macro_block_size=1,
                                    ffmpeg_log_level='error') as writer:
                recorder = AttentionPolicy(adapter, trace, destination, writer,
                                           control_hz=sim.control_hz, color_max=color_max)
                result = run_episode(sim, recorder, duration=duration)
    finally:
        sim.close()
    for row in recorder.frames:
        index = row['frame']
        row['execution_status'] = (
            'completed' if index < result['completed_control_steps'] else
            'attempted_not_completed' if index < result['attempted_control_steps'] else
            'not_attempted')
    report = dict(
        started_at=started_at, position=name, seed=seed, config=cfg, policy=adapter.metadata(), rollout=result,
        attention=dict(
            method='last_decoder_cross_attention', head_reduction='mean',
            native_grid='measured from each backbone forward', color_scale=[0, color_max],
            display_interpolation='bilinear', display_saturation='weights above color_max clipped only for display',
            interpretation='Action-query routing to encoder memory. Encoder mixing and CNN receptive fields prevent causal pixel attribution.',
            timing=('Each step shows query 0 of the latest prediction on the current image; '
                    'the executed action also blends historical predictions, reported separately.'
                    if coefficient is not None else
                    'Heatmaps use the source image of the executed cached action, not the live image.'),
            displayed_query_role=('newest_contributor_only' if coefficient is not None else 'selected_cached_action'),
            temporal_ensemble=dict(enabled=coefficient is not None, coefficient=coefficient,
                                   weight_order='oldest_to_newest',
                                   weight_formula='normalized exp(-coefficient * rank)',
                                   spatial_alignment='Historical attention maps are retained on their own source images; not averaged onto the current image.'),
            action_scope='Entire 8-D action vector; not an independent map for each joint.',
            frame_timing='Each video frame is the observation before the selected 100 ms action interval.',
            chunks=recorder.chunks, frames=recorder.frames),
        video='attention.mp4', preview='preview.png', scene='scene.xml')
    (destination / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(destination), frames=len(recorder.frames),
                          chunks=len(recorder.chunks), stop_reason=result['stop_reason'],
                          task_success=result['task_success']), ensure_ascii=False), flush=True)
    if result['runtime_error'] is not None or not recorder.frames:
        raise RuntimeError(f"Attention rollout failed: {result['runtime_error']}")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Local ACT step or pretrained_model directory')
    parser.add_argument('--output', help='New output directory (never overwritten)')
    parser.add_argument('--position', default='0', help='Fixed layout 0..9 or config')
    parser.add_argument('--config', help='Task scene YAML override')
    parser.add_argument('--duration', type=float, default=20., help='Maximum simulation seconds')
    parser.add_argument('--n-action-steps', type=int, help='Defaults to 1 with temporal ensembling, otherwise checkpoint setting')
    parser.add_argument('--temporal-ensemble-coeff', type=float,
                        help='Enable temporal ensembling, e.g. 0.01; 0 means uniform; omitted preserves checkpoint')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--backend', choices=('osmesa', 'egl', 'glfw'), default='osmesa')
    parser.add_argument('--color-max', type=float, default=.05, help='Fixed maximum attention weight on the color scale')
    parser.add_argument('--torch-threads', type=int, default=2)
    args = vars(parser.parse_args(argv))
    os.environ['MUJOCO_GL'] = args.pop('backend')
    run_visualization(**args)


if __name__ == '__main__':
    main()
