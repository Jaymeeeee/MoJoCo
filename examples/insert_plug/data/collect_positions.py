#!/usr/bin/env python3
"""Collect fixed episodes for each vetted 2 cm grid layout, without jitter."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
import math
import multiprocessing
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import yaml

from mujoco_tools.paths import WORKSPACE_ROOT


def emit(**values):
    print(json.dumps(values, ensure_ascii=False), flush=True)


def episode_config(base, position, sample_hz, grid_spacing, control_hz=None):
    config = copy.deepcopy(base)
    config['board']['source_xy'] = list(position['source_xy'])
    config['board']['target_xy'] = list(position['target_xy'])
    config.setdefault('recording', {})['sample_hz'] = sample_hz
    if control_hz is not None:
        config['task']['control_hz'] = control_hz
    config['cameras'].setdefault('depth', {})['enabled'] = True
    config['collection'] = dict(
        position_id=position['name'], grid_spacing_m=grid_spacing,
        base_source_xy=list(base['board']['source_xy']),
        base_target_xy=list(base['board']['target_xy']),
        source_steps=position['source_steps'], target_steps=position['target_steps'],
        layout_fixed_within_position=True,
    )
    return config


def existing_episode(directory, stem, config):
    """Only resume verified, complete episodes from exactly this configuration."""
    if not directory.exists():
        return None
    from examples.insert_plug.data.dataset import validate_episode
    expected = {f'{stem}.{suffix}' for suffix in ('h5', 'yaml', 'xml', 'mp4')}
    if {p.name for p in directory.iterdir()} != expected:
        raise RuntimeError(f'Incomplete/extraneous files in {directory}; preserved for inspection')
    saved = yaml.safe_load((directory / f'{stem}.yaml').read_text())
    if saved != config:
        raise RuntimeError(f'Existing episode configuration differs: {directory}')
    report = validate_episode(directory / f'{stem}.h5')
    if not report['valid'] or not report['success']:
        raise RuntimeError(f'Existing episode did not pass validation/success: {directory}')
    return report['steps']


def collect_position(position, base, output, episodes, sample_hz, backend, grid_spacing, control_hz=None):
    # Imports in each spawned process avoid sharing an OpenGL context.
    from examples.insert_plug.cli import collect
    from examples.insert_plug.data.dataset import validate_episode

    config = episode_config(base, position, sample_hz, grid_spacing, control_hz)
    index = int(position['name'].removeprefix('pos'))
    destination = Path(output) / position['name']
    with TemporaryDirectory(prefix=f'insert_plug_{position["name"]}_') as temp:
        config_path = Path(temp) / 'config.yaml'
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf8')
        for local in range(episodes):
            seed = index * episodes + local
            stem = f'episode_{seed:06d}'
            directory = destination / stem
            prior_frames = existing_episode(directory, stem, config)
            if prior_frames is not None:
                emit(event='episode_verified_existing', position=position['name'],
                     episode=seed, frames=prior_frames)
                continue
            args = SimpleNamespace(
                command='collect', config=str(config_path), output=str(destination),
                episodes=1, seed=seed, randomize_mm=0., backend=backend,
                width=None, control_hz=None, sample_hz=sample_hz,
                no_images=False, viewer=False, realtime=False, quiet=True, video=False,
            )
            captured = io.StringIO()
            try:
                with redirect_stdout(captured):
                    status = collect(args)
                if status != 0:
                    raise RuntimeError(f'Expert returned status {status}')
                report = validate_episode(directory / f'{stem}.h5')
                if not report['valid'] or not report['success']:
                    raise RuntimeError('Completed recording failed validation')
                emit(event='episode_complete', position=position['name'], episode=seed,
                     frames=report['steps'], success=True,
                     bytes=sum(p.stat().st_size for p in directory.iterdir()))
            except BaseException:
                emit(event='episode_failed', position=position['name'], episode=seed,
                     diagnostic=captured.getvalue()[-6000:])
                raise
    return position['name']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--positions-config', default=str(WORKSPACE_ROOT / 'config/insert_plug_positions.yaml'))
    parser.add_argument('--output', default=str(WORKSPACE_ROOT / 'outputs/datasets/insert_plug'))
    parser.add_argument('--episodes-per-position', type=int, default=10)
    parser.add_argument('--sample-hz', type=float, help='Recording rate; defaults to the position manifest')
    parser.add_argument('--control-hz', type=float, help='Expert target-update rate; defaults to manifest/base config')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--backend', choices=('egl', 'osmesa'), default='egl')
    parser.add_argument('--position', action='append', help='Optionally collect only named positions, e.g. pos0')
    args = parser.parse_args()
    if args.episodes_per_position < 1 or args.workers < 1:
        parser.error('episodes and workers must be positive')
    manifest = yaml.safe_load(Path(args.positions_config).read_text())
    base_path = Path(manifest['base_config'])
    if not base_path.is_absolute():
        base_path = WORKSPACE_ROOT / base_path
    base = yaml.safe_load(base_path.read_text())
    sample_hz = args.sample_hz if args.sample_hz is not None else float(manifest['recording_hz'])
    control_hz = (args.control_hz if args.control_hz is not None else
                  float(manifest.get('control_hz', base['task']['control_hz'])))
    if not all(math.isfinite(value) and value > 0 for value in (sample_hz, control_hz)):
        parser.error('sample-hz and control-hz must be finite and positive')
    expected_hash = manifest.get('base_config_sha256')
    if expected_hash and hashlib.sha256(base_path.read_bytes()).hexdigest() != expected_hash:
        parser.error('Base configuration changed after layout validation; regenerate the position manifest first')
    if not args.position and not manifest.get('selection_complete', False):
        parser.error('Position selection is incomplete; finish layout validation or explicitly select ready positions')
    positions = manifest['positions']
    names = [position['name'] for position in positions]
    if len(names) != len(set(names)) or any(name not in {f'pos{i}' for i in range(10)} for name in names):
        parser.error('Position names must be distinct pos0..pos9')
    if args.position:
        missing = set(args.position) - set(names)
        if missing:
            parser.error(f'Unknown positions: {sorted(missing)}')
        positions = [p for p in positions if p['name'] in args.position]
    if not positions:
        parser.error('No positions selected')
    output = Path(args.output).expanduser().resolve()
    emit(event='batch_start', positions=[p['name'] for p in positions],
         episodes_per_position=args.episodes_per_position, sample_hz=sample_hz, control_hz=control_hz,
         workers=args.workers, output=str(output))
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = [executor.submit(collect_position, p, base, str(output),
                                   args.episodes_per_position, sample_hz, args.backend,
                                   manifest['grid_spacing_m'], control_hz) for p in positions]
        for future in as_completed(futures):
            emit(event='position_complete', position=future.result())
    emit(event='batch_complete', positions=len(positions),
         episodes=len(positions) * args.episodes_per_position)


if __name__ == '__main__':
    main()
