"""Discrete socket layouts with full-mesh camera and physical-task checks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
import yaml

from mujoco_tools.cameras import camera_metadata
from mujoco_tools.paths import WORKSPACE_ROOT
from ..control.expert import Expert
from .scene import build_scene, load_config
from .simulation import Simulation


VISUAL_GEOMS = ('source_holder_visual', 'target_holder_visual', 'peg_visual')


class PositionsDumper(yaml.SafeDumper):
    """Keep coordinate pairs easy to edit without flattening layout mappings."""


def _represent_list(dumper, values):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', values,
                                     flow_style=all(not isinstance(x, (list, dict)) for x in values))


PositionsDumper.add_representer(list, _represent_list)


def manifest_row(row):
    """Keep verified coordinates and concise evidence, not per-vertex reports."""
    result = {key: row[key] for key in ('name', 'source_xy', 'target_xy', 'source_steps', 'target_steps')}
    result['visibility'] = {}
    for phase in ('initial', 'final'):
        report = row['visibility'][phase]
        margin = (min(item['edge_margin_px'] for item in report['objects'].values())
                  if 'objects' in report else report['min_edge_margin_px'])
        result['visibility'][phase] = dict(all_inside=report['all_inside'],
                                           min_edge_margin_px=round(float(margin), 3))
    physics = row['physics_check']
    result['physics_check'] = {key: physics[key] for key in (
        'success', 'duration_seconds', 'control_steps', 'radial_error_m', 'insertion_depth_m')}
    checks = physics.get('execution_checks', physics)
    result['physics_check'].update(
        collision_checks_enabled=checks.get('collision_checks_enabled', True),
        contact_checks=checks.get('checks', checks.get('contact_checks')),
        unsafe_contact_count=checks['unsafe_contact_count'],
    )
    return result


def mesh_world_vertices(model, data, geom_name):
    """Return actual compiled visual vertices, including MuJoCo mesh centering."""
    geom = model.geom(geom_name).id
    if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
        raise ValueError(f'{geom_name} must be a mesh')
    mesh = int(model.geom_dataid[geom])
    start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    vertices = model.mesh_vert[start:start + count].astype(float)
    return vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


def head_visibility(sim, margin_px=5.0):
    """Project every visual-mesh vertex into head RGB's complete view frustum.

    This tests frame containment, not occlusion by the moving robot. Frustum
    half-spaces are convex, so containing all vertices also contains the mesh
    triangles. It uses rendered mesh geometry, not proxy socket dimensions.
    """
    width, height = (int(sim.config['cameras'][key]) for key in ('width', 'height'))
    camera = camera_metadata(sim.model, sim.data, width, height)['head']
    matrix = np.asarray(camera['intrinsics'])
    pose = np.asarray(camera['world_from_camera_optical_at_capture'])
    result = {}
    for name in VISUAL_GEOMS:
        world = mesh_world_vertices(sim.model, sim.data, name)
        optical = (world - pose[:3, 3]) @ pose[:3, :3]
        z = optical[:, 2]
        valid_depth = bool(np.all((z > camera['near_m']) & (z < camera['far_m'])))
        projected = optical @ matrix.T
        pixels = projected[:, :2] / projected[:, 2:3]
        low, high = pixels.min(axis=0), pixels.max(axis=0)
        edge_margin = float(min(low[0], low[1], width - 1 - high[0], height - 1 - high[1]))
        inside = valid_depth and np.isfinite(pixels).all() and edge_margin >= margin_px
        result[name] = dict(
            inside=bool(inside), vertices=int(len(world)),
            pixel_bounds=[float(low[0]), float(low[1]), float(high[0]), float(high[1])],
            edge_margin_px=edge_margin, depth_range_m=[float(z.min()), float(z.max())],
        )
    return dict(
        all_inside=all(item['inside'] for item in result.values()),
        required_margin_px=float(margin_px), width=width, height=height,
        method='all compiled visual mesh vertices projected into head optical camera',
        occlusion_checked=False, objects=result,
    )


def candidate_layouts(base_config, seed=20261107, grid_spacing_m=.02, max_grid_steps=2):
    """Yield deterministic random pairs on a finite integer grid about the base."""
    if not np.isfinite(grid_spacing_m) or grid_spacing_m <= 0 or max_grid_steps < 1:
        raise ValueError('Grid spacing and step range must be positive')
    offsets = [(x, y) for x in range(-max_grid_steps, max_grid_steps + 1)
               for y in range(-max_grid_steps, max_grid_steps + 1) if (x, y) != (0, 0)]
    pairs = [(source, target) for source in offsets for target in offsets]
    rng = np.random.default_rng(seed)
    for index in rng.permutation(len(pairs)):
        source, target = pairs[index]
        yield dict(
            source_steps=list(source), target_steps=list(target),
            source_xy=np.round(np.asarray(base_config['board']['source_xy']) + grid_spacing_m * np.asarray(source), 10).tolist(),
            target_xy=np.round(np.asarray(base_config['board']['target_xy']) + grid_spacing_m * np.asarray(target), 10).tolist(),
        )


def select_layouts(base_config, count=10, seed=20261107, grid_spacing_m=.02,
                   max_grid_steps=2, margin_px=5.0, accepted_callback=None):
    """Select unique socket positions after a complete collision-guarded run."""
    if count < 1 or count > (2 * max_grid_steps + 1) ** 2 - 1:
        raise ValueError('Requested count exceeds the number of unique nonzero grid offsets')
    selected, used_source, used_target = [], set(), set()
    rejected = dict(duplicate_position=0, geometry_or_planning=0, outside_view=0, execution=0)
    with TemporaryDirectory(prefix='insert-plug-layouts-') as folder:
        for candidate in candidate_layouts(base_config, seed, grid_spacing_m, max_grid_steps):
            if tuple(candidate['source_steps']) in used_source or tuple(candidate['target_steps']) in used_target:
                rejected['duplicate_position'] += 1
                continue
            config = copy.deepcopy(base_config)
            config['board'].update(source_xy=candidate['source_xy'], target_xy=candidate['target_xy'])
            sim = None
            try:
                sim = Simulation(build_scene(config, Path(folder)/'scene.xml'), config, render=False)
                initial = head_visibility(sim, margin_px)
                if not initial['all_inside']:
                    rejected['outside_view'] += 1
                    continue
                expert = Expert(sim, verbose=False)
                expert.prepare()
                result = expert.run()
                if not result['success']:
                    rejected['execution'] += 1
                    print(json.dumps(dict(rejected=candidate, reason=result['failure_reason'])), flush=True)
                    continue
                final = head_visibility(sim, margin_px)
                if not final['all_inside']:
                    rejected['outside_view'] += 1
                    continue
                row = dict(name=f'pos{len(selected)}', **candidate,
                    visibility=dict(initial=initial, final=final),
                    physics_check=dict(success=True, duration_seconds=float(sim.data.time),
                        control_steps=result['steps'], radial_error_m=result['metrics']['radial_error_m'],
                        insertion_depth_m=result['metrics']['insertion_depth_m'],
                        execution_checks=result['execution_checks']))
                selected.append(row)
                used_source.add(tuple(candidate['source_steps']))
                used_target.add(tuple(candidate['target_steps']))
                print(json.dumps(dict(accepted=row['name'], source_xy=row['source_xy'],
                                      target_xy=row['target_xy'], seconds=row['physics_check']['duration_seconds'])), flush=True)
                if accepted_callback is not None:
                    accepted_callback(selected, rejected)
                if len(selected) == count:
                    return selected, rejected
            except (ValueError, RuntimeError) as error:
                rejected['geometry_or_planning'] += 1
                print(json.dumps(dict(rejected=candidate, reason=f'{type(error).__name__}: {error}')), flush=True)
            finally:
                if sim is not None:
                    sim.close()
    raise RuntimeError(f'Only {len(selected)} of {count} layouts passed; rejected={rejected}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=None)
    parser.add_argument('--output', default=str(WORKSPACE_ROOT/'config'/'insert_plug_positions.yaml'))
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20261107)
    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser().resolve() if args.config else WORKSPACE_ROOT/'config'/'insert_plug.yaml'
    config = load_config(config_path)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    document = dict(
        format_version=1,
        base_config=str(config_path.relative_to(WORKSPACE_ROOT)) if config_path.is_relative_to(WORKSPACE_ROOT) else str(config_path),
        base_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        base_source_xy=config['board']['source_xy'], base_target_xy=config['board']['target_xy'],
        grid_spacing_m=.02, max_grid_steps=2, selection_seed=args.seed,
        episodes_per_position=10, recording_hz=config['recording']['sample_hz'],
        control_hz=config['task']['control_hz'], depth_enabled=True,
        required_edge_margin_px=5.0,
        visibility_method='all compiled visual mesh vertices projected into head optical camera',
        occlusion_checked=False, positions=[],
    )

    def save_progress(positions, rejected):
        document.update(positions=[manifest_row(row) for row in positions], rejected_candidates=rejected,
                        selection_complete=len(positions) == args.count)
        temporary = output.with_suffix('.yaml.partial')
        temporary.write_text(yaml.dump(document, Dumper=PositionsDumper, sort_keys=False), encoding='utf8')
        temporary.replace(output)

    select_layouts(config, count=args.count, seed=args.seed, accepted_callback=save_progress)
    return 0


if __name__ == '__main__':
    main()
