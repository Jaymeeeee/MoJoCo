"""Names for task diagnostics, without loading a model or simulator."""
from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import yaml

from mujoco_tools.paths import CONFIG_DIR


def experiment_timestamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def _layout_split(config):
    board = config.get('board', {})
    coordinates = [board.get(key) for key in ('source_xy', 'target_xy')]
    if any(not isinstance(values, (list, tuple)) or len(values) != 2 for values in coordinates):
        return None
    matches = set()
    for split, filename in (('ind', 'insert_plug_positions.yaml'),
                            ('ood', 'insert_plug_generalization_v1.yaml')):
        path = CONFIG_DIR / filename
        if not path.is_file():
            continue
        manifest = yaml.safe_load(path.read_text(encoding='utf-8'))
        for row in manifest.get('positions', []):
            if all(math.isclose(float(actual), float(expected), rel_tol=0., abs_tol=1e-9)
                   for values, key in zip(coordinates, ('source_xy', 'target_xy'), strict=True)
                   for actual, expected in zip(values, row[key], strict=True)):
                matches.add(split)
    return next(iter(matches)) if len(matches) == 1 else None


def experiment_output_path(output, *, parent, checkpoint, saved, config, kind, started_at):
    """Respect explicit paths; otherwise name only facts known from saved inputs."""
    if output is not None:
        return Path(output).expanduser().resolve()
    checkpoint = Path(checkpoint)
    version = next((match.group(1) for part in (checkpoint, *checkpoint.parents)
                    if (match := re.search(r'(?:^|_)insert_plug_(v\d+)(?:_|$)', part.name))), None)
    if version is None:
        raise ValueError('Cannot infer the task version from the checkpoint path; specify --output')
    model = saved.get('type')
    if model not in {'act', 'diffusion'}:
        raise ValueError('Cannot infer the model type from the checkpoint; specify --output')
    features = saved.get('input_features', {})
    if not any(key.startswith('observation.images.') for key in features):
        raise ValueError('Cannot infer RGB/RGB-D inputs from the checkpoint; specify --output')
    modality = 'rgbd' if saved.get('depth_camera_map') else 'rgb'
    if kind not in {None, 'ensemble', 'chunked', 'stateless'}:
        raise ValueError(f'Unknown diagnostic kind: {kind}')
    start = datetime.fromisoformat(started_at)
    if start.tzinfo is None:
        raise ValueError('Diagnostic started_at must include its timezone')
    parts = [model, 'insert_plug', version, modality]
    split = _layout_split(config)
    if split is not None:
        parts.append(split)
    if kind is not None:
        parts.append(kind)
    parts.append(start.astimezone(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S'))
    return Path(parent) / '_'.join(parts)
