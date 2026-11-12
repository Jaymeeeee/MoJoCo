"""Render the canonical initial scene without advancing physics or recording data.

MuJoCo is imported inside ``render_snapshot`` so the CLI can select its graphics
backend before any rendering modules are loaded.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory


def _free_camera(mujoco, model, points, *, padding, azimuth, elevation):
    import numpy as np

    points = np.asarray(points, dtype=float)
    low, high = points.min(axis=0) - padding, points.max(axis=0) + padding
    radius = max(float(np.linalg.norm(high - low)) / 2, 0.04)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (low + high) / 2
    camera.distance = 1.05 * radius / np.sin(np.deg2rad(model.vis.global_.fovy) / 2)
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def _inspection_cameras(mujoco, model, data, config):
    import numpy as np
    from mujoco_tools.cameras import camera_id

    board = config['board']
    centre = np.asarray(board['centre_xy'], dtype=float)
    half = np.asarray(board['half_size_xy'], dtype=float)
    bottom = float(board['bottom_z'])
    top = bottom + float(board['bottom_thickness']) + float(board['wall_height'])
    board_points = [[*(centre + half * [x, y]), z]
                    for x in (-1, 1) for y in (-1, 1) for z in (bottom, top)]
    if board.get('type') == 'stl_holders':
        board_points = [[*(np.asarray(board[key]) + .014 * np.array([x, y])), z]
                        for key in ('source_xy', 'target_xy')
                        for x in (-1, 1) for y in (-1, 1) for z in (bottom, top)]
    robot_points = [data.xpos[index].copy() for index in range(1, model.nbody)
                    if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) or '')
                    .startswith(('fr3_', 'gripper_', 'robotiq_', 'left_finger', 'right_finger'))]
    overview = _free_camera(mujoco, model, [*robot_points, *board_points],
                            padding=0.065, azimuth=-50, elevation=-25)
    detail_points = list(board_points)
    for name in ('peg_tip', 'peg_head'):
        detail_points.append(data.site(name).xpos.copy())
    tcp = data.site('tcp').xpos.copy()
    # Include nearby fingertips, while keeping a distant home pose from making
    # the fixture closeup as wide as the overview. Never move the robot to fit.
    if np.linalg.norm(tcp - [*centre, top]) < 0.20:
        detail_points.append(tcp)
    detail = _free_camera(mujoco, model, detail_points,
                          padding=0.015, azimuth=-50, elevation=-42)
    cameras = {'overview': overview, 'detail': detail,
               'head': camera_id(model, 'head'), 'wrist': camera_id(model, 'wrist')}
    if config['gripper'].get('type') == 'robotiq_2f85':
        points = [data.xpos[index].copy() for index in range(1, model.nbody)
                  if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) or '').startswith('robotiq_')]
        points.extend([data.site('tcp').xpos.copy(), data.body('gripper_mount').xpos.copy()])
        side = data.body('gripper_mount').xmat.reshape(3, 3)[:, 1]
        cameras['gripper'] = _free_camera(mujoco, model, points, padding=.025,
                            azimuth=float(np.rad2deg(np.arctan2(side[1], side[0]))), elevation=-15)
    return cameras


def _contact_sheet(images, output):
    from PIL import Image, ImageDraw, ImageFont

    width, height = next(iter(images.values())).size
    label_height = max(32, width // 22)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', max(14, width // 48))
    except OSError:
        font = ImageFont.load_default()
    sheet = Image.new('RGB', (2 * width, 2 * (height + label_height)), '#18212a')
    draw = ImageDraw.Draw(sheet)
    labels = {'overview': 'Overview - initial scene', 'detail': 'Fixture detail - initial scene',
              'head': 'Head (third-person) camera - initial scene',
              'wrist': 'Wrist camera - initial scene'}
    for index, (name, image) in enumerate(images.items()):
        x, y = (index % 2) * width, (index // 2) * (height + label_height)
        draw.text((x + 12, y + max(6, label_height // 5)), labels[name],
                  font=font, fill='#f2f4f6')
        sheet.paste(image, (x, y + label_height))
    sheet.save(output)


def _depth_preview(depths, output, *, far_m):
    """Save a shared metric color scale; raw depth is never quantized here."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    valid = np.concatenate([values[np.isfinite(values) & (values > 0)]
                            for values in depths.values()])
    if not len(valid):
        raise ValueError('Depth images contain no finite positive values')
    # Background pixels can reach the far plane. Exclude them when choosing a
    # useful display range; keep them and every original value in the NPY files.
    foreground = valid[valid < far_m * .999]
    limits = np.percentile(foreground if len(foreground) else valid, [2, 98])
    low, high = float(limits[0]), float(limits[1])
    if high <= low:
        high = low + max(abs(low) * .01, .001)
    stops = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140],
                      [94, 201, 98], [253, 231, 37]], dtype=float)

    def colorize(values):
        scaled = np.clip((values - low) / (high - low), 0, 1)
        scaled = np.where(np.isfinite(scaled), scaled, 0)
        rgb = np.stack([np.interp(scaled, np.linspace(0, 1, len(stops)), channel)
                        for channel in stops.T], axis=-1).astype(np.uint8)
        rgb[~np.isfinite(values) | (values <= 0)] = 0
        return rgb

    height, width = next(iter(depths.values())).shape
    panel_width = max(width, 360)
    panel_height = round(height * panel_width / width)
    header_height, footer_height = 38, 112
    sheet = Image.new('RGB', (2 * panel_width, header_height + panel_height + footer_height), '#18212a')
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 17)
        small_font = ImageFont.truetype('DejaVuSans.ttf', 14)
    except OSError:
        font = small_font = ImageFont.load_default()
    for index, name in enumerate(('head', 'wrist')):
        x = index * panel_width
        label = 'Head (third-person)' if name == 'head' else 'Wrist'
        draw.text((x + 12, 9), f'{label} depth', font=font, fill='white')
        preview = Image.fromarray(colorize(depths[name]))
        if preview.size != (panel_width, panel_height):
            preview = preview.resize((panel_width, panel_height), Image.Resampling.NEAREST)
        sheet.paste(preview, (x, header_height))
    left, right = 24, sheet.width - 24
    bar_y = header_height + panel_height + 36
    bar = np.linspace(low, high, right - left, dtype=np.float32)[None, :].repeat(18, axis=0)
    sheet.paste(Image.fromarray(colorize(bar)), (left, bar_y))
    draw.text((left, bar_y - 28), 'Optical-axis Z depth (meters) - shared scale', font=font, fill='white')
    for fraction in (0, .25, .5, .75, 1):
        x = left + int(fraction * (right - left - 1))
        draw.line((x, bar_y + 18, x, bar_y + 23), fill='white')
        anchor = 'lt' if fraction == 0 else 'rt' if fraction == 1 else 'mt'
        draw.text((x, bar_y + 26), f'{low + fraction * (high - low):.3f} m',
                  font=small_font, fill='white', anchor=anchor)
    draw.text((left, bar_y + 53), 'Preview clips P2-P98 of foreground. Raw .npy files retain metric depth.',
              font=small_font, fill='#c6d0da')
    sheet.save(output)
    return dict(range_m=[low, high], raw_range_m=[float(valid.min()), float(valid.max())],
                range_selection='2nd to 98th percentile, excluding far-plane background',
                clipped_for_visualization=True)


def render_snapshot(config: dict, output: str | Path, *, width: int = 960) -> dict:
    """Save scene/camera views and a gripper detail of the unchanged home keyframe.

    RGB views, optional float32 metric depth, and camera metadata are retained.
    Scene XML is temporary, and camera resolution is changed in a private copy
    of the configuration. No physics step separates RGB and depth rendering.
    """
    if width < 1:
        raise ValueError('Snapshot width must be positive')

    import mujoco
    import numpy as np
    from PIL import Image
    from mujoco_tools.cameras import CAMERA_NAMES, camera_metadata, render_observations
    from .scene import build_scene

    config = copy.deepcopy(config)
    config['cameras']['width'] = config['cameras']['height'] = int(width)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    images = {}
    depths = {}
    depth_enabled = bool(config['cameras'].get('depth', {}).get('enabled', True))
    with TemporaryDirectory(prefix='mujoco-scene-snapshot-') as folder:
        scene_path = build_scene(config, Path(folder) / 'scene.xml')
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if key < 0:
            raise ValueError('Scene has no canonical home keyframe')
        mujoco.mj_resetDataKeyframe(model, data, key)
        mujoco.mj_forward(model, data)
        cameras = _inspection_cameras(mujoco, model, data, config)
        renderer = mujoco.Renderer(model, height=width, width=width)
        try:
            for name, camera in cameras.items():
                if name in CAMERA_NAMES:
                    continue
                renderer.update_scene(data, camera=camera)
                images[name] = Image.fromarray(renderer.render().copy())
            rgb, depths = render_observations(renderer, data, depth_enabled=depth_enabled)
            images.update({name: Image.fromarray(rgb[name]) for name in CAMERA_NAMES})
        finally:
            renderer.close()
        simulation_time = float(data.time)
        calibration = camera_metadata(model, data, width=width, height=width)

    paths = {}
    for name, image in images.items():
        path = output / f'{name}.png'
        image.save(path)
        paths[name] = str(path)
    preview = output / 'preview.png'
    _contact_sheet({name: images[name] for name in ('overview', 'detail', 'head', 'wrist')}, preview)
    paths['preview'] = str(preview)
    depth_paths = {}
    depth_metadata = dict(enabled=depth_enabled, dtype='float32', units='m',
                          convention='camera_optical_axis_z', shape=[width, width])
    if depth_enabled:
        for name, values in depths.items():
            path = output / f'{name}_depth.npy'
            np.save(path, values, allow_pickle=False)
            depth_paths[name] = str(path)
        depth_preview = output / 'depth_preview.png'
        depth_metadata['preview'] = _depth_preview(depths, depth_preview,
                                                  far_m=calibration['head']['far_m'])
        depth_metadata['preview']['path'] = str(depth_preview)
    calibration_path = output / 'calibration.json'
    calibration_path.write_text(json.dumps(calibration, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    result = dict(output=str(output), state='canonical home keyframe',
                  simulation_time=simulation_time, physics_steps=0,
                  image_size=[width, width], images=paths, depth=depth_paths,
                  depth_metadata=depth_metadata,
                  camera_bindings={name: calibration[name]['physical_camera'] for name in CAMERA_NAMES},
                  calibration=str(calibration_path))
    metadata_path = output / 'snapshot_metadata.json'
    metadata_path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    result['metadata'] = str(metadata_path)
    return result
