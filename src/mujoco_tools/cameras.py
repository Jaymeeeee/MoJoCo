"""Aligned RGB and metric depth from named MuJoCo cameras.

The policy-facing name ``head`` identifies the fixed third-person camera; it
does not imply that the camera is attached to the robot's head or flange.
"""
from __future__ import annotations

import numpy as np
import mujoco

CAMERA_NAMES = ('head', 'wrist')
RGB_FEATURES = tuple(f'observation.images.{name}' for name in CAMERA_NAMES)
DEPTH_FEATURES = tuple(f'observation.depth.{name}' for name in CAMERA_NAMES)


def camera_id(model, name):
    """Resolve the public head name to a new or existing scene camera."""
    aliases = ('head', 'third_person') if name == 'head' else (name,)
    for alias in aliases:
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, alias)
        if index >= 0:
            return index
    raise ValueError(f'Missing camera for observation {name!r}')


def render_observations(renderer, data, depth_enabled=True):
    """Render both modalities at one unchanged physics state.

    MuJoCo Renderer already converts its depth buffer to optical-axis Z in
    metres. Far-plane/background pixels remain finite at approximately far_m;
    they should be masked before point-cloud reconstruction. No RGB scaling or
    depth quantization is performed here.
    """
    images, depth = {}, {}
    renderer.disable_segmentation_rendering()
    renderer.disable_depth_rendering()
    try:
        for name in CAMERA_NAMES:
            renderer.update_scene(data, camera=camera_id(renderer.model, name))
            images[name] = renderer.render().copy()
            if depth_enabled:
                renderer.enable_depth_rendering()
                try:
                    depth[name] = np.asarray(renderer.render(), dtype=np.float32).copy()
                finally:
                    renderer.disable_depth_rendering()
    finally:
        renderer.disable_depth_rendering()
    return images, depth


def camera_metadata(model, data, width, height):
    """Calibration at the supplied state, with camera axes matching RGB pixels.

    Intrinsics use integer pixel-centre coordinates. Optical axes are +X right,
    +Y down and +Z forward; MuJoCo's camera frame instead has +Y up/-Z forward.
    Wrist extrinsics move during a rollout. Stored simulation/qpos and the
    episode XML can recover its per-frame transforms from forward kinematics.
    """
    if width <= 0 or height <= 0:
        raise ValueError('Camera dimensions must be positive')
    result = {}
    for name in CAMERA_NAMES:
        index = camera_id(model, name)
        fovy = float(model.cam_fovy[index])
        focal = height/(2*np.tan(np.deg2rad(fovy)/2))
        transform = np.eye(4)
        transform[:3, :3] = data.cam_xmat[index].reshape(3, 3) @ np.diag([1., -1., -1.])
        transform[:3, 3] = data.cam_xpos[index]
        result[name] = dict(
            physical_camera=model.camera(index).name,
            width=int(width), height=int(height), fovy_degrees=fovy,
            intrinsics=[[float(focal), 0., (width-1)/2], [0., float(focal), (height-1)/2], [0., 0., 1.]],
            near_m=float(model.vis.map.znear*model.stat.extent),
            far_m=float(model.vis.map.zfar*model.stat.extent),
            depth_units='m', depth_convention='camera_optical_axis_z',
            background='far clipping plane; mask values near far_m when reconstructing points',
            optical_axes='x right, y down, z forward; integer pixel-centre coordinates',
            world_from_camera_optical_at_capture=transform.tolist(),
            capture_time_s=float(data.time),
            moving_camera=bool(model.cam_bodyid[index] != 0),
        )
    return result
