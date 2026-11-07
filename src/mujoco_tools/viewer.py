"""Interactive inspection of a static MuJoCo scene, without physics rollout."""
from __future__ import annotations

import time


def view_static(model, data, *, camera="free", lookat=None):
    """Keep a passive viewer open until closed; never step physics or record.

    MuJoCo is imported here so callers can choose the graphics backend first.
    Forward kinematics refreshes GUI pose edits without advancing simulation time.
    """
    import mujoco
    import mujoco.viewer

    camera_id = None
    if camera != "free":
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if camera_id < 0:
            raise ValueError(f"Unknown scene camera: {camera}")
    with mujoco.viewer.launch_passive(model, data) as handle:
        with handle.lock():
            if camera_id is None:
                handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                if lookat is not None:
                    handle.cam.lookat[:] = lookat
                handle.cam.distance = 1.25
                handle.cam.azimuth = -50
                handle.cam.elevation = -25
            else:
                handle.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                handle.cam.fixedcamid = camera_id
        while handle.is_running():
            with handle.lock():
                mujoco.mj_forward(model, data)
            handle.sync()
            time.sleep(1 / 60)
