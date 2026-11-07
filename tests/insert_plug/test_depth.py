"""Analytic RGB-D checks in isolated software-rendering processes."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


def run_osmesa(source):
    # Other tests may already have imported MuJoCo with a different GL backend.
    # Select OSMesa before importing it in a fresh process, without a GPU/display.
    env = dict(os.environ, MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa")
    prefix = """
import mujoco
import numpy as np
from mujoco_tools.cameras import CAMERA_NAMES, camera_metadata, render_observations
"""
    result = subprocess.run(
        [sys.executable, "-c", prefix + textwrap.dedent(source)],
        cwd=Path(__file__).resolve().parents[2],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("physical_head", ["third_person", "head"])
def test_metric_depth_alignment_calibration_and_unchanged_state(physical_head):
    run_osmesa(f"""
        model = mujoco.MjModel.from_xml_string('''
        <mujoco>
          <statistic extent="1"/>
          <visual><map znear=".01" zfar="8"/></visual>
          <worldbody>
            <light pos="0 0 4" diffuse="1 1 1"/>
            <camera name="{physical_head}" pos="0 0 2" fovy="60"/>
            <camera name="wrist" pos="0 0 3" fovy="60"/>
            <geom type="plane" size="0 0 .1" rgba=".5 .5 .5 1"/>
            <geom type="box" pos=".4 .4 .5" size=".12 .12 .1" rgba="1 0 0 1"/>
            <body pos="5 5 5">
              <freejoint/>
              <geom type="sphere" size=".1"/>
            </body>
          </worldbody>
        </mujoco>''')
        data = mujoco.MjData(model)
        data.qvel[:] = np.arange(model.nv) + .1
        data.time = 1.25
        mujoco.mj_forward(model, data)
        qpos, qvel, time = data.qpos.copy(), data.qvel.copy(), data.time
        height, width = 80, 120
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            rgb_before, no_depth = render_observations(renderer, data, depth_enabled=False)
            assert no_depth == {{}}
            images, depth = render_observations(renderer, data)
            rgb_after, _ = render_observations(renderer, data, depth_enabled=False)
            assert CAMERA_NAMES == ("head", "wrist")
            assert tuple(images) == tuple(depth) == CAMERA_NAMES
            metadata = camera_metadata(model, data, width, height)
            assert metadata["head"]["physical_camera"] == "{physical_head}"

            for name, camera_z in (("head", 2.), ("wrist", 3.)):
                assert images[name].shape == (height, width, 3)
                assert images[name].dtype == np.uint8
                assert depth[name].shape == (height, width)
                assert depth[name].dtype == np.float32
                assert np.isfinite(depth[name]).all()
                np.testing.assert_array_equal(images[name], rgb_before[name])
                np.testing.assert_array_equal(images[name], rgb_after[name])

                # Every lower-half pixel sees the z=0 plane. Its axial depth is
                # constant even far away from the camera's principal point.
                np.testing.assert_allclose(depth[name][height // 2:, :], camera_z, atol=2e-5)
                focal = height / (2 * np.tan(np.deg2rad(60.) / 2))
                expected_k = [[focal, 0, (width - 1) / 2],
                              [0, focal, (height - 1) / 2], [0, 0, 1]]
                intrinsic = np.asarray(metadata[name]["intrinsics"])
                np.testing.assert_allclose(intrinsic, expected_k, atol=1e-10)
                transform = np.asarray(metadata[name]["world_from_camera_optical_at_capture"])

                # A known red box above/right of centre checks that RGB and
                # depth use the same top-to-bottom image orientation.
                box_depth = camera_z - .6
                u = round((width - 1) / 2 + focal * .4 / box_depth)
                v = round((height - 1) / 2 - focal * .4 / box_depth)
                assert u > width // 2 and v < height // 2
                red, green, blue = images[name][v, u].astype(int)
                assert red > green + 50 and red > blue + 50
                np.testing.assert_allclose(depth[name][v, u], box_depth, atol=2e-5)
                optical = np.linalg.solve(intrinsic, [u, v, 1.]) * depth[name][v, u]
                world = transform @ np.r_[optical, 1.]
                np.testing.assert_allclose(world[:3], [.4, .4, .6], atol=.025)

                # Unproject an off-axis plane pixel. Euclidean range must be
                # longer than the stored Z, while world Z still equals zero.
                v, u = height - 5, 5
                optical = np.linalg.solve(intrinsic, [u, v, 1.]) * depth[name][v, u]
                assert np.linalg.norm(optical) > camera_z * 1.15
                world = transform @ np.r_[optical, 1.]
                np.testing.assert_allclose(world[2], 0., atol=2e-5)
                assert world[0] < 0 and world[1] < 0
                np.testing.assert_allclose(metadata[name]["near_m"], .01)
                np.testing.assert_allclose(metadata[name]["far_m"], 8.)

        np.testing.assert_array_equal(data.qpos, qpos)
        np.testing.assert_array_equal(data.qvel, qvel)
        assert data.time == time
    """)


def test_background_retains_finite_metric_far_plane():
    run_osmesa("""
        model = mujoco.MjModel.from_xml_string('''
        <mujoco>
          <statistic extent="2"/>
          <visual><map znear=".02" zfar="7.5"/></visual>
          <worldbody>
            <camera name="third_person"/>
            <camera name="wrist"/>
            <geom type="sphere" pos="0 0 1" size=".1"/>
          </worldbody>
        </mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, height=24, width=32) as renderer:
            _, depth = render_observations(renderer, data)
            for image in depth.values():
                assert np.isfinite(image).all()
                np.testing.assert_allclose(image, 15., atol=2e-3)
    """)


def test_depth_render_failure_restores_rgb_mode():
    run_osmesa("""
        model = mujoco.MjModel.from_xml_string('''
        <mujoco><worldbody>
          <camera name="third_person" pos="0 0 2"/>
          <camera name="wrist" pos="0 0 3"/>
          <geom type="plane" size="0 0 .1"/>
        </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, height=24, width=32) as renderer:
            original_render = renderer.render
            def fail_on_depth():
                if renderer._depth_rendering:
                    raise RuntimeError("injected depth failure")
                return original_render()
            renderer.render = fail_on_depth
            try:
                render_observations(renderer, data)
            except RuntimeError as exc:
                assert str(exc) == "injected depth failure"
            else:
                raise AssertionError("Depth failure should propagate")
            renderer.render = original_render
            # A direct render must be RGB even after the helper fails.
            image = renderer.render()
            assert image.dtype == np.uint8 and image.shape == (24, 32, 3)
    """)
