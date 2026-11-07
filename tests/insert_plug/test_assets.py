"""Physical geometry regressions for the supplied STLs and native Robotiq."""
from __future__ import annotations

import hashlib
import json

import mujoco
import numpy as np
import pytest

from examples.insert_plug.scene import build_scene, load_config
from examples.insert_plug.simulation import Simulation
from mujoco_tools.paths import WORKSPACE_ROOT


@pytest.fixture(scope='module')
def asset_scene(tmp_path_factory):
    config = load_config()
    path = build_scene(config, tmp_path_factory.mktemp('asset-scene') / 'scene.xml')
    return path, config


@pytest.fixture
def sim(asset_scene):
    path, config = asset_scene
    simulation = Simulation(path, config, render=False)
    try:
        yield simulation
    finally:
        simulation.close()


def test_stl_collision_cavity_is_open_and_its_wall_stops_the_bolt(sim):
    """A collision-free bore and a colliding wall need separate positive checks."""
    manifest = json.loads((WORKSPACE_ROOT / sim.config['board']['collision_manifest']).read_text())
    for name in ('holder', 'bolt'):
        source = WORKSPACE_ROOT / manifest[name]['source']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest[name]['sha256']

    target = sim.site('target_hole')
    address = int(sim.model.joint('peg_free').qposadr[0])
    # This is an isolated geometry probe, not a rollout. Lift the head 2 mm
    # above the rim while leaving most of the shaft inside the target cavity.
    tip_z = target[2] - sim.config['peg']['shaft_length'] + 0.002
    sim.data.qpos[address:address + 7] = [*target[:2], tip_z, 1, 0, 0, 0]
    sim.data.qvel[:] = 0
    mujoco.mj_forward(sim.model, sim.data)

    def penetrating_contacts():
        return [contact for contact in sim.data.contact
                if sim.peg_id in sim.model.geom_bodyid[[contact.geom1, contact.geom2]]
                and contact.dist < -1e-6]

    assert not penetrating_contacts(), 'Centered shaft collides inside the nominally open bore'
    clearance = sim.config['board']['hole_radius'] - sim.config['peg']['shaft_radius']
    sim.data.qpos[address] += clearance + 0.002
    mujoco.mj_forward(sim.model, sim.data)
    assert penetrating_contacts(), 'Off-axis shaft passes through the holder wall'


def _pad_faces(sim):
    """Read actual box collider surfaces, independently of observation sites."""
    faces = []
    for side in ('left', 'right'):
        geom = sim.model.geom(f'robotiq_{side}_pad1').id
        rotation = sim.data.geom_xmat[geom].reshape(3, 3)
        faces.append(sim.data.geom_xpos[geom]
                     + rotation @ np.array([0, -sim.model.geom_size[geom, 1], 0]))
    return np.asarray(faces)


def test_robotiq_aperture_matches_colliders_and_reports_both_contact_sides(sim):
    """Metre-valued commands must operate the coupled native mechanism."""
    gripper = sim.gripper
    assert gripper is not None
    for width in (0.075, 0.024):
        gripper.reset(sim.data, width)
        mujoco.mj_forward(sim.model, sim.data)
        faces = _pad_faces(sim)
        measured_width = np.linalg.norm(faces[1] - faces[0])
        assert measured_width == pytest.approx(width, abs=1e-8)
        assert sim.gripper_width() == pytest.approx(measured_width, abs=1e-8)
        equality = sim.data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        assert np.any(equality)
        assert np.max(np.abs(sim.data.efc_pos[equality])) < 1e-8

    gripper.reset(sim.data, 0.075)
    mujoco.mj_forward(sim.model, sim.data)
    action = np.r_[sim.data.qpos[sim.arm_qadr], 0.024]
    for _ in range(round(2.0 / sim.dt)):
        sim.step(action)
    faces = _pad_faces(sim)
    assert np.linalg.norm(faces[1] - faces[0]) == pytest.approx(0.024, abs=0.0005)
    assert sim.gripper_width() == pytest.approx(np.linalg.norm(faces[1] - faces[0]), abs=1e-5)
    for width in (-0.001, 0.086):
        with pytest.raises(ValueError):
            sim.step(np.r_[action[:7], width])

    # Put the supplied hex head between the pads in an isolated contact probe.
    # Its 24 mm flats must touch both jaws at a slightly smaller aperture.
    gripper.reset(sim.data, 0.0235)
    mujoco.mj_forward(sim.model, sim.data)
    midpoint = _pad_faces(sim).mean(axis=0)
    mount = sim.data.xmat[gripper.mount_id].reshape(3, 3)
    bolt_rotation = mount @ np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, bolt_rotation.ravel())
    head_offset = sim.config['peg']['shaft_length'] + sim.config['peg']['head_height'] / 2
    tip = midpoint - bolt_rotation[:, 2] * head_offset
    address = int(sim.model.joint('peg_free').qposadr[0])
    sim.data.qpos[address:address + 7] = np.r_[tip, quaternion]
    sim.data.qvel[:] = 0
    mujoco.mj_forward(sim.model, sim.data)
    assert gripper.contact_sides(sim.data, sim.peg_id) == {'left', 'right'}
