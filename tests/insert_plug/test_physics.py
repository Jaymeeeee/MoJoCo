"""End-to-end checks for physical transfer and seating, without rendering."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from examples.insert_plug.expert import Expert
from examples.insert_plug.scene import build_scene, load_config
from examples.insert_plug.simulation import Simulation


@pytest.fixture(scope="module")
def scene_resources(tmp_path_factory):
    config = load_config()
    scene = build_scene(config, tmp_path_factory.mktemp("physics") / "scene.xml")
    return scene, config


@pytest.fixture
def sim(scene_resources):
    path, config = scene_resources
    simulation = Simulation(path, config, render=False)
    try:
        yield simulation
    finally:
        simulation.close()


def test_expert_physically_transfers_free_peg(sim, monkeypatch):
    """The real expert must grasp, extract, release and leave the peg seated.

    Monitoring qpos at the actual mj_step boundary catches direct teleports
    between control steps, while phase observations distinguish a real lift
    and release from merely starting or ending with the peg in the target.
    """
    peg_joint = sim.model.joint("peg_free").id
    assert sim.model.jnt_type[peg_joint] == mujoco.mjtJoint.mjJNT_FREE
    assert sim.model.body_parentid[sim.peg_id] == 0
    assert sim.model.body_mocapid[sim.peg_id] == -1
    assert sim.model.body_gravcomp[sim.peg_id] == 0
    for index in range(sim.model.neq):
        if sim.model.eq_type[index] in (mujoco.mjtEq.mjEQ_WELD, mujoco.mjtEq.mjEQ_CONNECT):
            assert sim.peg_id not in (sim.model.eq_obj1id[index], sim.model.eq_obj2id[index])

    source = sim.site("source_hole")
    target = sim.site("target_hole")
    np.testing.assert_allclose(sim.site("peg_tip")[:2], source[:2], atol=1e-9)
    assert np.linalg.norm(source[:2] - target[:2]) > 0.05

    original_step = mujoco.mj_step
    previous_physical_qpos = sim.data.qpos.copy()
    physical_steps = 0
    interval_ctrl = None

    def monitored_step(model, data, *args, **kwargs):
        nonlocal previous_physical_qpos, physical_steps, interval_ctrl
        if data is sim.data:
            np.testing.assert_array_equal(
                data.qpos, previous_physical_qpos,
                err_msg="Rollout qpos changed outside mj_step (possible teleport)",
            )
            if physical_steps % sim.substeps == 0:
                interval_ctrl = data.ctrl.copy()
            np.testing.assert_array_equal(
                data.ctrl, interval_ctrl,
                err_msg="Actuator command changed inside one control interval",
            )
        result = original_step(model, data, *args, **kwargs)
        if data is sim.data:
            previous_physical_qpos = data.qpos.copy()
            physical_steps += int(kwargs.get('nstep', 1))
        return result

    monkeypatch.setattr(mujoco, "mj_step", monitored_step)
    trace = []

    class ObservedExpert(Expert):
        def tick(self, target_qpos, width, phase):
            super().tick(target_qpos, width, phase)
            trace.append({
                "phase": phase,
                "tip": self.sim.site("peg_tip"),
                "finger_contacts": self.sim.gripper_contacts(),
                "time": float(self.sim.data.time),
            })

    expert = ObservedExpert(sim, verbose=False)
    result = expert.run()
    assert result["success"], result
    assert result["metrics"]["stable_for_final_250ms"]
    assert physical_steps == result["steps"] * sim.substeps
    assert result["steps"] == len(trace)
    # The default fixed scene must finish in strictly less than 20 simulated
    # seconds, including grasp settling, release and final success checks.
    assert result['steps'] < 200
    assert physical_steps > 100
    np.testing.assert_allclose(np.diff([row["time"] for row in trace]), sim.dt, atol=1e-10)

    grasp = [row for row in trace if row["phase"] == "grasp"]
    assert any(len(row["finger_contacts"]) >= 2 for row in grasp), "Peg was never grasped by both fingers"
    extracted = [row for row in trace if row["phase"] == "extract"]
    assert extracted[-1]["tip"][2] > source[2] + 0.01
    np.testing.assert_allclose(extracted[-1]["tip"][:2], source[:2], atol=0.002)

    shallow = [row for row in trace if row["phase"] == "shallow_insert"]
    clearance = sim.config["board"]["hole_radius"] - sim.config["peg"]["shaft_radius"]
    assert np.linalg.norm(shallow[-1]["tip"][:2] - target[:2]) < clearance
    final = trace[-1]
    assert not final["finger_contacts"]
    # The open-gripper interval must allow a further gravity-driven drop; the
    # expert only inserts the tip shallowly before releasing the object.
    assert shallow[-1]["tip"][2] - final["tip"][2] > 0.005
    np.testing.assert_allclose(final["tip"][:2], target[:2], atol=clearance)
    assert np.linalg.norm(final["tip"][:2] - source[:2]) > 0.05


def test_tilted_peg_cannot_pass_with_tip_centered_at_hole_bottom(sim):
    """A tip in the hole does not imply the entire inserted shaft fits it."""
    target = sim.site("target_hole")
    floor = max(sim.config["board"]["bottom_z"] + sim.config["board"]["bottom_thickness"],
                target[2] - sim.config["peg"]["shaft_length"])
    joint = sim.model.joint("peg_free").id
    address = int(sim.model.jnt_qposadr[joint])

    # Construct a diagnostic state, not an expert trajectory. The upright
    # reference must pass so this test cannot pass merely because the gripper
    # or robot is incorrectly positioned for both configurations.
    sim.data.qpos[address:address + 7] = [target[0], target[1], floor, 1, 0, 0, 0]
    sim.data.qvel[:] = 0
    mujoco.mj_forward(sim.model, sim.data)
    upright = sim.metrics()
    assert upright["success"], upright

    angle = 0.09
    sim.data.qpos[address + 3:address + 7] = [np.cos(angle / 2), 0, np.sin(angle / 2), 0]
    mujoco.mj_forward(sim.model, sim.data)
    tilted = sim.metrics()
    clearance = sim.config["board"]["hole_radius"] - sim.config["peg"]["shaft_radius"]
    assert tilted["radial_error_m"] < 1e-10
    assert tilted["seating_error_m"] < 1e-10
    assert tilted["tilt_rad"] < 0.10  # This passed the old angle-only criterion.
    assert tilted["mouth_axis_error_m"] > 3 * clearance
    assert not tilted["success"], tilted
