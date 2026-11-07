"""Offline plan invariants and actual-contact rejection using real colliders."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from examples.insert_plug.execution_guard import ExecutionSafetyError, MotionGuard
from examples.insert_plug.planning import PlanningError, TaskPlanner
from examples.insert_plug.scene import build_scene, load_config
from examples.insert_plug.simulation import Simulation


@pytest.fixture(scope='module')
def planning_scene(tmp_path_factory):
    config = load_config()
    path = build_scene(config, tmp_path_factory.mktemp('planning-scene') / 'scene.xml')
    return path, config


@pytest.fixture
def sim(planning_scene):
    path, config = planning_scene
    simulation = Simulation(path, config, render=False)
    try:
        yield simulation
    finally:
        simulation.close()


def _physics_state(sim):
    specification = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(sim.model, specification))
    mujoco.mj_getState(sim.model, sim.data, state, specification)
    return state


def _phase_positions(plan, phase):
    positions = plan.tcp_position[plan.phase == phase]
    assert len(positions), f'Missing planned stage: {phase}'
    return positions


def _assert_lift_then_transport(plan, source, target):
    extraction = _phase_positions(plan, 'extract')
    transfer = _phase_positions(plan, 'transfer')
    np.testing.assert_allclose(extraction[:, :2], np.broadcast_to(source[:2], extraction[:, :2].shape), atol=1e-8)
    assert np.all(np.diff(extraction[:, 2]) >= -1e-10)
    assert np.ptp(extraction[:, 2]) > 0.05
    np.testing.assert_allclose(transfer[:, 2], extraction[-1, 2], atol=1e-10)
    np.testing.assert_allclose(transfer[-1, :2], target[:2], atol=1e-8)
    # The held tip, rather than just the TCP, has to clear both rims.
    local_tip = np.asarray(plan.validation['held_tip_in_tcp'])
    tip = transfer + np.einsum('nij,j->ni', plan.tcp_rotation[plan.phase == 'transfer'], local_tip)
    assert np.min(tip[:, 2]) >= max(source[2], target[2]) + 0.015 - 1e-8


def test_complete_plan_does_not_step_or_mutate_live_physics(sim, monkeypatch, tmp_path):
    before = _physics_state(sim)
    before_target = sim.last_target.copy()
    before_collisions = (sim.model.geom_contype.copy(), sim.model.geom_conaffinity.copy())

    def forbidden_step(*args, **kwargs):
        pytest.fail('Offline planning must not step the live or a hidden physics rollout')

    monkeypatch.setattr(mujoco, 'mj_step', forbidden_step)
    plan = TaskPlanner(sim).plan()
    np.testing.assert_array_equal(_physics_state(sim), before)
    np.testing.assert_array_equal(sim.last_target, before_target)
    np.testing.assert_array_equal(sim.model.geom_contype, before_collisions[0])
    np.testing.assert_array_equal(sim.model.geom_conaffinity, before_collisions[1])
    assert plan.validation['valid']
    assert plan.validation['collision_samples'] >= len(plan.time)
    assert plan.actions.shape == (len(plan.time), 8)
    assert np.isfinite(plan.actions).all()
    np.testing.assert_allclose(np.diff(plan.time), sim.dt, atol=1e-12)
    assert [stage['phase'] for stage in plan.stages] == [
        'settle_initial', 'approach', 'descend_to_grasp', 'grasp',
        'extract', 'transfer', 'align', 'shallow_insert',
        'insertion_check', 'release', 'retract', 'settle_final',
    ]
    _assert_lift_then_transport(plan, sim.site('source_hole'), sim.site('target_hole'))
    limits = sim.model.jnt_range[sim.arm_jids]
    assert np.all(plan.arm_qpos >= limits[:, 0]) and np.all(plan.arm_qpos <= limits[:, 1])
    velocity = np.diff(np.vstack((plan.initial_arm_qpos, plan.arm_qpos)), axis=0) / sim.dt
    assert np.all(np.max(abs(velocity), axis=0) <= np.asarray(plan.validation['joint_velocity_limits_rad_s']) * 1.0001)
    paths = plan.save(tmp_path)
    with open(paths['json'], encoding='utf8') as stream:
        metadata = json.load(stream)
    assert metadata['samples'] == len(plan.time)
    with np.load(paths['npz'], allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['action'], plan.actions)
        np.testing.assert_array_equal(arrays['phase'], plan.phase)


def test_plan_uses_moved_fixture_positions_outside_old_board_rectangle(tmp_path):
    config = load_config()
    config['board']['source_xy'] = [0.55, -0.09]
    config['board']['target_xy'] = [0.50, 0.10]
    # These are independent STL holders supported by the table. The obsolete
    # procedural board rectangle is not their physical boundary.
    assert config['board']['source_xy'][0] + config['board']['socket_half_width'] > (
        config['board']['centre_xy'][0] + config['board']['half_size_xy'][0])
    path = build_scene(config, tmp_path / 'moved.xml')
    sim = Simulation(path, config, render=False)
    try:
        plan = TaskPlanner(sim).plan()
        _assert_lift_then_transport(plan, sim.site('source_hole'), sim.site('target_hole'))
        np.testing.assert_allclose(plan.validation['source_site_xyz'][:2], config['board']['source_xy'])
        np.testing.assert_allclose(plan.validation['target_site_xyz'][:2], config['board']['target_xy'])
    finally:
        sim.close()


def test_unreachable_robot_base_rejected_without_execution(tmp_path):
    config = load_config()
    config['robot']['base_position'] = [2.0, 0.0, 0.0]
    path = build_scene(config, tmp_path / 'unreachable.xml')
    sim = Simulation(path, config, render=False)
    try:
        before = _physics_state(sim)
        with pytest.raises(PlanningError, match='unreachable'):
            TaskPlanner(sim).plan()
        np.testing.assert_array_equal(_physics_state(sim), before)
    finally:
        sim.close()


def test_obstacle_in_transport_corridor_is_rejected_before_execution(tmp_path):
    config = load_config()
    path = build_scene(config, tmp_path / 'obstructed.xml')
    tree = ET.parse(path)
    ET.SubElement(tree.getroot().find('worldbody'), 'geom',
                  name='planning_obstacle', type='box', pos='0.45 0 0.16',
                  size='0.045 0.02 0.08', contype='1', conaffinity='1')
    tree.write(path)
    sim = Simulation(path, config, render=False)
    try:
        before = _physics_state(sim)
        with pytest.raises(PlanningError, match='collision.*planning_obstacle|planning_obstacle.*collision'):
            TaskPlanner(sim).plan()
        np.testing.assert_array_equal(_physics_state(sim), before)
    finally:
        sim.close()


def test_actual_guard_rejects_shaft_wall_collision_even_during_insertion(sim):
    target = sim.site('target_hole')
    address = int(sim.model.joint('peg_free').qposadr[0])
    # Diagnostic configuration only: center first, then put the shaft through
    # a real holder collider. This is not an executable expert trajectory.
    sim.data.qpos[address:address + 7] = [*target[:2], target[2] - 0.015, 1, 0, 0, 0]
    mujoco.mj_forward(sim.model, sim.data)
    guard = MotionGuard(sim)
    guard.phase = 'shallow_insert'
    guard()
    sim.data.qpos[address] += 0.002
    mujoco.mj_forward(sim.model, sim.data)
    with pytest.raises(ExecutionSafetyError, match='wall or rim'):
        guard()
    assert guard.unsafe_contact_count == 1
    assert guard.peak_shaft_fixture_penetration_m > guard.fixture_tolerance


def test_actual_guard_distinguishes_rim_support_from_wrong_stage_contact(sim):
    target = sim.site('target_hole')
    address = int(sim.model.joint('peg_free').qposadr[0])
    tip_z = target[2] - sim.config['peg']['shaft_length'] - 0.0001
    sim.data.qpos[address:address + 7] = [*target[:2], tip_z, 1, 0, 0, 0]
    mujoco.mj_forward(sim.model, sim.data)
    guard = MotionGuard(sim)
    guard.phase = 'release'
    guard()
    assert guard.peak_support_penetration_m > guard.fixture_tolerance
    assert guard.peak_shaft_fixture_penetration_m == 0
    guard.phase = 'transfer'
    with pytest.raises(ExecutionSafetyError, match='outside its allowed stage'):
        guard()


def test_safety_callback_observes_every_physics_substep_and_stops_immediately(sim):
    times = []

    def observe():
        times.append(float(sim.data.time))

    sim.safety_callback = observe
    action = np.r_[sim.last_target, sim.config['gripper']['open_width']]
    sim.step(action)
    assert len(times) == sim.substeps
    np.testing.assert_allclose(np.diff(times), sim.model.opt.timestep, atol=1e-12)
    assert times[-1] == pytest.approx(sim.dt)

    def stop():
        raise ExecutionSafetyError('Synthetic unsafe contact at the first physics sample')

    sim.safety_callback = stop
    before = float(sim.data.time)
    with pytest.raises(ExecutionSafetyError, match='first physics sample'):
        sim.step(action)
    assert float(sim.data.time) - before == pytest.approx(sim.model.opt.timestep)


def test_changed_diagonal_layout_refreshes_grasp_alignment_before_insertion(tmp_path):
    """A transported compliant grasp must be remeasured before a tight bore.

    This layout previously stopped above the target with 0.385 mm shaft-axis
    error, exceeding the unchanged 0.350 mm entry budget.
    """
    from examples.insert_plug.expert import Expert

    config = load_config()
    config['board']['source_xy'] = [0.40, -0.10]
    config['board']['target_xy'] = [0.48, 0.08]
    path = build_scene(config, tmp_path / 'diagonal.xml')
    simulation = Simulation(path, config, render=False)
    snapshots = {}

    class ObservedExpert(Expert):
        def _execute_stage(self, plan, stage, *args, **kwargs):
            super()._execute_stage(plan, stage, *args, **kwargs)
            if stage['phase'] in ('transfer', 'insertion_check'):
                snapshots[stage['phase']] = (_physics_state(self.sim), self.sim.last_target.copy())

    try:
        result = ObservedExpert(simulation, verbose=False).run()
        assert result['success'], result
        assert result['metrics']['stable_for_final_250ms']
        assert result['execution_checks']['unsafe_contact_count'] == 0
        assert result['execution_checks']['checks'] == result['steps'] * simulation.substeps
        assert result['execution_checks']['peak_robot_fixture_penetration_m'] == 0
        assert result['execution_checks']['peak_shaft_fixture_penetration_m'] < 5e-5
        assert not simulation.gripper_contacts()
    finally:
        simulation.close()

    # Inspect captured physical states in an independent diagnostic simulation;
    # the expert above is never repositioned or reset during its rollout.
    probe = Simulation(path, config, render=False)
    try:
        state, last_target = snapshots['transfer']
        mujoco.mj_setState(probe.model, probe.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        probe.last_target = last_target.copy()
        mujoco.mj_forward(probe.model, probe.data)
        before = _physics_state(probe)
        plan = TaskPlanner(probe).plan(start_phase='align')
        np.testing.assert_array_equal(_physics_state(probe), before)
        np.testing.assert_array_equal(probe.last_target, last_target)
        assert plan.stages[0]['phase'] == 'align'
        assert not {'grasp', 'extract', 'transfer'}.intersection(plan.phase)

        state, last_target = snapshots['insertion_check']
        mujoco.mj_setState(probe.model, probe.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        probe.last_target = last_target.copy()
        mujoco.mj_forward(probe.model, probe.data)
        assert len(probe.gripper_contacts()) == 2
        assert probe.site('peg_tip')[2] < probe.site('target_hole')[2]
        before = _physics_state(probe)
        with pytest.raises(PlanningError, match='too low for a lateral alignment correction'):
            TaskPlanner(probe).plan(start_phase='align')
        np.testing.assert_array_equal(_physics_state(probe), before)
    finally:
        probe.close()
