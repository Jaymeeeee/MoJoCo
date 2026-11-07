"""Replay diagnostics classify actual MuJoCo contacts without changing physics."""
from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from examples.insert_plug.action_replay_checks import ReplayContactMonitor
from examples.insert_plug.execution_guard import MotionGuard
from examples.insert_plug.scene import build_scene, load_config
from examples.insert_plug.simulation import Simulation


@pytest.fixture(scope='module')
def replay_scene(tmp_path_factory):
    config = load_config()
    path = build_scene(config, tmp_path_factory.mktemp('replay-contacts') / 'scene.xml')
    return path, config


@pytest.fixture
def sim(replay_scene):
    path, config = replay_scene
    simulation = Simulation(path, config, render=False)
    try:
        yield simulation
    finally:
        simulation.close()


def _place_peg(sim, xyz):
    address = int(sim.model.joint('peg_free').qposadr[0])
    sim.data.qpos[address:address + 7] = [*xyz, 1, 0, 0, 0]
    mujoco.mj_forward(sim.model, sim.data)


def _physics_state(sim):
    specification = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(sim.model, specification))
    mujoco.mj_getState(sim.model, sim.data, state, specification)
    return state


@pytest.mark.parametrize('side', ['source', 'target'])
def test_both_real_holder_rims_allow_head_support_without_stages(sim, side):
    rim = sim.site(f'{side}_hole')
    _place_peg(sim, [*rim[:2], rim[2] - sim.config['peg']['shaft_length'] - 0.0001])
    monitor = ReplayContactMonitor(sim)
    reference = MotionGuard(sim)
    before = _physics_state(sim)
    monitor()
    assert monitor.phase == 'action_replay'
    assert monitor.unsafe_contact_count == 0
    assert monitor.peak_support_penetration_m > monitor.fixture_tolerance
    assert monitor.peak_shaft_fixture_penetration_m == 0
    for name in ('fixture_tolerance', 'support_tolerance', 'pad_tolerance', 'self_tolerance'):
        assert getattr(monitor, name) == getattr(reference, name)
    np.testing.assert_array_equal(_physics_state(sim), before)


def test_real_shaft_wall_collision_is_counted_on_every_tick_and_contact(sim):
    target = sim.site('target_hole')
    _place_peg(sim, [target[0] + 0.002, target[1], target[2] - 0.015])
    monitor = ReplayContactMonitor(sim)
    monitor()
    count = monitor.unsafe_contact_count
    assert count > 1  # Multiple actual convex-holder contacts in this state.
    assert monitor.peak_shaft_fixture_penetration_m > monitor.fixture_tolerance
    assert monitor.violations_by_reason == {'peg strikes holder wall or rim': count}
    monitor.VIOLATION_SAMPLE_LIMIT = count
    sim.data.time = 0.001
    monitor()
    summary = monitor.summary()
    assert summary['checks'] == 2
    assert summary['unsafe_contact_count'] == 2 * count
    assert summary['first_violation']['time_s'] == 0
    assert summary['last_violation']['time_s'] == 0.001
    assert summary['violations_by_reason']['peg strikes holder wall or rim'] == 2 * count
    assert summary['violation_samples_retained'] == count
    assert summary['violation_samples_truncated']


def test_real_support_penetration_limit_still_applies(sim):
    target = sim.site('target_hole')
    _place_peg(sim, [*target[:2], target[2] - sim.config['peg']['shaft_length'] - 0.0015])
    monitor = ReplayContactMonitor(sim)
    monitor()
    assert monitor.peak_support_penetration_m > monitor.support_tolerance
    assert monitor.violations_by_reason['excessive head penetration into supporting rim'] > 0


def test_peg_table_contacts_do_not_abort_other_contact_checks(sim):
    _place_peg(sim, [0.6, 0.0, -0.001])
    monitor = ReplayContactMonitor(sim)
    before = _physics_state(sim)
    monitor()
    count = monitor.unsafe_contact_count
    assert count > 0
    assert monitor.first_violation['reason'] == 'peg contacts a fixture other than source or target holder'
    assert len(monitor.violations) == count
    monitor()
    assert monitor.unsafe_contact_count == 2 * count
    np.testing.assert_array_equal(_physics_state(sim), before)


def _small_physical_scene(robot_xml, peg_geom='peg_head', peg_position='0.0195 0 1', fixture_xml=''):
    """Small real collider models isolate pad, robot and self-contact cases."""
    model = mujoco.MjModel.from_xml_string(f'''
        <mujoco><option gravity="0 0 0"/><worldbody>
          <body name="fr3_link0">{robot_xml}</body>
          <body name="peg" pos="{peg_position}">
            <freejoint/><geom name="{peg_geom}" size="0.01" mass="0.01"/>
          </body>
          <body name="source_socket" pos="3 0 1"><site name="source_hole"/></body>
          <body name="target_socket" pos="4 0 1"><site name="target_hole"/></body>
          {fixture_xml}
        </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return SimpleNamespace(model=model, data=data, config={'task': {}},
                           peg_id=model.body('peg').id,
                           site=lambda name: data.site(name).xpos.copy())


@pytest.mark.parametrize(('body', 'peg_geom', 'separation', 'reason'), [
    ('left_finger', 'peg_head', 0.0195, None),
    ('left_finger', 'peg_head', 0.019, 'excessive grasp pad compression'),
    ('left_finger', 'peg_shaft', 0.0195, 'peg contacts robot outside grasp pads'),
    ('fr3_link1', 'peg_head', 0.0195, 'peg contacts robot outside grasp pads'),
])
def test_real_pad_contact_requires_head_and_preserves_compression_limit(body, peg_geom, separation, reason):
    sim = _small_physical_scene(
        f'<body name="{body}" pos="0 0 1"><joint/><geom name="robot_geom" size="0.01" mass="1"/></body>',
        peg_geom=peg_geom, peg_position=f'{separation} 0 1')
    assert sim.data.ncon == 1
    monitor = ReplayContactMonitor(sim)
    monitor()
    assert monitor.unsafe_contact_count == (0 if reason is None else 1)
    if reason is not None:
        assert monitor.first_violation['reason'] == reason


def test_real_robot_fixture_contact_is_rejected():
    sim = _small_physical_scene(
        '<body name="left_finger" pos="0 0 1"><joint/><geom name="robot_geom" size="0.01" mass="1"/></body>',
        peg_position='5 0 1',
        fixture_xml='<geom name="obstacle" pos="0.0195 0 1" size="0.01"/>')
    assert sim.data.ncon == 1
    monitor = ReplayContactMonitor(sim)
    monitor()
    assert monitor.violations_by_reason == {'robot or gripper contacts a fixture': 1}
    assert monitor.peak_robot_fixture_penetration_m > 0


def test_real_nonadjacent_robot_collision_is_rejected():
    sim = _small_physical_scene('''
        <body name="fr3_link1" pos="0 0 1"><joint/>
          <geom name="link1_geom" size="0.1" mass="1"/>
          <body name="fr3_link2" pos="0 0 0.4"><joint/>
            <geom name="link2_geom" size="0.1" mass="1"/>
            <body name="fr3_link3" pos="0.15 0 -0.4"><joint/>
              <geom name="link3_geom" size="0.1" mass="1"/>
            </body>
          </body>
        </body>''', peg_position='5 0 1')
    assert sim.data.ncon == 1
    monitor = ReplayContactMonitor(sim)
    monitor()
    assert monitor.violations_by_reason == {'nonadjacent robot self collision': 1}
    assert monitor.peak_self_penetration_m > monitor.self_tolerance
