"""Deployment timing, reset, privacy and termination checked with real physics."""
from __future__ import annotations

import copy

import mujoco
import numpy as np
import pytest

from examples.insert_plug.rollout import (
    PhysicalSuccessMonitor, run_episode, select_layouts, validate_contract,
)
from examples.insert_plug.scene import build_scene, load_config
from examples.insert_plug.simulation import Simulation


class HoldPolicy:
    def __init__(self, config, action):
        self.input_features = {
            'observation.state': {'type': 'STATE', 'shape': [8]},
            **{f'observation.images.{camera}': {'type': 'VISUAL',
                 'shape': [3, config['cameras']['height'], config['cameras']['width']]}
               for camera in ('head', 'wrist')},
        }
        self.output_features = {'action': {'type': 'ACTION', 'shape': [8]}}
        self.action = action
        self.reset_count = 0
        self.observations = []

    def reset(self):
        self.reset_count += 1

    def select_action(self, observation):
        self.observations.append(observation)
        return self.action.copy()


@pytest.fixture(scope='module')
def scene_resources(tmp_path_factory):
    cfg = load_config()
    return build_scene(cfg, tmp_path_factory.mktemp('policy-rollout') / 'scene.xml'), cfg


@pytest.fixture
def sim(scene_resources):
    scene, cfg = scene_resources
    simulation = Simulation(scene, cfg, render=False)
    # Images are deliberately synthetic here: the test concerns control timing
    # and filtering, with real model dynamics and contact/stability callbacks.
    def observe(data=None):
        data = simulation.data if data is None else data
        return {
            'observation.state': np.r_[data.qpos[simulation.arm_qadr], simulation.gripper_width(data)].astype(np.float32),
            'observation.images.head': np.zeros((256, 256, 3), dtype=np.uint8),
            'observation.images.wrist': np.zeros((256, 256, 3), dtype=np.uint8),
            'observation.depth.head': np.ones((256, 256), dtype=np.float32),
            'privileged.peg_pose': np.ones(7),
        }
    simulation.observe = observe
    try:
        yield simulation
    finally:
        simulation.close()


def hold_policy(sim):
    return HoldPolicy(sim.config, np.r_[sim.data.qpos[sim.arm_qadr], sim.gripper_width()].astype(np.float32))


def test_layout_selection_preserves_base_and_uses_vetted_pairs():
    cfg = load_config()
    before = copy.deepcopy(cfg)
    layouts = select_layouts(cfg, 'all')
    assert len(layouts) == 10
    assert layouts[0][0] == 'pos0'
    assert layouts[0][1]['board']['source_xy'] == [.49, -.07]
    assert layouts[9][1]['board']['target_xy'] == [.47, .05]
    assert select_layouts(cfg, '2')[0] == layouts[2]
    layouts[0][1]['board']['source_xy'][0] = 999
    assert cfg == before
    assert select_layouts(cfg)[0] == ('config', before)
    assert select_layouts(cfg, 'config')[0] == ('config', before)
    with pytest.raises(ValueError, match='Unknown position'):
        select_layouts(cfg, '10')


def test_contract_rejects_rate_camera_and_output_mismatch(sim):
    policy = hold_policy(sim)
    validate_contract(sim.config, policy)
    cfg = copy.deepcopy(sim.config)
    cfg['task']['control_hz'] = 50
    with pytest.raises(ValueError, match='10 Hz'):
        validate_contract(cfg, policy)
    policy.input_features['observation.images.head']['shape'] = [3, 224, 224]
    with pytest.raises(ValueError, match='observation contract'):
        validate_contract(sim.config, policy)
    policy = hold_policy(sim)
    policy.output_features['action']['shape'] = [7]
    with pytest.raises(ValueError, match='eight absolute targets'):
        validate_contract(sim.config, policy)


def test_live_policy_uses_ten_hz_actual_targets_and_only_public_inputs(sim, scene_resources):
    policy = hold_policy(sim)
    reference = Simulation(*scene_resources, render=False)
    try:
        result = run_episode(sim, policy, duration=.3, quiet=True)
        for _ in range(3):
            reference.step(policy.action)
        assert policy.reset_count == 1
        assert result['completed_control_steps'] == 3
        assert result['physics_checks'] == 300
        assert result['success_physics_samples'] == 300
        assert result['simulation_seconds'] == pytest.approx(.3)
        assert result['stop_reason'] == 'timeout'
        assert not result['success']
        assert result['contact_clean']
        assert len(policy.observations) == 3
        assert set(policy.observations[0]) == {
            'observation.state', 'observation.images.head', 'observation.images.wrist'}
        np.testing.assert_array_equal(sim.data.qpos, reference.data.qpos)
        np.testing.assert_array_equal(sim.data.qvel, reference.data.qvel)
        assert sim.safety_callback is None
        assert sim.depth_enabled == bool(sim.config['cameras']['depth']['enabled'])
    finally:
        reference.close()


def test_invalid_action_is_reported_without_moving_or_clipping(sim):
    policy = hold_policy(sim)
    policy.action[0] = 100
    before = sim.data.qpos.copy()
    result = run_episode(sim, policy, duration=.1, quiet=True)
    assert result['stop_reason'] == 'runtime_error'
    assert 'Joint target outside' in result['runtime_error']
    assert result['last_action'][0] == 100
    assert result['simulation_seconds'] == 0
    assert result['physics_checks'] == 0
    assert not result['partial_final_control_interval']
    assert not result['success']
    np.testing.assert_array_equal(sim.data.qpos, before)


def test_nonfinite_rejected_action_still_produces_standard_json(sim):
    import json
    policy = hold_policy(sim)
    policy.action[0] = np.nan
    result = run_episode(sim, policy, duration=.1, quiet=True)
    assert result['stop_reason'] == 'runtime_error'
    assert result['last_action'][0] == 'nan'
    assert not result['partial_final_control_interval']
    json.dumps(result, allow_nan=False)


def test_success_span_is_continuous_and_inspection_does_not_mutate_live_state(sim, monkeypatch):
    monitor = PhysicalSuccessMonitor(sim)
    good = True
    monkeypatch.setattr(monitor.snapshot, 'metrics', lambda: {'success': good})
    for tick in range(250):
        sim.data.time = tick * .001
        monitor.sample()
    assert not monitor.stable
    sim.data.time = .250
    before = sim.data.qacc_warmstart.copy()
    monitor.sample()
    assert monitor.stable
    assert monitor.continuous_success_seconds == pytest.approx(.25)
    np.testing.assert_array_equal(sim.data.qacc_warmstart, before)
    good = False
    sim.data.time = .251
    monitor.sample()
    assert not monitor.stable
    good = True
    sim.data.time = .252
    monitor.sample()
    assert monitor.continuous_success_seconds == 0


def test_real_collision_stops_on_first_physics_tick(sim):
    joint = sim.model.joint('peg_free').qposadr[0]
    target = sim.site('target_hole')
    sim.data.qpos[joint:joint + 7] = [target[0] + .002, target[1], target[2] - .015, 1, 0, 0, 0]
    mujoco.mj_forward(sim.model, sim.data)
    result = run_episode(sim, hold_policy(sim), duration=.1, quiet=True)
    assert result['stop_reason'] == 'contact_violation'
    assert result['physics_checks'] == 1
    assert result['simulation_seconds'] == pytest.approx(.001)
    assert result['completed_control_steps'] == 0
    assert result['partial_final_control_interval']
    assert not result['contact_clean']
    assert not result['success']
    assert result['contacts']['first_violation']['phase'] == 'policy_rollout'


def test_duration_cannot_silently_change_control_period(sim):
    with pytest.raises(ValueError, match='exact multiple'):
        run_episode(sim, hold_policy(sim), duration=.15)
