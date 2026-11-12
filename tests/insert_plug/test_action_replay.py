"""Saved-action replay retains physical success checks with optional contacts."""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from examples.insert_plug.diagnostics import action_replay_checks, check_action_replay
from examples.insert_plug.environment.scene import build_scene, load_config
from examples.insert_plug.environment.simulation import Simulation


@pytest.fixture(scope='module')
def replay_source(tmp_path_factory):
    root = tmp_path_factory.mktemp('saved-action-replay')
    config = load_config()
    path = build_scene(config, root / 'scene.xml')
    sim = Simulation(path, config, render=False)
    try:
        action = np.r_[sim.data.qpos[sim.arm_qadr], sim.gripper_width()].astype(np.float32).astype(float)
        qpos, qvel, joint, peg, times = [], [], [], [], []
        for _ in range(3):
            qpos.append(sim.data.qpos.copy())
            qvel.append(sim.data.qvel.copy())
            joint.append(sim.data.qpos[sim.arm_qadr].copy())
            peg.append(sim.data.xpos[sim.peg_id].copy())
            times.append(float(sim.data.time))
            sim.step(action)
    finally:
        sim.close()
    source = dict(model_path=path, config=config, control_steps_per_action=1, action_hold_s=.1,
                  actions=np.tile(action, (3, 1)),
                  metadata=dict(mujoco_version=mujoco.__version__, control_hz=10, physics_timestep=.001,
                                sample_hz=10, action_timing='held_until_next_observation',
                                execution_result=dict(steps=3)),
                  arrays={'simulation/qpos': np.array(qpos), 'simulation/qvel': np.array(qvel),
                          'observations/joint_position': np.array(joint),
                          'privileged/peg_pose': np.array(peg), 'timestamp': np.array(times),
                          'phases': ['settle_final'] * 3})
    return root, source


@pytest.mark.parametrize('diagnostics', [False, True])
@pytest.mark.parametrize('physical_success', [False, True])
def test_replay_contact_diagnostics_do_not_change_dynamics_or_physical_success(
        replay_source, monkeypatch, diagnostics, physical_success):
    root, source = replay_source
    monkeypatch.setattr(check_action_replay, 'ROOT', root)
    monkeypatch.setattr(check_action_replay, 'load_episode', lambda *args: source)
    # Isolate the contact policy from geometric success: actual MuJoCo motion,
    # its comparison with the saved states, and all 251 stability samples run.
    monkeypatch.setattr(Simulation, 'metrics', lambda self: dict(success=physical_success))

    class Diagnostic:
        def __init__(self, sim):
            assert diagnostics, 'Default replay must not instantiate a contact monitor'
            self.checks = 0
        def __call__(self):
            self.checks += 1
        def summary(self):
            return dict(checks=self.checks, unsafe_contact_count=1,
                        violations_by_reason={'synthetic diagnostic threshold': 1})

    monkeypatch.setattr(action_replay_checks, 'ReplayContactMonitor', Diagnostic)
    report = check_action_replay.replay_episode(
        root / 'pos0/episode_000000/episode_000000.h5', contact_diagnostics=diagnostics)
    assert report['success'] is physical_success
    assert report['task_success'] is physical_success
    assert report['physics_checks'] == 300
    assert report['final']['final_stability_physics_samples'] == 251
    assert report['completed_control_steps'] == 3
    assert report['max_joint_deviation_from_source_rad'] == 0
    assert report['max_peg_deviation_from_source_m'] == 0
    assert report['contact_clean'] is (False if diagnostics else None)
    assert report['contacts']['enabled'] is diagnostics
    assert report['contacts']['checks'] == (300 if diagnostics else 0)
    assert report['contacts']['unsafe_contact_count'] == (1 if diagnostics else None)
    summary = check_action_replay.summarize([report])
    assert summary['passed'] == int(physical_success)
    assert summary['contact_checked_episodes'] == int(diagnostics)
    assert summary['contact_clean'] == (0 if diagnostics else None)
