"""Intervention, full noise pairing and absolute-command execution contracts."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from examples.insert_plug.policy import translation_probe as probe


def test_translation_conditions_are_signed_xy_only():
    conditions = dict(probe.translation_conditions(.03))
    assert list(conditions) == ['baseline', 'x_plus', 'x_minus', 'y_plus', 'y_minus']
    np.testing.assert_array_equal(conditions['baseline'], [0, 0, 0])
    np.testing.assert_array_equal(conditions['x_plus'], [.03, 0, 0])
    np.testing.assert_array_equal(conditions['x_minus'], [-.03, 0, 0])
    np.testing.assert_array_equal(conditions['y_plus'], [0, .03, 0])
    np.testing.assert_array_equal(conditions['y_minus'], [0, -.03, 0])


@pytest.mark.parametrize('distance', [0, -1, np.nan, np.inf])
def test_invalid_displacement_rejected(distance):
    with pytest.raises(ValueError):
        probe.translation_conditions(distance)


def test_only_source_xy_config_changes_and_original_remains_owned():
    from examples.insert_plug.environment.scene import load_config
    original = load_config()
    saved = deepcopy(original)
    shifted = probe.shifted_config(original, [.03, 0, 0])
    assert original == saved
    np.testing.assert_allclose(shifted['board']['source_xy'], np.array(saved['board']['source_xy']) + [.03, 0])
    shifted['board']['source_xy'] = saved['board']['source_xy']
    assert shifted == saved
    shifted['robot']['home_qpos'][0] = 999
    assert original == saved


@pytest.mark.parametrize('delta', [[0, 0, .03], [0, np.nan, 0], [.03, 0], [0, 0, np.inf]])
def test_invalid_source_intervention_rejected(delta):
    with pytest.raises(ValueError):
        probe.shifted_config({'board': {'source_xy': [0, 0]}}, delta)


def test_built_intervention_moves_screw_and_source_only(tmp_path):
    """Check the actual scene builder, including unchanged wrist/head poses."""
    from examples.insert_plug.environment.scene import load_config, build_scene
    from examples.insert_plug.environment.simulation import Simulation
    config = load_config()
    baseline = Simulation(build_scene(config, tmp_path / 'base.xml'), config, render=False)
    try:
        for direction, delta in probe.translation_conditions(.03)[1:]:
            shifted = probe.shifted_config(config, delta)
            sim = Simulation(build_scene(shifted, tmp_path / f'{direction}.xml'), shifted, render=False)
            try:
                peg_adr = int(sim.model.joint('peg_free').qposadr[0])
                expected = baseline.data.qpos.copy()
                expected[peg_adr:peg_adr + 3] += delta
                np.testing.assert_allclose(sim.data.qpos, expected, rtol=0, atol=1e-12)
                np.testing.assert_allclose(sim.site('source_hole') - baseline.site('source_hole'), delta, atol=1e-12)
                np.testing.assert_allclose(sim.site('peg_head') - baseline.site('peg_head'), delta, atol=1e-12)
                for site in ('tcp', 'target_hole'):
                    np.testing.assert_array_equal(sim.site(site), baseline.site(site))
                for field in ('qvel', 'ctrl', 'cam_xpos', 'cam_xmat'):
                    np.testing.assert_array_equal(getattr(sim.data, field), getattr(baseline.data, field))
                assert sim.gripper_width() == baseline.gripper_width()
            finally:
                sim.close()
    finally:
        baseline.close()


def stochastic_adapter():
    """Tiny conditional sampler draws noise repeatedly like a DDPM scheduler."""
    torch = pytest.importorskip('torch')

    class Adapter:
        _torch = torch
        policy_type = 'diffusion'
        n_obs_steps = 1
        chunk_size = 64
        input_features = ('observation.state',)

        def __init__(self):
            self.policy = SimpleNamespace(predict_action_chunk=self.predict)
            self.cached = 'stale observation'
            self.resets = 0

        def reset(self):
            self.cached = None
            self.resets += 1

        def _prepare_observation(self, observation):
            return {'observation.state': torch.as_tensor(observation['observation.state']).clone()}

        def preprocessor(self, batch):
            return {'observation.state': batch['observation.state'].unsqueeze(0)}

        def predict(self, batch):
            assert self.cached is None
            assert tuple(batch['observation.state'].shape) == (1, 1, 8)
            self.cached = 'new observation'
            sample = torch.randn((1, 64, 8))
            for _ in range(4):
                sample = .5 * sample + torch.randn_like(sample)
            return sample + batch['observation.state'][:, :1, :]

        def postprocessor(self, action):
            # An in-place processor must not corrupt captured normalized data.
            return action.mul_(2).add_(3)

    return Adapter()


def test_complete_ddpm_noise_stream_paired_and_queues_reset():
    adapter = stochastic_adapter()
    base = {'observation.state': np.zeros(8, dtype=np.float32)}
    shifted = {'observation.state': np.full(8, .125, dtype=np.float32)}
    first = probe.predict_chunk(adapter, base, 123)
    second = probe.predict_chunk(adapter, shifted, 123)
    repeated = probe.predict_chunk(adapter, base, 123)
    assert adapter.resets == 3
    for key in ('rng_start', 'rng_end', 'cuda_rng_start', 'cuda_rng_end'):
        np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_array_equal(first[key], repeated[key])
    assert not np.array_equal(first['rng_start'], first['rng_end'])
    np.testing.assert_array_equal(first['actions'], repeated['actions'])
    np.testing.assert_allclose(second['normalized'] - first['normalized'], .125, atol=5e-7)
    np.testing.assert_allclose(first['actions'], first['normalized'] * 2 + 3, atol=5e-7)
    changed_seed = probe.predict_chunk(adapter, base, 124)
    assert not np.array_equal(changed_seed['actions'], first['actions'])
    np.testing.assert_array_equal(base['observation.state'], np.zeros(8))


def test_partial_diffusion_chunk_is_rejected_instead_of_assumed_full():
    adapter = stochastic_adapter()
    adapter.policy.predict_action_chunk = lambda batch: adapter._torch.zeros((1, 32, 8))
    with pytest.raises(ValueError, match='full chunk'):
        probe.predict_chunk(adapter, {'observation.state': np.zeros(8, dtype=np.float32)}, 1)


def test_fk_uses_absolute_commands_and_never_mutates_live_state(tmp_path):
    from examples.insert_plug.environment.scene import load_config, build_scene
    from examples.insert_plug.environment.simulation import Simulation
    config = load_config()
    sim = Simulation(build_scene(config, tmp_path / 'scene.xml'), config, render=False)
    try:
        action = np.r_[sim.data.qpos[sim.arm_qadr], sim.gripper_width()]
        modified = action.copy()
        modified[0] += .1
        commands = np.array([action, modified, modified, action])
        snapshots = {key: getattr(sim.data, key).copy() for key in ('qpos', 'qvel', 'ctrl', 'qacc_warmstart')}
        sim_time = sim.data.time
        tcp, rotations = probe.forward_kinematics(sim, commands)
        np.testing.assert_allclose(tcp[0], sim.site('tcp'), atol=1e-15)
        np.testing.assert_array_equal(tcp[0], tcp[-1])
        np.testing.assert_array_equal(tcp[1], tcp[2])
        assert np.linalg.norm(tcp[1] - tcp[0]) > .001
        assert rotations.shape == (4, 3, 3)
        for key, value in snapshots.items():
            np.testing.assert_array_equal(getattr(sim.data, key), value)
        assert sim.data.time == sim_time
    finally:
        sim.close()


def test_replay_holds_original_targets_at_physical_rate(tmp_path):
    from examples.insert_plug.environment.scene import load_config, build_scene
    from examples.insert_plug.environment.simulation import Simulation
    config = load_config()
    sim = Simulation(build_scene(config, tmp_path / 'scene.xml'), config, render=False)
    try:
        commands = np.tile(np.r_[sim.data.qpos[sim.arm_qadr], sim.gripper_width()], (3, 1))
        commands[1:, 0] += .005
        copied = commands.copy()
        result = probe.replay_chunk(sim, commands)
        assert result['failure'] is None
        assert result['tcp'].shape == (3, 3)
        assert result['qpos'].shape == (3, sim.model.nq)
        assert sim.data.time == pytest.approx(.3, abs=1e-12)
        np.testing.assert_array_equal(sim.data.ctrl[sim.arm_aids], commands[-1, :7])
        np.testing.assert_array_equal(commands, copied)
    finally:
        sim.close()


def test_replay_reports_invalid_command_without_clipping_or_continuing():
    class Simulation:
        def __init__(self):
            self.data = SimpleNamespace(qpos=np.zeros(8))
            self.received = []

        def reset(self):
            self.reset_called = True

        def step(self, action):
            self.received.append(action.copy())
            if np.max(np.abs(action[:7])) > 10:
                raise ValueError('Joint target outside robot limits')

        def site(self, name):
            return np.zeros(3)

        def gripper_width(self):
            return .075

        def gripper_contacts(self):
            return []

    sim = Simulation()
    actions = np.zeros((4, 8))
    actions[:, 7] = .075
    actions[2, 0] = 999
    result = probe.replay_chunk(sim, actions)
    assert sim.reset_called
    assert len(sim.received) == 3
    np.testing.assert_array_equal(sim.received[-1], actions[2])
    assert result['tcp'].shape == (2, 3)
    assert result['failure']['query'] == 2
    assert 'Joint target outside robot limits' in result['failure']['error']
