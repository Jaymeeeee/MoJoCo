"""Independent sampling must preserve real timing and the live dynamics."""

import numpy as np
import pytest

from examples.insert_plug.expert import Expert
from examples.insert_plug.recording import ALIGNED_ACTION_TIMING, LEGACY_ACTION_TIMING, RecordingSampler
from examples.insert_plug.scene import build_scene, load_config
from examples.insert_plug.simulation import Simulation


def test_fifteen_hz_uses_physics_ticks_without_control_step_jitter_or_drift():
    sampler = RecordingSampler(15, .001)
    times = [tick * .001 for tick in range(1000) if sampler.due(tick * .001)]
    assert len(times) == 15
    np.testing.assert_allclose(times[:4], [0, .067, .133, .200], atol=1e-12)
    assert set(np.round(np.diff(times), 3)) == {.066, .067}
    assert np.max(np.abs(np.asarray(times) - np.arange(15) / 15)) <= .0005
    metadata = sampler.metadata(50)
    assert metadata['sample_hz'] == 15 and metadata['control_hz'] == 50
    assert metadata['action_timing'] == LEGACY_ACTION_TIMING
    assert 'action_hold_duration_s' not in metadata


@pytest.mark.parametrize('origin', [0., .1, 4.2])
def test_equal_rates_on_the_zero_origin_control_grid_declare_the_actual_hold(origin):
    metadata = RecordingSampler(10, .001, start_time=origin).metadata(10)
    assert metadata['action_timing'] == ALIGNED_ACTION_TIMING
    assert metadata['action_hold_duration_s'] == .1
    assert metadata['sample_origin_s'] == origin


@pytest.mark.parametrize('sample_hz,control_hz,origin', [
    (10, 50, 0), (10, 10, .001), (10, 10, .15), (15, 15, 0),
])
def test_unaligned_sampling_does_not_claim_a_full_action_hold(sample_hz, control_hz, origin):
    metadata = RecordingSampler(sample_hz, .001, start_time=origin).metadata(control_hz)
    assert metadata['action_timing'] == LEGACY_ACTION_TIMING
    assert 'action_hold_duration_s' not in metadata


@pytest.mark.parametrize('sample_hz', [0, -1, float('nan'), 1001])
def test_impossible_sampling_rates_are_rejected(sample_hz):
    with pytest.raises(ValueError):
        RecordingSampler(sample_hz, .001)


def test_missing_a_scheduled_tick_is_reported_instead_of_retiming_the_frame():
    sampler = RecordingSampler(15, .001)
    assert sampler.due(0)
    with pytest.raises(RuntimeError, match='Missed recording tick'):
        sampler.due(.068)


@pytest.mark.parametrize('sample_hz,control_hz,expected_times', [
    (15, 50, [0, .067, .133, .200, .267, .333, .400, .467]),
    (10, 10, np.arange(25) / 10),
])
def test_substep_recording_aligns_render_state_and_does_not_change_physics(
        tmp_path, monkeypatch, sample_hz, control_hz, expected_times):
    cfg = load_config()
    cfg['recording'] = {'sample_hz': sample_hz}
    cfg['task']['control_hz'] = control_hz
    cfg['cameras']['depth'] = {'enabled': True}
    path = build_scene(cfg, tmp_path / 'scene.xml')
    recorded = Simulation(path, cfg, render=False)
    reference = Simulation(path, cfg, render=False)
    frames = []

    class Writer:
        def append(self, frame):
            frames.append(frame)

    def render_rgbd(data=None):
        # A renderer sees a refreshed copy of exactly this physics timestamp.
        assert data is not None and data is not recorded.data
        np.testing.assert_array_equal(data.qpos, recorded.data.qpos)
        np.testing.assert_array_equal(data.qvel, recorded.data.qvel)
        assert data.time == recorded.data.time
        code = int(round(data.time * 1000)) % 256
        rgb = {name: np.full((4, 4, 3), code, np.uint8) for name in ('head', 'wrist')}
        depth = {name: np.full((4, 4), 1 + data.qpos[0], np.float32) for name in ('head', 'wrist')}
        return rgb, depth

    monkeypatch.setattr(recorded, 'render_rgbd', render_rgbd)
    try:
        expert = Expert(recorded, writer=Writer(), verbose=False)
        home = np.asarray(cfg['robot']['home_qpos'])
        for step in range(25):
            target = home.copy()
            target[0] += .001 * step
            aperture = cfg['gripper']['open_width']
            expert.tick(target, aperture, 'storage_test')
            reference.step(np.r_[target, aperture].astype(np.float32).astype(np.float64))
            # Includes solver state: a forward pass on live data would differ.
            for field in ('qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart'):
                np.testing.assert_array_equal(getattr(recorded.data, field), getattr(reference.data, field))
        np.testing.assert_allclose([frame['timestamp'] for frame in frames],
                                   expected_times, atol=1e-10)
        assert expert.steps == 25 and expert.recording_sampler.samples == len(expected_times)
        for frame in frames:
            time = frame['timestamp']
            expected_control_step = int(np.floor((time + 1e-9) * control_hz))
            assert frame['action'][0] == pytest.approx(.001 * expected_control_step)
            np.testing.assert_array_equal(frame['action'], frame['action'].astype(np.float32).astype(np.float64))
            np.testing.assert_array_equal(frame['ctrl'][recorded.arm_aids], frame['action'][:7])
            np.testing.assert_array_equal(frame['joint_position'], frame['qpos'][recorded.arm_qadr])
            assert frame['images']['head'][0, 0, 0] == int(round(time * 1000)) % 256
            assert frame['depth']['head'][0, 0] == np.float32(1 + frame['qpos'][0])
    finally:
        recorded.close()
        reference.close()
