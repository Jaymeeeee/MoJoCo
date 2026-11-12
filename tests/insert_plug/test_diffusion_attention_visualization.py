"""Source-image timing and passive inference validation for Diffusion overlays."""
from types import SimpleNamespace

import numpy as np
import pytest

from examples.insert_plug.policy import visualize_diffusion_attention as visualization


CAMERAS = ('observation.images.head', 'observation.images.wrist')


def capture():
    weights = np.arange(1, 13, dtype=np.float32).reshape(2, 2, 3)
    weights /= weights.sum(axis=(1, 2), keepdims=True)
    return SimpleNamespace(camera_weights={key: weights.copy() for key in CAMERAS},
                           mean_weights={key: weights.mean(axis=0) for key in CAMERAS},
                           keypoints={key: np.array([[-1, -1], [1, 1]], dtype=np.float32) for key in CAMERAS},
                           grid_shapes={key: (2, 3) for key in CAMERAS},
                           keypoint_counts={key: 2 for key in CAMERAS},
                           keypoint_reconstruction_max_abs_error={key: 0. for key in CAMERAS})


class Trace:
    capture_count = 7
    latest = None

    def clear(self):
        self.latest = None


class Adapter:
    def __init__(self, trace, n_action_steps=3):
        self.trace, self.n_action_steps = trace, n_action_steps
        self.capture_interval = n_action_steps
        self.capture_increment = 1
        self.reset()

    def reset(self):
        self.calls, self.actions = 0, []

    def select_action(self, observation):
        if self.capture_interval is not None and self.calls % self.capture_interval == 0:
            self.trace.latest = capture()
            self.trace.capture_count += self.capture_increment
        action = np.arange(8, dtype=np.float32) + self.calls / 100
        self.actions.append(action)
        self.calls += 1
        return action


class Writer:
    def __init__(self):
        self.frames = []

    def append_data(self, frame):
        self.frames.append(frame.copy())


def observation(value=10):
    return {'observation.state': np.arange(8, dtype=np.float32) + value,
            **{key: np.full((8, 8, 3), value + index, dtype=np.uint8) for index, key in enumerate(CAMERAS)}}


@pytest.fixture
def fast_render(monkeypatch):
    rendered = []

    def render(live, source, captured, offset, **kwargs):
        rendered.append(dict(live={key: live[key].copy() for key in CAMERAS},
                             source={key: source[key].copy() for key in CAMERAS},
                             capture=captured, offset=offset, **kwargs))
        return np.zeros((16, 16, 3), dtype=np.uint8)

    monkeypatch.setattr(visualization, 'render_frame', render)
    monkeypatch.setattr(visualization, 'render_keypoint_atlas', lambda *a, **k: visualization.Image.new('RGB', (16, 16)))
    return rendered


@pytest.mark.parametrize('n_action_steps', [1, 3, 32])
def test_source_alignment_raw_arrays_and_original_actions(tmp_path, fast_render, n_action_steps):
    trace = Trace()
    adapter, writer = Adapter(trace, n_action_steps), Writer()
    recorder = visualization.DiffusionAttentionPolicy(adapter, trace, tmp_path, writer, color_max=.07)
    recorder.reset()
    live = observation()
    count = n_action_steps + 2
    expected_observations = []
    for step in range(count):
        for index, key in enumerate(CAMERAS):
            live[key].fill(step + index)
        live['observation.state'][:] = np.arange(8) + step
        before = {key: value.copy() for key, value in live.items()}
        expected_observations.append(before)
        returned = recorder.select_action(live)
        assert returned is adapter.actions[-1]
        for key in before:
            np.testing.assert_array_equal(live[key], before[key])
    assert len(writer.frames) == len(recorder.frames) == count
    for step, (frame, row) in enumerate(zip(fast_render, recorder.frames, strict=True)):
        source_step = step - step % n_action_steps
        assert row['source_step'] == source_step
        assert row['action_offset'] == step % n_action_steps
        assert row['source_age_s'] == pytest.approx((step - source_step) / 10)
        assert row['new_prediction'] is (step % n_action_steps == 0)
        assert frame['color_max'] == .07
        for index, key in enumerate(CAMERAS):
            np.testing.assert_array_equal(frame['source'][key], expected_observations[source_step][key])
            np.testing.assert_array_equal(frame['live'][key], expected_observations[step][key])
    for chunk in recorder.chunks:
        with np.load(tmp_path / chunk['weights']) as arrays:
            for index, key in enumerate(CAMERAS):
                np.testing.assert_array_equal(arrays[f'attention_{index}'], capture().camera_weights[key])
                np.testing.assert_array_equal(arrays[f'keypoints_{index}'], capture().keypoints[key])
                np.testing.assert_array_equal(arrays[f'image_{index}'], expected_observations[chunk['source_step']][key])
    recorder.save_trace()
    with np.load(tmp_path / 'trace.npz') as arrays:
        np.testing.assert_array_equal(arrays['actions'], adapter.actions)
        for index, key in enumerate(CAMERAS):
            np.testing.assert_array_equal(arrays[f'images_{index}'], [row[key] for row in expected_observations])
    with pytest.raises(RuntimeError, match='fresh'):
        recorder.reset()


@pytest.mark.parametrize('capture_interval,error_step', [(None, 0), (1, 1), (4, 3)])
def test_missing_early_or_late_capture_rejected(tmp_path, fast_render, capture_interval, error_step):
    trace = Trace()
    adapter = Adapter(trace)
    adapter.capture_interval = capture_interval
    recorder = visualization.DiffusionAttentionPolicy(adapter, trace, tmp_path, Writer())
    recorder.reset()
    for _ in range(error_step):
        recorder.select_action(observation())
    with pytest.raises(RuntimeError, match='capturing|queue timing'):
        recorder.select_action(observation())


def test_multiple_encodings_per_action_rejected(tmp_path, fast_render):
    trace = Trace()
    adapter = Adapter(trace)
    adapter.capture_increment = 2
    recorder = visualization.DiffusionAttentionPolicy(adapter, trace, tmp_path, Writer())
    recorder.reset()
    with pytest.raises(RuntimeError, match='exactly one'):
        recorder.select_action(observation())


def test_keypoints_use_feature_cell_centers():
    points = np.array([[-1, -1], [0, 0], [1, 1]])
    actual = visualization.keypoint_pixels(points, (8, 8), (256, 256))
    np.testing.assert_array_equal(actual, [[15.5, 15.5], [127.5, 127.5], [239.5, 239.5]])
    np.testing.assert_array_equal(visualization.keypoint_pixels(np.array([[0., 0.]]), (2, 4), (80, 40)), [[39.5, 19.5]])


def test_fixed_mean_display_and_all_keypoint_atlas():
    cap = capture()
    obs = observation()
    frame = visualization.render_frame(obs, obs, cap, 2, time_s=.2, source_time_s=0., chunk_index=0, color_max=.05)
    assert frame.shape == (1008, 688, 3) and frame.dtype == np.uint8
    atlas = visualization.render_keypoint_atlas(obs[CAMERAS[0]], cap.camera_weights[CAMERAS[0]], cap.keypoints[CAMERAS[0]],
                                               name='head', grid_shape=(2, 3))
    assert atlas.size == (688, 268)
    # A constant probability has the same displayed color at every frame, not a per-map rescaling.
    rgb = np.full((8, 8, 3), 100, dtype=np.uint8)
    low = visualization._overlay(rgb, np.full((2, 2), .01, dtype=np.float32), .05)
    high = visualization._overlay(rgb, np.full((2, 2), .04, dtype=np.float32), .05)
    assert not np.array_equal(low, high)


def test_replay_checks_all_actions_and_final_rng():
    import torch

    class RandomAdapter:
        def reset(self):
            self.queue = []

        def select_action(self, observation):
            if not self.queue:
                self.queue = list(torch.randn(3, 8).numpy())
            return self.queue.pop(0)

    adapter = RandomAdapter()
    adapter.reset()
    torch.manual_seed(17)
    initial = visualization._rng_snapshot(torch)
    observations = [observation(i) for i in range(7)]
    actions = [adapter.select_action(row).copy() for row in observations]
    final = visualization._rng_snapshot(torch)
    result, replay_rng = visualization.verify_actions(adapter, observations, actions, seed=17,
        expected_initial_rng=initial, expected_final_rng=final, seed_function=torch.manual_seed, torch=torch)
    assert result['passed'] and result['actions_checked'] == 7
    assert visualization._rng_equal(torch, final, replay_rng)
    actions[4][2] += 1
    result, _ = visualization.verify_actions(adapter, observations, actions, seed=17,
        expected_initial_rng=initial, expected_final_rng=final, seed_function=torch.manual_seed, torch=torch)
    assert not result['passed'] and result['mismatched_steps'] == [4]


def test_reference_validation_rejects_seed_and_model_mismatch(tmp_path):
    import json
    checkpoint = tmp_path / 'pretrained_model'
    checkpoint.mkdir()
    files = {'config.json': {'sha256': 'config-sha', 'bytes': 1}}
    r = dict(config={'board': 1}, seed=0, position='pos0', requested_duration_s=60., control_hz=10,
             action_hold_s=.1, physics_timestep_s=.001, stop_on_contact=False, contact_diagnostics=False,
             policy=dict(checkpoint=str(checkpoint), policy_type='diffusion', n_action_steps=32,
                         config_sha256='config-sha'))
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(r))
    args = dict(config=r['config'], checkpoint=checkpoint, seed=0, position='pos0', duration=60.,
                n_action_steps=32, files=files)
    assert visualization._validate_reference(path, **args)['result'] == r
    with pytest.raises(ValueError, match='seed'):
        visualization._validate_reference(path, **{**args, 'seed': 1})
    with pytest.raises(ValueError, match='checkpoint|execution'):
        visualization._validate_reference(path, **{**args, 'n_action_steps': 1})
