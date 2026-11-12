"""Attention diagnostics preserve action timing and the image that made a chunk."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from examples.insert_plug.policy import visualize_attention as visualization


CAMERAS = ('observation.images.head', 'observation.images.wrist')


def capture(chunk_size=4):
    weights = np.arange(1, chunk_size * 10 + 1, dtype=np.float32).reshape(chunk_size, 10)
    weights /= weights.sum(axis=1, keepdims=True)
    return SimpleNamespace(
        token_weights=weights,
        camera_weights={key: weights[:, 2 + index*4:6 + index*4].reshape(chunk_size, 2, 2)
                        for index, key in enumerate(CAMERAS)},
        special_token_weights={'latent': weights[:, 0], 'state': weights[:, 1]},
        grid_shapes={key: (2, 2) for key in CAMERAS},
        decoder_layer_index=0,
    )


class FakeTrace:
    def __init__(self):
        # The recorder must respect a monotonic counter, not assume it starts at 0.
        self.capture_count = 5
        self.latest = None

    def clear(self):
        self.latest = None


class QueuedAdapter:
    def __init__(self, trace, n_action_steps):
        self.trace = trace
        self.n_action_steps = n_action_steps
        self.calls = 0
        self.actions = []
        self.capture_interval = n_action_steps
        self.capture_increment = 1

    def reset(self):
        self.calls = 0
        self.actions = []

    def select_action(self, observation):
        if self.capture_interval is not None and self.calls % self.capture_interval == 0:
            self.trace.capture_count += self.capture_increment
            self.trace.latest = capture()
        action = np.arange(8, dtype=np.float32) + self.calls / 100
        self.calls += 1
        self.actions.append(action)
        return action


class MemoryWriter:
    def __init__(self):
        self.frames = []

    def append_data(self, frame):
        self.frames.append(frame.copy())


def observation(value=10):
    return {'observation.state': np.arange(8, dtype=np.float32) + value,
            **{key: np.full((8, 8, 3), value + index, dtype=np.uint8)
               for index, key in enumerate(CAMERAS)}}


def fast_renderer(monkeypatch):
    rendered = []

    def render(live, source, captured, query, **kwargs):
        rendered.append(dict(live={key: live[key].copy() for key in CAMERAS},
                             source={key: source[key].copy() for key in CAMERAS},
                             capture=captured, query=query, **kwargs))
        return np.zeros((16, 16, 3), dtype=np.uint8)

    monkeypatch.setattr(visualization, 'render_frame', render)
    return rendered


@pytest.mark.parametrize('n_action_steps', [1, 3])
def test_cached_queries_keep_prediction_images_and_actions_unchanged(tmp_path, monkeypatch, n_action_steps):
    rendered = fast_renderer(monkeypatch)
    trace = FakeTrace()
    adapter = QueuedAdapter(trace, n_action_steps)
    writer = MemoryWriter()
    recorder = visualization.AttentionPolicy(adapter, trace, tmp_path, writer, color_max=.07)
    recorder.reset()
    live = observation()
    for step in range(7):
        # Reuse and mutate simulator buffers between calls, as a renderer may do.
        for index, key in enumerate(CAMERAS):
            live[key].fill(10 + step + index)
        live['observation.state'][:] = np.arange(8) + 10 + step
        before = {key: value.copy() for key, value in live.items()}
        returned = recorder.select_action(live)
        assert returned is adapter.actions[-1]
        for key, value in live.items():
            np.testing.assert_array_equal(value, before[key])

    assert len(writer.frames) == len(rendered) == len(recorder.frames) == 7
    for step, (frame, row) in enumerate(zip(rendered, recorder.frames, strict=True)):
        source_step = step - step % n_action_steps
        assert row['query_index'] == frame['query'] == step % n_action_steps
        assert row['chunk_index'] == step // n_action_steps
        assert row['time_s'] == frame['time_s'] == step / 10
        assert row['source_time_s'] == frame['source_time_s'] == source_step / 10
        assert row['source_age_s'] == (step - source_step) / 10
        assert frame['color_max'] == .07
        for index, key in enumerate(CAMERAS):
            assert np.all(frame['live'][key] == 10 + step + index)
            assert np.all(frame['source'][key] == 10 + source_step + index)
        np.testing.assert_array_equal(row['action'], adapter.actions[step])
        assert sum(row['camera_mass'].values()) + sum(row['special_token_mass'].values()) == pytest.approx(1)

    for chunk in recorder.chunks:
        with np.load(tmp_path / chunk['weights']) as saved:
            assert saved['token_weights'].shape == (4, 10)
            np.testing.assert_array_equal(saved['token_weights'], capture().token_weights)
            for index, key in enumerate(CAMERAS):
                assert np.all(saved[f'image_{index}'] == 10 + chunk['source_step'] + index)
                np.testing.assert_array_equal(saved[f'attention_{index}'], capture().camera_weights[key])
            np.testing.assert_array_equal(saved['source_state'], np.arange(8) + 10 + chunk['source_step'])
    assert (tmp_path / 'preview.png').is_file()
    with pytest.raises(RuntimeError, match='fresh attention recorder'):
        recorder.reset()


def test_absolute_scale_distinguishes_uniform_attention_strength():
    image = np.full((32, 32, 3), 80, dtype=np.uint8)
    low = np.full((2, 2), .01, dtype=np.float32)
    high = np.full((2, 2), .02, dtype=np.float32)
    low_overlay = visualization._overlay(image, low, .05)
    high_overlay = visualization._overlay(image, high, .05)
    # A per-frame min/max normalization would erase the difference entirely.
    assert not np.array_equal(low_overlay, high_overlay)
    np.testing.assert_array_equal(low_overlay, visualization._overlay(image, high, .1))
    np.testing.assert_array_equal(image, visualization._overlay(image, np.zeros((2, 2)), .05))


def test_render_keeps_live_and_source_views_separate_and_camera_scale_fixed():
    source, live = observation(10), observation(180)
    captured = capture()
    for weights in captured.camera_weights.values():
        weights[:] = 0
    kwargs = dict(time_s=.2, source_time_s=0., chunk_index=0, color_max=.05)
    frame = visualization.render_frame(live, source, captured, 0, **kwargs)
    assert frame.shape == (912, 688, 3)
    for index, _ in enumerate(CAMERAS):
        x = 16 + 336*index
        np.testing.assert_array_equal(frame[109:429, x:x+320], np.full((320, 320, 3), 180 + index))
        np.testing.assert_array_equal(frame[472:792, x:x+320], np.full((320, 320, 3), 10 + index))
    captured.camera_weights[CAMERAS[0]][:] = .01
    head_only = visualization.render_frame(live, source, captured, 0, **kwargs)
    captured.camera_weights[CAMERAS[1]][:] = .4
    both = visualization.render_frame(live, source, captured, 0, **kwargs)
    np.testing.assert_array_equal(head_only[472:792, 16:336], both[472:792, 16:336])
    assert not np.array_equal(head_only[472:792, 352:672], both[472:792, 352:672])


@pytest.mark.parametrize('failure', ['missing', 'skipped_capture', 'stale_queue'])
def test_missing_or_mismatched_capture_fails_explicitly(tmp_path, monkeypatch, failure):
    fast_renderer(monkeypatch)
    trace = FakeTrace()
    adapter = QueuedAdapter(trace, 2)
    writer = MemoryWriter()
    recorder = visualization.AttentionPolicy(adapter, trace, tmp_path, writer)
    recorder.reset()
    if failure == 'missing':
        adapter.capture_interval = None
        match = 'without capturing'
    elif failure == 'skipped_capture':
        adapter.capture_increment = 2
        match = 'one ACT inference'
    else:
        adapter.capture_interval = 99
        recorder.select_action(observation())
        recorder.select_action(observation(11))
        match = 'action queue'
    frame_count = len(writer.frames)
    with pytest.raises(RuntimeError, match=match):
        recorder.select_action(observation(12))
    assert len(writer.frames) == frame_count


@pytest.mark.parametrize('coefficient', [-.3, 0., .3])
def test_temporal_contributions_reconstruct_vendor_actions_through_history_rollover(coefficient):
    torch = pytest.importorskip('torch')
    modeling = pytest.importorskip('lerobot.policies.act.modeling_act')
    chunk_size = 4
    ensemble = modeling.ACTTemporalEnsembler(coefficient, chunk_size)
    chunks = []
    for step in range(3 * chunk_size):
        # Distinct source/query patterns catch both indexing and weighting errors.
        chunk = np.array([[100 * step + 3 * query**2 + 1, step**2 + .25 * query]
                          for query in range(chunk_size)], dtype=np.float32)
        chunks.append(chunk)
        action = ensemble.update(torch.from_numpy(chunk[None])).numpy()[0]
        rows = visualization.temporal_contributions(step, chunk_size, coefficient)
        sources = list(range(max(0, step - chunk_size + 1), step + 1))
        assert len(rows) == len(sources)
        assert [row['source_step'] for row in rows] == sources
        assert [row['chunk_index'] for row in rows] == sources
        assert [row['query_index'] for row in rows] == [step - source for source in sources]
        assert [row['source_time_s'] for row in rows] == [source / 10 for source in sources]
        assert [row['source_age_s'] for row in rows] == [(step - source) / 10 for source in sources]
        weights = np.array([row['weight'] for row in rows])
        assert np.isfinite(weights).all() and (weights > 0).all()
        assert weights.sum() == pytest.approx(1)
        if coefficient == 0:
            np.testing.assert_allclose(weights, 1 / len(rows))
        elif coefficient > 0:
            assert np.all(np.diff(weights) < 0)
        else:
            assert np.all(np.diff(weights) > 0)
        reconstructed = sum(row['weight'] * chunks[row['source_step']][row['query_index']].astype(np.float64)
                            for row in rows)
        np.testing.assert_allclose(reconstructed, action, rtol=2e-6, atol=1e-5)


@pytest.mark.parametrize('coefficient', [0., .01, -.1])
def test_ensemble_heatmaps_use_current_query_and_image_without_fusing_history(tmp_path, monkeypatch, coefficient):
    rendered = fast_renderer(monkeypatch)
    trace = FakeTrace()
    adapter = QueuedAdapter(trace, 1)
    adapter.temporal_ensemble_coeff = coefficient
    adapter.chunk_size = 4
    writer = MemoryWriter()
    recorder = visualization.AttentionPolicy(adapter, trace, tmp_path, writer)
    recorder.reset()
    for step in range(9):
        live = observation(10 + step)
        action = recorder.select_action(live)
        assert action is adapter.actions[-1]
        row, frame = recorder.frames[-1], rendered[-1]
        assert row['query_index'] == frame['query'] == 0
        assert row['source_time_s'] == frame['source_time_s'] == step / 10
        assert row['source_age_s'] == 0
        for key in CAMERAS:
            np.testing.assert_array_equal(frame['source'][key], live[key])
            np.testing.assert_array_equal(frame['live'][key], live[key])
            # The visualized query is from this inference, not a sum over old captures.
            np.testing.assert_array_equal(frame['capture'].camera_weights[key][0], capture().camera_weights[key][0])
        metadata = row['temporal_ensemble']
        assert frame['ensemble'] == metadata
        assert metadata['coefficient'] == coefficient
        contributions = metadata['contributions']
        assert len(contributions) == min(step + 1, adapter.chunk_size)
        assert contributions[-1]['source_step'] == step
        assert contributions[-1]['query_index'] == 0
        assert metadata['current_prediction_weight'] == pytest.approx(contributions[-1]['weight'])
        assert metadata['historical_prediction_weight'] == pytest.approx(sum(item['weight'] for item in contributions[:-1]))
        assert metadata['current_prediction_weight'] + metadata['historical_prediction_weight'] == pytest.approx(1)
    assert len(recorder.chunks) == len(writer.frames) == 9
    assert all((tmp_path / row['weights']).is_file() for row in recorder.chunks)


def test_cli_forwards_requested_temporal_ensemble_settings(monkeypatch):
    received = {}
    monkeypatch.setenv('MUJOCO_GL', 'osmesa')
    monkeypatch.setattr(visualization, 'run_visualization', lambda **kwargs: received.update(kwargs))
    visualization.main(['--checkpoint', '/tmp/attention-test-checkpoint',
                        '--temporal-ensemble-coeff', '.01', '--n-action-steps', '1'])
    assert received['temporal_ensemble_coeff'] == .01
    assert received['n_action_steps'] == 1
