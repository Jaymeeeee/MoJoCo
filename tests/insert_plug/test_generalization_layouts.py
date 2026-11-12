"""Held-out geometry must stay new relative to raw training episodes."""
import json

import numpy as np
import pytest
import yaml

from examples.insert_plug.environment.generalization_layouts import audit_training_data, select_candidates


TRAINING = np.array([
    [.49, -.07, .41, .11], [.43, -.03, .49, .05], [.49, -.05, .41, .07],
    [.47, -.05, .47, .09], [.43, -.09, .47, .03], [.45, -.05, .41, .03],
    [.41, -.09, .45, .11], [.45, -.11, .41, .05], [.41, -.05, .45, .03],
    [.49, -.03, .47, .05],
])


def test_reproducible_unseen_individual_points_and_pairs_within_bounds():
    first, bounds, pool = select_candidates(TRAINING)
    assert (first, bounds, pool) == select_candidates(TRAINING)
    assert first != select_candidates(TRAINING, seed=20261110)[0]
    assert [row['name'] for row in first] == [f'pos{i}' for i in range(10)]
    assert pool == dict(source_points=16, target_points=16, pairs=256)
    pairs = np.array([row['source_xy'] + row['target_xy'] for row in first])
    assert np.all(pairs >= TRAINING.min(axis=0)) and np.all(pairs <= TRAINING.max(axis=0))
    for start in (0, 2):
        assert len(np.unique(pairs[:, start:start+2], axis=0)) == 10
        distances = np.linalg.norm(pairs[:, None, start:start+2] - TRAINING[None, :, start:start+2], axis=-1)
        assert distances.min() >= .01
    assert np.linalg.norm(pairs[:, None, :] - TRAINING[None, :, :], axis=-1).min() >= .02 - 1e-10
    np.testing.assert_allclose(pairs[0], [.48, -.08, .42, .04])


def test_novelty_uses_additional_actual_training_points():
    first = select_candidates(TRAINING)[0][0]
    augmented = np.vstack([TRAINING, first['source_xy'] + first['target_xy']])
    selected = select_candidates(augmented)[0]
    assert all(row['source_xy'] != first['source_xy'] for row in selected)
    assert all(row['target_xy'] != first['target_xy'] for row in selected)


@pytest.mark.parametrize('kwargs', [dict(count=17), dict(count=0), dict(grid_spacing_m=0),
                                    dict(minimum_point_distance_m=float('nan'))])
def test_rejects_impossible_or_invalid_selection(kwargs):
    with pytest.raises(ValueError):
        select_candidates(TRAINING, **kwargs)


def training_fixture(tmp_path):
    root = tmp_path/'dataset'
    episode = root/'pos0/episode_000000/episode_000000.yaml'
    episode.parent.mkdir(parents=True)
    episode.write_text(yaml.safe_dump(dict(board=dict(source_xy=[.45, -.07], target_xy=[.45, .07]))))
    metadata = root/'lerobot/meta/info.json'
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps(dict(total_episodes=1, fps=10)))
    manifest = tmp_path/'positions.yaml'
    manifest.write_text(yaml.safe_dump(dict(positions=[dict(name='pos0', source_xy=[.45, -.07], target_xy=[.45, .07])])))
    checkpoints = []
    for name in ('act', 'diffusion'):
        path = tmp_path/f'{name}.json'
        path.write_text(json.dumps(dict(dataset=dict(root=str(root/'lerobot'), episodes=None))))
        checkpoints.append(path)
    return root, manifest, checkpoints, episode


def test_raw_source_audit_hashes_every_episode_and_rejects_stale_manifest(tmp_path):
    root, manifest, checkpoints, episode = training_fixture(tmp_path)
    pairs, evidence = audit_training_data(root, manifest, checkpoints, expected_positions=1, episodes_per_position=1)
    np.testing.assert_allclose(pairs, [[.45, -.07, .45, .07]])
    assert len(evidence['raw_episode_configs'][0]['sha256']) == 64
    assert len(evidence['checkpoint_train_configs']) == 2
    episode.write_text(yaml.safe_dump(dict(board=dict(source_xy=[.46, -.07], target_xy=[.45, .07]))))
    with pytest.raises(ValueError, match='Raw episode disagrees'):
        audit_training_data(root, manifest, checkpoints, expected_positions=1, episodes_per_position=1)


def test_audit_rejects_checkpoint_filter_and_missing_raw_episode(tmp_path):
    root, manifest, checkpoints, episode = training_fixture(tmp_path)
    checkpoints[0].write_text(json.dumps(dict(dataset=dict(root=str(root/'lerobot'), episodes=[0]))))
    with pytest.raises(ValueError, match='filtering'):
        audit_training_data(root, manifest, checkpoints, expected_positions=1, episodes_per_position=1)
    episode.unlink()
    with pytest.raises(ValueError, match='episode counts'):
        audit_training_data(root, manifest, checkpoints, expected_positions=1, episodes_per_position=1)
