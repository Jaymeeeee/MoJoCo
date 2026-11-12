"""Camera interventions use separate physical units and fixed ensemble history."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from examples.insert_plug.policy.camera_ablation import (
    action_metrics, build_conditions, isolated_ensemble_delta, summarize_metrics,
)


def test_action_metrics_keep_joints_and_gripper_in_their_own_units():
    physical = np.zeros((2, 3, 4, 8), dtype=np.float64)
    physical[..., :7] = np.deg2rad(1.)
    physical[..., 7] = .002
    normalized = np.full_like(physical, 4.)
    metrics = action_metrics(physical, normalized)
    expected = dict(q0_normalized_rms=4., chunk_normalized_rms=4.,
                    q0_arm_rms_deg=1., q0_arm_max_deg=1.,
                    q0_gripper_abs_mm=2., chunk_arm_rms_deg=1.,
                    chunk_gripper_rms_mm=2.)
    for key, value in expected.items():
        assert np.shape(metrics[key]) == (2, 3)
        np.testing.assert_allclose(metrics[key], value, rtol=1e-12, atol=1e-12)

    # Gripper motion must not become a joint rotation, and vice versa.
    physical[..., :7] = 0
    grip_only = action_metrics(physical, normalized)
    np.testing.assert_array_equal(grip_only['q0_arm_rms_deg'], 0)
    np.testing.assert_array_equal(grip_only['q0_arm_max_deg'], 0)
    np.testing.assert_allclose(grip_only['q0_gripper_abs_mm'], 2)
    physical[..., :7] = np.pi / 2
    physical[..., 7] = 0
    arm_only = action_metrics(physical, normalized)
    np.testing.assert_allclose(arm_only['q0_arm_rms_deg'], 90)
    np.testing.assert_array_equal(arm_only['q0_gripper_abs_mm'], 0)


def test_action_metrics_distinguish_current_action_from_future_chunk():
    physical = np.zeros((2, 8), dtype=np.float64)
    physical[0, :7], physical[1, :7] = np.deg2rad(1.), np.deg2rad(3.)
    physical[:, 7] = [-.001, .005]
    normalized = np.empty_like(physical)
    normalized[0], normalized[1] = -2., 4.
    metrics = action_metrics(physical, normalized)
    assert metrics['q0_normalized_rms'] == pytest.approx(2.)
    assert metrics['chunk_normalized_rms'] == pytest.approx(np.sqrt(10))
    assert metrics['q0_arm_rms_deg'] == pytest.approx(1.)
    assert metrics['chunk_arm_rms_deg'] == pytest.approx(np.sqrt(5))
    assert metrics['q0_gripper_abs_mm'] == pytest.approx(1.)
    assert metrics['chunk_gripper_rms_mm'] == pytest.approx(np.sqrt(13))


@pytest.mark.parametrize('coefficient', [-.3, 0., .3])
def test_isolated_ensemble_change_matches_real_updater_with_identical_history(coefficient):
    torch = pytest.importorskip('torch')
    from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

    rng = np.random.default_rng(53)
    chunk_size = 4
    baseline = ACTTemporalEnsembler(coefficient, chunk_size)
    delta = np.linspace(-.4, .3, 8, dtype=np.float32)
    for step in range(3 * chunk_size):
        # Both interventions start from exactly the same prior prediction state.
        perturbed = copy.deepcopy(baseline)
        prediction = torch.from_numpy(rng.normal(size=(1, chunk_size, 8)).astype(np.float32))
        changed_prediction = prediction.clone()
        changed_prediction[:, 0] += torch.from_numpy(delta)
        actual = (perturbed.update(changed_prediction) - baseline.update(prediction))[0].numpy()
        predicted = isolated_ensemble_delta(delta, step, chunk_size, coefficient)
        np.testing.assert_allclose(predicted, actual, rtol=2e-5, atol=3e-7)
        # Leading dimensions let callers compare several cameras/mask repeats.
        stacked = isolated_ensemble_delta(np.stack([delta, delta * 2]), step, chunk_size, coefficient)
        np.testing.assert_allclose(stacked[0], predicted)
        np.testing.assert_allclose(stacked[1], predicted * 2)


def test_conditions_pair_camera_interventions_and_keep_one_baseline():
    conditions = build_conditions(3)
    assert len(conditions) == 1 + 4 + 2 * 3
    assert [row['index'] for row in conditions] == list(range(len(conditions)))
    assert len({row['name'] for row in conditions}) == len(conditions)
    assert conditions[0]['camera'] is None and conditions[0]['scheme'] == 'baseline'
    for scheme in ('full_mean', 'full_black'):
        selected = [row for row in conditions if row['scheme'] == scheme]
        assert {row['camera'] for row in selected} == {'head', 'wrist'}
        assert len(selected) == 2
    for repeat in range(3):
        selected = [row for row in conditions if row['scheme'] == 'patch_mean' and row['repeat'] == repeat]
        assert {row['camera'] for row in selected} == {'head', 'wrist'}
        assert len(selected) == 2


def test_summary_averages_mask_repeats_before_describing_frames():
    conditions = build_conditions(2)
    metric_names = ('q0_normalized_rms', 'chunk_normalized_rms', 'q0_arm_rms_deg',
                    'q0_arm_max_deg', 'q0_gripper_abs_mm', 'chunk_arm_rms_deg',
                    'chunk_gripper_rms_mm')
    metrics = {key: np.zeros((4, len(conditions)-1)) for key in metric_names}
    patch_columns = [row['index']-1 for row in conditions
                     if row['camera'] == 'head' and row['scheme'] == 'patch_mean']
    metrics['q0_normalized_rms'][:, patch_columns[0]] = [0, 2, 4, 6]
    metrics['q0_normalized_rms'][:, patch_columns[1]] = [4, 6, 8, 10]
    metrics['q0_arm_max_deg'][:, patch_columns[0]] = [0, 0, 0, 4]
    metrics['q0_arm_max_deg'][:, patch_columns[1]] = [4, 0, 0, 0]
    metrics['q0_gripper_abs_mm'][:, patch_columns] = np.array([0, 0, 2, 0])[:, None]
    summaries = summarize_metrics(metrics, conditions, np.array([0., 4.9, 5., 9.9]),
                                  joint_threshold_deg=1., gripper_threshold_mm=1.)
    assert len(summaries) == 6
    group = next(row for row in summaries if row['camera'] == 'head' and row['scheme'] == 'patch_mean')
    # Repeats are paired perturbations of four correlated execution frames,
    # not eight independent execution samples or additional statistical trials.
    assert group['samples'] == 4
    stats = group['metrics']['q0_normalized_rms']
    assert stats['mean'] == pytest.approx(5.)
    assert stats['median'] == pytest.approx(5.)
    assert stats['p95'] == pytest.approx(7.7)
    assert stats['max'] == pytest.approx(8.)
    assert group['obvious_change_fraction'] == pytest.approx(.75)
    bins = group['time_bins']
    assert [(row['start_s'], row['end_s'], row['samples']) for row in bins] == [(0., 5., 2), (5., 10., 2)]
    # A fully unperturbed camera condition remains zero in every statistic.
    untouched = next(row for row in summaries if row['camera'] == 'wrist' and row['scheme'] == 'full_mean')
    assert untouched['obvious_change_fraction'] == 0
    assert untouched['metrics']['q0_normalized_rms']['max'] == 0
