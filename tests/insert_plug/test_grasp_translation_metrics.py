"""Physical qualification excludes mere closure, transient contact and source support."""
import json

import numpy as np
import pytest

from examples.insert_plug.policy.grasp_translation_metrics import (
    GraspLiftMonitor, displacement_metrics, paired_grasp_metrics,
)


def feed(monitor, times, *, contact=True, height=.05, slip_per_second=0., rotation=None):
    rotation = np.eye(3) if rotation is None else rotation
    for t in times:
        tip = np.array([.4 + slip_per_second * t, -.05, height])
        tcp = np.array([.4, -.05, height + .03])
        monitor.sample(float(t), tcp, rotation, tip, .04, contact, int(t / .1))


def test_complete_two_hundred_milliseconds_requires_both_endpoints():
    monitor = GraspLiftMonitor()
    feed(monitor, np.arange(200) * .001)
    assert monitor.report()['grasp_event'] is None
    feed(monitor, [.200])
    report = monitor.report()
    assert report['passed']
    assert report['lift_event']['window_samples'] == 201
    assert report['lift_event']['window_seconds'] == pytest.approx(.2)
    assert report['grasp_event']['confirmed_action_index'] == 2
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('contacts', [False, [True, False], [False, True]])
def test_without_opposed_contact_lift_height_does_not_pass(contacts):
    monitor = GraspLiftMonitor()
    feed(monitor, np.arange(301) * .001, contact=contacts)
    assert monitor.report()['grasp_event'] is None
    assert not monitor.report()['passed']


def test_stable_grasp_while_seated_is_not_lift_success():
    monitor = GraspLiftMonitor()
    feed(monitor, np.arange(301) * .001, height=.0102)
    assert monitor.report()['grasp_event'] is not None
    assert not monitor.report()['passed']


def test_height_threshold_must_hold_for_complete_window():
    monitor = GraspLiftMonitor()
    feed(monitor, np.arange(201) * .001, height=.044)
    feed(monitor, np.arange(201, 401) * .001, height=.045)
    assert not monitor.report()['passed']
    feed(monitor, [.401], height=.045)
    assert monitor.report()['passed']
    assert monitor.report()['lift_event']['start_time_s'] == pytest.approx(.201)


def test_sliding_grasp_fails_despite_continuous_contact_and_clearance():
    monitor = GraspLiftMonitor()
    feed(monitor, np.arange(501) * .001, slip_per_second=.002)
    assert monitor.report()['grasp_event'] is None
    assert not monitor.report()['passed']


def test_rotating_rigid_grasp_uses_tcp_frame_instead_of_world_offset():
    monitor = GraspLiftMonitor()
    for t in np.arange(201) * .001:
        angle = float(t) * 3
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        tcp = np.array([.4, -.05, .08])
        tip = tcp + rotation @ [.01, 0, -.03]
        monitor.sample(float(t), tcp, rotation, tip, .04, True, int(t / .1))
    assert monitor.report()['passed']
    assert monitor.report()['lift_event']['max_relative_drift_m'] < 1e-14


def test_single_missing_contact_requires_a_new_full_window():
    monitor = GraspLiftMonitor()
    feed(monitor, np.arange(150) * .001)
    feed(monitor, [.150], contact=False)
    feed(monitor, np.arange(151, 351) * .001)
    assert not monitor.report()['passed']
    feed(monitor, [.351])
    assert monitor.report()['passed']


def test_missing_physics_samples_cannot_bridge_the_required_duration():
    monitor = GraspLiftMonitor()
    feed(monitor, np.arange(101) * .001)
    feed(monitor, np.arange(200, 400) * .001)
    assert not monitor.report()['passed']
    feed(monitor, [.400])
    assert monitor.report()['passed']


def test_earlier_abandoned_contact_is_not_selected_for_successful_lift():
    monitor = GraspLiftMonitor()
    feed(monitor, np.arange(251) * .001, height=.01)
    feed(monitor, [.251], contact=False, height=.01)
    feed(monitor, np.arange(252, 453) * .001)
    report = monitor.report()
    assert report['passed'] and len(report['grasp_events']) == 2
    assert report['first_observed_grasp_event']['start_time_s'] == 0
    assert report['grasp_event']['start_time_s'] == pytest.approx(.252)


def test_timestamp_roundoff_and_small_jitter_do_not_lose_valid_coverage():
    monitor = GraspLiftMonitor()
    times = np.arange(202) * .001 + np.sin(np.arange(202)) * 1e-7
    feed(monitor, times)
    assert monitor.report()['passed']
    assert monitor.report()['lift_event']['window_seconds'] >= .2 - 1e-10


def test_report_is_a_detached_snapshot():
    monitor = GraspLiftMonitor()
    feed(monitor, np.arange(201) * .001)
    first = monitor.report()
    first['grasp_event']['confirmed_tcp_m'][0] = 99
    assert monitor.report()['grasp_event']['confirmed_tcp_m'][0] == .4


def test_negative_axis_is_positive_when_tcp_follows_object():
    result = displacement_metrics([0, 0, 0], [.004, -.015, .003], [0, -.03, 0])
    assert result['gain'] == pytest.approx(.5)
    assert result['orthogonal_m'] == pytest.approx(.005)
    assert result['residual_m'] == pytest.approx(np.sqrt(.015**2 + .005**2))
    opposite = displacement_metrics([0, 0, 0], [0, .03, 0], [0, -.03, 0])
    assert opposite['gain'] == pytest.approx(-1)
    assert opposite['residual_m'] == pytest.approx(.06)


def test_equal_interval_and_separate_event_endpoints_remain_distinct():
    base = np.column_stack([np.arange(5) * .01, np.zeros((5, 2))])
    shifted = base + [.03, 0, 0]
    result = paired_grasp_metrics(base, shifted, base, shifted, [.03, 0, 0],
                                 {'confirmed_action_index': 1}, {'confirmed_action_index': 3})
    same = result['baseline_confirmed_grasp']
    independent = result['independent_confirmed_grasps']
    assert same['same_action_index'] and same['fk']['gain'] == pytest.approx(1)
    assert not independent['same_action_index']
    assert independent['actual']['along_m'] == pytest.approx(.05)
    json.dumps(result, allow_nan=False)


def test_failed_shift_has_no_invented_grasp_event_but_equal_time_metric_remains():
    base = np.zeros((5, 3))
    result = paired_grasp_metrics(base, base, base, base, [.03, 0, 0], {'confirmed_action_index': 1})
    assert result['baseline_confirmed_grasp']['actual']['gain'] == 0
    assert result['independent_confirmed_grasps'] is None


@pytest.mark.parametrize('kwargs', [dict(hold_seconds=0), dict(drift_tolerance_m=-1),
                                     dict(clearance_m=np.nan), dict(sample_period_s=0)])
def test_invalid_thresholds_fail(kwargs):
    with pytest.raises(ValueError):
        GraspLiftMonitor(**kwargs)


def test_non_monotonic_time_fails():
    monitor = GraspLiftMonitor()
    feed(monitor, [0, .001])
    with pytest.raises(ValueError):
        feed(monitor, [.001])


@pytest.mark.parametrize('delta', [[0, 0, 0], [np.nan, 0, 0], [0, 1]])
def test_invalid_displacements_fail(delta):
    with pytest.raises(ValueError):
        displacement_metrics([0, 0, 0], [0, 0, 0], delta)
