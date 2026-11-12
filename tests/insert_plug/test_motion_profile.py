"""Motion-profile smoothness and retiming bounds, independent of physics tuning."""
import numpy as np
import pytest

from examples.insert_plug.control.planning import PlanningError, _motion_profile, _profile_bounds, _quintic_positions


@pytest.mark.parametrize('profile,ramp', [
    ('quintic', .2), ('cosine_velocity', .05), ('cosine_velocity', .2),
    ('cosine_velocity', .5),
])
def test_profiles_move_monotonically_and_obey_retiming_bounds(profile, ramp):
    t = np.linspace(0., 1., 10001)
    position = _motion_profile(t, profile, ramp)
    velocity = np.gradient(position, t, edge_order=2)
    acceleration = np.gradient(velocity, t, edge_order=2)
    speed_bound, acceleration_bound = _profile_bounds(profile, ramp)
    assert position[0] == pytest.approx(0.)
    assert position[-1] == pytest.approx(1.)
    assert np.all(np.diff(position) >= -1e-12)
    assert np.max(np.abs(velocity)) <= speed_bound*(1+1e-6)
    assert np.max(np.abs(acceleration)) <= acceleration_bound*(1+1e-4)
    np.testing.assert_allclose(position, 1-position[::-1], atol=2e-14)


@pytest.mark.parametrize('profile,ramp', [
    ('quintic', .2), ('cosine_velocity', .05), ('cosine_velocity', .2),
    ('cosine_velocity', .5),
])
def test_profiles_start_and_stop_with_zero_speed_and_acceleration(profile, ramp):
    h = 1e-5
    left = _motion_profile(np.array([0., h, 2*h]), profile, ramp)
    right = _motion_profile(np.array([1-2*h, 1-h, 1.]), profile, ramp)
    speed_bound, acceleration_bound = _profile_bounds(profile, ramp)
    assert abs((left[1]-left[0])/h) < speed_bound*1e-6
    assert abs((right[-1]-right[-2])/h) < speed_bound*1e-6
    assert abs(np.diff(left, n=2)[0]/h**2) < acceleration_bound*1e-3
    assert abs(np.diff(right, n=2)[0]/h**2) < acceleration_bound*1e-3


@pytest.mark.parametrize('ramp', [.05, .2, .5])
def test_cosine_ramp_joins_have_continuous_velocity_and_acceleration(ramp):
    h = 1e-4
    speed_bound, acceleration_bound = _profile_bounds('cosine_velocity', ramp)
    for knot in (ramp, 1-ramp):
        values = _motion_profile(knot+np.arange(-2, 3)*h, 'cosine_velocity', ramp)
        left_speed = (values[2]-values[1])/h
        right_speed = (values[3]-values[2])/h
        assert left_speed == pytest.approx(right_speed, abs=speed_bound*1e-4)
        left_acceleration = (values[2]-2*values[1]+values[0])/h**2
        right_acceleration = (values[4]-2*values[3]+values[2])/h**2
        assert abs(left_acceleration) < acceleration_bound*.01
        assert abs(right_acceleration) < acceleration_bound*.01


@pytest.mark.parametrize('ramp', [0., -.1, .50001, np.nan, np.inf])
def test_invalid_ramp_is_rejected_before_retiming(ramp):
    with pytest.raises(PlanningError, match='ramp_fraction'):
        _motion_profile(np.array([0., .5, 1.]), 'cosine_velocity', ramp)
    with pytest.raises(PlanningError, match='ramp_fraction'):
        _profile_bounds('cosine_velocity', ramp)


def test_unknown_profile_cannot_silently_fall_back_to_another_curve():
    with pytest.raises(PlanningError, match='Unknown motion profile'):
        _profile_bounds('misspelled_profile', .2)


def test_quintic_position_segment_preserves_nonzero_endpoint_velocities():
    start = np.array([.45, -.07, .10])
    end = np.array([.45, -.07, .06])
    start_velocity = np.array([0., 0., -.006])
    end_velocity = np.array([0., 0., -.01])
    duration, step = 2.3, 1e-4
    t = np.array([0., step, 2*step, 3*step,
                  duration-3*step, duration-2*step, duration-step, duration])
    p = _quintic_positions(t/duration, start, end, start_velocity, end_velocity, duration)
    np.testing.assert_allclose(p[0], start, atol=1e-14)
    np.testing.assert_allclose(p[-1], end, atol=1e-14)
    np.testing.assert_allclose((p[1]-p[0])/step, start_velocity, atol=1e-8)
    np.testing.assert_allclose((p[-1]-p[-2])/step, end_velocity, atol=1e-8)
    # One-sided endpoint second derivatives; centred differences would measure
    # acceleration one sample inside the segment, where its jerk is nonzero.
    initial_acceleration = (2*p[0]-5*p[1]+4*p[2]-p[3])/step**2
    terminal_acceleration = (2*p[-1]-5*p[-2]+4*p[-3]-p[-4])/step**2
    np.testing.assert_allclose(initial_acceleration, 0., atol=2e-7)
    np.testing.assert_allclose(terminal_acceleration, 0., atol=2e-7)


def test_two_quintic_segments_cross_waypoint_without_stopping():
    start, via, end = np.array([[.40, 0., .12], [.45, 0., .10], [.45, 0., .06]])
    shared_velocity = np.array([0., 0., -.006])
    duration_a, duration_b, step = 2., 3., 1e-4
    before = _quintic_positions(
        (duration_a+np.array([-2., -1., 0.])*step)/duration_a,
        start, via, np.zeros(3), shared_velocity, duration_a)
    after = _quintic_positions(np.array([0., 1., 2.])*step/duration_b,
                               via, end, shared_velocity, np.zeros(3), duration_b)
    np.testing.assert_allclose(before[-1], after[0], atol=1e-14)
    left_velocity, right_velocity = (before[-1]-before[-2])/step, (after[1]-after[0])/step
    np.testing.assert_allclose(left_velocity, shared_velocity, atol=1e-8)
    np.testing.assert_allclose(right_velocity, shared_velocity, atol=1e-8)
    assert np.linalg.norm(left_velocity) > .0059
    np.testing.assert_allclose(np.diff(before, n=2, axis=0)[0]/step**2, 0., atol=5e-5)
    np.testing.assert_allclose(np.diff(after, n=2, axis=0)[0]/step**2, 0., atol=5e-5)
