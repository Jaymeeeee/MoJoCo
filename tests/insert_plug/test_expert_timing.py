"""Control-rate-independent expert timing and executed action precision."""
from types import SimpleNamespace

import numpy as np
import pytest

from examples.insert_plug.control.expert import Expert
from examples.insert_plug.control.execution_guard import ExecutionSafetyError


def _result_expert(final_checks):
    expert = Expert.__new__(Expert)
    expert.sim = SimpleNamespace(metrics=lambda: {'success': True}, dt=.1)
    expert.final_checks = final_checks
    expert.steps = 0
    expert.phase_results = []
    expert.plan_files = {}
    expert.plans = []
    expert.guard = SimpleNamespace(summary=lambda: {})
    expert.max_fit_error = 0.
    expert.max_insertion_error = 0.
    expert.alignment_corrections = 0
    return expert


@pytest.mark.parametrize('timestamps,expected', [
    ([.1, .2], False),
    ([.1, .2, .3], False),
    ([.1, .2, .3, .4], True),
    (list(np.arange(.02, .28, .02)), False),
    (list(np.arange(.02, .30, .02)), True),
])
def test_final_stability_uses_elapsed_time_between_observations(timestamps, expected):
    expert = _result_expert([(time, True) for time in timestamps])
    result = expert.result('')
    assert result['success'] is expected
    assert result['metrics']['stable_for_final_250ms'] is expected
    assert result['metrics']['final_success_observation_span_seconds'] == pytest.approx(
        timestamps[-1]-timestamps[0])


def test_failed_final_observation_restarts_stability_duration():
    expert = _result_expert([(.1, True), (.2, False), (.3, True), (.4, True), (.5, True)])
    assert not expert.result('')['success']
    expert.final_checks.append((.6, True))
    assert expert.result('')['success']


def test_hold_rounds_up_to_at_least_the_requested_duration():
    expert = Expert.__new__(Expert)
    expert.sim = SimpleNamespace(last_target=np.zeros(7), dt=.1)
    calls = []
    expert.tick = lambda target, width, phase: calls.append((target, width, phase))
    expert.hold(.25, .02, 'grasp')
    assert len(calls) == 3


def test_tick_executes_exact_float32_dataset_values():
    expert = Expert.__new__(Expert)
    actions = []
    expert.sim = SimpleNamespace(step=lambda action: actions.append(action.copy()))
    expert.guard = SimpleNamespace(phase='')
    expert.writer = None
    expert.steps = 0
    expert.realtime = False
    target = np.arange(7)*.012345678901234
    expert.tick(target, .075, 'settle_initial')
    expected = np.r_[target, .075].astype(np.float32).astype(np.float64)
    np.testing.assert_array_equal(actions[0], expected)
    np.testing.assert_array_equal(actions[0].astype(np.float32).astype(np.float64), actions[0])


def _adaptive_grasp_expert(history, updates, timeout=.8):
    expert = Expert.__new__(Expert)
    expert.sim = SimpleNamespace(last_target=np.zeros(7), dt=.1,
                                 data=SimpleNamespace(time=0.))
    expert.closed = .02
    expert.options = dict(adaptive_grasp_settle=True, grasp_settle_seconds=timeout)
    expert.grasp_history = list(history)
    pending = iter(updates)

    def tick(target, width, phase):
        assert phase == 'grasp'
        assert width == expert.closed
        expert.sim.data.time += expert.sim.dt
        expert.grasp_history.append(next(pending))

    expert.tick = tick
    return expert


def _grasp_sample(offset=0., contacts=('left', 'right')):
    return np.array([offset, 0., 0.]), set(contacts)


def test_adaptive_grasp_requires_elapsed_200ms_of_opposed_contact():
    expert = _adaptive_grasp_expert([_grasp_sample()], [_grasp_sample(), _grasp_sample()])
    result = expert._settle_grasp()
    # Three end-of-control observations span two 100 ms intervals.
    assert result['grasp_wait_seconds'] == pytest.approx(.2)
    assert result['both_jaws_stable_seconds'] >= .2


def test_adaptive_grasp_counts_stable_closing_history_without_extra_hold():
    expert = _adaptive_grasp_expert([_grasp_sample() for _ in range(3)], [])
    assert expert._settle_grasp()['grasp_wait_seconds'] == 0.
    assert expert.sim.data.time == 0.


def test_adaptive_grasp_lost_contact_restarts_the_window():
    expert = _adaptive_grasp_expert(
        [_grasp_sample(), _grasp_sample(contacts=('left',))],
        [_grasp_sample() for _ in range(3)])
    assert expert._settle_grasp()['grasp_wait_seconds'] == pytest.approx(.3)


def test_adaptive_grasp_does_not_accept_sliding_despite_two_contacts():
    expert = _adaptive_grasp_expert(
        [_grasp_sample(i*.001) for i in range(3)],
        [_grasp_sample(i*.001) for i in range(3, 5)], timeout=.2)
    with pytest.raises(ExecutionSafetyError, match='still slipping'):
        expert._settle_grasp()
    assert expert.sim.data.time == pytest.approx(.2)


def test_adaptive_grasp_timeout_does_not_continue_without_opposed_contact():
    expert = _adaptive_grasp_expert([], [_grasp_sample(contacts=('left',))]*3, timeout=.3)
    with pytest.raises(ExecutionSafetyError, match='both jaws'):
        expert._settle_grasp()
    assert expert.sim.data.time == pytest.approx(.3)


def _tracking_expert(statuses, stable_seconds=.2, timeout=.5):
    expert = Expert.__new__(Expert)
    expert.sim = SimpleNamespace(dt=.1, data=SimpleNamespace(time=1.))
    expert.options = dict(tracking_stable_seconds=stable_seconds, tracking_timeout=timeout)
    remaining = iter(statuses)
    expert._tracking_status = lambda stage: (next(remaining), dict(
        position_error_m=0., rotation_error_rad=0., max_joint_speed_rad_s=0.))

    def tick(target, width, phase):
        expert.sim.data.time += expert.sim.dt

    expert.tick = tick
    return expert


def test_tracking_counts_existing_stability_without_mandatory_extra_ticks():
    expert = _tracking_expert([True])
    result = expert._settle_tracking({'phase': 'align', 'width': .02}, np.zeros(7), stable_since=.8)
    assert result['tracking_wait_seconds'] == 0.
    assert result['tracking_stable_seconds'] == pytest.approx(.2)


def test_stage_executor_passes_stable_tail_to_the_tracking_gate():
    expert = _tracking_expert([True]*4)
    expert.sim.data.time = 0.
    expert.sim.metrics = lambda: {}
    expert.phase_results = []
    expert.verbose = False
    stage = dict(phase='approach', width=.075, start_index=0, end_index=3)
    plan = SimpleNamespace(arm_qpos=np.zeros((3, 7)), gripper_width=np.full(3, .075))
    expert._execute_stage(plan, stage)
    assert expert.sim.data.time == pytest.approx(.3)
    assert expert.phase_results[0]['checks']['tracking_wait_seconds'] == 0.


def test_tracking_failed_status_discards_existing_stability_window():
    expert = _tracking_expert([False, True, True, True])
    result = expert._settle_tracking({'phase': 'align', 'width': .02}, np.zeros(7), stable_since=.5)
    assert result['tracking_wait_seconds'] == pytest.approx(.3)
    assert result['tracking_stable_seconds'] == pytest.approx(.2)


def test_tracking_timeout_cannot_bypass_actual_pose_gate():
    expert = _tracking_expert([False]*4, timeout=.3)
    with pytest.raises(ExecutionSafetyError, match='tracking did not settle'):
        expert._settle_tracking({'phase': 'align', 'width': .02}, np.zeros(7), stable_since=.5)
    assert expert.sim.data.time == pytest.approx(1.3)


@pytest.mark.parametrize('phase', ['descend_to_grasp', 'transfer', 'align', 'shallow_insert'])
def test_continuous_free_space_preserves_contact_and_measurement_tracking_gates(phase):
    expert = _tracking_expert([False]*7)
    expert.options['continuous_free_space'] = True
    expert._check_fit = lambda: {}
    stage = dict(phase=phase, width=.02, start_index=0, end_index=1)
    plan = SimpleNamespace(arm_qpos=np.zeros((1, 7)), gripper_width=np.array([.02]))
    with pytest.raises(ExecutionSafetyError, match='tracking did not settle'):
        expert._execute_stage(plan, stage)


def test_continuous_extraction_still_requires_actual_tip_clearance():
    expert = _tracking_expert([])
    expert.options['continuous_free_space'] = True
    expert.sim.site = lambda name: np.array([0., 0., .045 if name == 'peg_tip' else .04])
    stage = dict(phase='extract', width=.02, start_index=0, end_index=1)
    plan = SimpleNamespace(arm_qpos=np.zeros((1, 7)), gripper_width=np.array([.02]))
    with pytest.raises(ExecutionSafetyError, match='transport clearance'):
        expert._execute_stage(plan, stage)


@pytest.mark.parametrize('phase', ['approach', 'transfer', 'align'])
def test_continuous_hover_executes_next_segment_without_settling(phase):
    # Empty tracking statuses would fail immediately if a hover wait were used.
    expert = _tracking_expert([])
    expert.options['continuous_hover'] = True
    expert.sim.metrics = lambda: {}
    expert.phase_results = []
    expert.verbose = False
    expert._check_fit = lambda: {'fit_checked': True}
    stage = dict(phase=phase, width=.02, start_index=0, end_index=1)
    plan = SimpleNamespace(arm_qpos=np.zeros((1, 7)), gripper_width=np.array([.02]))
    expert._execute_stage(plan, stage)
    assert expert.sim.data.time == pytest.approx(1.1)
    assert 'tracking_wait_seconds' not in expert.phase_results[0]['checks']
    if phase == 'align':
        assert expert.phase_results[0]['checks']['fit_checked']
