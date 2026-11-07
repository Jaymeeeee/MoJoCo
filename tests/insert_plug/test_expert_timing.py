"""Control-rate-independent expert timing and executed action precision."""
from types import SimpleNamespace

import numpy as np
import pytest

from examples.insert_plug.expert import Expert


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
