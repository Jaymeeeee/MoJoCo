"""Paired displacement metrics must retain direction, timing and absolute units."""
import json

import numpy as np
import pytest

from examples.insert_plug.policy.translation_metrics import first_closure_index, trajectory_metrics


def inputs(count=64):
    tcp = np.column_stack((np.linspace(.3, .45, count),
                           np.full(count, -.07), np.linspace(.2, .044, count)))
    actions = np.zeros((count, 8))
    actions[:, 7] = .075
    return dict(base_tcp=tcp, shifted_tcp=tcp + [.03, 0, 0], base_actions=actions,
                shifted_actions=actions.copy(), delta=np.array([.03, 0, 0]),
                initial_tcp=np.array([.3, -.07, .2]), base_grasp=tcp[-1],
                shifted_grasp=tcp[-1] + [.03, 0, 0])


def test_exact_positive_equivariance_has_unit_gain_and_no_vector_error():
    data = inputs()
    result = trajectory_metrics(**data)
    endpoint = result['endpoints']['last']
    assert endpoint['gain'] == pytest.approx(1)
    assert endpoint['along_m'] == pytest.approx(.03)
    assert endpoint['orthogonal_m'] < 1e-14
    assert endpoint['residual_m'] < 1e-14
    assert endpoint['shifted_grasp_xy_error_m'] == 0
    assert result['endpoints']['query31']['base_query'] == 31
    json.dumps(result, allow_nan=False)


def test_three_centimetres_in_wrong_direction_does_not_pass():
    data = inputs()
    data['shifted_tcp'] = data['base_tcp'] - [.03, 0, 0]
    endpoint = trajectory_metrics(**data)['endpoints']['last']
    assert endpoint['gain'] == pytest.approx(-1)
    assert endpoint['along_m'] == pytest.approx(-.03)
    assert endpoint['residual_m'] == pytest.approx(.06)


def test_negative_translation_has_correct_signed_gain_and_cross_axis_error():
    data = inputs()
    data['delta'] = np.array([0, -.03, 0])
    data['shifted_tcp'] = data['base_tcp'] + [.004, -.015, .003]
    endpoint = trajectory_metrics(**data)['endpoints']['last']
    assert endpoint['gain'] == pytest.approx(.5)
    assert endpoint['orthogonal_m'] == pytest.approx(.005)
    assert endpoint['residual_m'] == pytest.approx(np.sqrt(.015**2 + .005**2))


def test_closure_requires_two_adjacent_commands_and_no_end_transient():
    aperture = np.array([.075, .025, .075, .026, .026, .075])
    assert first_closure_index(aperture) == 3
    assert first_closure_index([.075, .025, .075, .025]) is None
    assert first_closure_index([]) is None


def test_no_closure_is_missing_data_not_last_query_substitution():
    result = trajectory_metrics(**inputs(count=5))
    assert not result['closure']['baseline_found']
    assert not result['closure']['shifted_found']
    assert result['endpoints']['baseline_first_closure'] is None
    assert result['endpoints']['independent_first_closures'] is None
    assert result['endpoints']['query31'] is None
    assert result['endpoints']['last']['base_query'] == 4


def test_separate_closure_times_are_labeled_without_changing_same_query_comparison():
    data = inputs()
    data['base_actions'][20:, 7] = .02
    data['shifted_actions'][25:, 7] = .02
    result = trajectory_metrics(**data)
    matched = result['endpoints']['baseline_first_closure']
    independent = result['endpoints']['independent_first_closures']
    assert (matched['base_query'], matched['shifted_query']) == (20, 20)
    assert matched['same_query'] and matched['gain'] == pytest.approx(1)
    assert (independent['base_query'], independent['shifted_query']) == (20, 25)
    assert not independent['same_query']
    assert independent['gain'] > 1
    assert independent['base_command_time_s'] == 2
    assert independent['shifted_command_time_s'] == 2.5


def test_baseline_closure_query_remains_available_if_shifted_never_closes():
    data = inputs()
    data['base_actions'][20:, 7] = .02
    result = trajectory_metrics(**data)
    assert result['endpoints']['baseline_first_closure']['shifted_query'] == 20
    assert result['endpoints']['independent_first_closures'] is None


def test_absolute_joint_commands_are_not_summed_for_cartesian_displacement():
    data = inputs()
    data['base_actions'][:, :7] = 2.
    data['shifted_actions'][:, :7] = -3.
    result = trajectory_metrics(**data)
    for condition, key in [('baseline', 'base_tcp'), ('shifted', 'shifted_tcp')]:
        summed = result['telescoping'][condition]
        expected = data[key][-1] - data['initial_tcp']
        np.testing.assert_allclose(summed['sum_successive_tcp_displacements_m'], expected, atol=1e-14)
        np.testing.assert_allclose(summed['terminal_minus_initial_m'], expected, atol=1e-14)
        assert summed['verified'] and summed['identity_error_m'] < 1e-14
    assert result['endpoints']['last']['gain'] == pytest.approx(1)


@pytest.mark.parametrize('key,value', [
    ('delta', [0, 0, 0]), ('control_hz', 0), ('closure_width_m', -1),
    ('base_tcp', [[1, 2, np.nan]]), ('shifted_tcp', np.zeros((63, 3))),
    ('base_actions', np.zeros((64, 7))), ('initial_tcp', [0, 0]),
])
def test_invalid_inputs_fail_explicitly(key, value):
    data = inputs()
    data[key] = value
    with pytest.raises(ValueError):
        trajectory_metrics(**data)
