"""Test the attribution mistakes the read-only prior diagnosis is meant to catch."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.diagnose_prior import (
    history_bin, main, prequential,
)
from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.calibration import ControlledCalibration


def test_target_units_cancel_but_input_units_change_the_penalty():
    # Fixed illustrative design: this is an algebra test, not a model candidate.
    x = np.array([[1., -.2], [1., .1], [1., .4]])
    target = np.array([0., -6., 3.])-7.5
    point = np.array([1., -.1])
    predict = lambda matrix, query, scale, penalty: 7.5+query@np.linalg.solve(
        matrix.T@matrix+penalty, matrix.T@(target/scale))*scale
    assert predict(x, point, 1., np.eye(2)) == pytest.approx(predict(x, point, 60., np.eye(2)), abs=1.e-12)
    scaled, scaled_point = x.copy(), point.copy()
    scaled[:, 1] /= 60.
    scaled_point[1] /= 60.
    scaled_fit = predict(scaled, scaled_point, 60., np.eye(2))
    raw_fit = predict(x, point, 60., np.diag([1., 3600.]))
    assert scaled_fit == pytest.approx(raw_fit, abs=1.e-12)
    assert abs(scaled_fit-predict(x, point, 60., np.eye(2))) > .01


def test_means_use_admitted_ids_instead_of_all_earlier_issues():
    def row(key, at, truth, admitted):
        return dict(id=key, issue_at=at, line_id='7X', admitted=True, admitted_ids=admitted,
            truth=[truth]*6, prediction=dict(shared=[1.]*6, history=[1.]*6))
    rows = [row('a', 1, 0., []), row('not_yet_admitted', 2, -100., []), row('c', 3, -6., ['a'])]
    scored = prequential(rows)
    assert scored[-1]['predictions']['past_line_mean'] == [0.]*6
    assert scored[-1]['first_negative_endpoints'] == list(range(6))
    damaged = deepcopy(rows)
    damaged[0]['admitted_ids'] = ['c']
    with pytest.raises(ValueError, match='current/future'):
        prequential(damaged)


def test_selected_issue_action_support_excludes_actual_floor():
    transform = ExecutionAction(dict(max_hold_seconds=60., scheduled_headway_scale_seconds=600.))
    calibration = ControlledCalibration.__new__(ControlledCalibration)
    calibration.transform = transform
    value = np.zeros(229, np.float32)
    event = value[5:].reshape(16, 14)[0]
    event[[3, 4, 5, 6, 8, 9, 11, 12]] = [1., 1., 1., 1., .1, .01, 1., 0.]
    for seconds, floor_active in ((1559, False), (3600, True), (3815, True), (18000, False)):
        value[0] = seconds/24000
        assert (calibration._action_for_hold(value, 0.) is None) == floor_active
    # The same clock window can still support both actions when load disables F's floor.
    value[0], event[8] = 3815/24000, .5
    assert all(calibration._action_for_hold(value, hold) is not None for hold in (0., 7.5))


def test_existing_summary_reproduction_and_reversal_unit():
    result = main()
    metrics = result['common_support_service_metrics']
    assert metrics['fixed_7p5']['mae_seconds'] == pytest.approx(2.1944444444444446)
    assert metrics['past_line_mean']['mae_seconds'] == pytest.approx(2.2083333333333335)
    assert all(value['states'] == 6 and value['endpoints'] == 36 for value in metrics.values())
    assert result['reversals']['first_negative_states'] == ['train/7X/response']
    assert result['reversals']['repeated_negative_after_own_line_negative_states'] == []
    support = result['sampling_support']
    assert [support['by_line'][line]['model_mean_joint_groups'] for line in ('7X', '102X', '705X')] == [2, 1, 1]
    assert support['bins_agree'] == 6
    assert history_bin(-7.5) == history_bin(7.5) == 'within_hold'
