"""Checks for causal training-only forecasts and endpoint-local histories."""
from copy import deepcopy

import numpy as np

from experiments.mcwm_closed_loop_local_history_v22_20261023.diagnostic import (
    endpoint_design,
    forecast,
)


def history(issued_at, residuals=(60., 120., -300.)):
    return dict(status='supported', mean_residual_scaled=float(np.mean(residuals))/60.,
        segments=[dict(available=True, latest=dict(residual=value,
            completed_at=issued_at-10., observed_at=issued_at-9., age=10.))
            for value in residuals])


def training_record(identifier, issued_at=20., available_at=50.):
    return dict(id=identifier, line_id='7X', role='train', selected_for_training=True,
        available_at=available_at, admitted_at=available_at, issued_at=issued_at,
        features=[1., 0., 0., .25], service_history=history(issued_at),
        service_targets=dict(arrival=[-1., -2., -3.], departure=[0., 1., 2.]))


def issue(line='7X', count=1):
    line_index = ('7X', '102X', '705X').index(line)
    features = [float(i == line_index) for i in range(3)]+[.5]
    return dict(id='train/'+line+'/post1', line_id=line, features=features,
        service_history=history(100.), selected=dict(started_at=100.),
        prefix=dict(counters=dict(service_shared_pairs=count)))


def test_unadmitted_and_future_training_truth_cannot_change_issue_forecast():
    records = [training_record('admitted'),
        training_record('available_but_not_admitted', issued_at=30., available_at=80.),
        training_record('future', issued_at=200., available_at=530.)]
    changed = deepcopy(records)
    for row in changed[1:]:
        row['service_targets'] = dict(arrival=[1e6]*3, departure=[-1e6]*3)
    for mode in ('shared', 'mean', 'local'):
        original = forecast(records, issue(), mode)
        modified = forecast(changed, issue(), mode)
        assert original['admitted_ids'] == modified['admitted_ids'] == ['admitted']
        assert original['support'] == modified['support'] == [1]*6
        np.testing.assert_allclose(original['prediction'], modified['prediction'])


def test_own_line_cold_start_and_endpoint_censoring_remain_unsupported():
    record = training_record('admitted')
    record['service_targets']['arrival'][1] = None
    record['service_targets']['departure'][1] = None
    for mode in ('shared', 'mean', 'local'):
        cold = forecast([record], issue(line='102X'), mode)
        assert cold['support'] == [0]*6
        assert cold['prediction'] == [None]*6
        supported = forecast([record], issue(), mode)
        assert supported['support'] == [1, 0, 1, 1, 0, 1]
        assert supported['prediction'][1] is None
        assert supported['prediction'][4] is None
        assert all(supported['prediction'][i] is not None for i in (0, 2, 3, 5))


def test_local_endpoint_input_excludes_other_segments():
    features = [1., 0., 0., .5]
    before = history(100.)
    after = history(100., residuals=(60., 1200., 3000.))
    first = endpoint_design(features, before, 0, 'local')
    np.testing.assert_array_equal(first, features+[1.])
    np.testing.assert_array_equal(first, endpoint_design(features, after, 0, 'local'))
    np.testing.assert_array_equal(endpoint_design(features, before, 2, 'local'),
        features+[-5.])
    assert endpoint_design(features, before, 0, 'mean')[-1] != \
        endpoint_design(features, after, 0, 'mean')[-1]
