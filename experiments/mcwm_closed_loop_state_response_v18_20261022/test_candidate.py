"""The single state addition preserves the control and budgeted-label contract."""
import copy

import numpy as np
import pytest

from experiments.mcwm_closed_loop_v1.stream import EI
from experiments.mcwm_closed_loop_state_response_v18_20261022.candidate import (
    RIDE_FEATURE_NAMES, StateResponseReadout, issue_state_features,
)
from experiments.mcwm_closed_loop_state_response_v18_20261022.difference import (
    DirectDifferenceReadout, FEATURE_NAMES, TARGET_SCALE, issue_features,
)


def records():
    rows = []
    for i, line in enumerate(('7X', '102X', '705X')):
        for j, (headway, load) in enumerate(((-.4, .1), (.2, .8), (.6, .3))):
            wait, ride = 20.*i+50.*headway, -10.*i+200.*load-25.*headway
            rows.append(dict(id=f'{line}/{j}', role='train', selected_for_training=True,
                available_at=430., admitted_at=600., line_id=line,
                features=[float(k == i) for k in range(3)]+[headway, load],
                outcomes={
                    'low': dict(truth=dict(status='complete', dt=300., wait=1000.+wait, ride=1000.+ride)),
                    'high': dict(truth=dict(status='complete', dt=300., wait=1000., ride=1000.))}))
    return rows


def test_issue_load_uses_current_event_and_preserves_original_four_exactly():
    event = np.zeros(max(EI.values())+1, np.float32)
    for key, value in dict(forward_headway_seconds=660., target_forward_headway_seconds=600.,
            backward_headway_seconds=540., target_backward_headway_seconds=600.,
            onboard_passengers=13., bus_capacity=50.).items():
        event[EI[key]] = value
    issued = dict(line_id='7X', event=event, started_at=18000., scenario='unused',
        future_outcomes='unused')
    features = issue_state_features(issued)
    assert features[:4] == issue_features(issued)
    assert features[4] == 13./50.
    event[EI['onboard_passengers']] = 1.
    assert issue_state_features(issued) == features[:4]+[1./50.]


def test_wait_is_exact_shared_control_and_only_ride_uses_the_added_column():
    shared, model = DirectDifferenceReadout(), StateResponseReadout()
    for fit in (shared, model):
        fit.accept(records(), 600.)
        fit.refit()
    for load in (.1, .9):
        features = [1., 0., 0., .25, load]
        actual = model.predict(features)
        assert actual['gain']['wait'] == shared.predict(features)['gain']['wait']
        assert actual['gain']['ride'] == float(np.asarray(features)@model.ride_weights*TARGET_SCALE)
        assert actual['gain']['cost'] == 2.*actual['gain']['wait']+actual['gain']['ride']
        assert actual['gain']['dt'] == 0.
        assert actual['fixed300_clock'] == dict(low=300., high=300.)
        assert actual['support_pairs'] == 3
    low = model.predict([1., 0., 0., .25, .1])['gain']
    high = model.predict([1., 0., 0., .25, .9])['gain']
    assert low['wait'] == high['wait'] and low['ride'] != high['ride']


def test_summary_reports_actual_endpoint_designs_coefficients_and_errors():
    model = StateResponseReadout()
    model.accept(records(), 600.)
    model.refit()
    summary = model.summary()
    assert summary['feature_names'] == dict(wait=list(FEATURE_NAMES), ride=list(RIDE_FEATURE_NAMES))
    assert summary['design']['wait']['columns'] == 4 and summary['design']['ride']['columns'] == 5
    assert summary['pairs'] == 9 and summary['outcome_labels'] == 18
    assert summary['pair_ids'] == [row['id'] for row in records()]
    errors = []
    for row in records():
        gain = model.predict(row['features'])['gain']
        for endpoint, width in (('wait', 4), ('ride', 5)):
            values = list(summary['coefficients'][endpoint].values())
            assert np.isclose(np.asarray(row['features'][:width])@values, gain[endpoint])
        errors.append([gain[key]-(row['outcomes']['low']['truth'][key]
            -row['outcomes']['high']['truth'][key]) for key in ('wait', 'ride')])
    errors = np.asarray(errors)
    cost = 2.*errors[:, 0]+errors[:, 1]
    assert summary['training_error'] == dict(wait_mae=float(np.mean(abs(errors[:, 0]))),
        ride_mae=float(np.mean(abs(errors[:, 1]))), cost_mae=float(np.mean(abs(cost))),
        cost_rmse=float(np.sqrt(np.mean(cost**2))))


def test_candidate_keeps_admitted_pair_support_and_rejects_heldout_or_future_labels():
    model = StateResponseReadout()
    assert model.predict([1., 0., 0., 0., .2])['gain'] is None
    model.accept(records()[:3], 600.)
    model.refit()
    prediction = model.predict([0., 1., 0., 0., .5])
    assert prediction['status'] == 'unsupported' and prediction['support_pairs'] == 0
    assert prediction['gain'] is None
    for change in (dict(role='heldout'), dict(available_at=601.), dict(selected_for_training=False)):
        row = copy.deepcopy(records()[3])
        row.update(change)
        with pytest.raises(ValueError):
            model.accept([row], 600.)
