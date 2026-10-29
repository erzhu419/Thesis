import copy
import json

import numpy as np
import pytest

from experiments.mcwm_closed_loop_cost_advantage_v85_20261027 import model


def state(z, index=2):
    value = np.zeros(229, dtype=np.float64)
    value[0], value[2] = .5, index/15
    event = value[5:].reshape(16, 14)[index]
    event[3], event[4], event[11], event[12] = 1-2*z, 1., 1., 1.
    return value.tolist()


def records():
    result = []
    for scene_index, scene in enumerate(model.SCENARIOS):
        for index, line in enumerate(model.LINES):
            z = (-.2 if scene_index == 0 else .2)+.025*index
            h_ref, h_candidate = 3.1+.1*index, 7.5
            beta = [1.5, 2., -2.][index]
            target = (h_candidate-h_ref)/7.5*(beta+2.5*z)
            result.append(dict(scenario=scene, line_id=line, state=state(z),
                ref_hold=h_ref, cand_hold=h_candidate,
                target_delta_person_seconds=target*15079,
                label_available_at=24030, source_scope='v84_train'))
    return result


def test_feature_uses_declared_slot_presence_and_clipping():
    value = np.asarray(state(.2))
    assert model.state_feature(value) == pytest.approx(.2)
    event = value[5:].reshape(16, 14)[2]
    event[3], event[4], event[11], event[12] = 2., 5., 1., 0.
    assert model.state_feature(value) == -.5
    event[12] = 1.
    assert model.state_feature(value) == 1.


def test_two_float64_solves_match_fixed_sum_loss_ridge_formula():
    rows = records()
    artifact = model.train(rows)
    assert artifact['gate']['all_pass']
    assert artifact['fits'] == 2
    x = np.asarray([model.design_row(row['line_id'], row['state'], row['ref_hold'], row['cand_hold'])
                    for row in rows], dtype=np.float64)
    y = np.asarray([row['target_delta_person_seconds']/15079 for row in rows])
    for name, columns in (('line', 3), ('state', 4)):
        design = x[:, :columns]
        expected = np.linalg.solve(design.T@design+np.eye(columns), design.T@y)
        np.testing.assert_allclose(artifact['models'][name]['coef'], expected, rtol=0, atol=0)
    assert artifact['models']['line']['design_rank'] == 3
    assert artifact['models']['state']['design_rank'] == 4
    json.dumps(artifact, allow_nan=False)


def test_reference_anchor_and_zero_prediction_tie_keep_reference():
    artifact = model.train(records())
    anchored = model.design_row('7X', state(.1), 3.3, 3.3)
    assert np.array_equal(anchored, np.zeros(4))
    assert anchored@np.asarray(artifact['models']['state']['coef']) == 0.
    for fitted in artifact['models'].values():
        fitted['coef'] = [0.]*len(fitted['coef'])
    prediction = model.predict(artifact, '7X', state(.1), 3.3, 7.5)
    assert prediction['supported']
    assert all(row == dict(predicted_delta=0., choice='reference') for row in prediction['models'].values())


@pytest.mark.parametrize('h_ref,h_candidate', [(3.3, 0.), (3.3, 3.3), (2.9, 7.5), (3.7, 7.5), (3.3, 60.)])
def test_support_failure_does_not_invent_a_zero_prediction(h_ref, h_candidate):
    prediction = model.predict(model.train(records()), '7X', state(.1), h_ref, h_candidate)
    assert prediction['supported'] is False
    assert prediction['reason'] is not None
    assert all(row == dict(predicted_delta=None, choice='reference') for row in prediction['models'].values())


def test_new_labels_wrong_maturity_and_duplicate_cases_cannot_fit():
    for key, value in (('source_scope', 'v85_holdout'), ('label_available_at', 24000)):
        data = records()
        data[0][key] = value
        with pytest.raises(ValueError):
            model.train(data)
    data = records()
    data[-1] = copy.deepcopy(data[0])
    with pytest.raises(ValueError):
        model.train(data)


def test_state_rank_deficiency_holds_all_prediction_without_feature_switch():
    data = records()
    for row in data:
        row['state'] = state(0.)
    artifact = model.train(data)
    assert artifact['fits'] == 2
    assert artifact['models']['line']['design_rank'] == 3
    assert artifact['models']['state']['design_rank'] == 3
    assert not artifact['gate']['all_pass']
    prediction = model.predict(artifact, '7X', state(0.), 3.3, 7.5)
    assert prediction['reason'] == 'training_gate_hold'
    assert prediction['models']['state']['predicted_delta'] is None


def test_nonfinite_input_holds_without_serializing_nan_coefficients():
    data = records()
    data[0]['state'][0] = float('nan')
    artifact = model.train(data)
    assert not artifact['gate']['all_pass']
    assert artifact['fits'] == 0
    assert artifact['models']['state']['coef'] is None
    json.dumps(artifact, allow_nan=False)


def test_predict_neither_mutates_fit_nor_consumes_randomness():
    artifact = model.train(records())
    before = copy.deepcopy(artifact)
    random_before = copy.deepcopy(np.random.get_state())
    prediction = model.predict(artifact, '705X', state(.1), 3.3, 7.5)
    random_after = np.random.get_state()
    assert artifact == before
    assert random_before[0] == random_after[0]
    np.testing.assert_array_equal(random_before[1], random_after[1])
    assert random_before[2:] == random_after[2:]
    assert prediction['supported']
    assert prediction['models']['line']['choice'] == 'candidate'
