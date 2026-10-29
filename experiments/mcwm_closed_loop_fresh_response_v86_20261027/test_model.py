import copy
import json
from unittest.mock import patch

import numpy as np
import pytest

from experiments.mcwm_closed_loop_cost_advantage_v85_20261027 import model as old_model
from experiments.mcwm_closed_loop_fresh_response_v86_20261027 import model


def state(z, index=2):
    value = np.zeros(229, dtype=np.float64)
    value[0], value[2] = .5, index/15
    event = value[5:].reshape(16, 14)[index]
    event[3], event[4], event[11], event[12] = 1-2*z, 1., 1., 1.
    return value.tolist()


def records(arm):
    result = []
    scope, seed = model.COHORTS[arm]
    for scene_index, scene in enumerate(model.SCENARIOS):
        for index, line in enumerate(model.LINES):
            z = (-.2 if scene_index == 0 else .2)+.025*index
            h_ref, h_candidate = 3.1+.1*index, 7.5
            beta = [1.5, 2., -2.][index]
            target = (h_candidate-h_ref)/7.5*(beta+2.5*z)
            if arm == 'fresh':
                target -= 1.
            result.append(dict(scenario=scene, line_id=line, state=state(z),
                ref_hold=h_ref, cand_hold=h_candidate, target_delta_person_seconds=target*15079,
                label_available_at=24030, source_scope=scope, external_seed=seed))
    return result


def cohorts():
    return {arm: records(arm) for arm in model.COHORTS}


def test_exactly_four_fixed_solves_and_stale_exactly_reproduces_v85():
    data = cohorts()
    original_solve = np.linalg.solve
    with patch.object(np.linalg, 'solve', wraps=original_solve) as solve:
        artifact = model.train(data)
    assert solve.call_count == artifact['fits'] == 4
    assert artifact['gate']['all_pass']
    old_rows = copy.deepcopy(data['stale'])
    for row in old_rows:
        row['source_scope'] = 'v84_train'
    old = old_model.train(old_rows)
    assert artifact['arms']['stale']['models'] == old['models']
    for arm in model.COHORTS:
        assert artifact['arms'][arm]['fits'] == 2
        assert artifact['arms'][arm]['models']['line']['design_rank'] == 3
        assert artifact['arms'][arm]['models']['state']['design_rank'] == 4
    json.dumps(artifact, allow_nan=False)


def test_fresh_targets_cannot_modify_stale_fit():
    data = cohorts()
    before = model.train(data)
    for row in data['fresh']:
        row['target_delta_person_seconds'] *= -2.
    after = model.train(data)
    assert after['arms']['stale'] == before['arms']['stale']
    assert after['arms']['fresh']['models']['state']['coef'] != before['arms']['fresh']['models']['state']['coef']


def test_presence_mask_and_reference_anchor_are_unchanged():
    value = np.asarray(state(.2))
    event = value[5:].reshape(16, 14)[2]
    event[3], event[4], event[11], event[12] = 2., 5., 1., 0.
    assert model.state_feature(value) == -.5
    event[12] = 1.
    assert model.state_feature(value) == 1.
    assert np.array_equal(model.design_row('7X', value, 3.3, 3.3), np.zeros(4))


@pytest.mark.parametrize('h_ref', [3., 3.6, 3.600001, 3.74, 3.749999])
def test_main_support_identical_and_legacy_changes_only_domain(h_ref):
    artifact = model.train(cohorts())
    prediction = model.predict(artifact, '7X', state(.1), h_ref, 7.5)
    assert prediction['supported']
    assert prediction['models']['stale']['supported']
    assert prediction['models']['fresh']['supported']
    legacy = prediction['models']['legacy_narrow']
    assert legacy['supported'] == (h_ref <= 3.6)
    if legacy['supported']:
        assert legacy == prediction['models']['stale']
    else:
        assert legacy['predicted_delta'] is None
        assert legacy['choice'] == 'reference'


@pytest.mark.parametrize('h_ref,h_candidate', [(3.3, 0.), (3.3, 3.3), (2.9, 7.5),
    (3.75, 7.5), (3.3, 60.), (3.3, 7.50002), (float('nan'), 7.5)])
def test_unsupported_actions_never_invent_zero_prediction(h_ref, h_candidate):
    prediction = model.predict(model.train(cohorts()), '7X', state(.1), h_ref, h_candidate)
    assert not prediction['supported']
    for value in prediction['models'].values():
        assert not value['supported']
        assert value['reason'] is not None
        assert value['predicted_delta'] is None
        assert value['choice'] == 'reference'


def test_candidate_tolerance_and_zero_tie():
    artifact = model.train(cohorts())
    for arm in artifact['arms'].values():
        arm['models']['state']['coef'] = [0.]*4
    prediction = model.predict(artifact, '7X', state(.1), 3.3, 7.500009)
    assert all(value['supported'] and value['predicted_delta'] == 0.
               and value['choice'] == 'reference' for value in prediction['models'].values())


@pytest.mark.parametrize('arm,key,value', [('stale', 'source_scope', 'v85_fresh'),
    ('fresh', 'external_seed', 316207), ('stale', 'external_seed', 216207),
    ('fresh', 'label_available_at', 24000)])
def test_wrong_cohort_heldout_or_unmatured_label_rejected_before_solves(arm, key, value):
    data = cohorts()
    data[arm][0][key] = value
    with patch.object(np.linalg, 'solve') as solve:
        with pytest.raises(ValueError):
            model.train(data)
        solve.assert_not_called()


def test_missing_or_duplicate_cases_rejected_before_solves():
    for arm in model.COHORTS:
        data = cohorts()
        data[arm][-1] = copy.deepcopy(data[arm][0])
        with pytest.raises(ValueError):
            model.train(data)
        data = cohorts()
        data[arm].pop()
        with pytest.raises(ValueError):
            model.train(data)


def test_bad_design_holds_both_arms_without_feature_switch():
    data = cohorts()
    for row in data['fresh']:
        row['state'] = state(0.)
    artifact = model.train(data)
    assert artifact['fits'] == 4
    assert artifact['arms']['fresh']['models']['state']['design_rank'] == 3
    assert not artifact['gate']['all_pass']
    prediction = model.predict(artifact, '7X', state(0.), 3.3, 7.5)
    assert all(value['reason'] == 'training_gate_hold' and value['predicted_delta'] is None
               for value in prediction['models'].values())


def test_nonfinite_training_skips_only_invalid_arm_without_nan_artifact():
    data = cohorts()
    data['fresh'][0]['state'][0] = float('nan')
    artifact = model.train(data)
    assert artifact['fits'] == 2
    assert artifact['arms']['fresh']['models']['state']['coef'] is None
    assert not artifact['gate']['all_pass']
    json.dumps(artifact, allow_nan=False)


def test_prediction_neither_mutates_parameters_nor_consumes_rng():
    artifact = model.train(cohorts())
    before = copy.deepcopy(artifact)
    random_before = copy.deepcopy(np.random.get_state())
    prediction = model.predict(artifact, '705X', state(.1), 3.3, 7.5)
    random_after = np.random.get_state()
    assert artifact == before
    assert random_before[0] == random_after[0]
    np.testing.assert_array_equal(random_before[1], random_after[1])
    assert random_before[2:] == random_after[2:]
    assert prediction['models']['stale']['choice'] == 'candidate'
