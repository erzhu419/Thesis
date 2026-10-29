import copy
from unittest.mock import patch

import numpy as np
import pytest

from experiments.mcwm_closed_loop_service_response_v88_20261028 import service


def context(load, margin):
    capacity = 100.
    seats = capacity-load*capacity
    queue = margin*capacity+seats
    if queue < 0:
        seats -= queue
        queue = 0.
    return dict(observed=True, reasons=[], capacity=capacity,
        clocks=dict(issue=12001., event=12000., obs_start=12001., obs_end=12001., atomic_at_issue=True),
        counts=dict(onboard=load*capacity, release_next=0., compatible_next_queue=queue,
                    seats_after_release=seats, next_queue=queue))


def covariates():
    rows = []
    for index, line in enumerate(service.LINES):
        for load_bin, load, margin in (('low', .2+.05*index, -.3+.1*index),
                                     ('high', .7+.03*index, .2-.04*index)):
            rows.append(dict(line_id=line, load_bin=load_bin, service=service.service_state(context(load, margin)),
                             ref_hold=3.05+.1*index+(load_bin == 'high')*.07, cand_hold=7.5))
    return rows


def training():
    rows = []
    for scene_index, scene in enumerate(service.SCENARIOS):
        for index, row in enumerate(covariates()):
            row = copy.deepcopy(row)
            if scene_index:
                load, margin = row['service']['features']
                row['service'] = service.service_state(context(load+.01, margin-.02))
            row.update(scenario=scene, source_scope='v88_train', external_seed=716207,
                physical_prefix='external716207/'+scene, sequence=index,
                observed_at=24000, label_available_at=24030, full_real_physical_pass=True,
                target_delta_person_seconds=(index-2.+scene_index*.5)*15079.)
            rows.append(row)
    return rows


def test_native_features_keep_event_and_issue_clocks_and_exact_load_boundary():
    state = service.service_state(context(.5, .3))
    assert state['features'] == pytest.approx([.5, .3])
    assert state['load_bin'] == 'high'
    assert state['issue_clock'] == 12001.
    assert state['event_clock'] == 12000.
    assert service.service_state(context(.4999, 2.))['load_bin'] == 'low'
    assert service.service_state(context(.2, 2.))['features'][1] == 1.
    assert service.service_state(context(.2, -2.))['features'][1] == -1.


@pytest.mark.parametrize('field', ['onboard','release_next','compatible_next_queue','seats_after_release'])
def test_missing_native_count_does_not_become_a_zero_feature(field):
    ctx = context(.7, .2)
    ctx['counts'][field] = None
    state = service.service_state(ctx)
    assert not state['observed']
    assert state['features'] is None and state['load_bin'] is None
    assert state['counts'][field] is None


def test_nonatomic_or_unobserved_context_cannot_enter_design():
    for kind in ('atomic', 'observed'):
        ctx = context(.2, -.3)
        if kind == 'atomic':
            ctx['clocks']['obs_end'] += 1
        else:
            ctx['observed'] = False
        row = covariates()[0]
        row['service'] = service.service_state(ctx)
        with pytest.raises(ValueError):
            service.make_design([row])


@pytest.mark.parametrize('h_ref,h_cand,step,action_step,expected', [
    (3.,7.5,1.,1.,True), (3.74999,7.5,1.,1.,True), (3.75,7.5,1.,1.,False),
    (3.3,0.,1.,1.,False), (3.3,7.50002,1.,1.,False), (3.3,7.5,1.,3.,False),
    (3.3,7.5,None,1.,False), (3.3,7.5,1.,0.,False)])
def test_existing_action_support_and_native_step_gap(h_ref,h_cand,step,action_step,expected):
    actual = service.action_support(h_ref,h_cand,step,action_step)
    assert actual['supported'] == expected
    assert (actual['reason'] is None) == expected


def test_covariate_design_rank_five_and_weighted_center_orthogonality_without_labels():
    class NeverRead:
        def __float__(self):
            raise AssertionError('Preflight must not consume a label')
    rows = covariates()
    for row in rows:
        row['target_delta_person_seconds'] = NeverRead()
    design = service.make_design(rows)
    assert design['ranks'] == dict(line=3, conditional=5)
    line = np.asarray(design['matrix']['line'])
    conditional = np.asarray(design['matrix']['conditional'])
    np.testing.assert_allclose(line.T@conditional[:, 3:], np.zeros((3,2)), rtol=0., atol=2e-16)
    for name in service.LINES:
        selected = [row for row in rows if row['line_id'] == name]
        a = np.asarray([(row['cand_hold']-row['ref_hold'])/7.5 for row in selected])
        x = np.asarray([row['service']['features'] for row in selected])
        expected = (a[:,None]**2*x).sum(axis=0)/(a**2).sum()
        np.testing.assert_allclose(design['line_means'][name],expected,rtol=0.,atol=0.)


def test_two_fixed_ridge_solves_preserve_line_coefficients():
    rows = training()
    original = np.linalg.solve
    with patch.object(np.linalg,'solve',wraps=original) as solve:
        artifact = service.train(rows)
    assert solve.call_count == artifact['fits'] == 2
    assert artifact['gate']['all_pass']
    np.testing.assert_allclose(artifact['models']['conditional']['coef'][:3],
                               artifact['models']['line']['coef'],rtol=0.,atol=1e-12)
    design = service.make_design(rows)
    target = np.asarray([row['target_delta_person_seconds']/15079 for row in rows])
    for name, columns in (('line',3),('conditional',5)):
        matrix = np.asarray(design['matrix'][name])
        expected = original(matrix.T@matrix+np.eye(columns),matrix.T@target)
        np.testing.assert_allclose(artifact['models'][name]['coef'],expected,rtol=0.,atol=1e-15)


@pytest.mark.parametrize('key,value', [('source_scope','v87_holdout'), ('external_seed',416207),
    ('external_seed',516207), ('external_seed',616207), ('observed_at',23999),
    ('label_available_at',24000), ('full_real_physical_pass',False)])
def test_old_truth_and_invalid_native_labels_are_rejected_before_fit(key,value):
    rows = training()
    rows[0][key] = value
    with patch.object(np.linalg,'solve') as solve:
        with pytest.raises(ValueError):
            service.train(rows)
        solve.assert_not_called()


def test_duplicate_physical_root_or_missing_case_is_not_replaced():
    rows = training()
    rows[1]['sequence'] = rows[0]['sequence']
    with pytest.raises(ValueError):
        service.train(rows)
    with pytest.raises(ValueError):
        service.train(training()[:-1])


def test_line_and_conditional_share_context_and_action_fallbacks():
    artifact = service.train(training())
    row = covariates()[0]
    for ctx, candidate in ((service.service_state({}),7.5),(row['service'],0.)):
        actual = service.predict(artifact,row['line_id'],ctx,row['ref_hold'],candidate,
                                 simulation_step=1.,action_step=1.)
        assert not actual['supported']
        for value in actual['models'].values():
            assert value['predicted_delta'] is None and value['choice'] == 'reference'
            assert value['reason'] == actual['reason']


def test_zero_gamma_preserves_line_predictions_and_zero_ties_keep_reference():
    artifact = service.train(training())
    artifact['models']['conditional']['coef'] = artifact['models']['line']['coef']+[0.,0.]
    row = covariates()[1]
    actual = service.predict(artifact,row['line_id'],row['service'],row['ref_hold'],row['cand_hold'],
                             simulation_step=1.,action_step=1.)
    assert actual['models']['line'] == actual['models']['conditional']
    for value in artifact['models'].values():
        value['coef'] = [0.]*len(value['coef'])
    actual = service.predict(artifact,row['line_id'],row['service'],row['ref_hold'],row['cand_hold'],
                             simulation_step=1.,action_step=1.)
    assert all(value['predicted_delta'] == 0. and value['choice']=='reference' for value in actual['models'].values())


def test_design_and_prediction_do_not_mutate_inputs_or_random_state():
    rows = covariates()
    before = copy.deepcopy(rows)
    rng = copy.deepcopy(np.random.get_state())
    service.make_design(rows)
    assert rows == before
    actual = np.random.get_state()
    assert rng[0] == actual[0] and rng[2:] == actual[2:]
    np.testing.assert_array_equal(rng[1],actual[1])
