"""Detect wrong cost units, unsupported endpoints and lost bridge components."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.mcwm_closed_loop_objective_alignment_v58_20261025.objective import (
    bridge_chain, objective_contract, pair_bridge, validate_real_chain,
)


def fixture():
    clocks = [100.,100.,220.,400.]
    rows = [dict(sequence=i, started_at=clocks[i],completed_at=clocks[i+1],
        dt=clocks[i+1]-clocks[i],s=[i],ns=[i+1],
        memory_s=dict(vector=[i]),memory_ns=dict(vector=[i+1]),
        r=r,terminal=False) for i,r in enumerate((0.,-700.,-1900.))]
    reward = np.asarray([0.,-.07,-.19],np.float32)
    b = np.asarray([2.,1.8,1.4],np.float32)
    h = np.asarray([.5,.6,.7],np.float32)
    d = np.ones(3,np.float32)
    v = np.asarray([.1,.2,.3],np.float32)
    qs = b-h-v
    arrays = dict(reward=reward,bootstrap=b,entropy=h,native=reward+b,
        matching_q=np.asarray([1.3,.9],np.float32),coefficients=np.asarray([1.,.8],np.float32),
        value_contribution=v,sac_q_contribution=qs,discount=d,
        q_exec=np.asarray([1.7,1.2,.85],np.float32))
    return rows,arrays


def test_real300_reward_units_and_additive_weighted_bridge():
    rows,a = fixture()
    proof = validate_real_chain(rows)
    assert proof['all_pass'] and proof['real_300_endpoint']
    assert proof['zero_duration_events'] == 1
    x = bridge_chain(rows,a,J300=2600.)
    assert x['raw_passenger_cost'] == 2600.
    assert x['learner_reward_sum'] == pytest.approx(-.26)
    assert abs(x['reward_scale_conversion_error']) < 1e-8
    assert abs(x['physical_identity_error']) < 1e-12
    assert abs(x['mixed_component_identity_error']) < 3e-8
    assert abs(x['Q_telescope']['identity_error']) < 1e-12
    q = x['Q_telescope']
    assert q['mixed_vs_recorded_target_Q_gap']+q['target_Q_minus_online_Q_gap'] == pytest.approx(q['mixed_continuation_gap'])
    assert abs(x['fixed300_Q_bridge']['identity_error']) < 1e-12
    assert not x['weighted_reward_is_undiscounted_cost']
    assert objective_contract()['native_gamma'] == 1.


def test_actual_weighted_recurrence_and_terminal_sensitivity():
    rows,a = fixture()
    x = bridge_chain(rows,a)
    p = float(a['native'][-1])
    for i in (1,0):
        p = float(a['native'][i])+float(a['coefficients'][i])*(p-float(a['matching_q'][i]))
    assert x['physical_recurrence64'] == p
    b = deepcopy(a)
    # A +1 actual endpoint bootstrap perturbation travels with the real links.
    b['bootstrap'] = b['bootstrap'].astype(float)
    b['native'] = b['native'].astype(float)
    b['bootstrap'][-1] += 1.
    b['native'][-1] += 1.
    y = bridge_chain(rows,b)
    assert y['bounded_trace64']-x['bounded_trace64'] == pytest.approx(x['trace_endpoint_bootstrap_sensitivity'])


def test_partial_clock_never_manufactures300_state():
    rows,a = fixture()
    rows[-1]['completed_at'] = 399.
    rows[-1]['dt'] = 179.
    proof = validate_real_chain(rows)
    assert proof['all_pass'] and not proof['real_300_endpoint']
    assert not proof['state_at300_supported']
    x = bridge_chain(rows,a)
    assert x['physical_span_seconds'] == 299.
    assert not x['endpoint_is_exact300']


def test_broken_native_link_is_not_legal_credit():
    rows,a = fixture()
    rows[1]['s'] = [-9]
    assert not validate_real_chain(rows)['all_pass']
    with pytest.raises(ValueError,match='legal native chain'):
        bridge_chain(rows,a)


def test_pair_retains_measured_endpoint_excess_without_inventing300_state():
    rows,a = fixture()
    x = bridge_chain(rows,a,J300=2600.)
    b = deepcopy(a)
    b['q_exec'][0] += .2
    rows[-1]['completed_at'] = 401.
    rows[-1]['dt'] = 181.
    y = bridge_chain(rows,b,J300=2500.)
    pair = pair_bridge(x,y)
    assert pair['supported'] and not pair['both_real_endpoints_exact300']
    assert not pair['action_Q_and300_sign_equivalence_established']
    assert pair['fixed300_Q_bridge']['high_action_benefit300'] == 100.
    assert abs(pair['fixed300_Q_bridge']['identity_error']) < 1e-12
    parts = pair['fixed300_Q_bridge']['high_minus_low_components']
    assert parts['passenger_cost300'] == pytest.approx(.01)
    assert parts['actual_endpoint_excess_cost'] == pytest.approx(-.01)


def test_no_importance_weights_means_only_Q_accounting_not_supported_trace():
    rows,a = fixture()
    del a['coefficients']
    x = bridge_chain(rows,a,J300=2600.)
    assert not x['weighted_trace_supported']
    assert 'physical_recurrence64' not in x
    assert 'physical_components' not in x
    assert 'bounded_trace64' not in x
    assert abs(x['Q_telescope']['identity_error']) < 1e-12
    assert abs(x['fixed300_Q_bridge']['identity_error']) < 1e-12
    pair = pair_bridge(x,x)
    assert not pair['weighted_trace_supported']
    assert 'physical_components_high_minus_low' not in pair
    assert abs(pair['fixed300_Q_bridge']['identity_error']) < 1e-12
