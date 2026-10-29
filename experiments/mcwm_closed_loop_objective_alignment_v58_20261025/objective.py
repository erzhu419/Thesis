"""Passenger-cost/soft-Q accounting on observed, adjacent event chains.

All decompositions use float64 as an accounting aid. Native targets are kept
as supplied, including their float32 assembly error. This module creates no
state, action outcome, terminal marker, or 300-second bootstrap.
"""
from __future__ import annotations

import math

import numpy as np


def _row(row):
    return row.get('transition', row) if isinstance(row, dict) else vars(row)


def objective_contract():
    return dict(
        cost='J=2*accrued_wait_seconds+accrued_in_vehicle_seconds',
        cost_unit='global weighted passenger-seconds; not seconds per passenger',
        raw_reward='r=-(2*delta_wait+delta_ride)', reward_unit=1.,
        learner_reward_scale=1e-4, native_gamma=1., finite_horizon_seconds=24000,
        mixed_bootstrap='.1*V+.9*E_F[Q-alpha*log_pi]',
        entropy_scope='proposal hybrid log density used by native backup; not an F-pushforward entropy',
        truth='fixed 300-second root intervention followed by the recorded frozen continuation',
        Q='long-horizon mixed soft critic; endpoints, continuation and Bellman gaps remain',
        action_identification='same observed root, context, exogenous seed, and continuation; distinct executed root actions')


def validate_real_chain(rows):
    """Check actual state/clock links and report available physical endpoints."""
    rows = [_row(row) for row in rows]
    if not rows:
        return dict(all_pass=False, status='missing', rows=0)
    checks = dict(
        durations=all(float(r['dt']) >= 0. and float(r['completed_at'])-
            float(r['started_at']) == float(r['dt']) for r in rows),
        actual_sequence=all(b['sequence'] == a['sequence']+1 for a,b in zip(rows,rows[1:])),
        real_state_links=all(a['completed_at'] == b['started_at'] and
            np.array_equal(a['ns'],b['s']) for a,b in zip(rows,rows[1:])),
        real_memory_links=all(a['memory_ns'] == b['memory_s'] for a,b in zip(rows,rows[1:])),
        no_synthetic_terminal=all(not bool(r['terminal']) for r in rows[:-1]))
    root = float(rows[0]['started_at'])
    endpoint = float(rows[-1]['completed_at'])
    exact = [i for i,r in enumerate(rows) if float(r['completed_at']) == root+300.]
    before = [i for i,r in enumerate(rows) if float(r['completed_at']) <= root+300.]
    return dict(all_pass=all(checks.values()), checks=checks, rows=len(rows),
        root_clock=root, final_real_clock=endpoint, span_seconds=endpoint-root,
        zero_duration_events=sum(float(r['dt']) == 0. for r in rows),
        real_300_endpoint=bool(exact), real_300_last_event=exact[-1] if exact else None,
        reaches_300_seconds=endpoint >= root+300.,
        last_real_endpoint_at_or_before300=(float(rows[before[-1]]['completed_at']) if before else None),
        state_at300_supported=bool(exact),
        exact300_ledger_does_not_imply_decision_state=True)


def bridge_chain(rows, arrays, *, J300=None, root=0, stop=None, reward_scale=1e-4, eta=9./270.):
    """Additive bridge for one root on its real recorded successor chain.

    Required arrays: native, reward, bootstrap, entropy (one per transition),
    matching_q (one per adjacent link). Optional arrays: coefficients,
    value_contribution, sac_q_contribution, discount, q_exec. The latter is
    for TD telescope accounting only; matching_q is the official recorded-next-
    action target Q, never the root action evaluated at an invented successor.
    """
    actual = [_row(row) for row in rows]
    proof = validate_real_chain(actual)
    if not proof['all_pass']:
        raise ValueError('The supplied rows do not form a legal native chain')
    stop = len(actual) if stop is None else int(stop)
    if not 0 <= root < stop <= len(actual):
        raise ValueError('Root and stop must select observed transitions')
    values = {k:np.asarray(v,dtype=np.float64).reshape(-1) for k,v in arrays.items()}
    for key in ('native','reward','bootstrap','entropy'):
        if len(values[key]) != len(actual):
            raise ValueError('Expected one actual transition component: '+key)
    for key in ('matching_q',):
        if len(values[key]) != len(actual)-1:
            raise ValueError('Expected one real adjacent-link component: '+key)
    if 'coefficients' in values and len(values['coefficients']) != len(actual)-1:
        raise ValueError('Expected one observed coefficient per adjacent link')
    s = slice(root,stop)
    r,b,h,y = (values[k][s] for k in ('reward','bootstrap','entropy','native'))
    qnext = values['matching_q'][root:stop-1]
    rounding = y-r-b
    raw_reward = math.fsum(float(r['r']) for r in actual[root:stop])
    unweighted_reward = float(r.sum())
    raw_cost = -raw_reward
    clock = float(actual[stop-1]['completed_at'])
    start = float(actual[root]['started_at'])
    result = dict(root_index=root, real_events=stop-root, root_clock=start,
        endpoint_clock=clock, physical_span_seconds=clock-start,
        endpoint_is_exact300=clock == start+300.,
        raw_passenger_cost=raw_cost, raw_reward_sum=raw_reward,
        learner_reward_sum=unweighted_reward,
        reward_scale_conversion_error=unweighted_reward+reward_scale*raw_cost,
        unweighted_endpoint_stitched=unweighted_reward+float(b[-1]),
        stitched_is_original_mixed_Bellman_target=False,
        root_native=float(y[0]),same_state_action_contrast=False,
        weighted_trace_supported='coefficients' in values)
    if 'coefficients' in values:
        c = values['coefficients'][root:stop-1]
        weights = np.concatenate(([1.],np.cumprod(c)))
        physical = float(y[-1])
        for i in range(len(y)-2,-1,-1):
            physical = float(y[i])+float(c[i])*(physical-float(qnext[i]))
        reward = float(np.dot(weights,r))
        entropy = float(np.dot(weights,h))
        endpoint_nonentropy = float(weights[-1]*(b[-1]-h[-1]))
        mixed_gap = float(np.dot(weights[:-1],b[:-1]-h[:-1]-qnext))
        coefficient_gap = float(np.dot(weights[:-1],(1.-c)*qnext))
        native_rounding = float(np.dot(weights,rounding))
        reconstructed = math.fsum((reward,entropy,endpoint_nonentropy,mixed_gap,
            coefficient_gap,native_rounding))
        result.update(physical_recurrence64=physical,
            bounded_trace64=float(y[0])+eta*(physical-float(y[0])),
            physical_components=dict(reward=reward,entropy=entropy,
                endpoint_nonentropy_bootstrap=endpoint_nonentropy,
                mixed_continuation_minus_recorded_Q=mixed_gap,
                coefficient_minus_unit_link=coefficient_gap,
                native_assembly_rounding=native_rounding),
            physical_identity_error=reconstructed-physical,
            trace_correction64=eta*(physical-float(y[0])),
            trace_endpoint_bootstrap_sensitivity=(eta*float(weights[-1]) if len(y)>1 else 0.),
            endpoint_unnormalized_path_weight=float(weights[-1]),
            weighted_reward_is_undiscounted_cost=False,
            future_residual_gain=float(eta*weights[1:].sum()))
        if all(k in values for k in ('value_contribution','sac_q_contribution','discount')):
            v,qs,d = (values[k][s] for k in ('value_contribution','sac_q_contribution','discount'))
            parts = dict(value_minus_recorded_Q=float(np.dot(weights[:-1],v[:-1]-.1*d[:-1]*qnext)),
                SAC_Q_minus_recorded_Q=float(np.dot(weights[:-1],qs[:-1]-.9*d[:-1]*qnext)),
                terminal_or_native_discount=float(np.dot(weights[:-1],(d[:-1]-1.)*qnext)))
            result['mixed_gap_components'] = parts
            result['mixed_component_identity_error'] = math.fsum(parts.values())-mixed_gap
    if 'q_exec' in values:
        q = values['q_exec'][s]
        td = float((y-q).sum())
        entropy_sum = float(h.sum())
        internal_gap = float((b[:-1]-h[:-1]-q[1:]).sum())
        tail = float(b[-1]-h[-1])
        telescope = math.fsum((unweighted_reward,tail,entropy_sum,
            internal_gap,-td,float(rounding.sum())))
        result['Q_telescope'] = dict(initial_Q=float(q[0]),
            reward=unweighted_reward,terminal_nonentropy_bootstrap=tail,
            entropy=entropy_sum,
            mixed_continuation_gap=internal_gap,minus_TD_residual_sum=-td,
            mixed_vs_recorded_target_Q_gap=float((b[:-1]-h[:-1]-qnext).sum()),
            target_Q_minus_online_Q_gap=float((qnext-q[1:]).sum()),
            native_assembly_rounding=float(rounding.sum()),identity_error=telescope-float(q[0]),
            recorded_current_Q_may_differ_from_target_matching_Q=True)
        if J300 is not None:
            excess = raw_cost-float(J300)
            parts = dict(passenger_cost300=-reward_scale*float(J300),
                actual_endpoint_excess_cost=-reward_scale*excess,
                reward_scale_rounding=unweighted_reward+reward_scale*raw_cost,
                entropy=entropy_sum,terminal_nonentropy_bootstrap=tail,
                mixed_continuation_gap=internal_gap,minus_TD_residual_sum=-td,
                native_assembly_rounding=float(rounding.sum()))
            result['fixed300_Q_bridge'] = dict(J300=float(J300),J_real_endpoint=raw_cost,
                endpoint_excess_cost=excess,reward_scale=float(reward_scale),
                root_Q=float(q[0]),components=parts,
                identity_error=math.fsum(parts.values())-float(q[0]),
                state_or_bootstrap_at300_created=False,
                interpretation='J300 is an observed ledger only; all Q and TD terms use actual event states.')
    return result


def pair_bridge(low, high):
    """Subtract already audited same-prefix action arms at actual endpoints.

    The caller audits root equality, seed and continuation, since those fields
    belong to the native issue records rather than these additive identities.
    Different actual endpoint clocks remain legal when their observed excess
    relative to the fixed300 ledger is kept explicitly in the bridge.
    """
    keys = ('unweighted_endpoint_stitched','raw_passenger_cost','learner_reward_sum')
    weighted = all(arm.get('weighted_trace_supported') for arm in (low,high))
    if weighted:
        keys += ('physical_recurrence64','bounded_trace64')
    contrast = {key:float(high[key])-float(low[key]) for key in keys}
    same_clock = low['endpoint_clock'] == high['endpoint_clock']
    exact300 = same_clock and low['endpoint_is_exact300'] and high['endpoint_is_exact300']
    result = dict(supported=True,same_state_action_contrast=True,
        equal_real_endpoint_clock=same_clock, both_real_endpoints_exact300=exact300,
        high_minus_low=contrast,weighted_trace_supported=weighted,
        action_Q_and300_sign_equivalence_established=False,
        interpretation='Real paired continuation contrast; unequal event clocks are not fixed-clock cost truth.')
    if weighted:
        result['physical_components_high_minus_low'] = {
            key:float(high['physical_components'][key])-float(low['physical_components'][key])
            for key in low['physical_components']}
    if all('fixed300_Q_bridge' in arm for arm in (low,high)):
        a,b = low['fixed300_Q_bridge'],high['fixed300_Q_bridge']
        diff = {key:b['components'][key]-a['components'][key] for key in a['components']}
        result['fixed300_Q_bridge'] = dict(high_minus_low_root_Q=b['root_Q']-a['root_Q'],
            high_minus_low_components=diff,
            high_action_benefit300=a['J300']-b['J300'],
            identity_error=math.fsum(diff.values())-(b['root_Q']-a['root_Q']))
    return result
