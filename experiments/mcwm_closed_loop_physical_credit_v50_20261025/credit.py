"""One physical-time residual target, evaluated only on cached real chains.

Y retains the original .1V/.9SAC backup.  The new residual propagation is
G_i = Y_i + lambda exp(-dt_i/300) c_(i+1) (G_(i+1)-Qbar_(i+1)).
This is a changed trace operator, not an equivalent expansion of the original
event-weighted trace or a new discount on the passenger-cost objective.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from experiments.mcwm_closed_loop_trace_target_v44_20261024.trace import decompose_trace


def physical_trace(rows, c, lambda_trace):
    """Preserve native Y bits at lambda=0 and every real endpoint at lambda=1.

    The float32 effective coefficient is an implementation input to the old
    arithmetic routine; it is not a new behavior-policy density ratio.
    """
    if len(c) != len(rows)-1:
        raise ValueError('Execution ratios must match real successor links')
    if any(float(row.get('gamma', 1.)) != 1. for row in rows):
        raise ValueError('This candidate is defined for the native gamma=1 contract')
    if any(not math.isfinite(float(row['dt'])) or float(row['dt']) < 0. for row in rows):
        raise ValueError('Physical durations must be finite and nonnegative')
    if any(value is not None and (not math.isfinite(float(value)) or not 0. <= float(value) <= 1.)
           for value in c):
        raise ValueError('Actual execution c must lie in [0, 1]')
    retention = [float(np.float32(math.exp(-float(row['dt'])/300.))) for row in rows[:-1]]
    effective = [None if value is None else float(np.float32(
        np.float32(keep)*np.float32(value))) for keep, value in zip(retention, c)]
    trace = decompose_trace(rows, effective, lambda_trace, mixed_path_weight=1.)
    for link, actual, keep, combined in zip(trace['links'], c, retention, effective):
        link.update(c=actual, actual_c=actual, physical_retention=keep,
                    effective_c=combined, effective_c_is_actual_RN=False)
    reach = trace['reach_weights']
    future = math.fsum(reach[1:])
    trace.update(original_backup_sac_weight=.9, original_backup_value_weight=.1,
        physical_time_constant_seconds=300., trace_kernel='exp(-source_dt/300)*actual_c',
        original_mixed_backup_preserved=True, original_trace_operator_preserved=False,
        gain=dict(sum_reach=math.fsum(reach), future_sum_reach=future,
            uniform_q_offset_gain=.9-.1*future,
            all_nonterminal_reference=not any(row['terminal'] for row in rows),
            scope='V fixed; all bootstrap Q and matching Q shifted together; gamma=1; '
                  'scalar nonterminal reference only, not deep-network stability'))
    return trace


def _trace(rows, c, trace_fn):
    return trace_fn(rows, c, 1.)


def _assemble(row):
    row['native_target'] = float(np.float32(np.float32(row['reward_scaled'])+
                                          np.float32(row['bootstrap_contribution'])))
    # These scalar probes do not reconstruct the native V/SAC component split.
    row.pop('value_contribution', None)
    row.pop('sac_contribution', None)


def _response(original, changed, c, trace_fn, baseline):
    result = _trace(changed, c, trace_fn)
    # This accounts for realized float32 row assembly and matching-Q changes;
    # recurrence rounding remains visible as the finite-difference residual.
    weights = baseline['reach_weights']
    native = math.fsum(weight*(float(after['native_target'])-float(before['native_target']))
        for weight, before, after in zip(weights, original, changed))
    matching = math.fsum(weights[index+1]*(float(changed[index]['recorded_next_target_q'])-
        float(original[index]['recorded_next_target_q'])) for index in range(len(c))
        if weights[index+1] and original[index].get('recorded_next_target_q') is not None)
    return dict(root_change=result['root']['trace_target']-baseline['root']['trace_target'],
        realized_unrolled_change=native-matching,
        changed_root=result['root']['trace_target'])


def probe_reward_band(rows, c, trace_fn=physical_trace):
    """Zero complete rewards whose real *completion* is in [30,45) seconds.

    No interval reward is split or prorated.  This is a numerical sensitivity
    probe, not the exact passenger cost incurred between those two clocks.
    """
    baseline = _trace(rows, c, trace_fn)
    changed, selected = copy.deepcopy(rows), []
    start = float(rows[0]['started_at'])
    for index, row in enumerate(changed):
        if 30. <= float(row['completed_at'])-start < 45.:
            selected.append(index)
            row['reward_scaled'] = 0.
            _assemble(row)
    result = _response(rows, changed, c, trace_fn, baseline)
    result.update(selected_count=len(selected), selected_nonzero_rewards=sum(
        float(rows[index]['reward_scaled']) != 0. for index in selected),
        algebraic_change=math.fsum(baseline['reach_weights'][index]*
            -float(rows[index]['reward_scaled']) for index in selected),
        endpoint_band_seconds=[30., 45.],
        scope='whole real rewards selected by completion clock; not exact band cost')
    return result


def probe_bootstrap_shift(rows, c, trace_fn=physical_trace):
    """Perturb only the final real bootstrap; never turn a cutoff terminal."""
    baseline = _trace(rows, c, trace_fn)
    if rows[-1]['terminal']:
        return dict(applicable=False, reason='true_terminal_has_no_bootstrap', perturbations=[])
    probes = []
    for delta in (-1., 1.):
        changed = copy.deepcopy(rows)
        before = np.float32(changed[-1]['bootstrap_contribution'])
        changed[-1]['bootstrap_contribution'] = float(np.float32(before+np.float32(delta)))
        _assemble(changed[-1])
        realized = changed[-1]['bootstrap_contribution']-float(before)
        value = _response(rows, changed, c, trace_fn, baseline)
        value.update(requested_shift=delta, realized_bootstrap_shift=realized,
            algebraic_change=baseline['reach_weights'][-1]*realized)
        probes.append(value)
    return dict(applicable=True, cutoff_weight=baseline['reach_weights'][-1],
        perturbations=probes, scope='scalar bootstrap sensitivity, not observed future reward')


def probe_uniform_q_shift(rows, c, trace_fn=physical_trace):
    """Shift all scalar Q references with V fixed; this is not model training.

    Each bootstrap moves by .9*dQ and each available matching Q by dQ.  The
    exact-reference gain is .9*sum(w)-sum(w[1:]); float32 assembly/recurrence
    can differ and is separately measured.  True terminals are inapplicable.
    """
    baseline = _trace(rows, c, trace_fn)
    if any(row['terminal'] for row in rows):
        return dict(applicable=False, reason='reference_requires_nonterminal_chain', perturbations=[])
    gain = .9-.1*math.fsum(baseline['reach_weights'][1:])
    probes = []
    for delta in (-1., 1.):
        changed = copy.deepcopy(rows)
        for row in changed:
            row['bootstrap_contribution'] = float(np.float32(
                np.float32(row['bootstrap_contribution'])+np.float32(.9)*np.float32(delta)))
            if row.get('recorded_next_target_q') is not None:
                row['recorded_next_target_q'] = float(np.float32(
                    np.float32(row['recorded_next_target_q'])+np.float32(delta)))
            _assemble(row)
        value = _response(rows, changed, c, trace_fn, baseline)
        value.update(requested_q_shift=delta, algebraic_change=gain*delta)
        probes.append(value)
    return dict(applicable=True, linear_reference_gain=gain, perturbations=probes,
        scope='uniform scalar Q offset with V fixed and frozen weights, not network stability')
