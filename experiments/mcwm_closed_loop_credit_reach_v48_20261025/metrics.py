"""Pure-array accounting on real v44 transition chains; no model or training.

``diagnose_chain`` and ``paired_diagnosis`` accept {'rows': [...], 'c': [...]};
the paired inputs are raw chains. ``summarize_trace`` also accepts an existing
v44 decomposition, so additional frozen sampling blocks need no prefix sweep.
All cutoffs retain the last real native bootstrap. Algebraic ablations are
labelled separately and are not admitted as new learning targets.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from experiments.mcwm_closed_loop_trace_target_v44_20261024.trace import decompose_trace


DEFAULT_PREFIXES = (1, 9, 17, 25, 33, 45)


def _clock(rows):
    start, end = rows[0].get('started_at'), rows[-1].get('completed_at')
    return dict(started_at=start, completed_at=end,
        physical_seconds=None if start is None or end is None else float(end)-float(start),
        duration_sum_seconds=math.fsum(float(row['dt']) for row in rows),
        zero_duration_events=sum(float(row['dt']) == 0. for row in rows),
        clock_link_mismatches=sum(current.get('completed_at') != following.get('started_at')
            for current, following in zip(rows, rows[1:])
            if 'completed_at' in current and 'started_at' in following))


def summarize_trace(trace, rows):
    """Root target = reward + bootstrap-minus-Q + explicit float32 rounding.

    The incoming matching Q of the last event is grouped with the intermediate
    contribution. The last *bootstrap* alone is the cutoff dependency, not a
    fabricated terminal and not an exact finite-difference gradient.
    """
    root, events = trace['root'], trace['per_event']
    parts, tail = root['components'], events[-1]
    terminal_bootstrap = tail['bootstrap_root_contribution']
    bootstrap_minus_q = parts['bootstrap']+parts['minus_matching_target_q']
    rounding = parts['native_rounding']+parts['recurrence_rounding']
    components = dict(reward=parts['reward'], bootstrap_minus_matching_q=bootstrap_minus_q,
                      rounding=rounding)
    total = math.fsum(components.values())
    return dict(event_count=len(rows), clock=_clock(rows), native_target=root['native_target'],
        trace_target=root['trace_target'], correction=root['correction'], components=components,
        component_sum=total, closure_residual=root['trace_target']-total,
        root_reward=float(events[0]['reward_scaled']), future_reward=root['future_reward'],
        raw_reward_sum=math.fsum(float(row['reward_scaled']) for row in rows),
        cutoff_bootstrap=terminal_bootstrap,
        intermediate_bootstrap_minus_matching_q=bootstrap_minus_q-terminal_bootstrap,
        cutoff_weight=trace['reach_weights'][-1],
        cutoff_native_bootstrap=float(rows[-1]['bootstrap_contribution']),
        cutoff_true_terminal=bool(rows[-1]['terminal']),
        supported_event_count=trace['reached_event_count'], first_stop=trace['first_stop'])


def _logs(trace, rows):
    links, cumulative = [], dict(gamma=0., mixing=0., c=0., multiplication_rounding=0.)
    reachable = True
    for index, link in enumerate(trace['links']):
        factors = dict(gamma=link['gamma'], mixing=link['mixed_path_weight'], c=link['c'])
        logs = {key: None if value is None or float(value) <= 0. else math.log(float(value))
                for key, value in factors.items()}
        active = link['active']
        residual = (math.log(link['coefficient'])-math.fsum(logs.values())
                    if active else None)
        reachable = reachable and active
        if reachable:
            for key in logs:
                cumulative[key] += logs[key]
            cumulative['multiplication_rounding'] += residual
        links.append(dict(from_event=index, to_event=index+1,
            source_dt=float(rows[index]['dt']), zero_duration=float(rows[index]['dt']) == 0.,
            active=active, stop_reason=link['reason'], coefficient=link['coefficient'],
            log_contributions=logs, log_multiplication_rounding=residual,
            cumulative_log_contributions=dict(cumulative) if reachable else None,
            root_reach_weight=trace['reach_weights'][index+1]))
    active_logs = [entry for entry in links if entry['active']]
    def subtotal(group):
        chosen = [entry for entry in active_logs if group is None or entry['zero_duration'] == group]
        return dict(link_count=len(chosen),
            gamma=math.fsum(entry['log_contributions']['gamma'] for entry in chosen),
            mixing=math.fsum(entry['log_contributions']['mixing'] for entry in chosen),
            c=math.fsum(entry['log_contributions']['c'] for entry in chosen),
            multiplication_rounding=math.fsum(entry['log_multiplication_rounding'] for entry in chosen))
    return links, dict(all_active_links=subtotal(None), zero_duration_links=subtotal(True),
        positive_duration_links=subtotal(False),
        first_stop=trace['first_stop'], final_root_weight=trace['reach_weights'][-1],
        cumulative_logs_are_full_chain=trace['full_chain_supported'],
        log_unit='natural_log_of_algebraic_link_weight_not_physical_time_discount')


def _cutoff_sensitivity(rows, c, trace):
    results = []
    for delta in (-1., 1.):
        changed = copy.deepcopy(rows)
        old = np.float32(changed[-1]['bootstrap_contribution'])
        changed[-1]['bootstrap_contribution'] = float(np.float32(old+np.float32(delta)))
        changed[-1]['native_target'] = float(np.float32(np.float32(changed[-1]['reward_scaled'])+
                                             np.float32(changed[-1]['bootstrap_contribution'])))
        # Component split is not reconstructed for this counterfactual scalar
        # probe. The original native target path is the sole recurrence.
        for row in changed:
            row.pop('value_contribution', None)
            row.pop('sac_contribution', None)
        perturbed = decompose_trace(changed, c, 1.)
        realized = changed[-1]['bootstrap_contribution']-float(old)
        actual = perturbed['root']['trace_target']-trace['root']['trace_target']
        algebraic = trace['reach_weights'][-1]*realized
        results.append(dict(requested_bootstrap_delta=delta, realized_bootstrap_delta=realized,
            root_target_delta=actual, unrounded_algebraic_response=algebraic,
            float32_response_minus_algebraic=actual-algebraic))
    return dict(cutoff_algebraic_weight=trace['reach_weights'][-1], perturbations=results,
        scope='scalar_bootstrap_sensitivity_only_not_an_observed_future_reward')


def diagnose_chain(chain, *, prefix_counts=DEFAULT_PREFIXES):
    """Diagnose every real event cutoff, returning six fixed prefixes by default.

    ``all_prefixes`` is resident detail; ``prefixes`` is the bounded public set.
    Every numeric target is produced by the unchanged v44 float32 recurrence.
    """
    rows, c = chain['rows'], chain['c']
    zero = decompose_trace(rows, c, 0.)
    full_trace = decompose_trace(rows, c, 1.)
    prefixes = [summarize_trace(decompose_trace(rows[:count], c[:count-1], 1.), rows[:count])
                for count in range(1, len(rows)+1)]
    links, attenuation = _logs(full_trace, rows)
    ablations = {}
    for name, ratios, mixing in (('c_one', [1.]*len(c), .9), ('mixing_one', c, 1.),
                                 ('c_and_mixing_one', [1.]*len(c), 1.)):
        trace = decompose_trace(rows, ratios, 1., mixed_path_weight=mixing)
        ablations[name] = summarize_trace(trace, rows)
    return dict(lambda_zero=dict(native_exact=zero['lambda_zero_native_exact'],
                root_target=zero['root']['trace_target'],
                native_target=zero['root']['native_target']),
        clock=_clock(rows), links=links, attenuation=attenuation,
        prefixes=[prefixes[count-1] for count in prefix_counts if count <= len(rows)],
        requested_prefix_counts=list(prefix_counts), all_prefixes=prefixes,
        full=summarize_trace(full_trace, rows), per_event=full_trace['per_event'],
        cutoff_sensitivity=_cutoff_sensitivity(rows, c, full_trace),
        first_nonzero_reward_event=next((index for index, row in enumerate(rows)
            if np.float32(row['reward_scaled']) != 0.), None),
        algebraic_ablations=dict(scope='accounting_only_not_legal_new_targets', cases=ablations),
        collector_cutoff_rule='retain_last_real_native_bootstrap', synthetic_states=0,
        new_training_updates=0, new_native_branches=0)


def _difference(low, high):
    keys = ('trace_target', 'native_target', 'correction', 'root_reward', 'future_reward',
            'raw_reward_sum', 'cutoff_bootstrap', 'intermediate_bootstrap_minus_matching_q')
    delta = {key: high[key]-low[key] for key in keys}
    components = {key: high['components'][key]-low['components'][key]
                  for key in low['components']}
    return dict(event_count_low=low['event_count'], event_count_high=high['event_count'],
        low_clock=low['clock'], high_clock=high['clock'], high_minus_low=delta,
        component_difference=components,
        difference_closure_residual=delta['trace_target']-math.fsum(components.values()),
        same_physical_endpoint=(low['clock']['completed_at'] is not None and
                               low['clock']['completed_at'] == high['clock']['completed_at']))


def paired_diagnosis(low, high, *, prefix_counts=DEFAULT_PREFIXES):
    """Compare raw low/high chains at fixed event counts, never a fake 300s state.

    Equal event counts may end on different real clocks. The root runner owns
    the separate shared-clock passenger-cost ledger and its physical endpoint.
    """
    lo, hi = diagnose_chain(low, prefix_counts=prefix_counts), diagnose_chain(high, prefix_counts=prefix_counts)
    indexed = {row['event_count']: row for row in hi['prefixes']}
    pairs = [_difference(row, indexed[row['event_count']]) for row in lo['prefixes']
             if row['event_count'] in indexed]
    return dict(low=lo, high=hi, fixed_event_prefix_differences=pairs,
        full_difference=_difference(lo['full'], hi['full']),
        comparison_scope='same_event_count_not_necessarily_shared_physical_clock',
        first_paired_reward_difference_event=next((index for index, (a, b) in enumerate(
            zip(low['rows'], high['rows']))
            if np.float32(a['reward_scaled']) != np.float32(b['reward_scaled'])), None))
