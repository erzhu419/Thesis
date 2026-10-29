"""Read-only accounting on actual native decision chains and passenger clocks.

The stitched return sum(r_scaled)+mixed_bootstrap is a diagnostic under the
recorded common continuation. It is not the original mixed Bellman operator
iterated along an on-policy trajectory, nor a synthetic state at 300 seconds.
"""
from __future__ import annotations

from collections import defaultdict
import math


MODELS = ('zero_memory', 'per_head_level', 'shared_level')
LEDGER_FIELDS = ('time_seconds', 'accrued_wait_seconds', 'accrued_in_vehicle_seconds')


def _cost(left, right):
    wait = float(right['accrued_wait_seconds']-left['accrued_wait_seconds'])
    ride = float(right['accrued_in_vehicle_seconds']-left['accrued_in_vehicle_seconds'])
    return dict(wait=wait, ride=ride, cost=2.*wait+ride)


def _close(left, right, *, absolute=2e-7):
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=absolute)


def _memory_equal(left, right):
    return (left['observed_at'] == right['observed_at']
            and left['vector'] == right['vector'])


def _transition(row):
    return row['transition']


def _model_point(rows, count, model, start_ledger):
    if count == 0:
        return None
    subset = rows[:count]
    scores = [row['scores'][model] for row in subset]
    reward = math.fsum(score['reward_scaled'] for score in scores)
    td = math.fsum(score['target']-score['q_min'] for score in scores)
    continuation_gap = math.fsum(scores[index]['bootstrap_contribution']-scores[index+1]['q_min']
        for index in range(len(scores)-1))
    behavior_gap = math.fsum(scores[index]['bootstrap_contribution']-scores[index+1]['target_q_min']
        for index in range(len(scores)-1))
    lag_gap = math.fsum(score['target_q_min']-score['q_min'] for score in scores[1:])
    # Native targets add float32 reward and bootstrap. Keep that rounding in
    # the accounting rather than mistaking a few ULP for a temporal defect.
    rounding = math.fsum(score['target']-(score['reward_scaled']+score['bootstrap_contribution'])
        for score in scores)
    last = scores[-1]
    stitched = reward+last['bootstrap_contribution']
    reconstructed = stitched+continuation_gap-td+rounding
    raw = math.fsum(_transition(row)['r'] for row in subset)
    physical = _cost(start_ledger, subset[-1]['ledger_ns'])
    return dict(events=count, clock=_transition(subset[-1])['completed_at'],
        physical=physical, reward_raw_sum=raw, reward_scaled_sum=reward,
        final_value_contribution=last['value_contribution'],
        final_sac_contribution=last['sac_contribution'],
        final_sac_q_contribution=.9*last['sac_q_without_entropy'],
        final_entropy_contribution=.9*last['sac_entropy_bonus'],
        final_bootstrap=last['bootstrap_contribution'], stitched_return=stitched,
        continuation_gap_sum=continuation_gap, native_td_min_sum=td,
        mixed_vs_recorded_target_q_gap_sum=behavior_gap, target_lag_gap_sum=lag_gap,
        native_addition_roundoff_sum=rounding,
        initial_q_min=scores[0]['q_min'], reconstructed_initial_q=reconstructed,
        identity_error=reconstructed-scores[0]['q_min'])


def analyze_arm(rows, ticks, *, selected, exact300, endpoint):
    """Check a single real chain and account for its legal endpoints.

    ``rows`` are native event dictionaries with transition, ledger_s/ns, and
    scores[model]. ``ticks`` contain real clocks and issue-relative welfare.
    All full event arrays remain resident; the caller chooses compact exports.
    """
    if not rows:
        return dict(status='missing_chain', integrity=dict(all_pass=False,
            checks=dict(nonempty_chain=False)), events=0, models={})
    transitions = [_transition(row) for row in rows]
    first, last = transitions[0], transitions[-1]
    start = selected['issue_ledger']
    issue_clock = float(selected['started_at'])
    horizon_clock = issue_clock+300.
    endpoint_observed = endpoint['status'] == 'observed'
    checks = dict(
        initial_issue=first['sequence'] == selected['sequence']
            and first['started_at'] == issue_clock and first['s'] == selected['s']
            and _memory_equal(first['memory_s'], selected['memory_s']),
        native_event_kinds=all(row['kind'] in ('global_transition', 'endpoint_actual_without_issue') for row in rows),
        legal_durations=all(row['dt'] >= 0 and row['completed_at']-row['started_at'] == row['dt'] for row in transitions),
        no_fabricated_terminal=all(row['terminal'] is False for row in transitions),
        sequence_continuity=all(right['sequence'] == left['sequence']+1
            for left, right in zip(transitions, transitions[1:])),
        state_clock_continuity=all(left['completed_at'] == right['started_at']
            and left['ns'] == right['s'] and left['raw_ns'] == right['raw_s']
            for left, right in zip(transitions, transitions[1:])),
        memory_causal=all(len(row[name]['vector']) == 84
            and row[name]['observed_at'] == row[clock]
            for row in transitions for name, clock in (('memory_s', 'started_at'), ('memory_ns', 'completed_at'))),
        memory_continuity=all(_memory_equal(left['memory_ns'], right['memory_s'])
            for left, right in zip(transitions, transitions[1:])),
        ledger_clocks=all(row['ledger_s']['time_seconds'] == transition['started_at']
            and row['ledger_ns']['time_seconds'] == transition['completed_at']
            for row, transition in zip(rows, transitions)),
        ledger_continuity=all(all(left['ledger_ns'][key] == right['ledger_s'][key]
            for key in LEDGER_FIELDS) for left, right in zip(rows, rows[1:])),
        initial_ledger=all(rows[0]['ledger_s'][key] == start[key] for key in LEDGER_FIELDS),
        reward_ledger=all(_close(transition['r'], -_cost(row['ledger_s'], row['ledger_ns'])['cost'])
            for row, transition in zip(rows, transitions)),
        complete_three_frozen_scores=all(set(row['scores']) == set(MODELS) for row in rows),
        scores_match_actual_transition=all(score['sequence'] == transition['sequence']
            and score['started_at'] == transition['started_at']
            and score['completed_at'] == transition['completed_at']
            and score['executed_action'] == transition['a'][0]
            for row, transition in zip(rows, transitions) for score in row['scores'].values()),
        exact300_ledger=exact300['status'] == 'complete' and exact300['dt'] == 300.,
        endpoint_not_issued=endpoint['issued'] is False and endpoint['executed'] is False,
    )
    if endpoint_observed:
        checks['real_endpoint'] = (last['completed_at'] == endpoint['clock']
            and horizon_clock <= endpoint['clock'] <= horizon_clock+120.
            and last['ns'] == endpoint['s'] and _memory_equal(last['memory_ns'], endpoint['memory'])
            and rows[-1]['kind'] == 'endpoint_actual_without_issue')
    clocks = [float(row['clock_seconds']) for row in ticks]
    by_clock = {row['clock_seconds']: row for row in ticks}
    h_tick = by_clock.get(horizon_clock)
    checks['complete_native_ticks'] = (bool(ticks) and clocks[0] == issue_clock
        and len(by_clock) == len(ticks) and all(right-left == 1. for left, right in zip(clocks, clocks[1:]))
        and (clocks[-1] == last['completed_at'] if endpoint_observed else
             horizon_clock <= clocks[-1] <= horizon_clock+120.))
    checks['tick_ledger'] = all(row['clock_seconds'] == row['ledger']['time_seconds']
        and row['relative_seconds'] == row['clock_seconds']-issue_clock
        and all(_close(row[key], _cost(start, row['ledger'])[key]) for key in ('wait', 'ride', 'cost'))
        for row in ticks)
    checks['fixed300_matches_tick'] = h_tick is not None and all(
        _close(exact300[key], h_tick[key]) for key in ('wait', 'ride', 'cost'))
    result = dict(status='complete' if endpoint_observed else 'endpoint_missing',
        integrity=dict(all_pass=all(checks.values()), checks=checks), events=len(rows),
        zero_duration_events=sum(row['dt'] == 0 for row in transitions),
        first_completion=first['completed_at'], last_completion=last['completed_at'],
        endpoint_observed=endpoint_observed,
        endpoint_overrun_seconds=endpoint['clock']-horizon_clock if endpoint_observed else None,
        exact300=dict(exact300), models={})
    if not result['integrity']['all_pass']:
        result['status'] = 'integrity_failed'
        return result
    before_count = sum(row['completed_at'] <= horizon_clock for row in transitions)
    before_ledger = rows[before_count-1]['ledger_ns'] if before_count else start
    partial = dict(clock_from=before_ledger['time_seconds'], clock_to=horizon_clock,
        **_cost(before_ledger, h_tick['ledger']), bootstrap=None, state=None)
    result['last_legal_before300_events'] = before_count
    result['partial_to300'] = partial
    result['physical_300_to_endpoint'] = _cost(h_tick['ledger'], rows[-1]['ledger_ns']) if endpoint_observed else None
    for model in MODELS:
        points = [_model_point(rows, count, model, start) for count in range(1, len(rows)+1)]
        score0 = rows[0]['scores'][model]
        identity_pass = all(_close(point['reconstructed_initial_q'], score0['q_min']) for point in points)
        result['models'][model] = dict(first_step=points[0],
            last_legal_before300=points[before_count-1] if before_count else None,
            endpoint=points[-1] if endpoint_observed else None,
            last_closed=points[-1], identity_all_pass=identity_pass,
            maximum_identity_error=max(abs(point['identity_error']) for point in points),
            first_step_target=score0['target'], initial_q_min=score0['q_min'],
            maximum_native_addition_roundoff=max(abs(row['scores'][model]['target']-
                (row['scores'][model]['reward_scaled']+row['scores'][model]['bootstrap_contribution'])) for row in rows))
        checks['identity_'+model] = identity_pass
    result['integrity']['all_pass'] = all(checks.values())
    return result


def _indexed_events(rows):
    occurrences, indexed = defaultdict(int), {}
    for index, row in enumerate(rows):
        transition = _transition(row)
        base = tuple(str(transition[key]) for key in ('line_id', 'vehicle_id', 'trip_id', 'station_id'))
        occurrence = occurrences[base]
        occurrences[base] += 1
        indexed[base+(occurrence,)] = (index, row)
    return indexed


def _witness(identity, left, right):
    a, b = _transition(left), _transition(right)
    return dict(identity=list(identity), issue_clocks=[a['started_at'], b['started_at']],
        completion_clocks=[a['completed_at'], b['completed_at']],
        equal_issue_clock=a['started_at'] == b['started_at'],
        equal_completion_clock=a['completed_at'] == b['completed_at'],
        same_next_vehicle=a['next_vehicle_id'] == b['next_vehicle_id'])


def _decision(advantage, benefit):
    high = advantage > 0.
    return dict(advantage_high_minus_low=advantage,
        choose_high=high, direction_correct=(None if benefit == 0. else
            (advantage > 0.) == (benefit > 0.) if advantage != 0. else False),
        regret=max(-benefit, 0.) if high else max(benefit, 0.))


def analyze_pair(low_rows, high_rows, low_ticks, high_ticks, *, selected,
                 exact300_by_arm, endpoint_by_arm):
    """Compare actual fixed-clock costs and logically matched event endpoints.

    Semantic event alignment may pair different clocks. Such contrasts are
    identified explicitly and never passed off as fixed-clock contrasts.
    """
    raw = dict(low=low_rows, high=high_rows)
    ticks = dict(low=low_ticks, high=high_ticks)
    arms = {arm: analyze_arm(raw[arm], ticks[arm], selected=selected,
        exact300=exact300_by_arm[arm], endpoint=endpoint_by_arm[arm]) for arm in ('low', 'high')}
    result = dict(arms=arms, integrity=dict(all_pass=all(
        value['integrity']['all_pass'] for value in arms.values())), models={})
    if not result['integrity']['all_pass']:
        result['status'] = 'integrity_failed'
        return result
    left, right = _indexed_events(low_rows), _indexed_events(high_rows)
    shared = sorted(left.keys() & right.keys(), key=lambda key: (
        max(_transition(left[key][1])['completed_at'], _transition(right[key][1])['completed_at']), key))
    first = {}
    for identity in shared:
        _, a = left[identity]
        _, b = right[identity]
        ta, tb = _transition(a), _transition(b)
        witness = _witness(identity, a, b)
        for field in ('s', 'ns', 'r', 'dt'):
            if ta[field] != tb[field] and field not in first:
                first[field] = dict(witness)
                if field in ('r', 'dt'):
                    first[field]['high_minus_low'] = tb[field]-ta[field]
        for field in ('memory_s', 'memory_ns'):
            if ta[field]['vector'] != tb[field]['vector'] and field not in first:
                first[field] = dict(witness)
        for model in MODELS:
            for field in ('q_min', 'reward_scaled', 'bootstrap_contribution', 'target',
                          'value_contribution', 'sac_q_without_entropy', 'sac_entropy_bonus'):
                name = model+'/'+field
                av, bv = a['scores'][model][field], b['scores'][model][field]
                if av != bv and name not in first:
                    first[name] = dict(witness, high_minus_low=bv-av)
    result['semantic_alignment'] = dict(matched=len(shared),
        unmatched_low=len(left)-len(shared), unmatched_high=len(right)-len(shared),
        different_completion_clocks=sum(_transition(left[key][1])['completed_at'] !=
            _transition(right[key][1])['completed_at'] for key in shared),
        first_differences=first)
    low_clock = {row['clock_seconds']: row for row in low_ticks}
    high_clock = {row['clock_seconds']: row for row in high_ticks}
    shared_clocks = sorted(low_clock.keys() & high_clock.keys())
    onset = {}
    for clock in shared_clocks:
        for field in ('wait', 'ride', 'cost'):
            difference = high_clock[clock][field]-low_clock[clock][field]
            if difference != 0. and field not in onset:
                onset[field] = dict(clock=clock, relative_seconds=clock-selected['started_at'],
                    high_minus_low=difference)
    result['fixed_clock_cost'] = dict(common_ticks=len(shared_clocks), first_differences=onset)
    low300, high300 = exact300_by_arm['low'], exact300_by_arm['high']
    benefit = low300['cost']-high300['cost']
    result['fixed300'] = dict(benefit_low_minus_high=benefit,
        high_minus_low={key: high300[key]-low300[key] for key in ('wait', 'ride', 'cost')},
        zero_cost_benefit=benefit == 0.)
    for model in MODELS:
        a, b = arms['low']['models'][model], arms['high']['models'][model]
        point = dict(initial_q=_decision(b['initial_q_min']-a['initial_q_min'], benefit),
            first_step_target=_decision(b['first_step_target']-a['first_step_target'], benefit))
        for name in ('last_legal_before300', 'endpoint'):
            av, bv = a[name], b[name]
            if av is None or bv is None:
                point[name] = None
                continue
            fields = ('reward_scaled_sum', 'final_value_contribution', 'final_sac_q_contribution',
                'final_entropy_contribution', 'final_bootstrap', 'stitched_return',
                'continuation_gap_sum', 'mixed_vs_recorded_target_q_gap_sum', 'target_lag_gap_sum',
                'native_td_min_sum', 'native_addition_roundoff_sum')
            point[name] = dict(clocks=[av['clock'], bv['clock']], same_clock=av['clock'] == bv['clock'],
                high_minus_low={field: bv[field]-av[field] for field in fields},
                stitched_vs_fixed300_descriptive=_decision(bv['stitched_return']-av['stitched_return'], benefit))
        result['models'][model] = point
    result['status'] = 'complete' if all(value['endpoint_observed'] for value in arms.values()) else 'endpoint_missing'
    result['interpretation'] = ('300-second cost is fixed-clock common-continuation truth. '
        'Stitched returns use observed decision endpoints and the original frozen mixed bootstrap; '
        'they are diagnostic, not a replacement mixed Bellman target. Event contrasts with unequal '
        'clocks remain event-aligned contrasts. No state or bootstrap is invented at300 seconds.')
    return result
