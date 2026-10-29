"""Read-only reward visibility on genuine v43 event closure clocks.

An ordinal reward difference can be a different partition of the same cost.
This helper compares cumulative costs only at clocks reached by both arms.
No state, reward, ledger endpoint, or300-second outcome is interpolated.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


TOL = 1e-7


def _records(value):
    if isinstance(value, (str, Path)):
        with Path(value).open() as source:
            return [json.loads(line) for line in source if line.strip()]
    return list(value)


def _events(value):
    return [row.get('transition', row) for row in _records(value)]


def _clock_costs(rows):
    result, accrued = {}, []
    last_reward = 0
    for index, row in enumerate(rows):
        accrued.append(-float(row['r']))
        if row['r'] != 0.:
            last_reward = index
        result[float(row['completed_at'])] = dict(cost=math.fsum(accrued),
            closed_rows=index+1, links=last_reward)
    return result


def _ledgers(trace):
    result = {}
    for row in _records(trace):
        if 'issue_ledger' not in row:
            continue
        ledger = row['issue_ledger']
        clock = float(ledger['time_seconds'])
        value = (float(ledger['accrued_wait_seconds']),
                 float(ledger['accrued_in_vehicle_seconds']))
        if clock in result and result[clock] != value:
            raise ValueError('One genuine clock has inconsistent issue ledgers')
        result[clock] = value
    return result


def _ledger_deltas(rows, trace, coefficients):
    ledgers = _ledgers(trace)
    root = float(rows[0]['started_at'])
    if root not in ledgers:
        raise ValueError('The suffix root issue ledger is missing')
    base = ledgers[root]
    wait, ride, unit = coefficients
    deltas = {clock: dict(wait=(value[0]-base[0])*wait/unit,
        ride=(value[1]-base[1])*ride/unit,
        cost=((value[0]-base[0])*wait+(value[1]-base[1])*ride)/unit)
        for clock, value in ledgers.items()}
    costs = _clock_costs(rows)
    errors = [abs(value['cost']-deltas[clock]['cost']) for clock, value in costs.items()
              if clock in deltas]
    return deltas, dict(root_ledger=list(base), genuine_ledger_clocks=len(ledgers),
        closure_clocks_cross_checked=len(errors), maximum_reward_ledger_error=max(errors, default=None),
        reward_ledger_consistent=bool(errors) and max(errors) <= TOL,
        final_closure_has_ledger=float(rows[-1]['completed_at']) in deltas)


def _credit(packet, line_id, arm, links):
    chain = next(row for row in packet['chains'] if row['line_id'] == line_id and row['arm'] == arm)
    scores = chain['rows']
    if not 0 <= links <= len(scores):
        raise ValueError('Required real reward lies outside the captured continuation')
    chosen = scores[:links]
    actual = math.fsum(row['log_c'] for row in chosen)
    geom = math.fsum(row['expected_log_c'] for row in chosen)
    arith = math.fsum(row['log_overlap'] for row in chosen)
    path = links*math.log(.9)
    return [packet['updates'], arm, links, actual, geom, arith, path,
            math.exp(actual+path), math.exp(geom+path), math.exp(arith+path)]


def summarize_pair(lowrows, highrows, lowtrace, hightrace, credit_packets, *,
                   line_id=None, cost_config=None):
    """Accept resident paths or loaded records; return a small JSON-safe result.

    Cost coefficients are the v7 native contract (waiting2, riding1, unit1),
    and may be explicitly supplied from the resident contract. Credit rows
    exclude the root intervention. L_actual is the realized action product;
    the geometric/arithmetic columns remain conditional fixed-state proxies.
    """
    low, high = _events(lowrows), _events(highrows)
    if len(low) != 45 or len(high) != 45:
        raise ValueError('Expected the45 genuine v43 closures in both arms')
    if low[0]['started_at'] != high[0]['started_at']:
        raise ValueError('Paired roots do not share a physical clock')
    line_id = line_id or low[0]['line_id']
    cfg = cost_config or {}
    coeff = (cfg.get('waiting_time_value', 2.), cfg.get('in_vehicle_time_value', 1.),
             cfg.get('reward_unit_passenger_seconds', 1.))
    costs = {'low': _clock_costs(low), 'high': _clock_costs(high)}
    ledgers, checks = {}, {}
    for arm, rows, trace in (('low', low, lowtrace), ('high', high, hightrace)):
        ledgers[arm], checks[arm] = _ledger_deltas(rows, trace, coeff)
    common = sorted(set(costs['low']) & set(costs['high']))
    if not common:
        raise ValueError('No real closure clock is shared by both arms')
    first = next((clock for clock in common
        if abs(costs['high'][clock]['cost']-costs['low'][clock]['cost']) > TOL), None)

    def endpoint(clock):
        if clock is None:
            return None
        value = dict(clock=clock, after_issue_seconds=clock-float(low[0]['started_at']),
            high_minus_low_cost=costs['high'][clock]['cost']-costs['low'][clock]['cost'],
            low_closed_rows=costs['low'][clock]['closed_rows'],
            high_closed_rows=costs['high'][clock]['closed_rows'],
            low_links=costs['low'][clock]['links'], high_links=costs['high'][clock]['links'])
        if all(clock in ledgers[arm] for arm in ('low', 'high')):
            value['ledger_high_minus_low'] = {key: ledgers['high'][clock][key]-ledgers['low'][clock][key]
                for key in ('wait', 'ride', 'cost')}
        else:
            value['ledger_high_minus_low'] = None
        return value

    by_updates = {packet['updates']: packet for packet in _records(credit_packets)}
    first_credit = []
    if first is not None:
        for update in (24, 32, 80, 456):
            if update in by_updates:
                for arm in ('low', 'high'):
                    first_credit.append(_credit(by_updates[update], line_id, arm, costs[arm][first]['links']))
    last_credit = []
    if 456 in by_updates:
        for arm in ('low', 'high'):
            last_credit.append(_credit(by_updates[456], line_id, arm, costs[arm][common[-1]]['links']))
    ordinal = next((index for index, (a, b) in enumerate(zip(low, high)) if a['r'] != b['r']), None)
    return dict(line=line_id, root_clock=low[0]['started_at'],
        real_span_seconds=[rows[-1]['completed_at']-rows[0]['started_at'] for rows in (low, high)],
        zero_duration_rows=[sum(row['dt'] == 0. for row in rows) for rows in (low, high)],
        first_ordinal_reward_difference=ordinal, common_real_closure_clocks=len(common),
        first_common_cost_difference=endpoint(first), last_common_closure=endpoint(common[-1]),
        ledger_checks=checks, first_difference_credit=first_credit,
        last_common_credit_at456=last_credit,
        credit_columns=['updates', 'arm', 'links', 'L_actual', 'L_cond_geom', 'L_cond_arith',
            'mixed_path_log_weight', 'actual_weight_with_mixed_path',
            'conditional_geom_proxy_with_mixed_path', 'conditional_arith_proxy_with_mixed_path'],
        missing_300s=True, root_intervention_excluded=True,
        conditional_metrics_are_trajectory_expectations=False,
        native_gamma=1., mixed_path_weight=.9)
