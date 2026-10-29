"""Small resident audit of native paired service responses at common clocks."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import groupby
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_execution_phase_v19_20261022.analyze import (
    source_day, read, audit, load_branch, PREFIX_CHECKS,
)
from experiments.mcwm_closed_loop_state_response_v18_20261022.analyze import (
    ENDPOINTS, METRICS, MODELS, compact, inspect_pair, prediction_gain,
)

LINES = ('7X', '102X', '705X')
ARMS = ('low', 'high')
LEDGER_FIELDS = ('waiting_passengers', 'in_vehicle_passengers',
                 'accrued_wait_seconds', 'accrued_in_vehicle_seconds')
EVENT_FIELDS = ('kind', 'clock_seconds', 'line_id', 'bus_id',
    'station_id', 'onboard', 'capacity', 'effective_capacity', 'waiting',
    'eligible_waiting', 'passenger_count', 'native_detection_time',
    'observed_at', 'person_sample_interval', 'station_interval_uncertainty', 'station_at_interval_end_confirmed')


def arm_clock(rows, started_at, truth):
    checks = dict(fixed_grid=len(rows) == 301 and
        [r.get('relative_seconds') for r in rows] == list(range(301)),
        native_clock=all(r.get('time_seconds') == started_at+r.get('relative_seconds', -1)
            for r in rows),
        ledger_fields=bool(rows) and all(all(k in r for k in LEDGER_FIELDS) for r in rows))
    ready = all(checks.values())
    totals = None
    if ready:
        checks['cumulative_monotone'] = all(b[k] >= a[k] for a, b in zip(rows, rows[1:])
            for k in LEDGER_FIELDS[2:])
        checks['stocks_nonnegative'] = all(r[k] >= 0 for r in rows for k in LEDGER_FIELDS[:2])
        wait = rows[-1]['accrued_wait_seconds']-rows[0]['accrued_wait_seconds']
        ride = rows[-1]['accrued_in_vehicle_seconds']-rows[0]['accrued_in_vehicle_seconds']
        totals = dict(dt=300., wait=wait, ride=ride, cost=2*wait+ride)
        checks['fixed300_ledger_exact'] = truth.get('status') == 'complete' and all(
            totals[k] == truth.get(k) for k in METRICS)
    return dict(integrity=audit(checks), points=len(rows), totals=totals)


def paired_clock(low, high):
    """Use cumulative ledgers; post-tick stock is not a left/right rectangle."""
    series = []
    for i, (lo, hi) in enumerate(zip(low, high)):
        wait = (lo['accrued_wait_seconds']-low[0]['accrued_wait_seconds'])-(
            hi['accrued_wait_seconds']-high[0]['accrued_wait_seconds'])
        ride = (lo['accrued_in_vehicle_seconds']-low[0]['accrued_in_vehicle_seconds'])-(
            hi['accrued_in_vehicle_seconds']-high[0]['accrued_in_vehicle_seconds'])
        previous = series[-1] if series else [0, 0, 0, 0, 0, 0]
        series.append([i, lo['waiting_passengers']-hi['waiting_passengers'],
            lo['in_vehicle_passengers']-hi['in_vehicle_passengers'], wait, ride, 2*wait+ride,
            wait-previous[3], ride-previous[4]])
    stages = []
    for start, stop in ((0, 75), (75, 150), (150, 225), (225, 300)):
        w, r = series[stop][3]-series[start][3], series[stop][4]-series[start][4]
        stages.append([start, stop, w, r, 2*w+r])
    nonzero_signs = [1 if r[5] > 0 else -1 for r in series if r[5] != 0]
    summary = dict(
        field_order=('relative_seconds', 'waiting_stock_low_minus_high', 'onboard_stock_low_minus_high',
            'cumulative_wait_gain', 'cumulative_ride_gain', 'cumulative_cost_gain',
            'interval_wait_gain', 'interval_ride_gain'),
        every30_seconds=series[::30], stage_fields=('start', 'stop', 'wait_gain', 'ride_gain', 'cost_gain'),
        stages=stages, final_gain=dict(dt=0., wait=series[-1][3], ride=series[-1][4], cost=series[-1][5]),
        cumulative_cost_sign_changes=sum(a != b for a, b in zip(nonzero_signs, nonzero_signs[1:])),
        first_positive_cost_second=next((r[0] for r in series if r[5] > 0), None),
        first_negative_cost_second=next((r[0] for r in series if r[5] < 0), None),
        largest_waiting_stock_difference=max(series, key=lambda r: abs(r[1]))[:3],
        largest_onboard_stock_difference=max(series, key=lambda r: abs(r[2]))[:3],
        reconstruction_error={k:sum(r[i] for r in stages)-series[-1][j]
            for k, i, j in (('wait', 2, 3), ('ride', 3, 4), ('cost', 4, 5))})
    return summary, series


def events_profile(events, selected):
    focus = [r for r in events if r.get('bus_id') == selected['trip_id']]
    station = [r for r in events if r.get('station_id') == selected['station_id']
        and r.get('kind') == 'arrival']
    station.sort(key=lambda r:r['clock_seconds'])
    def small(rows):
        return dict(total=len(rows), shown=min(len(rows), 6),
            events=[[r.get(k) for k in EVENT_FIELDS] for r in rows[:6]])
    groups = []
    for t, indices in groupby(range(len(station)), key=lambda i:station[i]['clock_seconds']):
        members = list(indices)
        groups.append([t, len(members), [i for i in members if i < 6], sum(i >= 6 for i in members)])
    return dict(counts=dict(Counter(r.get('kind', 'unknown') for r in events)),
        controlled_trip=small(focus), shared_station_arrivals=small(station),
        shared_station_clock_groups=dict(total=len(groups), shown=min(len(groups), 6), groups=groups[:6]))


def compare_events(low, high):
    def indexed(rows):
        counts, result = Counter(), {}
        for r in rows:
            identity = tuple(r.get(k) for k in ('kind', 'line_id', 'bus_id', 'station_id'))
            key = identity+(counts[identity],)
            counts[identity] += 1
            result[key] = r
        return result
    lo, hi = indexed(low), indexed(high)
    common = lo.keys() & hi.keys()
    differences = []
    for key in common:
        a, b = lo[key], hi[key]
        fields = [k for k in EVENT_FIELDS if k not in ('kind', 'line_id', 'bus_id', 'station_id')
            and a.get(k) != b.get(k)]
        if fields:
            differences.append(dict(identity=list(key), changed=fields,
                low={k:a.get(k) for k in EVENT_FIELDS if k in a},
                high={k:b.get(k) for k in EVENT_FIELDS if k in b}))
    differences.sort(key=lambda r:(min(r['low'].get('clock_seconds', float('inf')),
                                      r['high'].get('clock_seconds', float('inf'))),
                                  tuple(str(x) for x in r['identity'])))
    compact_differences = [dict(identity=r['identity'], changed=r['changed'],
        low_values=[r['low'].get(k) for k in r['changed']], high_values=[r['high'].get(k) for k in r['changed']])
        for r in differences[:4]]
    return dict(matched=len(common), unmatched_low=len(lo.keys()-hi.keys()),
        unmatched_high=len(hi.keys()-lo.keys()), matched_changed=len(differences),
        first_four_changed_events=compact_differences)


def arm_evidence(path, anchor, old):
    value = load_branch(path)
    delta = value.get('suffix', {}).get('training_delta')
    seconds = 0. if path.name == 'low' else 7.5
    checks = dict(branch_complete=value.get('status') == 'complete',
        branch_integrity=value.get('integrity', {}).get('all_pass') is True,
        selected_exact=value.get('selected') == anchor['selected'],
        prefix_exact=value.get('prefix') == anchor['prefix'],
        prediction_exact=value.get('prediction') == anchor['prediction'],
        old_native_complete=old.get('status') == 'complete' and old.get('integrity', {}).get('all_pass') is True,
        old_native_truth_exact=all(value.get(k) == old.get(k) for k in ('truth', 'global_transition', 'execution')),
        same_fork_batch_actions=value.get('native', {}).get('other_batch_actions') is not None and
            value.get('native', {}).get('other_batch_actions') == old.get('native', {}).get('other_batch_actions'),
        zero_new_training_labels=value.get('budget', {}).get('new_target_training_labels') == 0,
        truth_not_fit=value.get('budget', {}).get('truth_labels_enter_training') == 0,
        suffix_frozen=isinstance(delta, dict) and bool(delta) and all(x == 0 for x in delta.values()),
        suffix300=value.get('suffix', {}).get('seconds') == 300,
        actual_command=value.get('execution') == dict(requested_hold_seconds=seconds, executed_hold_seconds=seconds),
        one_new_diagnostic_outcome=value.get('budget', {}).get('new_diagnostic_action_outcomes') == 1,
        fixed_zero_exact=value.get('fixed_zero_prediction') == dict(dt=0., wait=0., ride=0., cost=0.),
        native_mapping=value.get('phase_intervention', {}).get('mapping') == 'native_all_suffix',
        event_clock_exact=value.get('phase_intervention', {}).get('actor_real_clock') is True)
    for name in PREFIX_CHECKS+('old_native_truth_exact', 'service_observation_complete',
                             'service_ledger_truth_exact', 'native_mapping_unchanged'):
        checks['runner/'+name] = value.get('integrity', {}).get('checks', {}).get(name) is True
    service = value.get('service', {})
    clock = read(path/'service_clock.json', {})
    rows = clock.get('rows', [])
    clocks = arm_clock(rows, anchor['selected']['started_at'], value.get('truth', {}).get('fixed300', {}))
    checks['service_clock'] = clocks['integrity']['all_pass']
    checks['service_initial_ledger'] = bool(rows) and all(rows[0].get(k) ==
        anchor['selected']['issue_ledger'].get(k) for k in ('time_seconds',)+LEDGER_FIELDS)
    checks['service_integrity'] = service.get('integrity', {}).get('all_pass') is True
    checks['service_file_clock'] = (clock.get('start_time') == anchor['selected']['started_at'] and
        clock.get('cutoff_time') == anchor['selected']['started_at']+300)
    events_path = path/'service_events.json'
    checks['service_events_present'] = events_path.is_file()
    events = read(events_path, {}).get('events', [])
    return value, dict(integrity=audit(checks), clock=clocks,
        service={k:service.get(k) for k in ('status', 'clock_rows', 'event_counts', 'line_event_counts', 'missing_counts')},
        event_profile=events_profile(events, anchor['selected'])), rows, events


def analyze(source, root, *, preflight=False):
    root = Path(root)
    day, anchors, source_info = source_day(source)
    line_names, arm_names = (LINES[:1], ARMS[:1]) if preflight else (LINES, ARMS)
    cases, budgets, statuses, resident_series = [], [], [], {}
    for line in line_names:
        anchor = anchors.get('train/'+line+'/late')
        if anchor is None:
            cases.append(dict(line_id=line, status='selection_missing', integrity=audit({'selected':False})))
            continue
        values, evidence, clocks, events = {}, {}, {}, {}
        for arm in arm_names:
            old = read(day/'training_branches'/str(anchor['index'])/arm/'summary.json', {})
            values[arm], evidence[arm], clocks[arm], events[arm] = arm_evidence(root/'branches/native'/line/arm, anchor, old)
            statuses.append(values[arm].get('status'))
            budgets.append(values[arm].get('budget', {}))
        checks = {arm+'/all_checks':ev['integrity']['all_pass'] for arm, ev in evidence.items()}
        row = dict(line_id=line, target_id=anchor['id'],
            selected={k:anchor['selected'][k] for k in ('sequence', 'started_at', 'vehicle_id', 'trip_id', 'station_id')},
            external_seed=anchor['prefix']['external_seed'],
            branch_status={arm:v.get('status') for arm,v in values.items()}, arms=evidence,
            prediction_supported={m:prediction_gain(anchor['prediction'], m) is not None for m in MODELS})
        if not preflight and all(ev['integrity']['all_pass'] for ev in evidence.values()):
            pair = inspect_pair(dict(seed=6201, scenario='stable', line_id=line, window_name='late'),
                                anchor, values['low'], values['high'])
            checks.update({'pair/'+k:v for k,v in pair['integrity']['checks'].items()})
            row.update(truth=pair['truth'], duration=pair['duration'],
                effects=pair['effects'] if pair['integrity']['all_pass'] else None)
            if pair['integrity']['all_pass']:
                clock_result, series = paired_clock(clocks['low'], clocks['high'])
                expected = pair['truth']['fixed300']['gain']
                checks['paired_clock_reconstruction'] = expected is not None and all(
                    clock_result['final_gain'][key] == expected[i] for i,key in enumerate(METRICS))
                checks['stages_reconstruct'] = all(x == 0 for x in clock_result['reconstruction_error'].values())
                row.update(clock_decomposition=clock_result,
                    service_comparison=compare_events(events['low'], events['high']))
                resident_series[line] = series
        row['integrity'] = audit(checks)
        cases.append(row)
    expected = 1 if preflight else 6
    terminal = len(statuses) == expected and all(s not in ('missing', None) for s in statuses)
    checks = dict(source_contract=source_info['integrity']['all_pass'],
        fixed_selections=len(cases) == len(line_names) and all(c.get('status') != 'selection_missing' for c in cases),
        branches_terminal=terminal, all_service_checks=all(c['integrity']['all_pass'] for c in cases),
        diagnostic_truth_not_fit=all(b.get('truth_labels_enter_training') == 0 for b in budgets),
        suffix_budget=len(budgets) == expected and all(b.get('suffix_sim_seconds') == 300 for b in budgets))
    status = 'incomplete' if not terminal else 'complete' if all(checks.values()) else 'integrity_failed'
    value = dict(status=status, mode='native_first_low_preflight' if preflight else 'native_common_clock_service_diagnostic',
        integrity=dict(all_pass=all(checks.values()), checks=checks), source=source_info,
        budget=dict(planned_new_branch_outcomes=expected, completed_new_branch_outcomes=sum(s == 'complete' for s in statuses),
            new_main_days=0, new_training_labels=0, new_training_pairs=0,
            suffix_seconds_planned=300*expected, suffix_seconds_observed=sum(b.get('suffix_sim_seconds', 0) for b in budgets),
            prefix_replay_seconds=sum(b.get('prefix_sim_seconds', 0) for b in budgets),
            replayed_target_labels=sum(b.get('replay_target_labels', 0) for b in budgets),
            replayed_policy_updates=sum(b.get('replay_policy_updates', 0) for b in budgets),
            truth_labels_enter_training=sum(b.get('truth_labels_enter_training', 0) for b in budgets),
            preflight_is_first_of_six=True),
        branch_statuses=dict(Counter(statuses)), cases=cases,
        schema=dict(exported_event_fields=EVENT_FIELDS,
            shared_station_group_fields=('clock_seconds', 'total_members', 'shown_arrival_indices', 'unshown_members'),
            gain='low(0s) minus high(7.5s), cost=2*wait+ride',
            service_order='Same-clock observations are ties; observation_order is logging order. Person changes are discrete native sample intervals, not continuous exact boarding times.',
            event_time_sources=dict(arrival='native_bus_arrival_history', departure='native_bus_departure_history',
                service_ready='bridge.service_completion_history',
                board_alight='person_sample_interval at discrete native person observations'),
            person_station_source='Native current/just-served stop with interval uncertainty, or current TraCI stop only when isAtBusStop confirms it; otherwise station remains missing. Do not order unknown-location person changes as definite stop services.',
            cumulative_clock='Each interval is differenced from the native cumulative ledger; displayed stocks are separate instantaneous observations.'),
        decision=dict(model_change='not_selected', independent_confirmation='not_submitted',
            next='Review matched-clock service order, capacity and queue associations before selecting one issue-observable model change.'),
        limitations='Only three predeclared stable seed6201 late states. Service associations do not uniquely identify a cost mechanism; future service events remain diagnostic truth, never fitting inputs.')
    if not preflight:
        selected = [c for c in cases if 'truth' in c]
        value['missing'] = dict(selection=sum(c.get('status') == 'selection_missing' for c in cases),
            ungated_pairs=sum('truth' not in c and c.get('status') != 'selection_missing' for c in cases),
            branch_outcomes=sum(s == 'missing' for s in statuses),
            endpoints={e:sum(c['truth'][e]['gain'] is None for c in selected) for e in ENDPOINTS})
        value['unsupported'] = {m:sum(not c['prediction_supported'][m] for c in cases
            if 'prediction_supported' in c) for m in MODELS}
        value['duration_counts'] = dict(Counter(c['duration'] for c in selected))
    return compact(value), resident_series


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    while True:
        value, series = analyze(args.source, args.root, preflight=args.preflight)
        if value['status'] == 'integrity_failed' or any(k in value['branch_statuses'] for k in ('failed', 'integrity_failed')):
            value['status'] = 'integrity_failed'
            break
        if value['status'] == 'complete' or not args.wait:
            break
        time.sleep(20)
    data = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'
    if len(data.encode()) > (12000 if args.preflight else 30000):
        raise RuntimeError('Reduce resident service summary before export')
    if value['integrity']['all_pass'] and series:
        with (args.out.parent/'paired_clock_resident.json').open('x') as f:
            json.dump(dict(series=series, field_order=value['cases'][0]['clock_decomposition']['field_order']), f,
                      ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            f.write('\n')
    with args.out.open('x') as f:
        f.write(data)
    print(json.dumps(dict(status=value['status'], integrity=value['integrity'], bytes=len(data.encode()))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        path = args.root/'analysis_failure.json'
        if not path.exists():
            with path.open('x') as f:
                json.dump(dict(status='failed', output=str(args.out),
                    type=type(error).__name__, message=str(error)), f, ensure_ascii=False)
                f.write('\n')
        raise


if __name__ == '__main__':
    main()
