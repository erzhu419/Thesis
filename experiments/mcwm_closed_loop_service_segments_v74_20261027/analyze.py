"""One resident-only service readout; no simulator, learner or checkpoint import."""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path

from core import load_services, intervals, physical_inventory, stats

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
V70 = EXPERIMENTS/'mcwm_closed_loop_recent30_mixed_cql_v70_20261026'
V71 = EXPERIMENTS/'mcwm_closed_loop_tail_policy_exchange_v71_20261026'
V73 = EXPERIMENTS/'mcwm_closed_loop_decoupled_feedback_v73_20261026'
DENOMINATOR = 15079
MAX_BYTES = 48000


def read(path):
    return json.loads(path.read_text())


def run_dir(root, seed, arm):
    return root/f'runs/executed_cql/development/seed{seed}/stable/{arm}'


def source_paths(seed):
    fresh = run_dir(V73, seed, 'fresh')
    baseline = run_dir(V73, seed, 'no_fresh')
    if seed == 6201:
        gate = read(V73/'regression_gate.json')
        if gate.get('all_pass') is not True:
            raise ValueError('The saved complete v70 baseline regression did not pass')
        return fresh, [fresh/'service_trace.jsonl'], [baseline/'service_trace.jsonl'], baseline/'prefix_trace.jsonl'
    parent = V71/f'runs/seed{seed}/no_fresh'
    tail = parent/'suffixes/high'
    prefix, suffix = read(parent/'prefix_summary.json'), read(tail/'summary.json')
    if not (prefix['status'] == suffix['status'] == 'complete'
            and prefix['integrity']['all_pass'] and suffix['integrity']['all_pass']
            and all(suffix['diagonal_reference'].get(k) is True for k in
                    ('ledger_exact', 'metrics_exact', 'full_remaining_trace_exact'))):
        raise ValueError('Saved v71 original no_fresh path has no exact diagonal provenance')
    return fresh, [fresh/'service_trace.jsonl'], [parent/'service_trace.jsonl', tail/'service_trace.jsonl'], run_dir(V70, seed, 'no_fresh')/'prefix_trace.jsonl'


def event_key(row):
    return tuple(row[k] for k in ('bus_id', 'line_id', 'station_id', 'kind', 'occurrence'))


def inventory_comparison(left, right):
    counters = Counter()
    jaccard, delta = [], []
    for key in left.keys() & right.keys():
        if key[3] != 'departure':
            continue
        a, b = physical_inventory(left[key]), physical_inventory(right[key])
        counters['matched_departures'] += 1
        if a is None or b is None:
            counters['unscorable_missing_or_lagged'] += 1
            continue
        counters['scorable_same_tick_departures'] += 1
        counters['identical_membership'] += a == b
        counters['different_membership'] += a != b
        counters['fresh_only_memberships'] += len(a-b)
        counters['baseline_only_memberships'] += len(b-a)
        delta.append(len(a)-len(b))
        jaccard.append(len(a & b)/len(a | b) if a | b else 1.)
    return dict(counts=dict(counters), onboard_count_delta=stats(delta), jaccard=stats(jaccard),
                scope='Same service identity at each arm own clock; memberships are event sums, not unique people.')


def issue_actions(path, segments):
    """Anchor a logged command to its native trip's unique physical stop visit."""
    visits = defaultdict(list)
    for key, segment in segments.items():
        if segment['kind'] == 'dwell':
            visits[key[0], key[1], key[3]].append((key, segment))
    assigned = defaultdict(list)
    counts = Counter()
    all_seconds = []
    for raw in path.open():
        row = json.loads(raw)
        command = dict(clock=row['started_at'], seconds=(row['a'][0]+1)*30)
        all_seconds.append(command['seconds'])
        # vehicle_id is a reused physical fleet ID; trip_id is the native bus ID.
        base = tuple(row[k] for k in ('trip_id', 'line_id', 'station_id'))
        matches = [key for key, segment in visits.get(base, ())
                   if segment['start']['clock_seconds'] <= command['clock']
                   <= segment['end']['clock_seconds']]
        if not matches:
            counts['outside_physical_dwell_issues'] += 1
        elif len(matches) != 1:
            counts['ambiguous_physical_dwell_issues'] += 1
        else:
            assigned[matches[0]].append(command)
    result = {key: values[0] for key, values in assigned.items() if len(values) == 1}
    audit = dict(total_logged_issues=len(all_seconds), physical_dwell_visits=sum(map(len, visits.values())),
                 single_issue_anchored_visits=len(result),
                 physical_dwell_visits_without_issue=sum(map(len, visits.values()))-len(assigned),
                 duplicate_issue_visits=sum(len(values) > 1 for values in assigned.values()),
                 issues_in_duplicate_visits=sum(len(values) for values in assigned.values() if len(values) > 1),
                 outside_physical_dwell_issues=counts['outside_physical_dwell_issues'],
                 ambiguous_physical_dwell_issues=counts['ambiguous_physical_dwell_issues'])
    totals = dict(count=len(all_seconds), sum_seconds=sum(all_seconds))
    return result, audit, totals


def holding_comparison(fresh_path, baseline_path, fresh_segments, baseline_segments):
    left, laudit, ltotals = issue_actions(fresh_path, fresh_segments)
    right, raudit, rtotals = issue_actions(baseline_path, baseline_segments)
    groups = defaultdict(list)
    all_deltas, clock_deltas = [], []
    for key in left.keys() & right.keys():
        a, b = left[key], right[key]
        delta = a['seconds']-b['seconds']
        period = 'post18300' if min(a['clock'], b['clock']) >= 18300 else 'pre18300'
        if min(a['clock'], b['clock']) < 18300 <= max(a['clock'], b['clock']):
            period = 'opposite_sides_of18300'
        groups[key[1], period].append(delta)
        all_deltas.append(delta)
        clock_deltas.append(a['clock']-b['clock'])
    return dict(matched=len(all_deltas), unmatched_fresh=len(left.keys()-right.keys()),
                unmatched_baseline=len(right.keys()-left.keys()), executed_seconds_delta=stats(all_deltas),
                issue_clock_delta=stats(clock_deltas), columns=['line', 'period', 'n', 'sum_delta', 'mean_delta', 'max_abs_delta'],
                rows=[[line, period, len(values), sum(values), sum(values)/len(values), max(map(abs, values))]
                      for (line, period), values in sorted(groups.items())],
                alignment_audit=dict(fresh=laudit, baseline=raudit),
                all_commands=dict(fresh=ltotals, baseline=rtotals,
                    sum_difference_seconds=ltotals['sum_seconds']-rtotals['sum_seconds']),
                alignment_scope='Native trip_id,line,station and unique physical arrival<=issue<=departure visit; '
                    'cross-arm pairs use that dwell identity. Missing or duplicate-issue visits are not zero-filled.',
                action_conversion='Executed normalized a maps to(a+1)*30 seconds under frozen max_hold60; not physical dwell.')


def segment_comparison(left, right):
    common = left.keys() & right.keys()
    by_group = defaultdict(list)
    by_line = defaultdict(list)
    details, rider_counts = [], Counter()
    rider_full = rider_tail = 0.
    for key in sorted(common):
        a, b = left[key], right[key]
        delta = a['duration']-b['duration']
        tail = a['tail_overlap']-b['tail_overlap']
        row = dict(bus=key[0], line=key[1], kind=key[2], start_station=key[3], end_station=key[4], occurrence=key[5],
                   duration_delta=delta, tail_overlap_delta=tail,
                   start_delta=a['start']['clock_seconds']-b['start']['clock_seconds'],
                   end_delta=a['end']['clock_seconds']-b['end']['clock_seconds'],
                   fresh_start_end=[a['start']['clock_seconds'], a['end']['clock_seconds']],
                   baseline_start_end=[b['start']['clock_seconds'], b['end']['clock_seconds']],
                   crosses18300=a['crosses_boundary'] or b['crosses_boundary'])
        periods = ['all']
        if a['tail_overlap'] > 0 or b['tail_overlap'] > 0:
            periods.append('tail_intersect')
            details.append(row)
        if row['crosses18300']:
            periods.append('cross18300')
        for period in periods:
            by_group[key[2], period].append(row)
        if 'tail_intersect' in periods:
            by_line[key[1], key[2]].append(row)
        if key[2] == 'run':
            inventories = [physical_inventory(x) for x in (a['start'], a['end'], b['start'], b['end'])]
            if any(x is None for x in inventories):
                rider_counts['missing_or_lagged_endpoint_segments'] += 1
            else:
                shared = set.intersection(*inventories)
                rider_counts['four_exact_tick_endpoint_segments'] += 1
                rider_counts['common_through_rider_segment_memberships'] += len(shared)
                rider_full += len(shared)*delta
                rider_tail += len(shared)*tail
    grouped = {kind+'/'+period: dict(n=len(rows), duration_delta=stats([x['duration_delta'] for x in rows]),
                start_clock_delta=stats([x['start_delta'] for x in rows]),
                end_clock_delta=stats([x['end_delta'] for x in rows]),
                tail_overlap_delta=stats([x['tail_overlap_delta'] for x in rows]))
               for (kind, period), rows in sorted(by_group.items())}
    tops = {kind: sorted([x for x in details if x['kind'] == kind], key=lambda x: (-x['duration_delta'], x['bus']))[:5]
            for kind in ('run', 'dwell')}
    return dict(matched_segments=len(common), unmatched_fresh=len(left.keys()-right.keys()),
                unmatched_baseline=len(right.keys()-left.keys()), groups=grouped,
                line_columns=['line', 'kind', 'n_tail_intersect', 'duration_delta_sum', 'tail_overlap_delta_sum', 'duration_delta_mean', 'arrival_delta_mean'],
                line_rows=[[line, kind, len(rows), sum(x['duration_delta'] for x in rows),
                    sum(x['tail_overlap_delta'] for x in rows), sum(x['duration_delta'] for x in rows)/len(rows),
                    sum(x['end_delta'] for x in rows)/len(rows)] for (line, kind), rows in sorted(by_line.items())],
                largest_positive_duration_differences_in_tail=tops,
                common_rider_running_time_proxy=dict(counts=dict(rider_counts), full_person_seconds=rider_full,
                    tail_person_seconds=rider_tail, full_seconds_per_cohort=rider_full/DENOMINATOR,
                    tail_seconds_per_cohort=rider_tail/DENOMINATOR,
                    scope='Shared onboard IDs at all four same-tick endpoints only. Observed running-segment proxy, not a complete passenger ledger decomposition; dwell and changed riders excluded.'))


def service_opportunities(rows):
    counts = Counter()
    groups = defaultdict(Counter)
    for row in rows:
        if row['kind'] != 'departure' or row['clock_seconds'] < 16200:
            continue
        period = 'pre18300' if row['clock_seconds'] < 18300 else 'post18300'
        group = groups[row['line_id'], period]
        group['departures'] += 1
        if physical_inventory(row) is None:
            group['missing_or_lagged'] += 1
            continue
        if row.get('eligible_waiting') is None or row.get('free_capacity') is None:
            group['missing_capacity_or_queue'] += 1
            continue
        group['scorable'] += 1
        group['full_with_compatible_queue'] += row['free_capacity'] <= 0 and row['eligible_waiting'] > 0
        stranded = row.get('native_stranded_count')
        if stranded is None:
            group['missing_stranded_observations'] += 1
        else:
            group['stranded_observations'] += 1
            group['native_stranded_event_sum'] += stranded
    for group in groups.values():
        counts.update(group)
    return dict(total=dict(counts), columns=['line', 'period', 'departures', 'scorable', 'full_with_compatible_queue', 'native_stranded_event_sum', 'missing_or_lagged', 'missing_stranded_observations'],
                rows=[[line, period, *[group[k] for k in ('departures', 'scorable', 'full_with_compatible_queue')],
                       group['native_stranded_event_sum'] if group['stranded_observations'] else None,
                       group['missing_or_lagged'], group['missing_stranded_observations']]
                      for (line, period), group in sorted(groups.items())])


def analyze():
    development = read(V73/'results/development.json')
    cells = []
    for seed in (6201, 6203):
        fresh_dir, fresh_paths, baseline_paths, baseline_issues = source_paths(seed)
        left, laudit = load_services(fresh_paths)
        right, raudit = load_services(baseline_paths)
        li, lsegments = intervals(left)
        ri, rsegments = intervals(right)
        cost = next(c for c in development['cells'] if c['identity'] == seed and c['scenario'] == 'stable')
        cells.append(dict(seed=seed, baseline_provenance='v73 exact full no_fresh regression' if seed == 6201 else 'v71 exact v70 original no_fresh diagonal',
            source_paths=dict(fresh=[str(p) for p in fresh_paths], baseline=[str(p) for p in baseline_paths]),
            observed_service_audits=dict(fresh=laudit, baseline=raudit), interval_audits=dict(fresh=lsegments, baseline=rsegments),
            passenger_ledger_cost_differences={period: row['matched'] for period, row in cost['periods'].items()},
            segments=segment_comparison(li, ri), departure_memberships=inventory_comparison(
                {event_key(x): x for x in left}, {event_key(x): x for x in right}),
            holding=holding_comparison(fresh_dir/'prefix_trace.jsonl', baseline_issues, li, ri),
            service_opportunities=dict(fresh=service_opportunities(left), baseline=service_opportunities(right))))
    return dict(status='complete', cells=cells, native_runs=0, training_updates=0,
                snapshots_or_truth_refit=False, algorithm_modified=False,
                interpretation='Frozen v73 remains an old-gate failure and a four-cell full-day development gain. This readout selects no new model, window or risk quota.',
                limitations=['All service inventories belong to observed_at; lagged or missing inventories are unscored.',
                    '6203 baseline inventory detail starts16200; clocks cover the full day.',
                    'Matched services occur at each arm own times, not identical physical states.',
                    'Running-time proxies do not reconstruct individual whole journeys or add up to full ledger cost.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = analyze()
    raw = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    if len(raw.encode()) > MAX_BYTES:
        raise ValueError('Readout exceeds the declared48KB export budget')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        handle.write(raw+'\n')
    print(json.dumps(dict(status='complete', bytes=len(raw.encode()), native_runs=0, training_updates=0)))


if __name__ == '__main__':
    main()
