"""Observe common native stop visits; keep clocks, memberships and legacy counters distinct."""
from collections import Counter, defaultdict

from experiments.mcwm_closed_loop_service_segments_v74_20261027.core import intervals, stats


def physical_set(row, field):
    if row is None or row.get('observed_at') != row.get('clock_seconds'):
        return None
    value = row.get(field)
    return None if value is None else set(value)


def physical_time(row):
    return row['clock_seconds'] if row.get('observed_at') == row.get('clock_seconds') else None


def visits(services):
    """Construct full-day visits before applying any analysis time window."""
    rows = []
    for raw in services:
        if raw.get('record') == 'service':
            row = dict(raw)
            row['_ordinal'] = len(rows)
            rows.append(row)
    segments, audit = intervals(rows)
    ready = defaultdict(list)
    for row in rows:
        if row['kind'] == 'service_ready':
            ready[row['bus_id'], row['line_id'], row['station_id']].append(row)
    result = {}
    for key, value in segments.items():
        if value['kind'] != 'dwell':
            continue
        visit_key = (key[0], key[1], key[3], key[5])
        candidates = [row for row in ready.get(visit_key[:3], ())
            if value['start']['clock_seconds'] <= row['clock_seconds'] <= value['end']['clock_seconds']]
        result[visit_key] = dict(value, ready=candidates[0] if len(candidates) == 1 else None,
            ready_count=len(candidates))
    return result, audit


def anchor_issues(issues, dwell):
    """trip_id is native SUMO identity; vehicle_id is a reusable fleet identity."""
    grouped, assigned = defaultdict(list), defaultdict(list)
    for key, visit in dwell.items():
        grouped[key[:3]].append((key, visit))
    audit = Counter()
    for row in issues:
        audit['logged_issues'] += 1
        candidates = [key for key, visit in grouped.get(
            (row['trip_id'], row['line_id'], row['station_id']), ())
            if visit['start']['clock_seconds'] <= row['started_at'] <= visit['end']['clock_seconds']]
        if len(candidates) != 1:
            audit['outside_visit' if not candidates else 'ambiguous_visit'] += 1
        elif physical_time(dwell[candidates[0]]['start']) is None or physical_time(dwell[candidates[0]]['end']) is None:
            audit['lagged_endpoint_issues'] += 1
        else:
            assigned[candidates[0]].append(dict(clock=row['started_at'], seconds=(row['a'][0]+1)*30))
    unique = {key: values[0] for key, values in assigned.items() if len(values) == 1}
    audit['unique_issue_visits'] = len(unique)
    audit['duplicate_issue_visits'] = sum(len(values) > 1 for values in assigned.values())
    audit['issues_in_duplicate_visits'] = sum(len(values) for values in assigned.values() if len(values) > 1)
    audit['visits_without_issue'] = len(dwell)-len(assigned)
    return unique, dict(audit)


def overlap(visit, lo, hi):
    return max(0., min(visit['end']['clock_seconds'], hi)-max(visit['start']['clock_seconds'], lo))


def timing(visit, command, lo, hi):
    start, end, ready = visit['start'], visit['end'], visit['ready']
    arrival, departure = physical_time(start), physical_time(end)
    both = arrival is not None and departure is not None
    logical_ready = None if ready is None else ready['clock_seconds']
    return dict(holding=None if command is None else command['seconds'],
        arrival=arrival, departure=departure, dwell=visit['duration'] if both else None,
        ready=logical_ready,
        ready_minus_arrival=None if logical_ready is None or arrival is None else logical_ready-arrival,
        departure_minus_ready=None if logical_ready is None or departure is None else departure-logical_ready,
        window_overlap=overlap(visit, lo, hi) if both else None)


def difference(left, right):
    return None if left is None or right is None else left-right


def measured(row, field):
    if row.get('observed_at') != row.get('clock_seconds'):
        return None, 'lagged'
    value = row.get(field)
    return (None, 'missing') if value is None else (value, 'observed')


def membership_changes(visit):
    before = physical_set(visit['start'], 'onboard_passenger_ids')
    after = physical_set(visit['end'], 'onboard_passenger_ids')
    if before is None or after is None:
        return None
    return dict(added=len(after-before), removed=len(before-after))


def risk(row):
    eligible, _ = measured(row, 'eligible_waiting')
    free, _ = measured(row, 'free_capacity')
    return None if eligible is None or free is None else eligible > 0 and free == 0


def simple_statistics(values):
    result = stats(values)
    return {key: result[key] for key in ('count', 'sum', 'mean', 'min', 'max')}


def summary(records, *, detailed=False):
    delta_fields = ('holding', 'arrival', 'ready', 'departure', 'dwell',
                    'ready_minus_arrival', 'departure_minus_ready', 'window_overlap')
    result = dict(visits=len(records), timing_delta={field: simple_statistics(
        [row['delta'][field] for row in records]) for field in delta_fields})
    observations = {}
    for endpoint in ('arrival', 'departure'):
        fields = {}
        for field in ('capacity', 'free_capacity', 'eligible_waiting', 'native_board_count',
                      'native_alight_count', 'native_stranded_count'):
            arms, deltas = {}, []
            for arm in ('fresh', 'baseline'):
                values, status = [], Counter()
                for record in records:
                    value, label = measured(record[arm]['start' if endpoint == 'arrival' else 'end'], field)
                    status[label] += 1
                    if value is not None:
                        values.append(value)
                calculated = stats(values)
                arms[arm] = dict(observed=status['observed'], missing=status['missing'],
                    lagged=status['lagged'], sum=calculated['sum'], mean=calculated['mean'])
            for record in records:
                a, _ = measured(record['fresh']['start' if endpoint == 'arrival' else 'end'], field)
                b, _ = measured(record['baseline']['start' if endpoint == 'arrival' else 'end'], field)
                deltas.append(difference(a, b))
            fields[field] = dict(**arms, paired_delta=simple_statistics(deltas))
        observations[endpoint] = fields
    inventory, risks = {}, {}
    for arm in ('fresh', 'baseline'):
        observed = [membership_changes(row[arm]) for row in records]
        valid = [value for value in observed if value is not None]
        inventory[arm] = dict(observed_visits=len(valid), missing_or_lagged=len(records)-len(valid),
            added_event_memberships=sum(value['added'] for value in valid) if valid else None,
            removed_event_memberships=sum(value['removed'] for value in valid) if valid else None)
        risk_values = [risk(row[arm]['end']) for row in records]
        risks[arm] = dict(scorable=sum(value is not None for value in risk_values),
            risk_events=sum(value is True for value in risk_values),
            missing_or_lagged=sum(value is None for value in risk_values))
    transitions = Counter()
    for row in records:
        a, b = risk(row['fresh']['end']), risk(row['baseline']['end'])
        if a is not None and b is not None:
            transitions['paired_scorable'] += 1
            transitions['both_risk'] += a and b
            transitions['fresh_only_risk'] += a and not b
            transitions['baseline_only_risk'] += b and not a
    changes = [(membership_changes(row['fresh']),membership_changes(row['baseline'])) for row in records]
    paired = [(a,b) for a,b in changes if a is not None and b is not None]
    inventory['paired_delta'] = dict(observed_visits=len(paired),
        added=simple_statistics([a['added']-b['added'] for a,b in paired]),
        removed=simple_statistics([a['removed']-b['removed'] for a,b in paired]))
    risks['paired'] = dict(transitions)
    result.update(inventory_changes=inventory, departure_risk=risks)
    if detailed:
        result['measurements'] = observations
    return result


GROUP_COLUMNS = ['identity', 'visits', 'holding_pairs', 'holding_delta_sum', 'holding_delta_mean',
    'arrival_delta_mean', 'ready_pairs', 'ready_delta_mean', 'departure_delta_mean', 'dwell_delta_sum',
    'ready_minus_arrival_delta_mean', 'departure_minus_ready_delta_mean',
    'fresh_risk_events', 'baseline_risk_events', 'paired_risk_visits', 'sum_abs_departure_shift']


def compact_group(identity, records):
    s = summary(records)
    delta = s['timing_delta']
    return [identity, len(records), delta['holding']['count'], delta['holding']['sum'],
        delta['holding']['mean'], delta['arrival']['mean'], delta['ready']['count'],
        delta['ready']['mean'], delta['departure']['mean'], delta['dwell']['sum'],
        delta['ready_minus_arrival']['mean'], delta['departure_minus_ready']['mean'],
        s['departure_risk']['fresh']['risk_events'], s['departure_risk']['baseline']['risk_events'],
        s['departure_risk']['paired'].get('paired_scorable', 0),
        sum(abs(row['delta']['departure']) for row in records if row['delta']['departure'] is not None)
            if delta['departure']['count'] else None]


def service_order(records, lo, hi):
    by_station = defaultdict(list)
    excluded = lagged = 0
    for row in records:
        if any(physical_time(row[arm]['end']) is None for arm in ('fresh','baseline')):
            lagged += 1
            continue
        clocks = [row[arm]['end']['clock_seconds'] for arm in ('fresh', 'baseline')]
        if all(lo <= clock < hi for clock in clocks):
            by_station[row['key'][2]].append(row)
        else:
            excluded += 1
    counts, headways, examples = Counter(), [], []
    for station, rows in sorted(by_station.items()):
        ordered = sorted(rows, key=lambda row: (row['baseline']['end']['clock_seconds'], row['key']))
        for first, second in zip(ordered, ordered[1:]):
            b0, b1 = (row['baseline']['end']['clock_seconds'] for row in (first, second))
            f0, f1 = (row['fresh']['end']['clock_seconds'] for row in (first, second))
            counts['adjacent_common_pairs'] += 1
            if b0 == b1 or f0 == f1:
                counts['tied_pairs'] += 1
            elif f1 < f0:
                counts['strict_adjacent_inversions'] += 1
                if len(examples) < 5:
                    examples.append(dict(station=station,
                        baseline_bus_order=[first['key'][0], second['key'][0]],
                        lines=[first['key'][1], second['key'][1]],
                        baseline_departures=[b0,b1], fresh_departures_of_same_trips=[f0,f1]))
            else:
                counts['same_order_pairs'] += 1
                headways.append((f1-f0)-(b1-b0))
    return dict(counts=dict(counts), excluded_boundary_visits=excluded, excluded_lagged_departure_visits=lagged,
        same_order_headway_delta=simple_statistics(headways), examples=examples,
        scope='Adjacent baseline-ordered common visits at one stop; both departures inside window. Ties are not inversions.')


def direct_opportunities(records):
    events = {'fresh_board_baseline_wait': [], 'baseline_board_fresh_wait': []}
    coverage = Counter()
    for row in records:
        a, b = row['fresh'], row['baseline']
        qa = physical_set(a['start'], 'queued_passenger_ids')
        qb = physical_set(b['start'], 'queued_passenger_ids')
        coverage['selected_visits'] += 1
        if qa is None or qb is None:
            coverage['missing_or_lagged_arrival_queue'] += 1
            continue
        common = qa & qb
        for name, boarded, waiting in [
            ('fresh_board_baseline_wait', physical_set(a['end'], 'onboard_passenger_ids'),
                physical_set(b['end'], 'queued_passenger_ids')),
            ('baseline_board_fresh_wait', physical_set(b['end'], 'onboard_passenger_ids'),
                physical_set(a['end'], 'queued_passenger_ids'))]:
            if boarded is None or waiting is None:
                coverage[name+'_missing_or_lagged_departure'] += 1
                continue
            coverage[name+'_scorable_visits'] += 1
            people = common & boarded & waiting
            if people:
                events[name].append((max(a['end']['clock_seconds'], b['end']['clock_seconds']), row, people))
    result = {}
    for name, values in events.items():
        seen, examples, first_events = set(), [], 0
        for clock, row, people in sorted(values, key=lambda value: (value[0],value[1]['key'])):
            pairs = {(person, row['key'][2]) for person in people}
            new = pairs-seen
            seen.update(pairs)
            if new:
                first_events += 1
                if len(examples) < 5:
                    examples.append(dict(bus=row['key'][0],line=row['key'][1],station=row['key'][2],
                        fresh_arrival=row['fresh']['start']['clock_seconds'],
                        baseline_arrival=row['baseline']['start']['clock_seconds'],
                        fresh_departure=row['fresh']['end']['clock_seconds'],
                        baseline_departure=row['baseline']['end']['clock_seconds'],
                        observed_count=len(people),new_unique_person_stop_count=len(new),
                        witness_clock=clock,boundary=row['boundary'],legacy_compatibility_available=False))
        result[name] = dict(observed_events=len(values), person_stop_event_memberships=sum(len(x[2]) for x in values),
            unique_person_stop_pairs=len(seen), first_witness_events=first_events, examples=examples)
    return dict(coverage=dict(coverage), directions=result,
        scope='Common arrival-queue IDs, then onboard in one arm and still queued in the other; not a claim that all queued people were eligible for this line. Each direction retains its earliest max(departure clocks) witness per person-stop; no IDs exported.')


def compare_services(fresh_services, baseline_services, fresh_issues, baseline_issues, lo=12000, hi=16200):
    left, laudit = visits(fresh_services)
    right, raudit = visits(baseline_services)
    lactions, lactions_audit = anchor_issues(fresh_issues, left)
    ractions, ractions_audit = anchor_issues(baseline_issues, right)
    common = left.keys() & right.keys()
    records, bounds = [], Counter()
    for key in sorted(common):
        a, b = left[key], right[key]
        if not (overlap(a,lo,hi) > 0 or overlap(b,lo,hi) > 0):
            continue
        av, bv = timing(a,lactions.get(key),lo,hi), timing(b,ractions.get(key),lo,hi)
        crosses_lo = any(v['start']['clock_seconds'] < lo < v['end']['clock_seconds'] for v in (a,b))
        crosses_hi = any(v['start']['clock_seconds'] < hi < v['end']['clock_seconds'] for v in (a,b))
        one_arm = (overlap(a,lo,hi) > 0) != (overlap(b,lo,hi) > 0)
        boundary = crosses_lo or crosses_hi or one_arm
        bounds['crosses_lower_boundary'] += crosses_lo
        bounds['crosses_upper_boundary'] += crosses_hi
        bounds['only_one_arm_overlaps'] += one_arm
        bounds['internal_both_arms'] += not boundary
        records.append(dict(key=key,fresh=a,baseline=b,boundary=boundary,
            delta={field:difference(av[field],bv[field]) for field in av}))
    by_line, by_station = defaultdict(list), defaultdict(list)
    for row in records:
        by_line[row['key'][1]].append(row)
        by_station[row['key'][2]].append(row)
    line_rows = [compact_group(key,rows) for key,rows in sorted(by_line.items())]
    station_rows = [compact_group(key,rows) for key,rows in by_station.items()]
    station_rows.sort(key=lambda row:(row[-1] is None,-row[-1] if row[-1] is not None else 0,row[0]))
    unmatched = {arm:sum(overlap(value,lo,hi) > 0 for key,value in table.items() if key not in common)
                 for arm,table in [('fresh',left),('baseline',right)]}
    ready_audit = {}
    for arm in ('fresh','baseline'):
        ready_audit[arm] = dict(missing=sum(row[arm]['ready_count'] == 0 for row in records),
            multiple=sum(row[arm]['ready_count'] > 1 for row in records),
            unique=sum(row[arm]['ready_count'] == 1 for row in records),
            lagged_unique=sum(row[arm]['ready'] is not None and row[arm]['ready']['observed_at']
                != row[arm]['ready']['clock_seconds'] for row in records))
    return dict(window=[lo,hi],alignment=dict(common_full_day_visits=len(common), selected_common_visits=len(records),
        unmatched_window_visits=unmatched,boundaries=dict(bounds),physical_interval_audit=dict(fresh=laudit,baseline=raudit),
        lagged_endpoint_visits={arm:sum(physical_time(row[arm]['start']) is None or
            physical_time(row[arm]['end']) is None for row in records) for arm in ('fresh','baseline')},
        issue_alignment=dict(fresh=lactions_audit,baseline=ractions_audit),ready=ready_audit),
        all=summary(records,detailed=True),by_line=dict(columns=GROUP_COLUMNS,rows=line_rows),
        top_stations=dict(rank_by='sum_abs_departure_shift',columns=GROUP_COLUMNS,rows=station_rows[:8]),
        service_order=service_order(records,lo,hi),direct_opportunities=direct_opportunities(records),
        limitations=['Membership changes are endpoint net observations, not complete boarding/alighting histories.',
            'Capacity and eligible queue scalars are used only at the exact observed event tick; legacy nulls remain missing.',
            'Ready clocks describe recorded logical service completion, not an observed end of physical boarding.',
            'Clock deltas compare the same service identity in different trajectories, not a same-state causal effect.'])
