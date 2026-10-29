"""Read-only physical service intervals; passenger inventories retain their own clocks."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median


def stats(values):
    values = sorted(float(value) for value in values if value is not None)
    if not values:
        return dict(count=0, sum=None, mean=None, median=None, p90=None, min=None, max=None)
    position = .9*(len(values)-1)
    low, high = math.floor(position), math.ceil(position)
    return dict(count=len(values), sum=math.fsum(values), mean=mean(values), median=median(values),
        p90=values[low]+(values[high]-values[low])*(position-low), min=values[0], max=values[-1])


def physical_inventory(row):
    """Return onboard IDs only at the exact physical event tick; missing stays missing."""
    ids = row.get('onboard_passenger_ids')
    if ids is None or row.get('observed_at') != row.get('clock_seconds'):
        return None
    return set(ids)


def load_services(paths):
    rows, occurrence = [], Counter()
    by_kind, detail, missing, lags = Counter(), Counter(), Counter(), defaultdict(list)
    orders = []
    for path in paths:
        with Path(path).open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get('record') != 'service':
                    continue
                kind = row['kind']
                key = (row['bus_id'], row['line_id'], row['station_id'], kind)
                row['occurrence'] = occurrence[key]
                row['_ordinal'] = len(rows)
                occurrence[key] += 1
                rows.append(row)
                by_kind[kind] += 1
                detail[kind] += 'onboard_passenger_ids' in row
                missing[kind] += row.get('onboard_passenger_ids') is None
                lags[kind].append(float(row['observed_at'])-float(row['clock_seconds']))
                orders.append(row['observation_order'])
    lag_summary = {kind: dict(stats(values), nonzero=sum(value != 0 for value in values),
        negative=sum(value < 0 for value in values), max_abs=max(map(abs, values)))
        for kind, values in sorted(lags.items())}
    audit = dict(service_events=len(rows), by_kind=dict(by_kind), detail_rows_by_kind=dict(detail),
        missing_inventory_by_kind=dict(missing), observed_lag_by_kind=lag_summary,
        physical_inventory_rows=sum(physical_inventory(row) is not None for row in rows),
        observation_order_continuous=all(right == left+1 for left, right in zip(orders, orders[1:])),
        observation_order_first=orders[0] if orders else None,
        observation_order_last=orders[-1] if orders else None,
        occurrence_groups=len(occurrence))
    return rows, audit


def intervals(rows):
    """Pair adjacent native arrivals/departures only; do not fabricate a terminal event."""
    buses = defaultdict(list)
    for row in rows:
        if row['kind'] in ('arrival', 'departure'):
            buses[row['bus_id'], row['line_id']].append(row)
    result, occurrences = {}, Counter()
    audit = dict(buses=len(buses), physical_endpoints=0, intervals=0,
        dwell=0, run=0, zero_duration=0, negative_duration=0,
        invalid_adjacency=0, initial_departures_without_arrival=0,
        final_arrivals_without_departure=0, final_departures_without_next_arrival=0,
        single_endpoint_buses=0, unused_endpoints=0, invalid_examples=[])
    for (bus, line), events in buses.items():
        events.sort(key=lambda row: (float(row['clock_seconds']), row['observation_order'], row['_ordinal']))
        audit['physical_endpoints'] += len(events)
        audit['single_endpoint_buses'] += len(events) == 1
        audit['initial_departures_without_arrival'] += events[0]['kind'] == 'departure'
        audit['final_arrivals_without_departure'] += events[-1]['kind'] == 'arrival'
        audit['final_departures_without_next_arrival'] += events[-1]['kind'] == 'departure'
        used = set()
        for start, end in zip(events, events[1:]):
            kind = None
            if start['kind'] == 'arrival' and end['kind'] == 'departure' and start['station_id'] == end['station_id']:
                kind = 'dwell'
            elif start['kind'] == 'departure' and end['kind'] == 'arrival':
                kind = 'run'
            if kind is None:
                audit['invalid_adjacency'] += 1
                continue
            base = (bus, line, kind, start['station_id'], end['station_id'])
            key = (*base, occurrences[base])
            occurrences[base] += 1
            lo, hi = float(start['clock_seconds']), float(end['clock_seconds'])
            duration = hi-lo
            if duration <= 0:
                audit['zero_duration' if duration == 0 else 'negative_duration'] += 1
                if len(audit['invalid_examples']) < 3:
                    audit['invalid_examples'].append(dict(key=key, start=lo, end=hi, duration=duration))
                continue
            result[key] = dict(bus_id=bus, line_id=line, kind=kind,
                start_station_id=start['station_id'], end_station_id=end['station_id'],
                occurrence=key[-1], start=start, end=end, duration=duration,
                tail_overlap=max(0., min(hi, 24000.)-max(lo, 18300.)),
                crosses_boundary=lo < 18300. < hi)
            audit[kind] += 1
            used.update((start['_ordinal'], end['_ordinal']))
        audit['unused_endpoints'] += sum(row['_ordinal'] not in used for row in events)
    audit['intervals'] = len(result)
    return result, audit
