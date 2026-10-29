"""Physical interval pairing and inventory clocks are distinct from passenger ledgers."""
import json

from experiments.mcwm_closed_loop_service_segments_v74_20261027.core import (
    intervals, load_services, physical_inventory, stats)


def event(kind, station, clock, order, *, bus='b1', lag=0, people=None):
    return dict(record='service', kind=kind, bus_id=bus, line_id='7X',
        station_id=station, clock_seconds=clock, observed_at=clock+lag,
        observation_order=order, onboard_passenger_ids=people)


def read(tmp_path, rows, name='services.jsonl'):
    path = tmp_path/name
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return load_services([path])


def test_same_duration_shift_changes_tail_overlap_without_becoming_delay(tmp_path):
    left, _ = read(tmp_path, [event('departure', 'A', 18290, 0),
        event('arrival', 'B', 18310, 1)], name='left.jsonl')
    right, _ = read(tmp_path, [event('departure', 'A', 18300, 0),
        event('arrival', 'B', 18320, 1)], name='right.jsonl')
    lhs, _ = intervals(left)
    rhs, _ = intervals(right)
    assert lhs.keys() == rhs.keys()
    key = next(iter(lhs))
    assert lhs[key]['duration'] == rhs[key]['duration'] == 20
    assert lhs[key]['tail_overlap'] == 10 and rhs[key]['tail_overlap'] == 20
    assert lhs[key]['crosses_boundary'] and not rhs[key]['crosses_boundary']


def test_lagged_inventory_stays_missing_for_physical_tick_and_lag_is_by_kind(tmp_path):
    rows, audit = read(tmp_path, [event('arrival', 'A', 100, 0, lag=5, people=['p']),
        event('departure', 'A', 110, 1, people=['p']),
        event('arrival', 'B', 120, 2)])
    assert physical_inventory(rows[0]) is None
    assert physical_inventory(rows[1]) == {'p'}
    assert physical_inventory(rows[2]) is None
    assert audit['physical_inventory_rows'] == 1
    assert audit['observed_lag_by_kind']['arrival']['nonzero'] == 1
    assert audit['observed_lag_by_kind']['arrival']['max_abs'] == 5
    assert audit['observed_lag_by_kind']['departure']['nonzero'] == 0
    assert audit['observation_order_continuous']


def test_repeated_station_visits_and_split_files_preserve_occurrence(tmp_path):
    first = tmp_path/'prefix.jsonl'
    second = tmp_path/'suffix.jsonl'
    values = [event('arrival', 'A', 10, 0), event('service_ready', 'A', 11, 1),
        event('departure', 'A', 12, 2), event('arrival', 'B', 20, 3),
        event('departure', 'B', 25, 4), event('arrival', 'A', 35, 5),
        event('departure', 'A', 38, 6)]
    first.write_text(''.join(json.dumps(x)+'\n' for x in values[:4]))
    second.write_text(''.join(json.dumps(x)+'\n' for x in values[4:]))
    rows, audit = load_services([first, second])
    assert audit['service_events'] == 7 and audit['observation_order_continuous']
    assert [x['occurrence'] for x in rows if x['kind'] == 'arrival' and x['station_id'] == 'A'] == [0, 1]
    result, pairing = intervals(rows)
    key = ('b1', '7X', 'dwell', 'A', 'A')
    assert result[(*key, 0)]['duration'] == 2
    assert result[(*key, 1)]['duration'] == 3
    assert pairing['dwell'] == 3 and pairing['run'] == 2
    assert pairing['final_departures_without_next_arrival'] == 1
    assert pairing['invalid_adjacency'] == 0


def test_missing_endpoint_and_zero_duration_are_not_filled_to_horizon(tmp_path):
    rows, _ = read(tmp_path, [event('arrival', 'A', 18200, 0),
        event('departure', 'A', 18200, 1), event('arrival', 'B', 18210, 2),
        event('departure', 'Z', 23999, 3, bus='b2')])
    result, audit = intervals(rows)
    assert len(result) == 1
    assert audit['zero_duration'] == 1
    assert audit['final_arrivals_without_departure'] == 1
    assert audit['final_departures_without_next_arrival'] == 1
    assert audit['single_endpoint_buses'] == 1
    assert all(row['end']['clock_seconds'] != 24000 for row in result.values())
    assert stats([None]) == dict(count=0, sum=None, mean=None, median=None, p90=None, min=None, max=None)
    assert stats([0, 10]) == dict(count=2, sum=10, mean=5, median=5, p90=9, min=0, max=10)
