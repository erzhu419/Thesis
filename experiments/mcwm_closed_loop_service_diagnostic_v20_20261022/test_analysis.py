"""Check fixed-clock cost reconstruction and rejection of disturbed replays."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v20_service_analysis_test', HERE/'analyze.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
old_spec = importlib.util.spec_from_file_location('v19_service_fixture',
    HERE.parent/'mcwm_closed_loop_execution_phase_v19_20261022/test_analysis.py')
old = importlib.util.module_from_spec(old_spec)
old_spec.loader.exec_module(old)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path):
    source, _ = old.fixture(tmp_path)
    root = tmp_path/'v20'
    day = source/'runs/stable/seed6201'
    acquisitions = json.loads((day/'acquisitions.json').read_text())
    for anchor in acquisitions['records']:
        selected = anchor['selected']
        selected['station_id'] = '7X13_102X26_705X27'
        selected['trip_id'] = selected['vehicle_id']+'_trip'
        selected['issue_ledger'] = dict(time_seconds=selected['started_at'], waiting_passengers=1,
            in_vehicle_passengers=1, accrued_wait_seconds=1000., accrued_in_vehicle_seconds=2000.)
        for arm in ('low', 'high'):
            original_path = day/'training_branches'/str(anchor['index'])/arm/'summary.json'
            value = json.loads(original_path.read_text())
            value['selected'] = deepcopy(selected)
            write(original_path, value)
            value = deepcopy(value)
            value['budget']['new_target_training_labels'] = 0
            value['budget']['new_diagnostic_action_outcomes'] = 1
            value['fixed_zero_prediction'] = dict(dt=0., wait=0., ride=0., cost=0.)
            value['phase_intervention'] = dict(mapping='native_all_suffix', actor_real_clock=True)
            value['integrity']['checks'].update(old_native_truth_exact=True,
                service_observation_complete=True, service_ledger_truth_exact=True, native_mapping_unchanged=True)
            value['service'] = dict(integrity=dict(all_pass=True), clock_rows=301, missing_counts={})
            path = root/'branches/native'/anchor['line_id']/arm
            write(path/'summary.json', value)
            endpoint = value['truth']['fixed300']
            rows = [dict(time_seconds=selected['started_at']+i, relative_seconds=i,
                waiting_passengers=1, in_vehicle_passengers=1,
                accrued_wait_seconds=1000.+endpoint['wait']*i/300.,
                accrued_in_vehicle_seconds=2000.+endpoint['ride']*i/300.) for i in range(301)]
            write(path/'service_clock.json', dict(start_time=selected['started_at'],
                cutoff_time=selected['started_at']+300, rows=rows))
            write(path/'service_events.json', dict(events=[dict(kind='arrival',
                clock_seconds=selected['started_at']+10, relative_seconds=10,
                line_id=anchor['line_id'], bus_id=selected['trip_id'], station_id=selected['station_id'],
                capacity=60, onboard=20, effective_capacity=40, waiting=5, eligible_waiting=3)]))
    write(day/'acquisitions.json', acquisitions)
    return source, root


def test_common_clock_phases_use_cumulative_ledger_and_reconstruct_exact_gain():
    low, high = [], []
    for i in range(301):
        low.append(dict(waiting_passengers=10, in_vehicle_passengers=10,
            accrued_wait_seconds=1000+10*i, accrued_in_vehicle_seconds=2000+10*i))
        high.append(dict(waiting_passengers=10, in_vehicle_passengers=10,
            accrued_wait_seconds=1000+(12*i if i <= 150 else 1800+8*(i-150)),
            accrued_in_vehicle_seconds=2000+9*i))
    value, series = analysis.paired_clock(low, high)
    assert value['final_gain'] == dict(dt=0., wait=0, ride=300, cost=300)
    assert value['stages'] == [[0,75,-150,75,-225], [75,150,-150,75,-225],
                              [150,225,150,75,375], [225,300,150,75,375]]
    assert value['reconstruction_error'] == dict(wait=0, ride=0, cost=0)
    assert value['first_negative_cost_second'] == 1
    assert value['first_positive_cost_second'] == 241
    assert value['cumulative_cost_sign_changes'] == 1
    assert len(series) == 301 and series[-1][1:3] == [0,0]
    assert sum(row[6] for row in series) == 0 and sum(row[7] for row in series) == 300


def test_all_six_native_replays_fit_one_budget_and_preserve_zero_global_duration(tmp_path):
    source, root = fixture(tmp_path)
    for arm in ('low', 'high'):
        for path in (root/'branches/native/705X'/arm/'summary.json',
            source/'runs/stable/seed6201/training_branches/8'/arm/'summary.json'):
            value = json.loads(path.read_text())
            value['truth']['global_next']['dt'] = 0.
            write(path, value)
    value, series = analysis.analyze(source, root)
    assert value['status'] == 'complete' and value['integrity']['all_pass']
    assert value['budget']['planned_new_branch_outcomes'] == 6
    assert value['budget']['suffix_seconds_observed'] == 1800
    assert value['budget']['new_training_labels'] == value['budget']['truth_labels_enter_training'] == 0
    assert value['duration_counts'] == dict(positive_dt=2, zero_dt=1)
    assert value['missing']['selection'] == 0 and all(x == 0 for x in value['unsupported'].values())
    assert set(series) == set(analysis.LINES)
    assert all(row['clock_decomposition']['final_gain']['cost'] == 10. for row in value['cases'])
    assert len(json.dumps(value, separators=(',', ':')).encode()) < 30000


def test_first_low_is_preflight_and_is_reused_without_needing_other_five(tmp_path):
    source, root = fixture(tmp_path)
    for path in (root/'branches/native').glob('*/*/summary.json'):
        if path.parent.parent.name != '7X' or path.parent.name != 'low':
            path.unlink()
    value, series = analysis.analyze(source, root, preflight=True)
    assert value['status'] == 'complete' and value['integrity']['all_pass']
    assert value['budget']['planned_new_branch_outcomes'] == 1
    assert value['budget']['suffix_seconds_observed'] == 300
    assert series == {} and 'effects' not in value['cases'][0]


@pytest.mark.parametrize('change', ('clock_gap', 'old_truth', 'training', 'native_mapping'))
def test_disturbed_observation_or_replay_blocks_scientific_pair_output(tmp_path, change):
    source, root = fixture(tmp_path)
    path = root/'branches/native/7X/low'
    value = json.loads((path/'summary.json').read_text())
    if change == 'clock_gap':
        clocks = json.loads((path/'service_clock.json').read_text())
        del clocks['rows'][100]
        write(path/'service_clock.json', clocks)
    elif change == 'old_truth':
        value['truth']['fixed300']['wait'] += 1
    elif change == 'training':
        value['suffix']['training_delta']['policy'] = 1
    else:
        value['phase_intervention']['mapping'] = 'tail_after_fork_batch'
    write(path/'summary.json', value)
    result, _ = analysis.analyze(source, root)
    assert result['status'] == 'integrity_failed' and not result['integrity']['all_pass']
    assert 'clock_decomposition' not in result['cases'][0]


def test_simultaneous_services_remain_ties_and_person_ids_are_not_exported():
    selected = dict(trip_id='bus1', station_id='shared')
    events = [dict(kind='arrival', clock_seconds=10., line_id=line, bus_id=bus,
        station_id='shared', passenger_ids=['resident-only'])
        for line, bus in [('7X','bus1'), ('102X','bus2')]]
    value = analysis.events_profile(events, selected)
    assert len(value['shared_station_clock_groups']['groups']) == 1
    assert value['shared_station_clock_groups']['groups'][0][1:3] == [2,[0,1]]
    assert 'resident-only' not in json.dumps(value)


def test_output_failure_is_recorded_for_continuation_to_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['analyze.py', '--source', str(tmp_path),
        '--root', str(tmp_path), '--out', str(tmp_path/'summary.json')])
    monkeypatch.setattr(analysis, 'run', lambda args: (_ for _ in ()).throw(RuntimeError('resident cap exceeded')))
    with pytest.raises(RuntimeError, match='cap exceeded'):
        analysis.main()
    failure = json.loads((tmp_path/'analysis_failure.json').read_text())
    assert failure['message'] == 'resident cap exceeded'
    assert not (tmp_path/'summary.json').exists()


def test_representative_full_service_schema_fits_export_caps(tmp_path):
    source, root = fixture(tmp_path)
    day = source/'runs/stable/seed6201'
    records = json.loads((day/'acquisitions.json').read_text())['records']
    for anchor in records:
        selected = anchor['selected']
        for arm in ('low', 'high'):
            path = root/'branches/native'/anchor['line_id']/arm
            value = json.loads((path/'summary.json').read_text())
            value['service'] = dict(status='complete', integrity=dict(all_pass=True,
                checks=dict(fixed_clock_complete=True, exact_native_cutoff=True, final_ledger_exact=True)),
                start_time=selected['started_at'], cutoff_time=selected['started_at']+300,
                clock_rows=301, event_counts=dict(arrival=7, departure=1, board=1, alight=1, service_ready=1, issue=1),
                line_event_counts={line:40 for line in ('7X','102X','705X','1X','2X','3X','4X','5X','6X','8X','9X','10X')},
                missing_counts=dict(passenger_boundary_baseline=1, vehicle_capacity=1),
                fixed300={k:value['truth']['fixed300'][k] for k in ('dt','wait','ride','cost')},
                paths=dict(clock='service_clock.json', events='service_events.json', trace='service_trace.jsonl'),
                clocks='Clock rows are ledger delivery times; service events retain exact/detection times and observed_at. Use cumulative ledger differences.',
                service_scope='All native bus service boundaries; each-second queue at the fork shared station. No future observation enters training.')
            write(path/'summary.json', value)
            events = []
            for i in range(12):
                at = selected['started_at']+10+i+(arm == 'high')
                events.append(dict(kind=('arrival','departure','service_ready','board','alight','issue')[i]
                    if i < 6 else 'arrival', clock_seconds=at, relative_seconds=at-selected['started_at'],
                    line_id=anchor['line_id'], bus_id=selected['trip_id'] if i < 6 else '705X_very_long_trip_name_'+str(i),
                    station_id=selected['station_id'], onboard=20+(arm == 'high'), capacity=60,
                    effective_capacity=40, free_capacity=40, waiting=5, eligible_waiting=3,
                    passenger_count=2, native_detection_time=at, observed_at=at+1, observation_order=i,
                    person_sample_interval=[at,at+1],
                    station_source='traci.vehicle.getStops_first_stoppingPlaceID',
                    time_source='bridge.service_completion_history',
                    station_interval_uncertainty=True, station_at_interval_end_confirmed=False,
                    passenger_ids=['resident-only-person-'+str(i)]))
            write(path/'service_events.json', dict(events=events))
    value, _ = analysis.analyze(source, root)
    preflight, _ = analysis.analyze(source, root, preflight=True)
    assert value['integrity']['all_pass'] and preflight['integrity']['all_pass']
    for row in value['cases']:
        assert len(row['service_comparison']['first_four_changed_events']) == 4
        for arm in ('low','high'):
            assert row['arms'][arm]['event_profile']['controlled_trip']['shown'] == 6
            assert row['arms'][arm]['event_profile']['shared_station_arrivals']['shown'] == 6
    assert len(json.dumps(value, separators=(',', ':')).encode()) < 30000
    assert len(json.dumps(preflight, separators=(',', ':')).encode()) < 12000
    assert 'resident-only-person-' not in json.dumps(value)
