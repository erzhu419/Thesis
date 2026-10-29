"""Exercise actual native detection/delivery ordering and passive observations."""
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v20_service_hook_test', HERE/'service_trace.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ServiceTrace = module.ServiceTrace


class Ledger:
    def __init__(self):
        self.time_seconds = 10.
        self.waiting_passengers, self.in_vehicle_passengers = 3, 2
        self.accrued_wait_seconds, self.accrued_in_vehicle_seconds = 30., 20.

    def snapshot(self):
        return SimpleNamespace(**vars(self))


class Bridge:
    def __init__(self):
        self.passenger_time_ledger = Ledger()
        self.BusCap = 45
        self.bus_obj_dic = {'B': SimpleNamespace(belong_line_id_s='7X',
            arriver_stop_time_d={}, depart_stop_time_d={}, service_completion_history=defaultdict(list),
            just_server_stop_data_d={}, bus_state_s='Run')}
        self.active_bus_ids = {'B'}
        self._bus_onboard_passenger_ids = {'B': {'stay', 'off'}}
        self.passenger_obj_dic = {'q': SimpleNamespace(passable_line_l=['7X'])}
        self.people = {'stay', 'off'}
        self.native_calls = []

    def _synchronize_bus_passenger_transition(self, bus, previous, current, obj, t):
        self.native_calls.append(('transition', bus, previous, current, t))
        return 'transition_result'

    def _collect_new_events(self, t):
        self.native_calls.append(('collect', t))
        obj = self.bus_obj_dic['B']
        if t == 12.:
            obj.service_completion_history['S'].append(11.5)
        return 'collect_result'

    def _synchronize_passenger_time_counts(self, t):
        self.native_calls.append(('counts', t))
        return 'counts_result'

    def _advance_one_step(self):
        ledger, obj = self.passenger_time_ledger, self.bus_obj_dic['B']
        t = ledger.time_seconds
        previous = obj.bus_state_s
        if t == 10.:
            obj.bus_state_s = 'Stop'
            obj.arriver_stop_time_d['S'] = t
            obj.just_server_stop_data_d = {'S': (10., 11.5)}
            self.people = {'stay'}  # Actual alighting at arrival.
        if t == 11.:
            self.people.add('new')  # Boarding inside dwell, before departure.
        if t == 13.:
            obj.bus_state_s = 'Run'
            obj.depart_stop_time_d['S'] = t
            obj.just_server_stop_data_d.clear()
        self._synchronize_passenger_time_counts(t)
        self._synchronize_bus_passenger_transition('B', previous, obj.bus_state_s, obj, t)
        self._collect_new_events(t)
        ledger.accrued_wait_seconds += ledger.waiting_passengers
        ledger.accrued_in_vehicle_seconds += ledger.in_vehicle_passengers
        ledger.time_seconds += 1.
        return 'native_tick_result'


def api(bridge, *, missing_capacity=False):
    def capacity(bus):
        if missing_capacity:
            raise RuntimeError('Vehicle no longer available')
        return 50
    return SimpleNamespace(vehicle=SimpleNamespace(getPersonIDList=lambda bus: tuple(bridge.people),
        getPersonCapacity=capacity, getStops=lambda bus, n: [SimpleNamespace(stoppingPlaceID='S')],
        isAtBusStop=lambda bus: bridge.bus_obj_dic['B'].bus_state_s == 'Stop'),
        busstop=SimpleNamespace(getPersonIDs=lambda stop: ('q',)))


def test_passive_full301_clock_and_real_inside_dwell_boarding(tmp_path):
    bare, observed = Bridge(), Bridge()
    hook = ServiceTrace(observed, tmp_path/'service_trace.jsonl', traci=api(observed))
    hook.activate(t0=10., bus='B', line='7X', station='S')
    for _ in range(300):
        assert bare._advance_one_step() == observed._advance_one_step() == 'native_tick_result'
    summary = hook.finish(observed.passenger_time_ledger.snapshot())
    assert observed.native_calls == bare.native_calls
    assert vars(observed.passenger_time_ledger) == vars(bare.passenger_time_ledger)
    assert summary['integrity']['all_pass'] is True
    assert summary['clock_rows'] == 301
    assert summary['fixed300'] == dict(dt=300., wait=900., ride=600., cost=2400.)
    rows = json.loads((tmp_path/'service_clock.json').read_text())['rows']
    assert [r['relative_seconds'] for r in rows] == list(range(301))
    events = json.loads((tmp_path/'service_events.json').read_text())['events']
    assert sorted((e['kind'], e['clock_seconds']) for e in events) == sorted([
        ('arrival', 10.), ('alight', 10.), ('board', 11.), ('service_ready', 11.5), ('departure', 13.)])
    ready = next(e for e in events if e['kind'] == 'service_ready')
    assert ready['native_detection_time'] == 12. and ready['clock_seconds'] == 11.5
    board = next(e for e in events if e['kind'] == 'board')
    assert board['passenger_ids'] == ['new'] and board['passenger_count'] == 1
    assert board['person_sample_interval'] == [10., 11.]
    assert board['capacity'] == 50 and board['configured_bus_capacity'] == 45
    assert board['onboard'] == 2 and board['free_capacity'] == board['effective_capacity'] == 48
    assert board['waiting'] == board['eligible_waiting'] == 1
    hook.close()


def test_missing_capacity_is_preserved_without_default_zero(tmp_path):
    bridge = Bridge()
    hook = ServiceTrace(bridge, tmp_path/'service_trace.jsonl', traci=api(bridge, missing_capacity=True), duration_seconds=1)
    hook.activate(t0=10., bus='B', line='7X', station='S')
    bridge._advance_one_step()
    result = hook.finish(bridge.passenger_time_ledger.snapshot())
    event = hook.events[0]
    assert event['capacity'] is None and event['free_capacity'] is None
    assert event['configured_bus_capacity'] == 45
    assert result['missing_counts']['vehicle_capacity'] > 0
    assert result['integrity']['checks']['real_capacity_observed'] is False
    hook.close()


def test_hook_chain_restores_existing_instance_wrapper(tmp_path):
    bridge = Bridge()
    original = bridge._advance_one_step
    calls = []
    def existing_clock_wrapper():
        calls.append('existing_clock')
        return original()
    bridge._advance_one_step = existing_clock_wrapper
    hook = ServiceTrace(bridge, tmp_path/'service_trace.jsonl', traci=api(bridge))
    assert bridge._advance_one_step() == 'native_tick_result'  # Inactive prefix delegates only.
    assert calls == ['existing_clock'] and hook.rows == {} and hook.events == []
    hook.close()
    assert bridge._advance_one_step is existing_clock_wrapper
    assert '_synchronize_bus_passenger_transition' not in bridge.__dict__
    assert '_collect_new_events' not in bridge.__dict__
    assert '_synchronize_passenger_time_counts' not in bridge.__dict__


def test_no_extra_sampling_when_same_clock_is_delivered_twice(tmp_path):
    bridge = Bridge()
    hook = ServiceTrace(bridge, tmp_path/'service_trace.jsonl', traci=api(bridge), duration_seconds=1)
    hook.activate(t0=10., bus='B', line='7X', station='S')
    hook._sample_clock()
    assert len(hook.rows) == 1
    bridge._advance_one_step()
    assert hook.finish(bridge.passenger_time_ledger.snapshot())['clock_rows'] == 2
    hook.close()


def test_final_cutoff_person_change_is_observed_without_an_extra_native_step(tmp_path):
    class FinalTickBridge(Bridge):
        def _advance_one_step(self):
            result = super()._advance_one_step()
            if self.passenger_time_ledger.time_seconds == 14.:
                self.people.add('last_tick_boarding')
            return result
    bridge = FinalTickBridge()
    hook = ServiceTrace(bridge, tmp_path/'service_trace.jsonl', traci=api(bridge), duration_seconds=4)
    hook.activate(t0=10., bus='B', line='7X', station='S')
    for _ in range(4):
        bridge._advance_one_step()
    assert not any('last_tick_boarding' in e.get('passenger_ids', []) for e in hook.events)
    summary = hook.finish(bridge.passenger_time_ledger.snapshot())
    last = next(e for e in hook.events if 'last_tick_boarding' in e.get('passenger_ids', []))
    assert last['clock_seconds'] == 14. and last['person_sample_interval'] == [13., 14.]
    assert bridge.passenger_time_ledger.time_seconds == 14.
    assert summary['integrity']['all_pass'] is True
    hook.close()


def test_dwell_end_person_change_uses_served_stop_not_the_next_stop(tmp_path):
    bridge = Bridge()
    obj = bridge.bus_obj_dic['B']
    obj.bus_state_s, obj.just_server_stop_data_d = 'Stop', {'OLD': (9., 10.)}
    traci = api(bridge)
    traci.vehicle.getStops = lambda bus, n: [SimpleNamespace(stoppingPlaceID='NEXT')]
    traci.vehicle.isAtBusStop = lambda bus: False
    hook = ServiceTrace(bridge, tmp_path/'service_trace.jsonl', traci=traci)
    hook.activate(t0=10., bus='B', line='7X', station='OLD')
    bridge.people.add('last_dwell_boarding')
    hook._observe_people(11.)
    event = hook.events[-1]
    assert event['station_id'] == 'OLD'
    assert event['station_source'] == 'native_current_or_just_served_stop'
    assert event['station_interval_uncertainty'] is True
    obj.bus_state_s, obj.just_server_stop_data_d = 'Run', {}
    bridge.people.add('unknown_station_boarding')
    hook._observe_people(12.)
    assert hook.events[-1]['station_id'] is None  # NEXT is not an observed exchange station.
    assert hook.events[-1]['station_source'] == 'not_observed'
    hook.close()
