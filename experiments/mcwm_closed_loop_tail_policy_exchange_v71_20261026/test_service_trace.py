"""Verify that passive evidence covers real services without advancing native time."""
from types import SimpleNamespace

import pytest

from experiments.mcwm_closed_loop_tail_policy_exchange_v71_20261026.service_trace import TailServiceTrace


class Bridge:
    def __init__(self):
        self.clock = 16000.
        self.transition_calls = []
        self.collect_calls = []
        self.stop_obj_dic = {'A': object(), 'B': object(), 'C': object()}
        self.line_obj_dic = {'7X': SimpleNamespace(stop_id_l=['A', 'B', 'C'])}
        person = lambda lines: SimpleNamespace(passable_line_l=lines,
            start_stop_id_s='A', transfer_stop_id_s='', end_stop_id_s='C',
            passenger_state_s='Stop', travel_data_l=[])
        self.passenger_obj_dic = {'p1': person(['7X']), 'p2': person(['7X'])}
        self.departed_passenger_times = {'p1': 15000., 'p2': 15100.}
        self.bus = SimpleNamespace(belong_line_id_s='7X', bus_state_s='Stop',
            just_server_stop_data_d={'C': [16200., 16204.]},
            arriver_stop_time_d={'C': 16200.}, depart_stop_time_d={},
            service_completion_history={}, board_num_d={}, alight_num_d={},
            strand_num_d={}, want_board_num_d={})
        self.active_bus_ids = {'bus1'}
        self.bus_obj_dic = {'bus1': self.bus}
        self.passenger_time_ledger = SimpleNamespace(snapshot=self.ledger)
        self.waiting = 2

    def ledger(self):
        return SimpleNamespace(time_seconds=self.clock, waiting_passengers=self.waiting,
            in_vehicle_passengers=3, accrued_wait_seconds=self.clock*2,
            accrued_in_vehicle_seconds=self.clock*3)

    def _synchronize_bus_passenger_transition(self, *args):
        self.transition_calls.append(args)
        return 'transition-result'

    def _collect_new_events(self, clock):
        self.collect_calls.append(clock)
        return 'collect-result'

    def _advance_one_step(self):
        self.clock += 1.
        return 'advance-result'


class API:
    def __init__(self):
        self.queues = {'A': (), 'B': (), 'C': ('p1', 'p2')}
        self.queue_calls = 0
        self.busstop = SimpleNamespace(getPersonIDs=self.ids)
        self.vehicle = SimpleNamespace(getPersonIDList=lambda bus: ('r1', 'r2', 'r3'),
            getPersonCapacity=lambda bus: 50)

    def ids(self, station):
        self.queue_calls += 1
        return self.queues[station]


def test_terminal_and_isolated_service_without_control_issue(tmp_path):
    bridge, api = Bridge(), API()
    trace = TailServiceTrace(bridge, tmp_path/'service.jsonl', traci=api)
    assert bridge._synchronize_bus_passenger_transition(
        'bus1', 'Edge', 'Stop', bridge.bus, 16200.) == 'transition-result'
    bridge.bus.service_completion_history['C'] = [16204.]
    assert bridge._collect_new_events(16204.) == 'collect-result'
    assert bridge._collect_new_events(16205.) == 'collect-result'
    bridge.bus.bus_state_s = 'Edge'
    bridge.bus.depart_stop_time_d['C'] = 16207.
    bridge.bus.board_num_d['C'] = 1
    bridge.bus.alight_num_d['C'] = 2
    bridge.bus.strand_num_d['C'] = 1
    bridge.bus.want_board_num_d['C'] = 2
    api.queues['C'] = ('p2',)
    bridge._synchronize_bus_passenger_transition('bus1', 'Stop', 'Edge', bridge.bus, 16207.)
    assert [row['kind'] for row in trace.services] == ['arrival', 'service_ready', 'departure']
    assert all(row['terminal'] for row in trace.services)
    assert trace.services[0]['waiting'] == 2
    assert trace.services[-1]['waiting'] == 1
    assert trace.services[-1]['native_board_count'] == 1
    assert trace.services[-1]['free_capacity'] == 47
    assert bridge.clock == 16000.
    assert len(bridge.transition_calls) == 2
    assert bridge.collect_calls == [16204., 16205.]
    trace.close()
    assert '_collect_new_events' not in bridge.__dict__
    assert bridge._collect_new_events(17000.) == 'collect-result'


def test_milestone_unique_people_and_child_resident_history(tmp_path):
    bridge, api = Bridge(), API()
    trace = TailServiceTrace(bridge, tmp_path/'prefix.jsonl', traci=api)
    bridge._synchronize_bus_passenger_transition('bus1', 'Edge', 'Stop', bridge.bus, 16200.)
    bridge.clock = 18300.
    milestone = trace.snapshot('18300', clock=18300.)
    assert milestone['waiting_matches_ledger']
    assert milestone['waiting_unique'] == 2
    trace.flush()
    trace.child_detach(tmp_path/'child'/'service.jsonl')
    bridge.clock = 24000.
    result = trace.finish()
    assert result['milestones']['18300']['time_seconds'] == 18300.
    group = result['final_waiting_groups'][0]
    assert group['passenger_ids'] == ['p1', 'p2']
    assert group['last_compatible_service']['arrival']['clock_seconds'] == 16200.
    assert group['last_compatible_service']['departure'] is None
    assert (tmp_path/'prefix.jsonl').read_text().count('"record": "service"') == 1
    assert '"record": "service"' not in (tmp_path/'child'/'service.jsonl').read_text()
    assert result['final']['waiting_passengers'][0]['departure_time'] == 15000.
    trace.close()


def test_missing_and_duplicate_station_membership_not_invented_zero(tmp_path):
    bridge, api = Bridge(), API()
    api.queues['A'] = ('p1',)
    api.queues['C'] = ('p1', 'p2', 'unknown')
    trace = TailServiceTrace(bridge, tmp_path/'service.jsonl', traci=api)
    bridge.clock = 20100.
    result = trace.snapshot('20100')
    assert result['waiting_raw_memberships'] == 4
    assert result['waiting_unique'] == 3
    assert result['duplicate_station_memberships'] == {'p1': ['A', 'C']}
    assert not result['waiting_matches_ledger']
    unknown = next(row for row in result['waiting_passengers'] if row['passenger_id'] == 'unknown')
    assert unknown['passable_lines'] is None
    assert not unknown['metadata_observed']
    del api.queues['B']
    missing = trace.snapshot('missing')
    assert missing['missing_stations'] == ['B']
    assert not missing['waiting_inventory_complete']
    with pytest.raises(ValueError, match='actual ledger clock'):
        trace.snapshot('wrong', clock=20101.)
    trace.close()


def test_prefix_service_clocks_cost_no_person_queries_before_detail_window(tmp_path):
    bridge, api = Bridge(), API()
    trace = TailServiceTrace(bridge, tmp_path/'service.jsonl', traci=api)
    bridge.bus.arriver_stop_time_d = {'A': 1000.}
    bridge._synchronize_bus_passenger_transition('bus1', 'Edge', 'Stop', bridge.bus, 1000.)
    assert api.queue_calls == 0
    assert len(trace.services) == 1
    assert 'waiting' not in trace.services[0]
    trace.close()


def test_native_post_advance_captures_exact_milestones_only(tmp_path):
    bridge, api = Bridge(), API()
    trace = TailServiceTrace(bridge, tmp_path/'service.jsonl', traci=api)
    bridge.clock = 18298.
    assert bridge._advance_one_step() == 'advance-result'
    assert api.queue_calls == 0
    api.queues['C'] = ('p1',)
    bridge.waiting = 1
    assert bridge._advance_one_step() == 'advance-result'
    assert trace.milestones['18300']['time_seconds'] == 18300.
    assert trace.milestones['18300']['waiting_unique'] == 1
    assert trace.milestones['18300']['waiting_matches_ledger']
    assert api.queue_calls == 3
    bridge.clock = 20099.
    bridge._advance_one_step()
    bridge.clock = 23999.
    bridge._advance_one_step()
    assert set(trace.milestones) == {'18300', '20100', '24000'}
    assert api.queue_calls == 9
    result = trace.finish()
    assert result['final']['time_seconds'] == 24000.
    assert api.queue_calls == 9
    assert (tmp_path/'service_summary.json').is_file()
    trace.close()
    assert '_advance_one_step' not in bridge.__dict__
