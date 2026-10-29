"""Causality, physical aliases and native event ordering for v32 context."""
import os
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault('THESIS_H2O_ROOT', str(Path(__file__).resolve().parents[1]/'sumo_eval/runtime/H2Oplus'))

from experiments.mcwm_closed_loop_action_fork_v7_20261020.test_fork_stream import (
    FakeBridge, FakeNativeEnv, policy, snapshot,
)
from experiments.mcwm_closed_loop_v1.stream import EventStream, TransitMDPConfigV7
from experiments.mcwm_closed_loop_critic_memory_v32_20261023.memory import ExecutionMemory, MemoryEventStream
from bus_h2o.passenger_time import PassengerTimeLedger


def bus(line='7X', phase='Stop'):
    return SimpleNamespace(belong_line_id_s=line, bus_state_s=phase, bus_speed_n=0.,
        distance_n=17., bus_cur_lane_s='lane')


class Bridge:
    def __init__(self):
        self.current_time = 10.
        self.line_obj_dic = {'7X': None, '102X': None}
        self.bus_obj_dic = {'A': bus(), 'B': bus('102X')}
        self.passenger_time_ledger = SimpleNamespace(snapshot=lambda: snapshot(self.current_time))
        self.calls = []

    def apply_action(self, event, command):
        self.calls.append(('apply', event.bus_id))
        event.metadata['applied_additional_hold_seconds'] = command[0]
        return 'applied'

    def _synchronize_bus_passenger_transition(self, trip, previous, current, obj, time):
        self.calls.append(('phase', trip))
        obj.bus_state_s = current
        return 'synchronized'


def apply(bridge, trip='A', hold=0.):
    event = SimpleNamespace(bus_id=trip, line_id=bridge.bus_obj_dic[trip].belong_line_id_s,
        stop_id='stop', metadata={})
    assert bridge.apply_action(event, [hold, 1.]) == 'applied'
    return event


def test_all_lines_and_vehicles_share_one_fixed_context_with_explicit_availability():
    bridge = Bridge()
    memory = ExecutionMemory(bridge)
    empty = memory.snapshot(10)
    assert empty['line_ids'] == ['102X', '7X']
    assert empty['vector'] == [0.]*14
    apply(bridge, 'A')
    apply(bridge, 'B', 5.)
    result = memory.snapshot(10)
    assert [line['recent_vehicle_count'] for line in result['lines']] == [1, 1]
    assert [line['available'] for line in result['lines']] == [True, True]
    assert result['metadata']['recent_trip_count'] == 2
    assert sorted(memory.records) == ['A', 'B']
    assert result['metadata']['selected_diagnostic_vehicle'] is False
    assert memory.summary()['line_apply_counts'] == {'102X': 1, '7X': 1}
    assert memory.memory_dim == 14 and memory.summary()['integrity']['all_pass']


def test_command_alias_does_not_manufacture_a_physical_difference():
    vectors, metadata = [], []
    for hold in (0., .9375):
        bridge = Bridge()
        memory = ExecutionMemory(bridge)
        apply(bridge, hold=hold)
        bridge.current_time = 11.
        bridge._synchronize_bus_passenger_transition('A', 'Stop', 'Edge', bridge.bus_obj_dic['A'], 11.)
        row = memory.snapshot(12)
        vectors.append(row['vector'])
        metadata.append(memory.records['A'].copy())
    assert vectors[0] == vectors[1]
    assert metadata[0]['command_hold_seconds'] != metadata[1]['command_hold_seconds']


def test_observed_departure_changes_context_and_survives_other_vehicle_events():
    bridge = Bridge()
    memory = ExecutionMemory(bridge)
    apply(bridge)
    before = memory.snapshot(11)
    bridge.current_time = 11.
    assert bridge._synchronize_bus_passenger_transition('A', 'Stop', 'Edge', bridge.bus_obj_dic['A'], 11.) == 'synchronized'
    apply(bridge, 'B')
    after = memory.snapshot(11)
    assert before['lines'][1]['departure_observed_count'] == 0
    assert after['lines'][1]['departure_observed_count'] == 1
    assert after['lines'][1]['stopped_count'] == 0
    assert after['lines'][0]['stopped_count'] == 1
    assert memory.snapshot(15)['lines'][1]['vector'][-1] == 4/300
    # Past objects are immutable snapshots, not references to live bus state.
    assert before['lines'][1]['stopped_count'] == 1


def test_past_snapshots_and_future_physical_event_backfill_are_rejected():
    bridge = Bridge()
    memory = ExecutionMemory(bridge)
    apply(bridge)
    with pytest.raises(ValueError, match='future observation'):
        memory.snapshot(9.)
    with pytest.raises(ValueError, match='own native event time'):
        memory.observe_phase('A', 'Stop', 'Edge', bridge.bus_obj_dic['A'], 12., 11.)
    # A history entry already present on an object is not a causal observation.
    bridge.bus_obj_dic['A'].depart_stop_time_d = {'stop': 15.}
    assert memory.snapshot(15)['lines'][1]['departure_observed_count'] == 0


def test_latest_visit_replaces_old_visit_and_expired_rows_are_unavailable():
    bridge = Bridge()
    memory = ExecutionMemory(bridge)
    apply(bridge)
    bridge.current_time = 20.
    bridge._synchronize_bus_passenger_transition('A', 'Stop', 'Edge', bridge.bus_obj_dic['A'], 20.)
    bridge.current_time = 30.
    bridge.bus_obj_dic['A'].bus_state_s = 'Stop'
    apply(bridge, hold=7.5)
    assert memory.snapshot(30)['lines'][1]['recent_vehicle_count'] == 1
    assert memory.snapshot(30)['lines'][1]['departure_observed_count'] == 0
    assert memory.snapshot(330)['lines'][1]['available']
    assert not memory.snapshot(331)['lines'][1]['available']


def test_close_restores_the_exact_prior_instance_wrapper_and_class_descriptor():
    bridge = Bridge()
    original_sync = bridge._synchronize_bus_passenger_transition
    calls = []

    def history_wrapper(*args):
        calls.append('history')
        return original_sync(*args)

    bridge._synchronize_bus_passenger_transition = history_wrapper
    memory = ExecutionMemory(bridge)
    apply(bridge)
    memory.close()
    assert 'apply_action' not in bridge.__dict__
    assert bridge._synchronize_bus_passenger_transition is history_wrapper
    bridge._synchronize_bus_passenger_transition('A', 'Stop', 'Edge', bridge.bus_obj_dic['A'], 10.)
    assert calls == ['history']
    # This is the same deletion performed by NativeRunHistory.close when it
    # originally wrapped the native class descriptor.
    del bridge._synchronize_bus_passenger_transition
    assert '_synchronize_bus_passenger_transition' not in bridge.__dict__
    assert bridge._synchronize_bus_passenger_transition.__func__ is Bridge._synchronize_bus_passenger_transition


def test_actual_native_tick_orders_ledger_then_phase_and_next_tick_before_apply():
    # Execute the shipped native tick body with deterministic native services.
    # Its actual ledger/phase ordering is the behavior under test; a full SUMO
    # process is unnecessary to detect a future-clock assertion here.
    source = Path(__file__).resolve().parents[1]/'sumo_eval/runtime/H2Oplus/bus_h2o/sumo_env/rl_bridge.py'
    tree = ast.parse(source.read_text())
    native_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SumoRLBridge')
    tick = next(node for node in native_class.body if isinstance(node, ast.FunctionDef) and node.name == '_advance_one_step')
    native_time = [18314.]
    simulation = SimpleNamespace(getTime=lambda: native_time[0], getDepartedIDList=lambda: [],
        getArrivedIDList=lambda: [], getDepartedPersonIDList=lambda: [], getArrivedPersonIDList=lambda: [])
    traci = SimpleNamespace(simulation=simulation, simulationStep=lambda: native_time.__setitem__(0, native_time[0]+1.))
    namespace = {'traci': traci}
    exec(compile(ast.Module(body=[tick], type_ignores=[]), str(source), 'exec'), namespace)
    bridge = Bridge()
    bridge.passenger_time_ledger = PassengerTimeLedger(18313.)
    bridge.done, bridge.steps = False, 1
    bridge.active_bus_ids = {'A'}
    bridge.active_passenger_ids = set()
    bridge.line_obj_dic = {'7X': None, '102X': None}
    bridge.bus_obj_dic['A'].bus_running = lambda *args: setattr(bridge.bus_obj_dic['A'], 'bus_state_s', 'Edge')
    for name in ('BusCap', 'AveAlightingTime', 'AveBoardingTime'):
        setattr(bridge, name, 0)
    for name in ('stop_obj_dic', 'signal_obj_dic', 'passenger_obj_dic', 'bus_arrstation_od_otd_dict',
                 'involved_tl_ID_l', 'busline_edge_order'):
        setattr(bridge, name, {})
    bridge._synchronize_passenger_time_counts = lambda time: bridge.passenger_time_ledger.set_counts(
        time, waiting_passengers=0, in_vehicle_passengers=0)
    bridge._collect_new_events = lambda time: None
    for name in ('_check_station_capacity', '_cleanup_departed_buses', '_check_termination'):
        setattr(bridge, name, lambda: None)
    memory = ExecutionMemory(bridge)
    apply(bridge)
    namespace['_advance_one_step'](bridge)
    row = memory.snapshot(18315.)
    assert memory.records['A']['departure_at'] == 18314.
    assert memory.records['A']['departure_observed_at'] == 18314.
    assert row['metadata']['latest_native_observation_at'] == 18314.
    assert bridge.current_time == bridge.passenger_time_ledger.snapshot().time_seconds == 18315.
    bridge.bus_obj_dic['A'].bus_state_s = 'Stop'
    event = apply(bridge, hold=.9375)
    event.sim_time = 18314.  # decision event's raw time can precede the ledger.
    assert memory.records['A']['applied_at'] == 18315.
    assert memory.summary()['integrity']['all_pass']


class MemoryFakeBridge(FakeBridge):
    def __init__(self, batches):
        super().__init__(batches)
        self.bus_obj_dic = {}

    def apply_action(self, event, command):
        super().apply_action(event, command)
        event.metadata = {'applied_additional_hold_seconds': float(command[0])}
        self.bus_obj_dic[event.bus_id] = bus('L')

    def _synchronize_bus_passenger_transition(self, trip, previous, current, obj, time):
        obj.bus_state_s = current

    def _advance_one_step(self):
        super()._advance_one_step()
        for trip, obj in self.bus_obj_dic.items():
            self._synchronize_bus_passenger_transition(trip, obj.bus_state_s, 'Edge', obj, self.current_time)


class MemoryNativeEnv(FakeNativeEnv):
    def reset(self):
        super().reset()
        self._bridge = MemoryFakeBridge(self.batches)


def test_stream_preserves_old_events_actions_rewards_and_same_clock_closure_order():
    batches = [(10, [('A', 0), ('B', 0)]), (20, [('C', 0)]), (30, [('A', 1)]), (40, [('B', 1)])]
    original_env, candidate_env = MemoryNativeEnv(batches), MemoryNativeEnv(batches)
    contract = TransitMDPConfigV7(horizon_seconds=24000)
    original = EventStream(original_env, [3], contract, execute_transform=False)
    issues = []
    candidate = MemoryEventStream(candidate_env, [3], contract, execute_transform=False,
        on_issue=lambda event: issues.append(event['memory_s']))
    original.reset()
    candidate.reset()
    candidate.memory = ExecutionMemory(candidate_env._bridge)
    actual_rows = []
    for _ in range(3):
        expected, actual = original.step(policy), candidate.step(policy)
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            for name in ('s', 'a', 'ns', 'raw_s', 'raw_ns'):
                np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
            assert (left.r, left.dt, left.sequence) == (right.r, right.dt, right.sequence)
            assert right.memory_s['observed_at'] == right.started_at
            assert right.memory_ns['observed_at'] == right.completed_at
        actual_rows.extend(actual)
    assert original_env._bridge.actions == candidate_env._bridge.actions
    assert original_env.native_advance_calls == candidate_env.native_advance_calls
    assert actual_rows[0].dt == 0.
    assert actual_rows[0].memory_s['vector'] == actual_rows[0].memory_ns['vector'] == [0.]*7
    assert issues[0]['vector'] == issues[1]['vector'] == [0.]*7
    assert actual_rows[1].memory_ns['lines'][0]['recent_vehicle_count'] == 2
    assert actual_rows[1].memory_ns['lines'][0]['departure_observed_count'] == 2
