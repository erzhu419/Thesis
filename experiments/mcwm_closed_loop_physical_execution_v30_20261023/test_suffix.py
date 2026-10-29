"""Native clocks, physical transitions, immutable feedback and exact endpoints."""
from types import SimpleNamespace
import math
import pytest

from experiments.mcwm_closed_loop_action_fork_v7_20261020.fork_stream import ForkEventStream
from experiments.mcwm_closed_loop_action_fork_v7_20261020.test_fork_stream import FakeNativeEnv, policy
from experiments.mcwm_closed_loop_v1.stream import TransitMDPConfigV7
from experiments.mcwm_closed_loop_physical_execution_v30_20261023.suffix import NativeTruthTriple, NativePhysicalObserver, HOLD_SECONDS


class AppliedNativeEnv(FakeNativeEnv):
    def reset(self):
        super().reset()
        bridge = self._bridge
        bridge.bus_obj_dic = {'A': SimpleNamespace(bus_state_s='Stop',
            just_server_stop_data_d={'S0': [8., 10.]}, next_stop_id_s='S1',
            arriver_stop_time_d={'S0': 8.}, depart_stop_time_d={})}
        self.depart_at = self.arrive_at = None
        self.registered_duration = 3600.
        self.missing_physical_api = False
        self.departure_enabled = self.arrival_enabled = True
        apply = self._bridge.apply_action

        def native_apply(event, command):
            apply(event, command)
            if event.bus_id == 'A' and self.depart_at is None:
                self.registered_duration = float(command[0])
                self.depart_at = bridge.current_time+1+int(float(command[0]))
                self.arrive_at = self.depart_at+40
            event.assigned_action = float(command[0])
            event.metadata = dict(applied_additional_hold_seconds=float(command[0]),
                predicted_departure_time_seconds=event.sim_time+float(command[0]))

        self._bridge.apply_action = native_apply
        advance = bridge._advance_one_step

        def native_advance():
            advance()
            obj = bridge.bus_obj_dic['A']
            if self.depart_at is not None and self.departure_enabled and bridge.current_time > self.depart_at:
                obj.depart_stop_time_d['S0'] = self.depart_at
                obj.bus_state_s = 'Edge'
            if self.arrive_at is not None and self.arrival_enabled and bridge.current_time > self.arrive_at:
                obj.arriver_stop_time_d['S1'] = self.arrive_at
                obj.bus_state_s = 'Stop'
                obj.just_server_stop_data_d = {'S1': [self.arrive_at, self.arrive_at+10]}
                obj.next_stop_id_s = 'S2'

        bridge._advance_one_step = native_advance

        def at_stop(bus):
            if self.missing_physical_api:
                raise RuntimeError('Selected trip removed')
            return (self.depart_at is None or not self.departure_enabled or bridge.current_time < self.depart_at
                or self.arrival_enabled and bridge.current_time >= self.arrive_at)

        def current_station():
            return 'S1' if self.arrive_at is not None and bridge.current_time >= self.arrive_at else 'S0'

        def position(bus):
            if self.depart_at is None:
                return 10.
            return 10.+max(0., min(bridge.current_time-self.depart_at, 40.))*5.

        self.traci = SimpleNamespace(
            simulation=SimpleNamespace(getTime=lambda: bridge.current_time, getDeltaT=lambda: 1.),
            vehicle=SimpleNamespace(isAtBusStop=at_stop, getActionStepLength=lambda bus: 1.,
                getLastActionTime=lambda bus: math.floor(bridge.current_time),
                getStopState=lambda bus: 17 if at_stop(bus) else 0,
                getStops=lambda bus, count: [SimpleNamespace(stoppingPlaceID=current_station(),
                    duration=self.registered_duration)],
                getSpeed=lambda bus: 0. if at_stop(bus) else 5., getLaneID=lambda bus: 'E_0',
                getLanePosition=position, getPosition=lambda bus: [position(bus), 0.]))


def run_suffix(tmp_path, arm, batches, mutate_counts=False, issue_lag_seconds=0):
    contract = TransitMDPConfigV7(horizon_seconds=24000)
    actions = {name: [2*seconds/contract.max_hold_seconds-1]
        for name, seconds in HOLD_SECONDS.items()}
    counts = dict(policy=456, feedback=285, fresh_shadow=248, no_fresh_shadow=248,
        world=0, trust=0)
    pair = NativeTruthTriple(tmp_path, dict(selected=dict(sequence=0), actions=actions),
        lambda: counts.copy(), 106202)
    # fork_children already has process/RNG/route-refill coverage. Here emulate
    # only the child continuation after that exact live-state fork returns.
    pair.in_child, pair.arm, pair.before, pair.started_wall = True, arm, counts.copy(), 1.
    env = AppliedNativeEnv(batches)
    stream = ForkEventStream(env, [3], contract, execute_transform=False,
        selector=lambda event: actions[arm], fork_duration_seconds=300.,
        on_issue=lambda event: pair.observe_issue(event, stream), delay_seconds=30.)
    stream.reset()
    # SUMO may queue a decision before the issue callback sees the ledger.
    env._bridge.current_time += issue_lag_seconds
    env._bridge.steps += issue_lag_seconds
    pair._install_native_observer(stream, traci=env.traci)
    while not stream.cutoff_reached:
        stream.step(policy)
        pair.observe_progress(stream)
    if mutate_counts:
        counts['policy'] += 1
    return pair, stream, pair.build_summary(stream)


@pytest.mark.parametrize('arm,seconds', list(HOLD_SECONDS.items()))
def test_actual_saved_command_and_same_time_next_event_keep_exact_300_ledger(tmp_path, arm, seconds):
    _, stream, summary = run_suffix(tmp_path, arm,
        [(10, [('A', 0), ('B', 0)]), (20, [('C', 0)]), (1000, [('D', 0)])])
    assert summary['integrity']['all_pass']
    assert stream.env._bridge.actions[0][2] == [seconds, 1.]
    assert summary['execution']['executed_hold_seconds'] == seconds
    assert summary['execution']['native_applied_hold_seconds'] == seconds
    assert summary['truth']['next_event'] == dict(status='complete', dt=0., wait=0.,
        ride=0., cost=0., completed_at=10., next_vehicle_id='B', terminal=False,
        zero_duration=True)
    assert summary['truth']['fixed300'] == dict(status='complete', dt=300.,
        wait=600., ride=900., cost=2100.)
    assert summary['budget']['suffix_sim_seconds'] == 300.
    assert all(value == 0 for value in summary['counters']['delta'].values())
    assert not stream.done


def test_next_event_exactly_at_cutoff_is_observed_without_an_extra_issue(tmp_path):
    _, stream, summary = run_suffix(tmp_path, 'mid',
        [(10, [('A', 0)]), (310, [('B', 0)]), (1000, [('C', 0)])])
    assert summary['integrity']['all_pass']
    assert summary['truth']['next_event']['dt'] == 300.
    assert summary['truth']['next_event']['cost'] == 2100.
    assert stream.issued_count == 1
    assert summary['global_transition']['next_vehicle_id'] == 'B'


def test_native_application_is_captured_when_event_clock_precedes_issue_ledger(tmp_path):
    _, _, summary = run_suffix(tmp_path, 'mid',
        [(10, [('A', 0)]), (20, [('B', 0)]), (1000, [('C', 0)])], issue_lag_seconds=1)
    assert summary['integrity']['all_pass']
    assert summary['selected']['started_at'] == 11.
    application = summary['execution']['native_application']
    assert application['time'] == 10.
    assert application['issue_ledger_time'] == 11.
    assert application['applied_hold_seconds'] == .9375
    assert summary['truth']['next_event']['dt'] == 9.
    assert summary['truth']['fixed300']['dt'] == 300.


def test_unobserved_next_event_stays_missing_instead_of_zero(tmp_path):
    _, stream, summary = run_suffix(tmp_path, 'low',
        [(10, [('A', 0)]), (1000, [('B', 0)])])
    assert summary['integrity']['all_pass']
    assert summary['truth']['next_event']['status'] == 'missing'
    assert summary['truth']['next_event']['dt'] is None
    assert summary['truth']['next_event']['cost'] is None
    assert summary['truth']['fixed300']['dt'] == 300.
    assert stream.now == 310.


def test_any_suffix_training_invalidates_the_branch(tmp_path):
    _, _, summary = run_suffix(tmp_path, 'low',
        [(10, [('A', 0)]), (20, [('B', 0)]), (1000, [('C', 0)])], mutate_counts=True)
    assert not summary['integrity']['all_pass']
    assert not summary['integrity']['checks']['suffix_zero_training']
    assert summary['counters']['delta']['policy'] == 1


@pytest.mark.parametrize('arm,expected_departure', [('low', 11.), ('mid', 11.), ('high', 12.)])
def test_real_departure_next_same_vehicle_arrival_and_short_native_ticks(tmp_path, arm, expected_departure):
    _, _, summary = run_suffix(tmp_path, arm,
        [(10, [('A', 0)]), (20, [('B', 0)]), (1000, [('C', 0)])])
    physical = summary['physical']
    assert physical['apply_before']['native_clock_seconds'] == 10.
    assert physical['apply_after']['native_clock_seconds'] == 10.
    assert physical['apply_before']['action_step_length_seconds'] == 1.
    assert physical['apply_before']['simulation_step_length_seconds'] == 1.
    assert physical['apply_before']['last_action_time_seconds'] == 10.
    assert physical['apply_before']['is_at_bus_stop'] is True
    assert physical['apply_before']['stop_state'] == 17
    assert physical['apply_before']['registered_stop_duration_seconds'] == 3600.
    assert physical['apply_after']['registered_stop_duration_seconds'] == HOLD_SECONDS[arm]
    assert physical['departure']['clock_seconds'] == expected_departure
    assert physical['departure']['station_id'] == 'S0'
    assert physical['departure']['time_source'] == 'traci_isAtBusStop_true_to_false'
    assert physical['native_history_departure']['clock_seconds'] == expected_departure
    assert physical['native_history_departure']['observed_at'] == expected_departure+1
    assert physical['next_same_vehicle_arrival']['clock_seconds'] == expected_departure+40
    assert physical['next_same_vehicle_arrival']['station_id'] == 'S1'
    assert physical['native_history_next_arrival']['clock_seconds'] == expected_departure+40
    assert [row['native_clock_seconds'] for row in physical['short_tick_samples']] == list(range(11, 23))
    assert physical['native_tick_count'] == 300
    assert physical['transition_rows_omitted'] == 0
    assert all(physical['coverage'].values())
    # The command's continuous metadata is explicitly not the physical clock.
    if arm == 'mid':
        assert summary['execution']['native_application']['predicted_departure_time_seconds'] == 10.9375
        assert physical['departure']['clock_seconds'] != 10.9375


def test_physical_clock_reads_current_native_time_not_old_queued_event(tmp_path):
    _, _, summary = run_suffix(tmp_path, 'high',
        [(10, [('A', 0)]), (20, [('B', 0)]), (1000, [('C', 0)])], issue_lag_seconds=1)
    application, physical = summary['execution']['native_application'], summary['physical']
    assert application['time'] == 10.
    assert application['real_clock_before'] == application['real_clock_after'] == 11.
    assert physical['apply_before']['native_clock_seconds'] == 11.
    assert physical['apply_after']['native_clock_seconds'] == 11.
    assert physical['departure']['clock_seconds'] == 13.


def test_missing_next_same_vehicle_arrival_remains_truncated_not_zero():
    env = AppliedNativeEnv([])
    observer = NativePhysicalObserver(env._bridge, traci=env.traci)
    observer.activate(dict(trip_id='A', station_id='S0', started_at=10.))
    observer.departure = observer._event('S0', 11., 'native_test_departure', 11.)
    value = observer.summary(True)
    assert value['next_same_vehicle_arrival']['status'] == 'truncated'
    assert value['next_same_vehicle_arrival']['clock_seconds'] is None
    assert not value['coverage']['next_arrival_observed']
    assert value['departure']['status'] == 'observed'
    assert observer.summary(False)['next_same_vehicle_arrival']['status'] == 'missing'


def test_removed_vehicle_queries_do_not_fabricate_departure():
    env = AppliedNativeEnv([])
    observer = NativePhysicalObserver(env._bridge, traci=env.traci)
    observer.activate(dict(trip_id='A', station_id='S0', started_at=10.))
    row = observer.sample()
    observer.observe(row)
    env.missing_physical_api = True
    env._bridge.current_time = 11.
    observer.observe(observer.sample(), tick=True)
    value = observer.summary(True)
    assert value['departure']['status'] == 'truncated'
    assert value['departure']['clock_seconds'] is None
    assert value['short_tick_samples'][0]['is_at_bus_stop'] is None
    assert value['short_tick_samples'][0]['registered_stop_duration_seconds'] is None
    assert value['missing']['is_at_bus_stop'] > 0


def test_departure_from_a_later_stop_is_not_selected_stop_departure():
    env = AppliedNativeEnv([])
    bridge = env._bridge
    bridge.current_time, env.depart_at, env.arrive_at = 10., 9., 11.
    observer = NativePhysicalObserver(bridge, traci=env.traci)
    observer.activate(dict(trip_id='A', station_id='S0', started_at=10.))
    observer.observe(observer.sample())
    bridge.current_time = 11.
    observer.observe(observer.sample(), tick=True)
    bridge.current_time = 12.
    env.arrival_enabled = False
    observer.observe(observer.sample(), tick=True)
    result = observer.summary(True)
    assert result['departure']['clock_seconds'] is None
    assert result['next_same_vehicle_arrival']['clock_seconds'] is None
