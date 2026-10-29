"""Dynamic executed endpoints, native physical truth and unchanged feedback."""
from types import SimpleNamespace
import json
import time

import numpy as np
import pytest

from experiments.mcwm_closed_loop_action_fork_v7_20261020.fork_stream import ForkEventStream
from experiments.mcwm_closed_loop_action_fork_v7_20261020.test_fork_stream import policy
from experiments.mcwm_closed_loop_physical_execution_v30_20261023.test_suffix import AppliedNativeEnv
from experiments.mcwm_closed_loop_v1.stream import TransitMDPConfigV7
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.suffix import NativeTruthPair, pair_comparison, selected_state


def run_suffix(arm, batches, *, holds=None, lag=0, mutate_counts=False, departure=True, arrival=True,
               quantize=False, registered_offset=0.):
    holds = holds or dict(low=2.8125, high=7.5)
    contract = TransitMDPConfigV7(horizon_seconds=24000)
    actions = {key: [2*value/60.-1] for key, value in holds.items()}
    counts = dict(policy=456, feedback=285, world=0, trust=0)
    pair = NativeTruthPair('.', dict(selected=dict(sequence=0), actions=actions, holds=holds),
        lambda: counts.copy(), 106202)
    pair.in_child, pair.arm, pair.before, pair.started_wall = True, arm, counts.copy(), 1.
    env = AppliedNativeEnv(batches)
    stream = ForkEventStream(env, [3], contract, execute_transform=False,
        selector=lambda event: actions[arm], fork_duration_seconds=300.,
        on_issue=lambda event: pair.observe_issue(dict(event, memory_s=dict(vector=[.2],
            observed_at=event['started_at'])), stream), delay_seconds=30.)
    stream.reset()
    if quantize or registered_offset:
        stops = env.traci.vehicle.getStops

        def quantized_stops(*args):
            rows = stops(*args)
            rows[0].duration = (round(rows[0].duration, 3) if quantize else rows[0].duration)+registered_offset
            return rows

        env.traci.vehicle.getStops = quantized_stops
    env._bridge.current_time += lag; env._bridge.steps += lag
    env.departure_enabled, env.arrival_enabled = departure, arrival
    pair._install_native_observer(stream, traci=env.traci)
    while not stream.cutoff_reached:
        stream.step(policy)
        pair.observe_progress(stream)
    if mutate_counts:
        counts['policy'] += 1
    return pair, stream, pair.build_summary(stream)


@pytest.mark.parametrize('arm,seconds,departure', [('low', 2.8125, 13.), ('high', 7.5, 18.)])
def test_dynamic_saved_actions_are_applied_once_and_fixed300_is_exact(arm, seconds, departure):
    _, stream, summary = run_suffix(arm,
        [(10, [('A', 0), ('B', 0)]), (20, [('C', 0)]), (1000, [('D', 0)])])
    assert summary['integrity']['all_pass']
    assert summary['execution']['executed_hold_seconds'] == seconds
    assert summary['execution']['native_applied_hold_seconds'] == seconds
    assert summary['execution']['registered_matches_requested'] is True
    assert stream.env._bridge.actions[0][2] == [seconds, 1.]
    assert summary['selected']['memory_s'] == dict(vector=[.2], observed_at=10.)
    assert summary['truth']['next_event']['zero_duration']
    assert summary['truth']['next_event']['cost'] == 0.
    assert summary['truth']['fixed300'] == dict(status='complete', dt=300., wait=600., ride=900., cost=2100.)
    assert summary['physical']['departure']['clock_seconds'] == departure
    assert summary['physical']['next_same_vehicle_arrival']['clock_seconds'] == departure+40
    assert summary['physical']['native_tick_count'] == 300
    assert all(value == 0 for value in summary['counters']['delta'].values())


def test_raw_event_clock_lag_does_not_change_real_application_clock():
    _, _, summary = run_suffix('high', [(10, [('A', 0)]), (20, [('B', 0)]), (1000, [('C', 0)])], lag=1)
    assert summary['integrity']['all_pass']
    application = summary['execution']['native_application']
    assert application['time'] == 10.
    assert application['real_clock_before'] == application['real_clock_after'] == 11.
    assert summary['physical']['departure']['clock_seconds'] == 19.


def test_native_stop_registration_allows_only_millisecond_quantization():
    _, _, summary = run_suffix('low', [(10, [('A', 0)]), (20, [('B', 0)]), (1000, [('C', 0)])],
        holds=dict(low=.9375, high=1.875), quantize=True)
    assert summary['integrity']['all_pass']
    assert summary['execution']['native_applied_hold_seconds'] == .9375
    assert summary['execution']['registered_hold_seconds'] == .938
    assert summary['execution']['registered_matches_requested'] is True


def test_observed_registered_duration_mismatch_invalidates_actual_execution():
    _, _, summary = run_suffix('high', [(10, [('A', 0)]), (20, [('B', 0)]), (1000, [('C', 0)])],
        registered_offset=.25)
    assert summary['execution']['native_applied_hold_seconds'] == 7.5
    assert summary['execution']['registered_hold_seconds'] == 7.75
    assert summary['execution']['registered_matches_requested'] is False
    assert not summary['integrity']['all_pass']
    assert not summary['integrity']['checks']['registered_hold_when_observed']


def test_missing_or_identical_departures_are_retained_without_integrity_failure():
    batches = [(10, [('A', 0)]), (20, [('B', 0)]), (1000, [('C', 0)])]
    aliases = {arm: run_suffix(arm, batches, holds=dict(low=0., high=.9375))[2] for arm in ('low', 'high')}
    assert all(value['integrity']['all_pass'] for value in aliases.values())
    result = pair_comparison(aliases)
    assert result['physical_separation']['status'] == 'nonseparable'
    assert result['physical_separation']['physical_separable'] is False
    missing = {arm: run_suffix(arm, batches, departure=False, arrival=False)[2] for arm in ('low', 'high')}
    assert all(value['integrity']['all_pass'] for value in missing.values())
    result = pair_comparison(missing)
    assert result['physical_separation']['status'] == 'physical_unknown'
    assert result['physical_separation']['physical_separable'] is None
    assert result['physical_separation']['departure_gap_seconds'] is None


def test_shared_departure_does_not_conceal_arrival_response_or_unknown_tail():
    batches = [(10, [('A', 0)]), (20, [('B', 0)]), (1000, [('C', 0)])]
    values = {arm: run_suffix(arm, batches, holds=dict(low=0., high=.9375))[2] for arm in ('low', 'high')}
    values['high']['physical']['next_same_vehicle_arrival']['clock_seconds'] += 1.
    result = pair_comparison(values)['physical_separation']
    assert result['departure_separable'] is False
    assert result['next_arrival_separable'] is True
    assert result['physical_separable'] is True and result['status'] == 'separable'
    values['high']['physical']['next_same_vehicle_arrival'].update(status='truncated', clock_seconds=None)
    result = pair_comparison(values)['physical_separation']
    assert result['departure_separable'] is False
    assert result['next_arrival_separable'] is None
    assert result['physical_separable'] is None and result['status'] == 'physical_unknown'


def test_physical_separation_and_absent_global_event_remain_distinct_endpoints():
    values = {arm: run_suffix(arm, [(10, [('A', 0)]), (1000, [('B', 0)])])[2] for arm in ('low', 'high')}
    assert all(value['integrity']['all_pass'] for value in values.values())
    comparison = pair_comparison(values)
    assert comparison['physical_separation']['physical_separable']
    assert comparison['physical_separation']['departure_gap_seconds'] == 5.
    assert comparison['high_minus_low']['next_event']['status'] == 'missing'
    assert comparison['high_minus_low']['next_event']['cost'] is None
    assert comparison['high_minus_low']['fixed300']['status'] == 'complete'
    assert comparison['high_minus_low']['fixed300']['cost'] == 0.


def test_any_suffix_training_invalidates_the_actual_branch():
    _, _, summary = run_suffix('low', [(10, [('A', 0)]), (20, [('B', 0)]), (1000, [('C', 0)])], mutate_counts=True)
    assert not summary['integrity']['all_pass']
    assert not summary['integrity']['checks']['suffix_zero_training']


def test_parent_can_fork_two_declared_states_without_entering_suffix_mode(tmp_path):
    env = AppliedNativeEnv([(10, [('A', 0)]), (20, [('B', 0)]),
        (100, [('A', 1)]), (1000, [('C', 0)]), (2000, [('D', 0)])])
    trace = SimpleNamespace(path=tmp_path/'parent.jsonl',
        writer=(tmp_path/'parent.jsonl').open('x'), reference=None)
    counts = dict(policy=456, feedback=285, world=0, trust=0)
    pairs, active = [], [None]

    class NativeTestPair(NativeTruthPair):
        def _install_native_observer(self, stream, *, traci=None):
            return super()._install_native_observer(stream, traci=stream.env.traci)

    def selector(event):
        if event['trip_id'] != 'A' or event['sequence'] not in (0, 2):
            return None
        event['memory_s'] = dict(observed_at=event['started_at'], vector=[.2])
        holds = dict(low=0., high=1.875)
        expected = dict(selected=selected_state(event, stream._snapshot()), holds=holds,
            actions={arm: [2*seconds/60.-1] for arm, seconds in holds.items()},
            extra=dict(memory_s=event['memory_s']))
        pair = NativeTestPair(tmp_path/f'state{len(pairs)}', expected, lambda: counts.copy(), 106202)
        pairs.append(pair); active[0] = pair
        # The simulated suffix is extremely short in wall time; this pause is
        # solely inside native-test children to permit a real overlap proof.
        return pair.fork(event, stream, trace, prepare_child=lambda arm: time.sleep(.06))

    def issue(event):
        trace.writer.write(json.dumps(dict(sequence=event['sequence'], action=event['a'].tolist()))+'\n')
        trace.writer.flush()
        if active[0] is not None:
            active[0].observe_issue(event, stream)

    stream = ForkEventStream(env, [3], TransitMDPConfigV7(horizon_seconds=24000),
        execute_transform=False, selector=selector, on_issue=issue, fork_duration_seconds=300.)
    stream.reset()
    original_apply = env._bridge.apply_action

    def apply_each_visit(event, command):
        original_apply(event, command)
        if event.bus_id == 'A':
            # The historical fake initializes registration only on its first
            # selected visit; the real bridge registers every native apply.
            env.registered_duration = float(command[0])

    env._bridge.apply_action = apply_each_visit
    try:
        while stream.issued_count < 3:
            stream.step(policy)
            if active[0] is not None and active[0].in_child:
                active[0].observe_progress(stream)
                if stream.cutoff_reached:
                    active[0].finish_child(stream)
        assert len(pairs) == 2
        assert stream.fork is None and not stream.cutoff_reached
        assert counts == dict(policy=456, feedback=285, world=0, trust=0)
        assert [pair.values['low']['selected']['sequence'] for pair in pairs] == [0, 2]
        for pair in pairs:
            assert pair.receipt['all_pass'] and pair.receipt['overlap_seconds'] > 0
            assert set(pair.receipt['child_pids']) == {'low', 'high'}
            assert pair.values['low']['selected'] == pair.values['high']['selected']
            assert pair.values['low']['selected']['memory_s']['vector'] == [.2]
    finally:
        trace.writer.close()
