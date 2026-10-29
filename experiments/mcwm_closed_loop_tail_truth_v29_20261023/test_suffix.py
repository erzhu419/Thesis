"""Ledger and action tests; live SUMO/fork parity is a resident native gate."""
import pytest

from experiments.mcwm_closed_loop_action_fork_v7_20261020.fork_stream import ForkEventStream
from experiments.mcwm_closed_loop_action_fork_v7_20261020.test_fork_stream import FakeNativeEnv, policy
from experiments.mcwm_closed_loop_v1.stream import TransitMDPConfigV7
from experiments.mcwm_closed_loop_tail_truth_v29_20261023.suffix import NativeTruthPair


class AppliedNativeEnv(FakeNativeEnv):
    def reset(self):
        super().reset()
        apply = self._bridge.apply_action

        def native_apply(event, command):
            apply(event, command)
            event.assigned_action = float(command[0])
            event.metadata = dict(applied_additional_hold_seconds=float(command[0]),
                predicted_departure_time_seconds=event.sim_time+float(command[0]))

        self._bridge.apply_action = native_apply


def run_suffix(tmp_path, arm, batches, mutate_counts=False, issue_lag_seconds=0):
    contract = TransitMDPConfigV7(horizon_seconds=24000)
    actions = {name: [2*seconds/contract.max_hold_seconds-1]
        for name, seconds in (('low', 0.), ('high', .9375))}
    counts = dict(policy=456, feedback=285, fresh_shadow=248, no_fresh_shadow=248,
        world=0, trust=0)
    pair = NativeTruthPair(tmp_path, dict(selected=dict(sequence=0), actions=actions),
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
    pair._install_native_observer(stream)
    while not stream.cutoff_reached:
        stream.step(policy)
        pair.observe_progress(stream)
    if mutate_counts:
        counts['policy'] += 1
    return pair, stream, pair.build_summary(stream)


@pytest.mark.parametrize('arm,seconds', [('low', 0.), ('high', .9375)])
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
    _, stream, summary = run_suffix(tmp_path, 'high',
        [(10, [('A', 0)]), (310, [('B', 0)]), (1000, [('C', 0)])])
    assert summary['integrity']['all_pass']
    assert summary['truth']['next_event']['dt'] == 300.
    assert summary['truth']['next_event']['cost'] == 2100.
    assert stream.issued_count == 1
    assert summary['global_transition']['next_vehicle_id'] == 'B'


def test_native_application_is_captured_when_event_clock_precedes_issue_ledger(tmp_path):
    _, _, summary = run_suffix(tmp_path, 'high',
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
