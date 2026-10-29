"""Catch unequal paired prefixes, sibling interventions and endpoint overrun."""

import os
from pathlib import Path

import numpy as np

os.environ.setdefault('THESIS_H2O_ROOT', str(Path(__file__).resolve().parents[1]/'sumo_eval/runtime/H2Oplus'))

from experiments.mcwm_closed_loop_action_fork_v7_20261020.test_fork_stream import FakeNativeEnv, policy
from experiments.mcwm_closed_loop_action_response_v8_20261021.action_response import ActionResponseMemory
from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.controlled_stream import ControlledStream
from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.endpoints import ResponseEndpointCollector
from experiments.mcwm_closed_loop_response_residual_v11_20261021.paired_stream import PairedControlledStream
from experiments.mcwm_closed_loop_v1.stream import EI, TransitMDPConfigV7


def make(env, *, selector=lambda event: None, fork_selector=None, callback=None,
         endpoints=None, cls=PairedControlledStream):
    kwargs = dict(selector=selector, endpoints=endpoints or ResponseEndpointCollector(),
        response_memory=ActionResponseMemory(), execute_transform=False,
        on_issue=callback, delay_seconds=30.)
    if cls is PairedControlledStream:
        kwargs['fork_selector'] = fork_selector
    result = cls(env, [3], TransitMDPConfigV7(horizon_seconds=24000), **kwargs)
    result.reset()
    result.attach_native_clock()
    return result


def test_repeated_calibration_prefix_matches_v10_without_a_fork():
    batches = [(10, [('A', 0), ('B', 0)]), (20, [('A', 1)]),
               (40, [('B', 1)]), (60, [('C', 0)])]
    issues = [[], []]
    select = lambda event: {0: [-.75], 2: [-1.]}.get(event['sequence'])
    left = make(FakeNativeEnv(batches), selector=select, callback=issues[0].append, cls=ControlledStream)
    right = make(FakeNativeEnv(batches), selector=select, callback=issues[1].append)
    for _ in range(3):
        expected, actual = left.step(policy), right.step(policy)
        assert len(expected) == len(actual)
        for a, b in zip(expected, actual):
            for name in ('s', 'a', 'ns', 'raw_s', 'raw_ns'):
                np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
            assert (a.dt, a.r, a.sequence, a.available_at) == (b.dt, b.r, b.sequence, b.available_at)
    assert left.env._bridge.actions == right.env._bridge.actions
    assert left.executed_controls == right.executed_controls
    assert left.native_ticks == right.native_ticks
    for a, b in zip(*issues):
        for name in ('s', 'raw_s', 'a', 'proposed_a'):
            np.testing.assert_array_equal(a[name], b[name])
    assert right.fork is None


def test_pair_preserves_calibrated_nominal_sibling_actions_and_actual_recorded_states():
    batches = [(10, [('A', 0), ('B', 0)]), (20, [('A', 1), ('B', 1)]),
               (40, [('C', 0)]), (400, [('D', 0)])]
    branches = []
    for action in (-1., -.75):
        issues, selector_inputs = [], []

        def selector(event):
            selector_inputs.append((event['sequence'], event['s'].copy()))
            return {0: [-.75], 2: [-.75]}.get(event['sequence'])

        candidate = make(FakeNativeEnv(batches), selector=selector,
            fork_selector=lambda event, action=action: [action] if event['sequence'] == 2 else None,
            callback=issues.append)
        candidate.step(policy)
        candidate.step(policy)
        branches.append((candidate, issues, selector_inputs))
    low, high = branches
    assert low[0].env._bridge.actions[:2] == high[0].env._bridge.actions[:2]
    assert low[0].fork_batch_other_controls == high[0].fork_batch_other_controls
    assert low[0].fork_batch_other_controls[0]['a'] == [-.375]
    assert low[1][3]['s'][18] == -1. and high[1][3]['s'][18] == -.75
    assert low[2][3][1][18] == high[2][3][1][18] == -.75
    np.testing.assert_array_equal(low[0].fork['s'], high[0].fork['s'])
    assert low[0].fork['nominal_a'].tolist() == high[0].fork['nominal_a'].tolist() == [-.75]
    assert low[0].fork_transition.dt == high[0].fork_transition.dt == 0.
    assert low[0].fork_transition.r == high[0].fork_transition.r == 0.
    assert low[0].env._bridge.actions[2][2] == [0., 1.]
    assert high[0].env._bridge.actions[2][2] == [7.5, 1.]


def registered_fork(env):
    endpoints = ResponseEndpointCollector(delay_seconds=30., duration_seconds=300.)
    candidate = None

    def register(event):
        if event['sequence'] == 0:
            endpoints.register(event, candidate._snapshot(), role='heldout', hold_seconds=7.5)

    candidate = make(env, endpoints=endpoints,
        fork_selector=lambda event: [-.75] if event['sequence'] == 0 else None, callback=register)
    return candidate, endpoints


def test_native_cutoff_preserves_unobserved_same_vehicle_endpoint_as_missing():
    candidate, endpoints = registered_fork(FakeNativeEnv([
        (10, [('A', 0)]), (20, [('B', 0)]), (500, [('A', 1)])]))
    rows = []
    while not candidate.cutoff_reached:
        rows.extend(candidate.step(policy))
    record = endpoints.records[0]
    assert candidate.now == candidate.cutoff_time == 310.
    assert candidate.native_suffix_steps == candidate.native_ticks == 300
    assert candidate.cutoff_snapshot.time_seconds == 310.
    assert record.endpoints['fixed300']['dt'] == 300.
    assert record.endpoints['fixed300']['wait'] == 600.
    assert record.endpoints['global_next']['dt'] == candidate.fork_transition.dt == 10.
    assert record.endpoints['same_vehicle_next']['status'] == 'pending'
    endpoints.finish(candidate.now, reason='paired_native_300_second_cutoff')
    assert record.endpoints['same_vehicle_next']['status'] == 'missing'
    assert record.endpoints['same_vehicle_next']['reason'] == 'paired_native_300_second_cutoff'
    assert not candidate.done and candidate.final_snapshot is None
    assert not any(row.terminal for row in rows)
    before = candidate.env._bridge.steps
    candidate.step(policy)
    assert candidate.env._bridge.steps == before


def test_actual_same_vehicle_event_at_cutoff_is_observed_without_new_control():
    candidate, endpoints = registered_fork(FakeNativeEnv([
        (10, [('A', 0)]), (310, [('A', 1)]), (500, [('B', 0)])]))
    issued = []
    register = candidate.on_issue
    candidate.on_issue = lambda event: (issued.append(event), register(event))
    candidate.step(policy)
    record = endpoints.records[0]
    assert candidate.cutoff_reached and candidate.now == 310.
    assert record.status == 'complete'
    assert record.endpoints['global_next']['dt'] == 300.
    assert record.endpoints['same_vehicle_next']['dt'] == 300.
    assert record.endpoints['same_vehicle_next']['next_station_id'] == 'S1'
    assert record.endpoints['fixed300']['dt'] == 300.
    assert len(issued) == candidate.issued_count == 1
    assert candidate.env._bridge.actions == [(10., 'A', [7.5, 1.])]
    assert candidate.fork_transition.available_at == 340.
    assert candidate.release() == []


def test_terminal_same_vehicle_event_at_cutoff_is_not_a_global_control():
    candidate, endpoints = registered_fork(FakeNativeEnv([
        (10, [('A', 2)]), (310, [('A', 3)]), (500, [('B', 0)])]))
    candidate.step(policy)
    record = endpoints.records[0]
    assert candidate.cutoff_reached
    assert record.endpoints['same_vehicle_next']['status'] == 'complete'
    assert record.endpoints['same_vehicle_next']['next_station_id'] == 'S3'
    assert record.endpoints['global_next']['status'] == 'pending'
    assert candidate.fork_transition is None
    assert candidate.issued_count == 1
    assert not candidate.done


def test_cutoff_global_event_counts_even_after_fork_transition_already_completed():
    candidate, endpoints = registered_fork(FakeNativeEnv([
        (10, [('A', 0)]), (20, [('B', 0)]), (310, [('C', 0)]), (500, [('A', 1)])]))
    candidate.step(policy)
    candidate.step(policy)
    assert candidate.fork_transition.next_vehicle_id == 'B'
    assert endpoints.records[0].endpoints['fixed300']['global_events'] == 2
    assert candidate.completed_count == 2 and candidate.issued_count == 2
    assert candidate._pending is None


def test_same_vehicle_endpoint_keeps_native_delivery_clock_instead_of_detection_clock():
    class DeliveredNextTickEnv(FakeNativeEnv):
        def _event_to_obs(self, event):
            observed = super()._event_to_obs(event)
            observed[EI['sim_time_seconds']] -= 1.
            return observed

    candidate, endpoints = registered_fork(DeliveredNextTickEnv([
        (10, [('A', 0)]), (20, [('A', 1)]), (500, [('B', 0)])]))
    candidate.step(policy)
    candidate.step(policy)
    endpoint = endpoints.records[0].endpoints['same_vehicle_next']
    assert endpoint['event_detected_at'] == 19.
    assert endpoint['event_delivered_at'] == 20.
    assert endpoint['delivery_lag_seconds'] == 1.
    assert endpoint['dt'] == 10.
    assert endpoint['wait'] == 20.
