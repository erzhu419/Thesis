"""Detect a logged-versus-executed intervention mismatch and wrong native clocks."""
from pathlib import Path
import os
import numpy as np

os.environ.setdefault('THESIS_H2O_ROOT', str(Path(__file__).resolve().parents[1]/'sumo_eval/runtime/H2Oplus'))

from experiments.mcwm_closed_loop_action_fork_v7_20261020.test_fork_stream import (
    FakeNativeEnv, policy,
)
from experiments.mcwm_closed_loop_v1.stream import EventStream, TransitMDPConfigV7
from experiments.mcwm_closed_loop_action_response_v8_20261021.action_response import ActionResponseMemory
from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.controlled_stream import ControlledStream
from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.endpoints import ResponseEndpointCollector


def make(env, selector, callback=None, endpoints=None):
    return ControlledStream(env, [3], TransitMDPConfigV7(horizon_seconds=24000),
        selector=selector, endpoints=endpoints or ResponseEndpointCollector(),
        response_memory=ActionResponseMemory(), execute_transform=False,
        on_issue=callback, delay_seconds=30.)


def test_no_control_replacement_preserves_native_prefix_exactly():
    batches = [(10, [('A', 0), ('B', 0)]), (20, [('A', 1)]),
               (30, [('B', 1)]), (40, [('C', 0)])]
    leftenv, rightenv = FakeNativeEnv(batches), FakeNativeEnv(batches)
    leftissues, rightissues = [], []
    baseline = EventStream(leftenv, [3], TransitMDPConfigV7(horizon_seconds=24000),
        execute_transform=False, on_issue=leftissues.append, delay_seconds=30.)
    candidate = make(rightenv, lambda e: None, rightissues.append)
    candidate.reset()
    candidate.attach_native_clock()
    for _ in range(3):
        expected, actual = baseline.step(policy), candidate.step(policy)
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left.s, right.s)
            np.testing.assert_array_equal(left.ns, right.ns)
            assert (left.dt, left.r, left.sequence) == (right.dt, right.r, right.sequence)
    assert leftenv._bridge.actions == rightenv._bridge.actions
    assert len(leftissues) == len(rightissues)


def test_repeated_controls_reach_native_commands_and_same_batch_prior():
    env = FakeNativeEnv([(10, [('A', 0), ('B', 0)]), (20, [('A', 1)]),
                         (40, [('B', 1)]), (60, [('C', 0)])])
    issues = []
    replacement = {0: [-.75], 2: [-1.]}
    candidate = make(env, lambda e: replacement.get(e['sequence']), issues.append)
    released = []
    for _ in range(3):
        released.extend(candidate.step(policy))
    assert [(r['sequence'], r['hold_seconds']) for r in candidate.executed_controls] == [(0, 7.5), (2, 0.)]
    assert env._bridge.actions[0][2] == [7.5, 1.]
    assert env._bridge.actions[2][2] == [0., 1.]
    assert issues[1]['s'][18] == -.75
    row = next(r for r in released if r.sequence == 0)
    assert row.dt == row.r == 0.
    assert row.a.tolist() == [-.75]


def test_exact_fixed_endpoint_observed_inside_native_event_gap():
    env = FakeNativeEnv([(10, [('A', 0)]), (20, [('A', 1)]),
                         (400, [('B', 0)]), (450, [('B', 1)])])
    endpoints = ResponseEndpointCollector(delay_seconds=30., duration_seconds=300.)
    candidate = None

    def callback(event):
        if event['sequence'] == 0:
            endpoints.register(event, candidate._snapshot(), role='train', hold_seconds=7.5)

    candidate = make(env, lambda e: [-.75] if e['sequence'] == 0 else None, callback, endpoints)
    candidate.reset()
    candidate.attach_native_clock()
    candidate.step(policy)
    candidate.step(policy)
    record = endpoints.records[0]
    assert candidate.now == 400.
    assert record.endpoints['fixed300']['completed_at'] == 310.
    assert record.endpoints['fixed300']['wait'] == 600.
    assert record.endpoints['global_next']['dt'] == record.endpoints['same_vehicle_next']['dt'] == 10.
    assert record.available_at == 340.
    assert [r.sequence for r in endpoints.release(candidate.now)] == [0]
    assert candidate.native_ticks == 390
