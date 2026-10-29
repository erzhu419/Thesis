"""Parent budget cutoff and child continuation on the real native tick path."""
import ast
from pathlib import Path

import numpy as np

from experiments.mcwm_closed_loop_supervision_plan_v84_20261027.test_stream import (
    stream_class as remaining_day_class, TerminalEnv,
)
from experiments.mcwm_closed_loop_action_fork_v7_20261020.test_fork_stream import (
    FakeNativeEnv, policy,
)
from experiments.mcwm_closed_loop_v1.stream import EventStream, TransitMDPConfigV7


def collection_class():
    path = Path(__file__).with_name('stream.py')
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.ClassDef) and node.name == 'CollectionStream')
    namespace = dict(RemainingDayStream=remaining_day_class())
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['CollectionStream']


def make_stream(env, issued=None, selector=lambda _: None, *, limit=25, horizon=24000):
    return collection_class()(env, [3], TransitMDPConfigV7(horizon_seconds=horizon),
        selector=selector, prefix_limit_seconds=limit, execute_transform=False,
        delay_seconds=30, on_issue=None if issued is None else issued.append)


def test_disabled_parent_limit_keeps_original_prefix_exact():
    batches = [(10, [('A', 0), ('B', 0)]), (20, [('A', 1)]),
               (30, [('B', 1)]), (40, [('C', 0)])]
    original_env, actual_env = FakeNativeEnv(batches), FakeNativeEnv(batches)
    original = EventStream(original_env, [3], TransitMDPConfigV7(),
                           execute_transform=False, delay_seconds=30)
    actual = make_stream(actual_env, limit=15)
    for _ in range(3):
        expected, observed = original.step(policy), actual.step(policy)
        assert len(expected) == len(observed)
        for left, right in zip(expected, observed):
            for key in ('s', 'a', 'ns', 'raw_s', 'raw_ns'):
                np.testing.assert_array_equal(getattr(left, key), getattr(right, key))
            for key in ('r', 'dt', 'terminal', 'available_at', 'sequence'):
                assert getattr(left, key) == getattr(right, key)
    assert actual_env._bridge.actions == original_env._bridge.actions
    assert actual.now > actual.prefix_limit_seconds
    assert actual_env.native_advance_calls == original_env.native_advance_calls


def test_parent_stops_at_real_tick_without_terminal_or_later_event():
    env = FakeNativeEnv([(10, [('A', 0)]), (20, [('B', 0)]), (1000, [('C', 0)])])
    stream = make_stream(env)
    stream.reset()
    stream.enable_prefix_limit()
    while not stream.prefix_limit_reached:
        stream.step(policy)
    assert stream.now == 25.
    assert not stream.done and stream.final_snapshot is None
    assert env._bridge.native_tick_calls == 25
    assert not any(row.terminal for row in stream._queue)
    assert stream._pending['vehicle_id'] == 'B'
    assert stream._pending['started_at'] == 20.
    steps = env._bridge.steps
    stream.step(policy)
    assert env._bridge.steps == steps


def test_issue_exactly_at_parent_boundary_is_not_selected():
    issued = []
    env = FakeNativeEnv([(10, [('A', 0)]), (25, [('B', 0)]), (1000, [('C', 0)])])
    stream = make_stream(env, issued)
    stream.reset()
    stream.enable_prefix_limit()
    stream.step(policy)
    assert stream.now == 25 and stream.prefix_limit_reached
    assert [row['trip_id'] for row in issued] == ['A']
    assert env._bridge.decision_queue[0].bus_id == 'B'
    assert stream.final_snapshot is None and not stream.done


def test_child_entering_suffix_inside_parent_step_runs_to_real_terminal():
    issued = []
    stream = None

    def selector(event):
        if event['sequence'] == 0:
            stream.enter_suffix()
            return [1.]
        return None

    env = TerminalEnv(40)
    stream = make_stream(env, issued, selector, limit=15, horizon=40)
    stream.reset()
    stream.enable_prefix_limit()
    while not stream.done:
        stream.step(policy)
    assert stream.now == stream.final_snapshot.time_seconds == 40.
    assert not stream.prefix_limit_reached
    assert issued[0]['a'].tolist() == [1.]
    assert issued[1]['s'][18] == 1.
    assert issued[1]['a'].tolist() == [.5]
    terminal = stream._queue[-1]
    assert terminal.terminal and terminal.available_at == 70.
    assert env.native_advance_calls > 1
