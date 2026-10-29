"""Actual-prior continuation and the simulator's sole real terminal endpoint."""
import ast
from pathlib import Path

import numpy as np
import pytest

from experiments.mcwm_closed_loop_action_fork_v7_20261020.test_fork_stream import (
    FakeNativeEnv, policy,
)
from experiments.mcwm_closed_loop_v1.stream import (
    EventStream, TransitMDPConfigV7, pack_event_batch_v7,
    policy_action_to_hold_seconds_v7, terminal_raw_state_v7,
)


def stream_class():
    # Import only the class; production run.py intentionally bootstraps resident
    # initializers and the scheduler environment, neither needed by a stream test.
    path = Path(__file__).with_name('run.py')
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.ClassDef) and node.name == 'RemainingDayStream')
    namespace = dict(EventStream=EventStream, np=np,
        pack_event_batch_v7=pack_event_batch_v7,
        policy_action_to_hold_seconds_v7=policy_action_to_hold_seconds_v7,
        terminal_raw_state_v7=terminal_raw_state_v7)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['RemainingDayStream']


def make_stream(env, issued, selector=lambda _: None, *, horizon=24000):
    return stream_class()(env, [3], TransitMDPConfigV7(horizon_seconds=horizon),
        selector=selector, execute_transform=False, delay_seconds=30,
        on_issue=issued.append)


def test_no_intervention_matches_original_prefix_actions_and_labels():
    batches = [(10, [('A', 0), ('B', 0)]), (20, [('A', 1)]),
               (30, [('B', 1)]), (40, [('C', 0)])]
    original_issues, actual_issues = [], []
    original_env, actual_env = FakeNativeEnv(batches), FakeNativeEnv(batches)
    original = EventStream(original_env, [3], TransitMDPConfigV7(horizon_seconds=24000),
        execute_transform=False, delay_seconds=30, on_issue=original_issues.append)
    actual = make_stream(actual_env, actual_issues)
    for _ in range(3):
        expected_rows, actual_rows = original.step(policy), actual.step(policy)
        assert len(expected_rows) == len(actual_rows)
        for expected, observed in zip(expected_rows, actual_rows):
            for field in ('s', 'a', 'ns', 'raw_s', 'raw_ns'):
                np.testing.assert_array_equal(getattr(expected, field), getattr(observed, field))
            for field in ('r', 'dt', 'terminal', 'sequence', 'available_at'):
                assert getattr(expected, field) == getattr(observed, field)
    assert original_env._bridge.actions == actual_env._bridge.actions
    assert len(original_issues) == len(actual_issues)
    for expected, observed in zip(original_issues, actual_issues):
        for key in ('s', 'raw_s', 'a', 'proposed_a'):
            np.testing.assert_array_equal(expected[key], observed[key])
        assert expected['sequence'] == observed['sequence']


def test_one_root_intervention_changes_later_same_batch_policy_observation():
    batches = [(10, [('A', 0), ('B', 0)]), (20, [('C', 0)]), (30, [('D', 0)])]
    runs = []
    for action in (-1., 1.):
        issued, replaced = [], []

        def selector(event):
            if event['sequence'] == 0:
                replaced.append(event['sequence'])
                return [action]
            return None

        env = FakeNativeEnv(batches)
        stream = make_stream(env, issued, selector)
        stream.step(policy)
        stream.step(policy)
        assert replaced == [0]
        assert [row['sequence'] for row in issued] == [0, 1, 2]
        assert issued[0]['a'].tolist() == [action]
        # This actor explicitly responds to the previous executed action.
        assert issued[1]['s'][18] == action
        assert issued[1]['a'].tolist() == [action*.5]
        assert env._bridge.actions[1][2][0] == (action*.5+1)*30
        runs.append(issued)
    np.testing.assert_array_equal(runs[0][0]['s'], runs[1][0]['s'])
    assert runs[0][1]['a'].tolist() != runs[1][1]['a'].tolist()


class TerminalEnv(FakeNativeEnv):
    def __init__(self, terminal_clock):
        self.terminal_clock = terminal_clock
        super().__init__([(10, [('A', 0), ('B', 0)]), (20, [('A', 1)]),
                          (30, [('B', 1)])])

    def step_to_event(self, commands):
        self.native_advance_calls += 1
        bridge = self._bridge
        for event in self._current_events:
            bridge.apply_action(event, commands.get(event.bus_id, [0., 1.]))
        while not bridge.decision_queue and bridge.current_time < self.terminal_clock:
            bridge._advance_one_step()
        self._current_events, bridge.decision_queue = bridge.decision_queue, []
        self.done = bridge.done = bridge.current_time == self.terminal_clock
        self.state = {event.bus_id: [[self._event_to_obs(event).tolist()]]
                      for event in self._current_events}
        return self.state, {}, self.done


def test_only_real_episode_terminal_closes_return_and_respects_label_delay():
    stream = make_stream(TerminalEnv(40), [], horizon=40)
    while not stream.done:
        stream.step(policy)
    assert stream.now == stream.final_snapshot.time_seconds == 40.
    terminal = stream._queue[-1]
    assert terminal.terminal
    assert terminal.completed_at == 40.
    assert terminal.available_at == 70.
    assert terminal.dt == 10.
    assert not any(row.terminal for row in stream.release())


def test_early_native_stop_is_not_completed_with_a_fabricated_terminal():
    stream = make_stream(TerminalEnv(39), [], horizon=40)
    with pytest.raises(RuntimeError, match='Native episode ended'):
        while not stream.done:
            stream.step(policy)
    assert stream.now == 39.
    assert not any(row.terminal for row in stream._queue)
