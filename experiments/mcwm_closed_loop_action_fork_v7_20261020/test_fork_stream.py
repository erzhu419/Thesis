"""Native callback-path tests for prefix parity and exact action interventions."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("THESIS_H2O_ROOT", str(Path(__file__).resolve().parents[1] / "sumo_eval/runtime/H2Oplus"))

from experiments.mcwm_closed_loop_action_fork_v7_20261020.fork_stream import ForkEventStream
from experiments.mcwm_closed_loop_v1.stream import EventStream, EI, TransitMDPConfigV7
from bus_h2o.mdp_contract_v7 import RAW_EVENT_STATE_DIM
from bus_h2o.passenger_time import PassengerTimeSnapshot


def snapshot(time):
    return PassengerTimeSnapshot(float(time), 2, 3, 2. * time, 3. * time)


def local(time, stop):
    row = np.zeros(RAW_EVENT_STATE_DIM, np.float32)
    for key, value in dict(line_id=0, station_index=stop, sim_time_seconds=time,
        forward_headway_seconds=300, backward_headway_seconds=360,
        target_forward_headway_seconds=360, target_backward_headway_seconds=360,
        bus_capacity=50, onboard_passengers=5, eligible_arrival_rate_per_second=.1,
        base_dwell_seconds=10, forward_bus_present=1, backward_bus_present=0,
        system_waiting_passengers=2, system_in_vehicle_passengers=3).items():
        row[EI[key]] = value
    return row


class FakeBridge:
    """Native callback surface; there is deliberately no save/load snapshot."""

    def __init__(self, batches):
        self.batches = dict(batches)
        self.current_time, self.steps, self.done = 0., 0, False
        self.decision_queue = []
        self.actions = []
        self.passenger_time_ledger = SimpleNamespace(snapshot=lambda: snapshot(self.current_time))
        self.line_obj_dic = {"L": SimpleNamespace(stop_id_l=["S0", "S1", "S2", "S3"])}
        self.fleet_obj_dic = {}
        self.native_fetch_calls = self.native_tick_calls = 0

    def apply_action(self, event, command):
        self.actions.append((self.current_time, event.bus_id, list(command)))

    def _advance_one_step(self):
        self.native_tick_calls += 1
        self.steps += 1
        self.current_time += 1.
        for bus, station in self.batches.get(self.current_time, []):
            self.decision_queue.append(SimpleNamespace(bus_id=bus, line_id="L", sim_time=self.current_time,
                stop_id=f"S{station}", stop_idx=station))

    def fetch_events(self):
        self.native_fetch_calls += 1
        while not self.decision_queue:
            self._advance_one_step()
        events, self.decision_queue = self.decision_queue, []
        return events, self.done, []

    def consume_completed_trip_passenger_times(self):
        return {}


class FakeNativeEnv:
    def __init__(self, batches):
        self.batches = batches
        self.native_advance_calls = 0
        self.reset()

    def reset(self):
        self._bridge = FakeBridge(self.batches)
        self._current_events, self.state, self.reward, self.done = [], {}, {}, False

    def _event_to_obs(self, event):
        return local(event.sim_time, event.stop_idx)

    def _compute_reward(self, _event):
        return 0.

    def step_to_event(self, commands):
        self.native_advance_calls += 1
        for event in self._current_events:
            self._bridge.apply_action(event, commands.get(event.bus_id, [0., 1.]))
        self._current_events, self.done, _ = self._bridge.fetch_events()
        self.state = {event.bus_id: [[self._event_to_obs(event).tolist()]] for event in self._current_events}
        return self.state, {}, self.done


def policy(state):
    # State's previous action changes a later simultaneous control unless the
    # stream deliberately passes the nominal-prior policy state.
    return np.array([state[18] * .5], np.float32)


def stream(env, selector=lambda issued: None, duration=300, issued=None):
    return ForkEventStream(env, [3], TransitMDPConfigV7(horizon_seconds=24000),
        selector=selector, fork_duration_seconds=duration, execute_transform=False,
        on_issue=None if issued is None else issued.append, delay_seconds=30.)


def test_prefix_preserves_original_native_actions_and_transitions_exactly():
    batches = [(10, [("A", 0), ("B", 0)]), (20, [("A", 1)]), (30, [("B", 1)]), (40, [("C", 0)])]
    original_env, fork_env = FakeNativeEnv(batches), FakeNativeEnv(batches)
    original_issues, fork_issues = [], []
    original = EventStream(original_env, [3], TransitMDPConfigV7(horizon_seconds=24000),
        execute_transform=False, delay_seconds=30, on_issue=original_issues.append)
    candidate = stream(fork_env, issued=fork_issues)
    for _ in range(3):
        expected, actual = original.step(policy), candidate.step(policy)
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left.s, right.s)
            np.testing.assert_array_equal(left.ns, right.ns)
            np.testing.assert_array_equal(left.a, right.a)
            assert (left.r, left.dt, left.sequence) == (right.r, right.dt, right.sequence)
    assert original_env._bridge.actions == fork_env._bridge.actions
    assert original_env.native_advance_calls == fork_env.native_advance_calls
    for left, right in zip(original_issues, fork_issues):
        np.testing.assert_array_equal(left["raw_s"], right["raw_s"])
        np.testing.assert_array_equal(left["a"], right["a"])
    assert candidate.fork is None


def test_only_selected_command_changes_and_actual_zero_time_global_event_is_retained():
    batches = [(10, [("A", 0), ("B", 0)]), (20, [("C", 0)]), (30, [("D", 0)])]
    branches = []
    for action in (-1., 1.):
        env, issued = FakeNativeEnv(batches), []
        branch = stream(env, selector=lambda event, action=action: [action], duration=15, issued=issued)
        branch.step(policy)
        assert branch.fork["sequence"] == 0
        assert branch.fork_transition.dt == branch.fork_transition.r == 0.
        assert (branch.fork_transition.vehicle_id, branch.fork_transition.next_vehicle_id) == ("A", "B")
        assert not branch.fork_transition.terminal
        assert branch.fork_transition.available_at == 40
        # Not yet available to training, but diagnostic observation exists.
        assert branch.release() == []
        branches.append((branch, issued))
    low, high = branches
    assert low[0].fork_batch_other_controls == high[0].fork_batch_other_controls
    assert low[0].fork_batch_other_controls[0]["vehicle_id"] == "B"
    np.testing.assert_array_equal(low[1][0]["s"], high[1][0]["s"])
    assert low[1][1]["a"].tolist() == high[1][1]["a"].tolist() == [-.25]
    assert low[1][1]["s"][18] == -1. and high[1][1]["s"][18] == 1.
    assert low[0].env._bridge.actions[0][2] == [0., 1.]
    assert high[0].env._bridge.actions[0][2] == [60., 1.]


def test_exact_300_seconds_ledger_cutoff_does_not_advance_to_later_event_or_fabricate_terminal():
    env = FakeNativeEnv([(10, [("A", 0)]), (20, [("B", 0)]), (1000, [("C", 0)])])
    candidate = stream(env, selector=lambda event: [-1.])
    rows = []
    while not candidate.cutoff_reached:
        rows.extend(candidate.step(policy))
    assert candidate.now == candidate.cutoff_time == 310.
    assert candidate.cutoff_snapshot.time_seconds == 310.
    assert candidate.native_suffix_steps == 300
    assert candidate.cutoff_snapshot.accrued_wait_seconds - snapshot(10).accrued_wait_seconds == 600
    assert candidate.cutoff_snapshot.accrued_in_vehicle_seconds - snapshot(10).accrued_in_vehicle_seconds == 900
    assert candidate.fork_transition.dt == 10.
    assert candidate.fork_transition.next_vehicle_id == "B"
    assert not candidate.done and not env._bridge.done
    assert candidate.final_snapshot is None
    assert all(not row.terminal for row in rows)
    assert candidate._pending["vehicle_id"] == "B"  # uncompleted, not fabricated
    assert env.native_advance_calls == 1  # reset; suffix used full bridge ticks
    before = env._bridge.steps
    candidate.step(policy)
    assert env._bridge.steps == before


def test_actual_next_global_at_exact_endpoint_is_observed_without_issuing_action():
    env = FakeNativeEnv([(10, [("A", 0)]), (310, [("B", 0)]), (1000, [("C", 0)])])
    issued = []
    candidate = stream(env, selector=lambda event: [1.], issued=issued)
    candidate.step(policy)
    assert candidate.cutoff_reached and candidate.now == 310.
    assert candidate.fork_transition.dt == 300.
    assert candidate.fork_transition.next_vehicle_id == "B"
    assert len(issued) == 1 and candidate.issued_count == 1
    assert candidate.fork_transition.available_at == 340.
    assert candidate.release() == []
    assert not candidate.done and candidate.final_snapshot is None
