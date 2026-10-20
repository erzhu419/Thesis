"""Verify feedback causality, event identities and the executed-action contract."""

from types import SimpleNamespace

import numpy as np
import pytest

from stream import EventStream, EI, TransitMDPConfigV7
from bus_h2o.mdp_contract_v7 import RAW_EVENT_STATE_DIM, current_event_v7
from bus_h2o.passenger_time import PassengerTimeSnapshot


def snapshot(t):
    # Constant system occupancy gives an exactly integrable cost rate: 2*2+3=7.
    return PassengerTimeSnapshot(float(t), 2, 3, 2.*t, 3.*t)


def local(t, station):
    row = np.zeros(RAW_EVENT_STATE_DIM, np.float32)
    for name, value in dict(line_id=0, station_index=station, sim_time_seconds=t,
        forward_headway_seconds=300, backward_headway_seconds=360,
        target_forward_headway_seconds=360, target_backward_headway_seconds=360,
        bus_capacity=50, onboard_passengers=5, eligible_arrival_rate_per_second=.1,
        base_dwell_seconds=10, forward_bus_present=1, backward_bus_present=0,
        system_waiting_passengers=2, system_in_vehicle_passengers=3).items():
        row[EI[name]] = value
    return row


class FakeSource:
    """Uses the supported source environment interface, including real trip IDs."""

    def __init__(self, batches=None, horizon=60):
        self.batches = batches or [(10, [(0, 0, "A"), (1, 0, "B")]),
            (20, [(2, 0, "C")]), (30, [(0, 1, "A")]), (40, [(2, 1, "C")])]
        self.horizon, self.commands = horizon, []
        line = SimpleNamespace(line_idx=0, stations=list(range(4)), bus_all=[],
                               passenger_time_ledger=SimpleNamespace(snapshot=lambda: snapshot(self.time)))
        self.line_map = {"L": line}
        self.reset()

    def reset(self):
        self.time, self.position, self.state, self.done = 0, -1, {}, False

    def passenger_time_at_event(self, agent):
        return snapshot(self.time)

    def step_to_event(self, commands):
        self.commands.append(commands)
        self.position += 1
        if self.position >= len(self.batches):
            self.time, self.state, self.done = self.horizon, {}, True
        else:
            self.time, events = self.batches[self.position]
            self.state = {("L", bus): [[local(self.time, stop)]] for bus, stop, _ in events}
            self.line_map["L"].bus_all = [SimpleNamespace(bus_id=bus, trip_id=trip,
                last_station=SimpleNamespace(station_name=f"S{stop}")) for bus, stop, trip in events]
        return self.state, {}, self.done


def collect(stream, policy=lambda s: np.array([-1.], np.float32)):
    rows, intervals = [], []
    while not stream.done or stream.episode_id < 0:
        rows.extend(stream.step(policy))
        intervals.extend(stream.release_intervals())
    return rows, intervals


def test_chronological_zero_time_cross_vehicle_and_exact_horizon_cost():
    stream = EventStream(FakeSource(), [3], TransitMDPConfigV7(horizon_seconds=60), execute_transform=False)
    rows, intervals = collect(stream)
    assert len(rows) == stream.issued_count == 5
    assert rows[0].dt == 0 and rows[0].r == 0
    assert (rows[0].vehicle_id, rows[0].next_vehicle_id) == ("L:0", "L:1")
    assert (rows[1].vehicle_id, rows[1].next_vehicle_id) == ("L:1", "L:2")
    assert rows[-1].terminal and rows[-1].completed_at == 60 and rows[-1].dt == 20
    assert sum(r.r for r in rows) == -7*(60-10)
    # Physical labels come from same vehicle+trip at the next stop, independently
    # of the global cross-vehicle chronological transition.
    assert [(p.vehicle_id, p.trip_id, p.dt) for p in intervals] == [("L:0", "A", 20), ("L:2", "C", 20)]
    assert all(p.semantics == "decision_ready_to_next_stop_decision_ready" for p in intervals)
    assert stream.uncompleted_interval_count == 3


def test_delay_and_thinning_gate_both_replay_and_physical_labels():
    stream = EventStream(FakeSource(), [3], TransitMDPConfigV7(horizon_seconds=60),
                         execute_transform=False, delay_seconds=15, record_every=2)
    assert stream.step(lambda s: [-1]) == []  # clock 20; first label is available at 25.
    assert stream.release_intervals() == []
    ready = stream.step(lambda s: [-1])
    assert [r.sequence for r in ready] == [0]
    assert ready[0].completed_at == 10 and ready[0].available_at == 25
    assert stream.step(lambda s: [-1]) == []
    assert stream.release_intervals() == []  # stop interval completes at 30, released at 45.
    ready = stream.step(lambda s: [-1])
    assert [r.sequence for r in ready] == [2]
    assert [p.sequence for p in stream.release_intervals()] == [0, 2]
    assert stream.pending_completed_count == 1  # terminal row available at 75, beyond horizon.
    assert stream.thinned_count == 2
    with pytest.raises(ValueError, match="beyond the observed"):
        stream.release(75)


def test_recorded_and_prior_batch_actions_are_the_executed_transform():
    issued = []
    env = FakeSource(horizon=60)
    stream = EventStream(env, [3], TransitMDPConfigV7(horizon_seconds=60), on_issue=issued.append)
    rows = stream.step(lambda s: [-1])
    assert len(issued) == 2 and len(rows) == 1
    assert issued[0]["proposed_a"][0] == -1 and issued[0]["a"][0] == 1
    assert env.commands[-1][("L", 0)] == [60., 1.]
    assert rows[0].a[0] == 1
    # Slot 0's action is already stored in the next within-batch v7 state.
    assert rows[0].ns[5+13] == 1
    np.testing.assert_array_equal(current_event_v7(issued[0]["raw_s"]), local(10, 0))


def test_trip_restart_and_skipped_stop_do_not_create_next_stop_labels():
    batches = [(10, [(0, 0, "A")]), (20, [(0, 1, "NEW")]), (30, [(0, 3, "NEW")])]
    stream = EventStream(FakeSource(batches), [3], TransitMDPConfigV7(horizon_seconds=60),
                         execute_transform=False)
    _, intervals = collect(stream)
    assert intervals == []


def test_stopping_at_an_event_leaves_an_uncompleted_control_transition():
    stream = EventStream(FakeSource(), [3], TransitMDPConfigV7(horizon_seconds=60), execute_transform=False)
    rows = stream.step(lambda s: [-1])
    assert len(rows) == 1 and not rows[0].terminal
    assert stream.completed_count == 1 and stream.issued_count == 2
    assert stream.release() == [] and not stream.done


def test_premature_env_done_does_not_fabricate_a_horizon_terminal():
    stream = EventStream(FakeSource(horizon=50), [3], TransitMDPConfigV7(horizon_seconds=60), execute_transform=False)
    with pytest.raises(RuntimeError, match="stopped before the configured horizon"):
        collect(stream)
