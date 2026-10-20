"""Chronological v7 replay and separate observed same-trip stop intervals.

The physical interval is decision-ready stop j to decision-ready stop j+1:
holding at j + motion/signals + service at j+1, including event discretization.
It is neither a pure segment-running-time nor a passenger-arrival-rate label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import os
import sys
from typing import Callable

import numpy as np

RUNTIME = Path(os.environ.get("THESIS_H2O_ROOT", "/home/erzhu419/mine_code/sumo-rl/H2Oplus"))
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

from bus_h2o.mdp_contract_v7 import (
    RAW_EVENT_STATE_DIM, RAW_EVENT_STATE_INDEX as EI, TransitMDPConfigV7,
    pack_event_batch_v7, policy_action_to_hold_seconds_v7,
    raw_state_to_canonical_v7, terminal_raw_state_v7, transition_reward_v7,
)
from bus_h2o.passenger_time import aggregate_passenger_time_snapshots


@dataclass
class Transition:
    s: np.ndarray
    a: np.ndarray
    ns: np.ndarray
    r: float
    dt: float
    terminal: bool
    episode_id: int
    domain: str
    started_at: float
    completed_at: float
    available_at: float
    vehicle_id: str
    line_id: str
    station_id: str
    next_vehicle_id: str
    sequence: int
    raw_s: np.ndarray | None = None
    raw_ns: np.ndarray | None = None
    trip_id: str = ""


@dataclass
class PhysicalInterval:
    s: np.ndarray
    a: np.ndarray
    event: np.ndarray
    next_event: np.ndarray
    dt: float
    started_at: float
    completed_at: float
    available_at: float
    vehicle_id: str
    line_id: str
    trip_id: str
    station_id: str
    next_station_id: str
    episode_id: int
    domain: str
    sequence: int
    semantics: str = "decision_ready_to_next_stop_decision_ready"


def _event(value):
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    result = np.asarray(value, np.float32).reshape(-1)
    if result.shape != (RAW_EVENT_STATE_DIM,):
        raise ValueError(f"Expected {RAW_EVENT_STATE_DIM}-field v7 local event, got {result.shape}")
    return result.copy()


class EventStream:
    """Advance one atomic event batch; expose only completed, available records.

    ``policy`` receives one canonical state. ``on_issue`` receives a dictionary
    with the executed action before advancing the simulator, so predictions can
    be registered before the eventual label is observed. Thinning chooses every
    ``record_every``-th issued control event, without changing executed controls.
    Both chronological and physical labels obey the same delay/thinning rule.
    """

    def __init__(self, env, stations, contract, domain="target", *,
                 delay_seconds=0., record_every=1, execute_transform=True,
                 on_issue: Callable[[dict], None] | None = None):
        if delay_seconds < 0 or record_every < 1 or int(record_every) != record_every:
            raise ValueError("delay_seconds must be nonnegative and record_every a positive integer")
        self.env, self.stations, self.contract = env, np.asarray(stations, np.float32), contract
        self.domain, self.delay_seconds = domain, float(delay_seconds)
        self.record_every, self.execute_transform = int(record_every), execute_transform
        self.on_issue = on_issue
        self.episode_id, self.issued_count = -1, 0
        self.completed_count = self.released_count = self.thinned_count = 0
        self.physical_intervals: list[PhysicalInterval] = []
        self._queue, self._interval_queue = [], []
        self._pending, self._vehicle_pending = None, {}
        self.initial_snapshot = self.final_snapshot = None
        self.uncompleted_interval_count = 0

    @property
    def done(self):
        return bool(self.env.done)

    @property
    def now(self):
        return float(self._snapshot().time_seconds)

    @property
    def pending_completed_count(self):
        return len(self._queue)

    def _snapshot(self, agent=None):
        if getattr(self.env, "_bridge", None) is not None:
            return self.env._bridge.passenger_time_ledger.snapshot()
        if agent is not None:
            return self.env.passenger_time_at_event(agent)
        return aggregate_passenger_time_snapshots(
            line.passenger_time_ledger.snapshot() for line in self.env.line_map.values())

    def _canonical(self, raw):
        return raw_state_to_canonical_v7(raw, self.stations, self.contract)

    def reset(self):
        self.env.reset()
        self.episode_id += 1
        self._pending, self._vehicle_pending = None, {}
        self._queue.clear()
        self._interval_queue.clear()
        self.physical_intervals.clear()
        self.uncompleted_interval_count = 0
        if not any(self.env.state.values()) and not self.done:
            self.env.step_to_event({})
        self.initial_snapshot, self.final_snapshot = self._snapshot(), None

    def _batch(self):
        """Return all real stop events, including SUMO's automatic terminus."""
        if getattr(self.env, "_bridge", None) is not None:
            result = []
            for ev in sorted(self.env._current_events,
                             key=lambda e: (e.sim_time, str(e.line_id), str(e.bus_id), e.stop_idx)):
                line = self.env._bridge.line_obj_dic[ev.line_id]
                terminal = ev.stop_id == line.stop_id_l[-1] or ev.stop_idx >= len(line.stop_id_l)-1
                fleet = self.env._bridge.fleet_obj_dic.get(ev.bus_id)
                result.append(dict(agent=ev.bus_id, vehicle_id=str(fleet.fleet_id if fleet else ev.bus_id),
                    line_id=str(ev.line_id), station_id=str(ev.stop_id), trip_id=str(ev.bus_id),
                    event=_event(self.env.state[ev.bus_id][-1]), terminal=terminal))
            return result
        result = []
        for agent, history in sorted(self.env.state.items(), key=lambda x: (str(x[0][0]), int(x[0][1]))):
            if not history:
                continue
            line = self.env.line_map[agent[0]]
            bus = next(bus for bus in line.bus_all if bus.bus_id == agent[1])
            result.append(dict(agent=agent, vehicle_id=f"{agent[0]}:{agent[1]}", line_id=str(agent[0]),
                station_id=str(getattr(bus.last_station, "physical_stop_id", bus.last_station.station_name)),
                trip_id=str(bus.trip_id), event=_event(history[-1]), terminal=False))
        return result

    @staticmethod
    def _vehicle_key(event):
        return event["line_id"], event["vehicle_id"], event["trip_id"]

    def _close_physical(self, event):
        previous = self._vehicle_pending.pop(self._vehicle_key(event), None)
        if previous is None:
            return
        raw, last = event["event"], previous["event"]
        # A skipped stop cannot supply a next-stop label. A trip restart cannot
        # close a previous trip because trip identity is part of the key.
        if int(round(raw[EI["station_index"]])) != int(round(last[EI["station_index"]])) + 1:
            return
        end = float(raw[EI["sim_time_seconds"]])
        if end <= previous["started_at"]:
            raise RuntimeError("A same-trip next-stop interval must advance physical time")
        if previous["sequence"] % self.record_every == 0:
            self._interval_queue.append(PhysicalInterval(
                s=previous["s"], a=previous["a"], event=last, next_event=raw.copy(),
                dt=end-previous["started_at"], started_at=previous["started_at"], completed_at=end,
                available_at=end+self.delay_seconds, vehicle_id=previous["vehicle_id"],
                line_id=previous["line_id"], trip_id=previous["trip_id"], station_id=previous["station_id"],
                next_station_id=event["station_id"], episode_id=self.episode_id,
                domain=self.domain, sequence=previous["sequence"]))

    def _close(self, raw_ns, snapshot, next_vehicle_id="", terminal=False):
        p = self._pending
        if p is None:
            return
        end = float(snapshot.time_seconds)
        dt = end-p["started_at"]
        if dt < 0:
            raise RuntimeError("Chronological passenger-time clock moved backwards")
        self.completed_count += 1
        if p["sequence"] % self.record_every == 0:
            self._queue.append(Transition(p["s"], p["a"], self._canonical(raw_ns),
                float(transition_reward_v7(p["snapshot"], snapshot, self.contract)), dt, terminal,
                self.episode_id, self.domain, p["started_at"], end, end+self.delay_seconds,
                p["vehicle_id"], p["line_id"], p["station_id"], next_vehicle_id, p["sequence"],
                p["raw_s"], raw_ns.copy(), p["trip_id"]))
        else:
            self.thinned_count += 1
        self._pending = None

    def step(self, policy) -> list[Transition]:
        if self.episode_id < 0:
            self.reset()
        if self.done:
            return self.release()
        batch, commands = self._batch(), {}
        for event in batch:
            self._close_physical(event)
            if event["terminal"]:
                commands[event["agent"]] = [0., 1.]
        controls = [event for event in batch if not event["terminal"]]
        if controls:
            snapshot = self._snapshot(controls[0]["agent"])
            local = np.stack([event["event"] for event in controls])
            prior = np.full(len(controls), -1., np.float32)
            for index, event in enumerate(controls):
                raw = pack_event_batch_v7(local, index, prior, self.contract)
                state = self._canonical(raw)
                self._close(raw, snapshot, event["vehicle_id"])
                proposed = np.asarray(policy(state), np.float32).reshape(-1)
                if proposed.shape != (1,) or not np.isfinite(proposed).all() or np.any(np.abs(proposed)>1.00001):
                    raise ValueError("Policy must return one finite holding action in [-1, 1]")
                action = np.clip(proposed, -1., 1.)
                if self.execute_transform:
                    from SimpleSAC.evaluate_transit_v19_completion_safety_reserve import completion_safety_action
                    action = completion_safety_action(raw, action, "quarter-load-sixty", self.contract)
                issued = dict(event, s=state.copy(), raw_s=raw.copy(), a=action.copy(),
                    proposed_a=proposed.copy(), started_at=float(snapshot.time_seconds),
                    sequence=self.issued_count, episode_id=self.episode_id, domain=self.domain)
                if self.on_issue is not None:
                    self.on_issue(issued)
                self._pending = dict(issued, snapshot=snapshot)
                self._vehicle_pending[self._vehicle_key(event)] = issued
                prior[index] = action[0]
                commands[event["agent"]] = [float(policy_action_to_hold_seconds_v7(action[0], self.contract)), 1.]
                self.issued_count += 1
        self.env.step_to_event(commands)
        if getattr(self.env, "_bridge", None) is not None:
            self.env._bridge.consume_completed_trip_passenger_times()
        if self.done:
            self.final_snapshot = self._snapshot()
            if self.final_snapshot.time_seconds != self.contract.horizon_seconds:
                raise RuntimeError("Simulator stopped before the configured horizon; no terminal label was fabricated")
            if self._pending is not None:
                self._close(terminal_raw_state_v7(self._pending["raw_s"], self.final_snapshot),
                            self.final_snapshot, terminal=True)
            self.uncompleted_interval_count = len(self._vehicle_pending)
        return self.release()

    def _release_time(self, now):
        current = self.now
        now = current if now is None else float(now)
        if now > current:
            raise ValueError("Cannot release feedback beyond the observed simulator clock")
        return now

    def release(self, now=None) -> list[Transition]:
        now = self._release_time(now)
        ready = [row for row in self._queue if row.available_at <= now]
        self._queue = [row for row in self._queue if row.available_at > now]
        self.released_count += len(ready)
        return ready

    def release_intervals(self, now=None) -> list[PhysicalInterval]:
        now = self._release_time(now)
        ready = [row for row in self._interval_queue if row.available_at <= now]
        self._interval_queue = [row for row in self._interval_queue if row.available_at > now]
        self.physical_intervals.extend(ready)
        return ready


def station_max_from_source(env) -> np.ndarray:
    stations = np.ones(len(env.line_map), np.float32)
    for line in env.line_map.values():
        stations[line.line_idx] = max(len(line.stations)-1, 1)
    return stations


def create_source_env(source_env_path, *, episode_seconds=24000,
                      passenger_arrival_end_seconds=18000, demand_scale=1., seed=6201):
    from bus_h2o.envs.all_lines_sim_env import AllLinesSimEnv
    random.seed(seed)
    np.random.seed(seed)
    return AllLinesSimEnv(str(Path(source_env_path).resolve()), episode_seconds=episode_seconds,
        passenger_arrival_end_seconds=passenger_arrival_end_seconds, demand_scale=demand_scale,
        observation_schema="v7", observation_speed_seed=seed)


def create_target_env(sumo_root, edge_xml, *, episode_seconds=24000, seed=7201,
                      sumo_cfg=None, passenger_obj_xml=None, backend="libsumo"):
    from bus_h2o.envs.all_lines_sumo_env import AllLinesSumoEnv
    return AllLinesSumoEnv(str(Path(sumo_root).resolve() / "online_control"),
        str(Path(edge_xml).resolve()), episode_seconds=int(episode_seconds), sumo_seed=seed,
        sumo_cfg=None if sumo_cfg is None else str(Path(sumo_cfg).resolve()),
        passenger_obj_xml=None if passenger_obj_xml is None else str(Path(passenger_obj_xml).resolve()),
        observation_schema="v5", sumo_backend=backend)
