"""Server-side same-vehicle data preparation and compact in-memory replay."""

import json
import os
import sys
import time

import h5py
import numpy as np
import torch

from station_common import (EXPERIMENTS, P, ROOT, action_dim, data_folder, observation,
                    neutral_action, physical_action, state_headway_reward, write)
from SimpleSAC.transit_v3.replay_buffer import _read_rows


class RowWriter:
    def __init__(self, path, mode, stations):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(path, "w")
        self.file.attrs["stations"] = json.dumps(np.asarray(stations).tolist())
        self.file.attrs["mode"] = mode
        self.rows, self.count = [], 0
        dim = action_dim(mode)
        for name, width in (("observations", 15+dim), ("actions", dim), ("next_observations", 15+dim),
                            ("headway_rewards", 0), ("passenger_rewards", 0), ("delta_t_seconds", 0), ("role", 0)):
            shape = (0, width) if width else (0,)
            maximum = (None, width) if width else (None,)
            self.file.create_dataset(name, shape=shape, maxshape=maximum, dtype="float32",
                                     chunks=True, compression="lzf")

    def append(self, s, a, ns, rh, rp, dt, role=0):
        if dt <= 0:
            raise ValueError("same-vehicle station transitions must advance physical time")
        self.rows.append((s.copy(), a.copy(), ns.copy(), rh, rp, dt, role))
        if len(self.rows) >= 4096:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        names = ("observations", "actions", "next_observations", "headway_rewards",
                 "passenger_rewards", "delta_t_seconds", "role")
        stop = self.count + len(self.rows)
        for name, values in zip(names, zip(*self.rows)):
            data = self.file[name]
            data.resize(stop, axis=0)
            data[self.count:stop] = np.asarray(values, np.float32)
        self.count = stop
        self.rows.clear()

    def close(self):
        self.flush()
        self.file.close()


def prepare_holding(smoke=False):
    folder = data_folder("holding", smoke)
    target = folder / "dataset.h5"
    if (folder / "ready.json").exists():
        return
    source = os.environ["THESIS_DATASET"]
    started, pending, last = time.monotonic(), {}, {}
    with h5py.File(source) as data:
        stations = np.array(json.loads(data.attrs["station_max_by_line"]), np.float32)
        writer = RowWriter(target, "holding", stations)
        n = min(50000, len(data["actions"])) if smoke else len(data["actions"])
        trajectory, total_wait, total_ride = None, 0., 0.
        for start in range(0, n, 8192):
            stop = min(n, start+8192)
            rows = {k: data[k][start:stop] for k in ("raw_observations", "actions", "trajectory_id",
                "event_line_index", "event_bus_code", "event_station_index", "terminals", "split_id",
                "wait_passenger_seconds", "in_vehicle_passenger_seconds")}
            raw_batch = rows["raw_observations"]
            slots = raw_batch[:, 5:].reshape(-1, 16, 15)
            current = np.rint(raw_batch[:, 2]).astype(int)
            local = slots[np.arange(len(slots)), current, 1:14]
            raw = np.column_stack((local[:, :2], raw_batch[:, 0], local[:, 2:], raw_batch[:, 3:5]))
            for i in range(len(raw)):
                tid = int(rows["trajectory_id"][i])
                if tid != trajectory:
                    pending.clear()
                    last.clear()
                    trajectory, total_wait, total_ride = tid, 0., 0.
                if rows["split_id"][i] != 0:
                    continue
                if rows["terminals"][i] < .5:
                    key = (int(rows["event_line_index"][i]), int(rows["event_bus_code"][i]))
                    state = observation(raw[i], last.get(key, neutral_action("holding")), stations)
                    previous = pending.pop(key, None)
                    if previous is not None and raw[i, 1] > previous["station"]:
                        welfare = -(2*(total_wait-previous["wait"]) + total_ride-previous["ride"])
                        writer.append(previous["s"], previous["a"], state,
                            float(state_headway_reward(state[None])[0]), welfare,
                            raw[i, 2]-previous["time"], tid % 10)
                    action = rows["actions"][i].copy()
                    pending[key] = dict(s=state, a=action, station=raw[i, 1], time=raw[i, 2],
                                        wait=total_wait, ride=total_ride)
                    last[key] = action
                total_wait += float(rows["wait_passenger_seconds"][i])
                total_ride += float(rows["in_vehicle_passenger_seconds"][i])
        writer.close()
    write(folder / "ready.json", dict(mode="holding", rows=writer.count,
        preparation_seconds=time.monotonic()-started, source="Existing v7 training partition only",
        settlement="same line, bus and trajectory; unfinished intervals discarded, no fabricated terminal feedback"))


def sumo_helpers():
    sys.path.insert(0, str(EXPERIMENTS / "mcwm_bus"))
    import evaluate_control
    return evaluate_control


def create_sumo(folder, scale, demand_seed, sumo_seed, horizon):
    helper = sumo_helpers()
    config, people = helper.build_scenario(folder, scale, demand_seed, P["departure_jitter_seconds"])
    env = helper.AllLinesSumoEnv(str(helper.SUMO_ROOT / "online_control"),
        str(ROOT / "bus_h2o/network_data/a_sorted_busline_edge.xml"), episode_seconds=horizon,
        sumo_seed=sumo_seed, sumo_cfg=config, passenger_obj_xml=people, observation_schema="v5",
        sumo_backend="libsumo")
    from environments import enable_sumo_speed
    enable_sumo_speed(env)
    return env, helper


def collect_dual(shard, smoke=False):
    from collect_policy.collect_transit_v7 import raw_event_observation, event_sort_key, is_terminal_event
    folder = data_folder("dual", smoke) / f"shard{shard:02d}"
    if (folder / "ready.json").exists():
        return
    seed, scale = P["offline_seed_start"]+shard, (.75, 1., 1.25)[shard % 3]
    horizon = 1800 if smoke else P["horizon_seconds"]
    with h5py.File(os.environ["THESIS_DATASET"]) as metadata:
        stations = np.array(json.loads(metadata.attrs["station_max_by_line"]), np.float32)
    env, helper = create_sumo(folder, scale, seed, seed+10000, horizon)
    writer = RowWriter(folder / "dataset.h5", "dual", stations)
    rng, pending, last = np.random.default_rng(seed), {}, {}
    started, decisions = time.monotonic(), 0
    try:
        env.reset()
        env.step_to_event({})
        while not env.done:
            commands = {}
            snapshot = env._bridge.passenger_time_ledger.snapshot()
            totals = (snapshot.accrued_wait_seconds, snapshot.accrued_in_vehicle_seconds)
            for event in sorted(env._current_events, key=event_sort_key):
                key = (event.line_id, event.bus_id)
                raw = raw_event_observation(env, event.bus_id)
                state = observation(raw, last.get(key, neutral_action("dual")), stations)
                previous = pending.pop(key, None)
                if previous is not None and raw[1] > previous["station"]:
                    welfare = -(2*(totals[0]-previous["totals"][0]) + totals[1]-previous["totals"][1])
                    writer.append(previous["s"], previous["a"], state,
                        float(state_headway_reward(state[None])[0]), welfare,
                        raw[2]-previous["time"], shard % 10)
                if is_terminal_event(env, event):
                    commands[event.bus_id] = [0., 1.]
                    continue
                behavior = shard // 3
                hold = 0. if behavior == 0 else float(rng.beta(1., 3.)*60) if behavior == 1 else float(np.clip(.25*(raw[4]-raw[3]), 0, 60))
                speed_raw = float(rng.uniform(-1., 1.)) if behavior in (0, 1) else float(np.clip((raw[3]-raw[4])/720., -1, 1))
                action = np.array([hold/30.-1, speed_raw], np.float32)
                if behavior == 3:
                    action[0] = -1. if rng.random() < .5 else rng.uniform(-1, 1)
                commands[event.bus_id] = physical_action(action)
                last[key] = action.copy()
                pending[key] = dict(s=state, a=action.copy(), totals=totals, station=raw[1], time=raw[2])
                decisions += 1
            env.step_to_event(commands)
            env._bridge.consume_completed_trip_passenger_times()
        writer.close()
        write(folder / "ready.json", dict(shard=shard, seed=seed, scale=scale, rows=writer.count,
            decisions=decisions, horizon=horizon, wall_seconds=time.monotonic()-started))
    finally:
        if writer.file.id.valid:
            writer.close()
        env.close()


def merge_dual(smoke=False):
    folder = data_folder("dual", smoke)
    shards = 2 if smoke else P["offline_dual_shards"]
    if (folder / "ready.json").exists():
        return
    for i in range(shards):
        if not (folder / f"shard{i:02d}/ready.json").exists():
            raise ValueError("dual collection is not complete")
    with h5py.File(folder / "shard00/dataset.h5") as data:
        stations = json.loads(data.attrs["stations"])
    writer = RowWriter(folder / "dataset.h5", "dual", stations)
    writer.flush()
    for i in range(shards):
        with h5py.File(folder / f"shard{i:02d}/dataset.h5") as data:
            n = len(data["actions"])
            for name in writer.file:
                target = writer.file[name]
                target.resize(writer.count+n, axis=0)
                target[writer.count:] = data[name][:]
            writer.count += n
    writer.close()
    write(folder / "ready.json", dict(mode="dual", rows=writer.count, shards=shards))


class Replay:
    def __init__(self, mode, objective, smoke=False):
        with h5py.File(data_folder(mode, smoke) / "dataset.h5") as data:
            n = len(data["actions"])
            ix = np.linspace(0, n-1, min(n, 200000), dtype=int)
            selected = np.zeros(n, dtype=bool)
            selected[ix] = True
            self.stations = np.array(json.loads(data.attrs["stations"]), np.float32)
            # Contiguous reads avoid HDF5's costly large point-selection query.
            self.offline = {k: _read_rows(data[k], selected, 65536)
                            for k in ("observations", "actions", "next_observations", "delta_t_seconds", "role")}
            self.raw_rewards = _read_rows(data[objective+"_rewards"], selected, 65536)
        self.objective, self.mode = objective, mode
        self.mean, self.std = float(self.raw_rewards.mean()), max(float(self.raw_rewards.std()), 1.)
        self.offline["rewards"] = self.normalize(self.raw_rewards)
        self.offline["discounts"] = np.full(len(ix), P["same_vehicle_discount"], np.float32)
        self.offline["terminals"] = np.zeros(len(ix), np.float32)
        self.offline_size, self._online_size = len(ix), 0
        self.online = {k: np.zeros((P["source_events"]+128, *v.shape[1:]), np.float32)
                       for k, v in self.offline.items() if k != "role"}

    @property
    def online_size(self):
        return self._online_size

    def normalize(self, reward):
        return (np.asarray(reward)-self.mean) * (10./self.std)

    def append(self, s, a, ns, rh, rp, dt):
        row = dict(observations=s, actions=a, next_observations=ns,
            rewards=float(self.normalize(rh if self.objective == "headway" else rp)),
            discounts=P["same_vehicle_discount"], terminals=0., delta_t_seconds=float(dt))
        for k, v in row.items():
            self.online[k][self._online_size] = v
        self._online_size += 1

    def sample(self, size, *, scope="offline", rng=None, partition=None):
        rng = rng or np.random.default_rng()
        source = self.offline if scope == "offline" else self.online
        count = self.offline_size if scope == "offline" else self.online_size
        ids = rng.integers(count, size=size)
        keys = ("observations", "actions", "next_observations", "rewards", "discounts", "terminals", "delta_t_seconds")
        return {k: torch.as_tensor(source[k][ids], dtype=torch.float32) for k in keys}

    def sample_mixed(self, size, fraction, *, rng):
        online = int(round(size*fraction)) if self.online_size else 0
        return (self.sample(size-online, rng=rng) if online < size else None,
                self.sample(online, scope="online", rng=rng) if online else None)
