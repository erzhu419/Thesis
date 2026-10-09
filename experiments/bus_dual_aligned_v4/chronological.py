"""One shared chronological dataset for both objectives and both controls."""

import json
import os
import time

import h5py
import numpy as np
import torch

from aligned_common import (
    P, CONFIG, STATE_DIM, data_folder, state, pack_event_batch_v7,
    terminal_raw_state_v7, ridge_of_state, physical_action, write,
)
from dataset import create_sumo
from SimpleSAC.transit_v3.replay_buffer import _read_rows

FIELDS = ("observations", "actions", "next_observations", "headway_rewards",
          "passenger_rewards", "delta_t_seconds", "terminals", "role")


class Writer:
    def __init__(self, path, stations):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(path, "w")
        self.file.attrs["stations"] = json.dumps(np.asarray(stations).tolist())
        self.count, self.rows = 0, []
        for name in FIELDS:
            width = STATE_DIM if name in ("observations", "next_observations") else 2 if name == "actions" else 0
            shape = (0,width) if width else (0,)
            self.file.create_dataset(name, shape=shape, maxshape=(None,width) if width else (None,),
                                     chunks=True, compression="lzf", dtype="float32")

    def append(self, s, a, ns, rh, rp, dt, terminal=False, role=0):
        self.rows.append((s.copy(), a.copy(), ns.copy(), rh, rp, dt, float(terminal), role))
        if len(self.rows) >= 2048:
            self.flush()

    def flush(self):
        if self.rows:
            stop = self.count+len(self.rows)
            for name, values in zip(FIELDS, zip(*self.rows)):
                self.file[name].resize(stop, axis=0)
                self.file[name][self.count:stop] = np.asarray(values, np.float32)
            self.count = stop
            self.rows.clear()

    def close(self):
        self.flush()
        self.file.close()


def close_pending(pending, ns, snapshot, terminal=False):
    before = pending["snapshot"]
    rp = -(2*(snapshot.accrued_wait_seconds-before.accrued_wait_seconds)
           + snapshot.accrued_in_vehicle_seconds-before.accrued_in_vehicle_seconds)
    dt = snapshot.time_seconds-before.time_seconds
    if dt < 0 or rp > 1e-3:
        raise ValueError("Chronological passenger ledger moved backwards")
    return (pending["state"], pending["action"], ns,
            0. if terminal else float(ridge_of_state(ns[None])[0]), float(rp), float(dt), terminal)


def collect(shard, smoke=False):
    from collect_policy.collect_transit_v7 import raw_event_observation, event_sort_key, is_terminal_event
    from rules import hold_seconds
    folder = data_folder(smoke) / f"shard{shard:02d}"
    if (folder / "ready.json").exists():
        return
    seed, scale = P["offline_seed_start"]+shard, (.75,1.,1.25)[shard % 3]
    with h5py.File(os.environ["THESIS_DATASET"]) as data:
        stations = np.asarray(json.loads(data.attrs["station_max_by_line"]), np.float32)
    horizon = 1800 if smoke else CONFIG.horizon_seconds
    env, _ = create_sumo(folder, scale, seed, seed+10000, horizon)
    writer, rng = Writer(folder / "dataset.h5", stations), np.random.default_rng(seed)
    pending, decisions, total_cost = None, 0, 0.
    started = time.monotonic()
    try:
        env.reset()
        env.step_to_event({})
        initial = env._bridge.passenger_time_ledger.snapshot()
        while not env.done:
            commands, controls = {}, []
            for event in sorted(env._current_events, key=event_sort_key):
                if is_terminal_event(env, event):
                    commands[event.bus_id] = [0.,1.]
                else:
                    controls.append((event, raw_event_observation(env, event.bus_id)))
            if controls:
                snapshot = env._bridge.passenger_time_ledger.snapshot()
                raw_batch = np.stack([r for _,r in controls])
                prior, speeds = np.full(len(controls), -1., np.float32), np.zeros(16, np.float32)
                for index, (event, raw) in enumerate(controls):
                    global_raw = pack_event_batch_v7(raw_batch, index, prior, CONFIG)
                    s = state(global_raw, speeds, stations)
                    if pending is not None:
                        row = close_pending(pending, s, snapshot)
                        writer.append(*row, role=shard % 10)
                        total_cost -= row[4]
                    behavior = shard // 3
                    if behavior == 0:
                        hold = 0.
                    elif behavior == 1:
                        hold = float(rng.beta(1.,3.)*60) if rng.random() < .65 else 0.
                    elif behavior == 2:
                        hold = hold_seconds("daganzo_duet", raw[3],raw[4],raw[5],raw[6],raw[11],raw[12])
                    else:
                        hold = float(rng.uniform(0,60)) if rng.random() < .5 else 0.
                    speed = float(np.clip((raw[3]-raw[4])/720., -1,1)) if behavior == 2 else float(rng.uniform(-1,1))
                    action = np.asarray([hold/30.-1, speed], np.float32)
                    commands[event.bus_id] = physical_action(action)
                    prior[index], speeds[index] = action
                    pending = dict(raw=global_raw, state=s, action=action, snapshot=snapshot)
                    decisions += 1
            env.step_to_event(commands)
            env._bridge.consume_completed_trip_passenger_times()
        final = env._bridge.passenger_time_ledger.snapshot()
        if pending is not None:
            raw = terminal_raw_state_v7(pending["raw"], final)
            row = close_pending(pending, state(raw, np.zeros(16,np.float32), stations), final, True)
            writer.append(*row, role=shard % 10)
            total_cost -= row[4]
        ledger_cost = 2*(final.accrued_wait_seconds-initial.accrued_wait_seconds)+final.accrued_in_vehicle_seconds-initial.accrued_in_vehicle_seconds
        if abs(total_cost-ledger_cost) > 1e-3:
            raise RuntimeError("Chronological passenger returns do not telescope")
        writer.close()
        write(folder / "ready.json", dict(shard=shard, seed=seed, rows=writer.count, decisions=decisions,
            horizon_seconds=horizon, wall_seconds=time.monotonic()-started, recorded_passenger_cost=total_cost,
            ledger_passenger_cost=ledger_cost, reward_balance_error=total_cost-ledger_cost))
    finally:
        if writer.file.id.valid:
            writer.close()
        env.close()


def merge(smoke=False):
    folder, count = data_folder(smoke), 2 if smoke else P["offline_shards"]
    if (folder / "ready.json").exists():
        return
    with h5py.File(folder / "shard00/dataset.h5") as data:
        stations = json.loads(data.attrs["stations"])
    writer = Writer(folder / "dataset.h5", stations)
    for shard in range(count):
        if not (folder / f"shard{shard:02d}/ready.json").exists():
            raise ValueError("Incomplete shared dual-action collection")
        with h5py.File(folder / f"shard{shard:02d}/dataset.h5") as data:
            n = len(data["actions"])
            for start in range(0,n,8192):
                stop = min(start+8192,n)
                for name in FIELDS:
                    writer.file[name].resize(writer.count+stop-start, axis=0)
                    writer.file[name][writer.count:] = data[name][start:stop]
                writer.count += stop-start
    writer.close()
    write(folder / "ready.json", dict(rows=writer.count, shards=count, state_dim=STATE_DIM, action_dim=2))


class Replay:
    def __init__(self, objective, smoke=False):
        with h5py.File(data_folder(smoke) / "dataset.h5") as data:
            n = len(data["actions"])
            selected = np.zeros(n, bool)
            selected[np.linspace(0,n-1,min(n,P["offline_max_rows"]),dtype=int)] = True
            self.stations = np.asarray(json.loads(data.attrs["stations"]), np.float32)
            self.offline = {k:_read_rows(data[k],selected,65536) for k in FIELDS if k not in ("headway_rewards","passenger_rewards")}
            self.offline["rewards"] = _read_rows(data[objective+"_rewards"],selected,65536)
        self.objective = objective
        self.offline["discounts"] = 1.-self.offline["terminals"]
        self.offline_size, self._online_size = len(self.offline["rewards"]), 0
        self.online = {k:np.zeros((P["source_events"]+128,*v.shape[1:]),np.float32) for k,v in self.offline.items() if k != "role"}

    @property
    def online_size(self):
        return self._online_size

    def append(self,s,a,ns,rh,rp,dt,terminal=False):
        row = dict(observations=s, actions=a, next_observations=ns, rewards=rh if self.objective == "headway" else rp,
                   delta_t_seconds=dt, terminals=float(terminal), discounts=1.-float(terminal))
        for k,v in row.items():
            self.online[k][self._online_size] = v
        self._online_size += 1

    def sample(self, size, *, scope="offline", rng=None, partition=None):
        rng = rng or np.random.default_rng()
        data, n = (self.offline,self.offline_size) if scope == "offline" else (self.online,self.online_size)
        ids = rng.integers(n, size=size)
        return {k:torch.as_tensor(data[k][ids],dtype=torch.float32) for k in self.online}

    def sample_mixed(self, size, fraction, *, rng):
        n = int(round(size*fraction)) if self.online_size else 0
        return (self.sample(size-n,rng=rng) if n < size else None,
                self.sample(n,scope="online",rng=rng) if n else None)


class Sampler:
    def __init__(self, env, replay):
        self.env, self.replay = env, replay
        self.pending, self.active = None, False

    def collect(self, policy, count):
        from SimpleSAC.transit_v7.sampler import _active_event_batch, _terminal_passenger_time
        events = 0
        while events < count:
            if not self.active:
                self.env.reset()
                self.env.step_to_event({})
                self.active = True
            active = _active_event_batch(self.env.state)
            commands = {}
            if active:
                snapshot = self.env.passenger_time_at_event(active[0][0])
                raw_batch = np.stack([raw for _,raw in active])
                prior, speeds = np.full(len(active),-1.,np.float32), np.zeros(16,np.float32)
                for index,(agent,_) in enumerate(active):
                    raw = pack_event_batch_v7(raw_batch,index,prior,CONFIG)
                    s = state(raw,speeds,self.replay.stations)
                    if self.pending is not None:
                        self.replay.append(*close_pending(self.pending,s,snapshot))
                    action = policy(s[None],deterministic=False)[0]
                    commands[agent] = physical_action(action)
                    prior[index], speeds[index] = action
                    self.pending = dict(raw=raw,state=s,action=action.copy(),snapshot=snapshot)
                    events += 1
            self.env.step_to_event(commands)
            if self.env.done:
                if self.pending is not None:
                    final = _terminal_passenger_time(self.env)
                    raw = terminal_raw_state_v7(self.pending["raw"],final)
                    self.replay.append(*close_pending(self.pending,state(raw,np.zeros(16,np.float32),self.replay.stations),final,True))
                    self.pending = None
                self.active = False
        return events
