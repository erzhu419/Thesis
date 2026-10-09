"""Retain actions until that same vehicle reaches a later station."""

import json
import os
import time

import h5py
import numpy as np

from headway_common import P, HERE, CONFIG, state, pack_event_batch_v7, physical_action, raw_reward, data_folder, write
import chronological as old
from chronological import Writer


def settle(previous, s, raw, terminal=False):
    dt = float(raw[2]-previous["time"])
    if not terminal and (dt<=0 or raw[0]!=previous["line"] or raw[1]<=previous["station"]):
        raise ValueError("Same-vehicle feedback must advance time and station on the same line")
    return previous["state"],previous["action"],s,0. if terminal else raw_reward(raw),0.,dt,terminal


def bookmark(s,a,raw):
    return dict(state=s.copy(),action=a.copy(),time=float(raw[2]),line=float(raw[0]),station=float(raw[1]))


def terminal_row(previous,snapshot):
    ns = previous["state"].copy()
    ns[0] = snapshot.time_seconds/24000
    ns[3:5] = np.minimum([snapshot.waiting_passengers/20000,snapshot.in_vehicle_passengers/20000],1.25)
    return previous["state"],previous["action"],ns,0.,0.,float(snapshot.time_seconds-previous["time"]),True


def collect(shard,smoke=False):
    from collect_policy.collect_transit_v7 import raw_event_observation,event_sort_key,is_terminal_event
    from rules import hold_seconds
    folder = data_folder(smoke) / f"shard{shard:02d}"
    if (folder / "ready.json").exists():
        return
    with h5py.File(os.environ["THESIS_DATASET"]) as data:
        stations = np.asarray(json.loads(data.attrs["station_max_by_line"]),np.float32)
    seed,scale = P["offline_seed_start"]+shard,(.75,1.,1.25)[shard%3]
    env,_ = old.create_sumo(folder,scale,seed,seed+10000,1800 if smoke else 24000)
    writer,rng,pending = Writer(folder / "dataset.h5",stations),np.random.default_rng(seed),{}
    started,decisions = time.monotonic(),0
    try:
        env.reset()
        env.step_to_event({})
        while not env.done:
            events = sorted(env._current_events,key=event_sort_key)
            commands = {}
            controls = [(e,raw_event_observation(env,e.bus_id)) for e in events if not is_terminal_event(env,e)]
            if controls:
                raw_batch = np.stack([r for _,r in controls])
                prior,speeds = np.full(len(controls),-1.,np.float32),np.zeros(16,np.float32)
                for index,(event,raw) in enumerate(controls):
                    s = state(pack_event_batch_v7(raw_batch,index,prior,CONFIG),speeds,stations)
                    key = (event.line_id,event.bus_id)
                    if key in pending:
                        writer.append(*settle(pending.pop(key),s,raw),role=shard%10)
                    behavior = shard//3
                    if behavior==0:
                        hold = 0.
                    elif behavior==1:
                        hold = float(rng.beta(1.,3.)*60) if rng.random()<.65 else 0.
                    elif behavior==2:
                        hold = hold_seconds("daganzo_duet",raw[3],raw[4],raw[5],raw[6],raw[11],raw[12])
                    else:
                        hold = float(rng.uniform(0,60)) if rng.random()<.5 else 0.
                    speed = float(np.clip((raw[3]-raw[4])/720,-1,1)) if behavior==2 else float(rng.uniform(-1,1))
                    a = np.asarray([hold/30.-1,speed],np.float32)
                    commands[event.bus_id] = physical_action(a)
                    prior[index],speeds[index] = a
                    pending[key] = bookmark(s,a,raw)
                    decisions += 1
            for event in events:
                if is_terminal_event(env,event):
                    commands[event.bus_id] = [0.,1.]
                    previous = pending.pop((event.line_id,event.bus_id),None)
                    if previous is not None:
                        raw = raw_event_observation(env,event.bus_id)
                        s = state(pack_event_batch_v7(raw[None],0,np.array([-1.],np.float32),CONFIG),np.zeros(16,np.float32),stations)
                        writer.append(*settle(previous,s,raw,True),role=shard%10)
            env.step_to_event(commands)
            env._bridge.consume_completed_trip_passenger_times()
        final = env._bridge.passenger_time_ledger.snapshot()
        for previous in pending.values():
            writer.append(*terminal_row(previous,final),role=shard%10)
        writer.close()
        write(folder / "ready.json",dict(shard=shard,rows=writer.count,decisions=decisions,wall_seconds=time.monotonic()-started,
            contract="same line and SUMO vehicle, strictly later station; terminal headway zero",seed=seed,smoke=smoke))
    finally:
        if writer.file.id.valid:
            writer.close()
        env.close()


def merge(smoke=False):
    previous = old.data_folder
    old.data_folder = data_folder
    try:
        old.merge(smoke)
    finally:
        old.data_folder = previous


class Replay(old.Replay):
    def __init__(self,objective="headway",smoke=False):
        previous = old.data_folder
        old.data_folder = data_folder
        try:
            super().__init__("headway",smoke)
        finally:
            old.data_folder = previous
        self.offline["discounts"] *= P["discount"]

    def append(self,*args):
        super().append(*args)
        self.online["discounts"][self._online_size-1] *= P["discount"]


class Sampler:
    def __init__(self,env,replay):
        self.env,self.replay = env,replay
        self.pending,self.active = {},False

    def collect(self,policy,count):
        from SimpleSAC.transit_v7.sampler import _active_event_batch,_terminal_passenger_time
        events = 0
        while events<count:
            if not self.active:
                self.env.reset()
                self.env.step_to_event({})
                self.active = True
            active = _active_event_batch(self.env.state)
            commands = {}
            if active:
                raw_batch = np.stack([raw for _,raw in active])
                prior,speeds = np.full(len(active),-1.,np.float32),np.zeros(16,np.float32)
                for index,(agent,raw) in enumerate(active):
                    line = self.env.line_map[agent[0]]
                    bus = next(b for b in line.bus_all if b.bus_id==agent[1])
                    key = (*agent,int(bus.trip_id))
                    s = state(pack_event_batch_v7(raw_batch,index,prior,CONFIG),speeds,self.replay.stations)
                    if key in self.pending:
                        self.replay.append(*settle(self.pending.pop(key),s,raw))
                    a = policy(s[None],deterministic=False)[0]
                    commands[agent] = physical_action(a)
                    prior[index],speeds[index] = a
                    self.pending[key] = bookmark(s,a,raw)
                    events += 1
            self.env.step_to_event(commands)
            if self.env.done:
                final = _terminal_passenger_time(self.env)
                for previous in self.pending.values():
                    self.replay.append(*terminal_row(previous,final))
                self.pending.clear()
                self.active = False
        return events
