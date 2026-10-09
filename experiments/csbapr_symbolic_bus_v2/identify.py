"""Collect controlled same-vehicle transitions once, with both cost labels."""

from contextlib import redirect_stdout
from io import StringIO
import time

import numpy as np

from settings import P, day_folder, write
from bus_runtime import environment, run_episode, seed_all, CSBAPRConfig


class Records:
    def __init__(self, collector):
        self.collector, self.buffer = collector, []

    @property
    def size(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        # Reject the legacy runner's recycled-bus trip boundaries.
        if next_state[2] <= state[2] or next_state[10] <= state[10]:
            return
        key = tuple(np.rint(state[:2]*[12, 25]).astype(int))
        previous = self.collector.marks[key]
        ledger = self.collector.env.ledger
        passenger = 2*(ledger.wait-previous[0])+ledger.ride-previous[1]
        self.buffer.append((state.copy(), action.copy(), next_state.copy(), -reward, passenger))


class Collector:
    def __init__(self, env, seed):
        self.env, self.rng = env, np.random.default_rng(seed)
        self.replay_buffer = Records(self)
        self.marks = {}

    def select_action(self, state, deterministic=False):
        key = tuple(np.rint(state[:2]*[12, 25]).astype(int))
        self.marks[key] = (self.env.ledger.wait, self.env.ledger.ride)
        hold = self.rng.choice(P["reference_hold_grid"])
        return np.array([hold/30-1], np.float32)

    def update(self, **kwargs):
        pass


def collect_day(seed, day):
    folder = day_folder(seed, day)
    if (folder / "ready.json").exists():
        return
    started = time.monotonic()
    rng_seed = seed+P["identification_seed_offset"]+day
    seed_all(rng_seed)
    with environment("headway") as env:
        collector = Collector(env, rng_seed)
        with redirect_stdout(StringIO()):
            run_episode(env, collector, train=True, config=CSBAPRConfig(batch_size=256))
        s, a, ns, headway, passenger = zip(*collector.replay_buffer.buffer)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "transitions.npz", states=s, actions=a, next_states=ns,
                            costs=np.column_stack((headway, passenger)))
        write(folder / "ready.json", dict(seed=seed, day=day, collection_seed=rng_seed,
            role="validation" if day == 3 else "fit", rows=len(s),
            horizon_seconds=env.current_time, wall_seconds=time.monotonic()-started,
            learner_updates=0, target="same vehicle, increasing station and physical time"))
