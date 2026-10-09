"""The frozen twelve-line event interface and independent passenger accounting."""

from contextlib import contextmanager, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile

import numpy as np
import torch

from settings import ENVROOT, EXPERIMENTS, P, module, use_snapshot

use_snapshot()
# Import the frozen v8 learner before the legacy environment runner sets paths.
from csbapr.agent import CSBAPRAgent
from csbapr.config import CSBAPRConfig

os.environ["THESIS_CSBAPR_ROOT"] = str(ENVROOT)
os.environ["OFFLINE_SUMO_ENV"] = str(ENVROOT / "offline-sumo/env")
sys.path.insert(0, str(EXPERIMENTS / "reward_cross_v1"))
from rewards import PassengerLedger, counts
recovery = module("symbolic_bus_recovery", EXPERIMENTS / "csbapr_bus/evaluate_recovery.py")
run_episode = recovery.run_episode_multiline


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ObjectiveEnv(recovery.ObservedEnv):
    objective = "headway"

    def reset(self):
        self.ledger = PassengerLedger()
        self.windows = {}
        return super().reset()

    def step(self, actions, **kwargs):
        before = self.current_time
        waiting, onboard = counts(list(self.line_map.values()))
        state, rewards, done = super().step(actions, **kwargs)
        self.ledger.advance(waiting, onboard, self.current_time-before)
        if self.objective == "passenger":
            rewards = {line:{bus:self.ledger.settle((line, bus)) for bus, obs in buses.items() if obs}
                       for line, buses in state.items()}
        return state, rewards, done


@contextmanager
def environment(objective):
    source = ENVROOT / "offline-sumo/env/calibrated_env"
    with tempfile.TemporaryDirectory(prefix="thesis-cs-symbolic-") as directory:
        directory = Path(directory)
        shutil.copy2(source / "config.json", directory / "config.json")
        os.symlink(source / "data", directory / "data", target_is_directory=True)
        ObjectiveEnv.objective = objective
        with redirect_stdout(StringIO()):
            env = ObjectiveEnv(str(directory))
        for line in env.line_map.values():
            line.max_time = max(line.max_time, P["horizon_seconds"]+line.time_step)
        yield env


class MeasuredPolicy(recovery.ObservedPolicy):
    def __init__(self, agent, env):
        super().__init__(agent, env)
        self.plan_sum = 0.

    def select_action(self, state, deterministic=False):
        self.plan_sum += .5*(abs(float(state[5]-state[8]))+abs(float(state[6]-state[8])))*600
        return super().select_action(state, deterministic=deterministic)


def service_metrics(env, policy, decisions):
    people = [p for line in env.line_map.values() for station in line.stations for p in station.total_passenger]
    n = len(people)
    action_count = sum(w["decisions"] for w in env.windows.values())
    if env.current_time != P["horizon_seconds"]:
        raise RuntimeError("Evaluation did not complete the frozen service horizon")
    return dict(horizon_seconds=env.current_time, decisions=decisions, generated_passengers=n,
        completion_rate=sum(p.arrived for p in people)/n,
        generalized_seconds_per_passenger=(2*env.ledger.wait+env.ledger.ride)/n,
        wait_passenger_seconds=env.ledger.wait, ride_passenger_seconds=env.ledger.ride,
        headway_plan_mae_seconds=policy.plan_sum/action_count,
        headway_imbalance_seconds=sum(w["imbalance_sum"] for w in env.windows.values())/action_count,
        mean_hold_seconds=sum(w["hold_sum"] for w in env.windows.values())/action_count)
