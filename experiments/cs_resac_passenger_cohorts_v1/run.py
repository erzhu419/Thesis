"""Evaluate preserved holding actors against labelled, paired passenger flows."""

import argparse
import ast
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import random
import resource
import shutil
import sys
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from cohorts import ArrivalFlow, install_arrivals, passenger_class, summarize_passengers

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
RUNTIME = HERE / "inputs/runtime"
sys.path.insert(0, str(RUNTIME / "CS-BAPR"))
sys.path.insert(0, str(RUNTIME / "offline-sumo/env"))

from csbapr.agent import CSBAPRAgent
from csbapr.config import CSBAPRConfig
from scripts.test_multiline_convergence import normalize_obs
from sim_core.passenger import Passenger
from sim_core.sim import MultiLineEnv


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def resac_actor(payload):
    source = HERE / "inputs/model_core.py"
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if
                 isinstance(node, ast.ClassDef) and node.name in ("EmbeddingLayer", "BusPolicy")]
    namespace = dict(__name__=__name__, np=np, torch=torch, nn=nn, Normal=Normal)
    exec(compile(tree, str(source), "exec"), namespace)
    state = payload["actor"]
    sizes = tuple(state[f"embedding.embeddings.{i}.weight"].shape[0] for i in range(5))
    actor = namespace["BusPolicy"](15, 1, state["mean.weight"].shape[1], sizes, 60.0)
    actor.load_state_dict(state)
    actor.eval()
    return actor


class FrozenPolicy:
    def __init__(self, method, seed):
        self.method = method
        self.path = HERE / "inputs/policies" / f"{method}_seed{seed}.pt"
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        if method == "resac":
            self.actor = resac_actor(payload)
        else:
            config = CSBAPRConfig(**payload["config"])
            config.device = "cpu"
            self.agent = CSBAPRAgent(15, 1, config)
            self.agent.actor.load_state_dict(payload["actor"])
            self.agent.actor.eval()

    def action(self, raw):
        state = normalize_obs(raw)
        if self.method == "resac":
            state[:5] = np.rint(state[:5] * [12.0, 25.0, 55.0, 24.0, 1.0])
            unit = self.actor.get_action(state, deterministic=True) / 30.0 - 1.0
        else:
            unit = self.agent.select_action(state, deterministic=True)
        holding = 30.0 * (float(np.clip(unit[0], -1.0, 1.0)) + 1.0)
        assert np.isfinite(holding) and 0 <= holding <= 60
        return np.array([holding, 1.0], dtype=np.float32)


class CohortEnv(MultiLineEnv):
    def reset(self):
        result = super().reset()
        tagged = passenger_class(Passenger)
        for line in self.line_map.values():
            flow = ArrivalFlow(self.evaluation_seed, line.line_idx, self.scene,
                               self.burst_start, self.burst_end, P["extra_multiplier"])
            install_arrivals(line, tagged, flow)
        return result

    def step(self, actions, **kwargs):
        if self.current_time >= self.horizon:
            self.done = True
            return self._aggregate_state(), self._aggregate_reward(), True
        states, reward, done = super().step(actions, **kwargs)
        return states, reward, done or self.current_time >= self.horizon


def evaluate(args):
    path = HERE / args.phase / args.method / f"seed{args.seed}" / (
        f"day{args.evaluation_seed}_{args.scene}.json"
    )
    if path.exists():
        return
    torch.set_num_threads(1)
    random.seed(args.evaluation_seed)
    np.random.seed(args.evaluation_seed)
    torch.manual_seed(args.evaluation_seed)
    policy = FrozenPolicy(args.method, args.seed)
    # Actor construction must not shift the environmental RNG stream.
    random.seed(args.evaluation_seed)
    np.random.seed(args.evaluation_seed)
    torch.manual_seed(args.evaluation_seed)
    horizon, start, end = P["horizon_seconds"], P["burst_start"], P["burst_end"]
    if args.phase == "preflight":
        horizon, start, end = 900, 300, 360
    CohortEnv.horizon, CohortEnv.burst_start, CohortEnv.burst_end = horizon, start, end
    CohortEnv.evaluation_seed, CohortEnv.scene = args.evaluation_seed, args.scene
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="thesis-passenger-cohorts-") as temporary:
        directory = Path(temporary)
        source = RUNTIME / "offline-sumo/env/calibrated_env"
        shutil.copy2(source / "config.json", directory / "config.json")
        (directory / "data").symlink_to(source / "data", target_is_directory=True)
        with redirect_stdout(StringIO()):
            env = CohortEnv(str(directory))
            for line in env.line_map.values():
                line.max_time = max(line.max_time, horizon + line.time_step)
            env.reset()
            states, _, _ = env.initialize_state()
            commands = {key: {bus: 0.0 for bus in range(line.max_agent_num)}
                        for key, line in env.line_map.items()}
            decisions = 0

            def choose(observations):
                nonlocal decisions
                for line, buses in observations.items():
                    for bus, values in buses.items():
                        if values:
                            commands[line][bus] = policy.action(values[-1])
                            decisions += 1

            choose(states)
            done = False
            with torch.no_grad():
                while not done:
                    states, _, done = env.step_to_event(commands)
                    commands = {line: {bus: 0.0 for bus in buses}
                                for line, buses in commands.items()}
                    choose(states)
        assert env.current_time == horizon
        people = (passenger for line in env.line_map.values()
                  for station in line.stations for passenger in station.total_passenger)
        groups = summarize_passengers(people, horizon, start, end)
        assert groups["baseline"]["passengers"] > 0
        assert groups["all"]["passengers"] == (
            groups["baseline"]["passengers"] + groups["extra"]["passengers"]
        )
        assert (groups["extra"]["passengers"] > 0) == (args.scene == "burst50")
        row = dict(method=args.method, training_seed=args.seed, evaluation_seed=args.evaluation_seed,
                   scene=args.scene, phase=args.phase, horizon_seconds=horizon,
                   burst_start=start, burst_end=end, checkpoint=str(policy.path), groups=groups,
                   decisions=decisions, wall_seconds=time.monotonic() - started,
                   peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        write(path, row)
    print(json.dumps({k: row[k] for k in ("method", "scene", "phase", "wall_seconds", "decisions")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=P["methods"], required=True)
    parser.add_argument("--seed", choices=P["training_seeds"], type=int, required=True)
    parser.add_argument("--evaluation-seed", choices=P["evaluation_seeds"], type=int, required=True)
    parser.add_argument("--scene", choices=P["scenes"], required=True)
    parser.add_argument("--phase", choices=("preflight", "evaluate"), default="evaluate")
    evaluate(parser.parse_args())
