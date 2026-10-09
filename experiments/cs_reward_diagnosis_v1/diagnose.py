"""Frozen-policy reward and holding diagnostics; no learner updates."""

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "cs_resac_passenger_cohorts_v1"
runtime = json.loads((HERE / "node_runtime.json").read_text())
os.environ.update(runtime["env"])
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1",
                  OFFLINE_SUMO_ENV=str(PRIOR / "inputs/runtime/offline-sumo/env"))
sys.path.insert(0, str(PRIOR))
import numpy as np
import torch
import run as paired


def distribution(values):
    x = np.asarray(values, dtype=float)
    return dict(count=len(x), mean=float(x.mean()), std=float(x.std()),
                p01=float(np.quantile(x, .01)), median=float(np.median(x)),
                p99=float(np.quantile(x, .99)), minimum=float(x.min()), maximum=float(x.max()))


class DiagnosticEnv(paired.CohortEnv):
    def reset(self):
        self.wait = self.ride = 0.
        self.marks, self.passenger_rewards, self.intervals = {}, {}, {}
        return super().reset()

    def step(self, commands, **kwargs):
        before = self.current_time
        waiting = sum(len(s.waiting_passengers) for line in self.line_map.values() for s in line.stations)
        onboard = sum(len(b.passengers) for line in self.line_map.values() for b in line.bus_all)
        states, reward, done = super().step(commands, **kwargs)
        elapsed = self.current_time - before
        self.wait += waiting * elapsed
        self.ride += onboard * elapsed
        total = 2 * self.wait + self.ride
        self.passenger_rewards, self.intervals = {}, {}
        for line, buses in states.items():
            for bus, observations in buses.items():
                if observations:
                    key = (line, bus)
                    previous_cost, previous_time = self.marks.get(key, (total, self.current_time))
                    self.passenger_rewards[key] = -(total - previous_cost)
                    self.intervals[key] = self.current_time - previous_time
                    self.marks[key] = (total, self.current_time)
        return states, reward, done


def evaluate(args):
    out = HERE / "results" / f"{args.arm}_{args.seed}_{args.scene}.json"
    if out.exists():
        return
    torch.set_num_threads(1)
    config = None
    if args.arm.startswith("fixed"):
        hold = float(args.arm.removeprefix("fixed"))
        choose = lambda raw: np.array([hold, 1.], dtype=np.float32)
        checkpoint = None
    else:
        checkpoint = HERE / "policies" / f"{args.arm}_seed{args.seed}.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert payload["objective"] == args.arm
        config = paired.CSBAPRConfig(**payload["config"])
        config.device = "cpu"
        agent = paired.CSBAPRAgent(15, 1, config)
        agent.actor.load_state_dict(payload["actor"])
        agent.actor.eval()

        def choose(raw):
            action = agent.select_action(paired.normalize_obs(raw), deterministic=True)
            return np.array([30 * (float(action[0]) + 1), 1.], dtype=np.float32)

    random.seed(78101)
    np.random.seed(78101)
    torch.manual_seed(78101)
    DiagnosticEnv.horizon = 18000
    DiagnosticEnv.burst_start, DiagnosticEnv.burst_end = 3600, 4200
    DiagnosticEnv.evaluation_seed, DiagnosticEnv.scene = 78101, args.scene
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="thesis-cs-reward-diagnosis-") as temporary:
        directory = Path(temporary)
        source = PRIOR / "inputs/runtime/offline-sumo/env/calibrated_env"
        shutil.copy2(source / "config.json", directory / "config.json")
        (directory / "data").symlink_to(source / "data", target_is_directory=True)
        with redirect_stdout(StringIO()), torch.no_grad():
            env = DiagnosticEnv(str(directory))
            for line in env.line_map.values():
                line.max_time = max(line.max_time, 18000 + line.time_step)
            env.reset()
            states, _, _ = env.initialize_state()
            commands = {key: {bus: 0. for bus in range(line.max_agent_num)}
                        for key, line in env.line_map.items()}
            pending, commands_seen, settled = {}, [], []

            def select(line, bus, raw):
                command = choose(raw)
                if not np.isfinite(command).all() or not 0 <= command[0] <= 60 or command[1] != 1:
                    raise ValueError("Incorrect physical holding/speed command")
                state = paired.normalize_obs(raw)
                pending[line, bus] = state
                commands[line][bus] = command
                commands_seen.append((float(command[0]), float(raw[10]),
                    .5 * (abs(float(raw[5] - raw[8])) + abs(float(raw[6] - raw[8]))),
                    abs(float(raw[5] - raw[6])), float(raw[7])))

            for line, buses in states.items():
                for bus, observations in buses.items():
                    if observations:
                        select(line, bus, observations[-1])
            done = False
            while not done:
                states, rewards, done = env.step_to_event(commands)
                commands = {line: {bus: 0. for bus in buses} for line, buses in commands.items()}
                for line, buses in states.items():
                    for bus, observations in buses.items():
                        if not observations:
                            continue
                        key, raw = (line, bus), observations[-1]
                        state = paired.normalize_obs(raw)
                        old = pending.get(key)
                        if old is not None and int(old[2] * 55) != int(state[2] * 55):
                            settled.append((float(rewards.get(line, {}).get(bus, 0)),
                                            env.passenger_rewards[key], env.intervals[key]))
                        select(line, bus, raw)
        if env.current_time != 18000:
            raise ValueError("Evaluation ended before the fixed horizon")
        people = (p for line in env.line_map.values() for station in line.stations
                  for p in station.total_passenger)
        groups = paired.summarize_passengers(people, 18000, 3600, 4200)
        actions, transitions = np.asarray(commands_seen), np.asarray(settled)
        phase_actions = {}
        for phase, mask in [("before", actions[:, 1] < 3600),
                            ("during", (actions[:, 1] >= 3600) & (actions[:, 1] < 4200)),
                            ("after", actions[:, 1] >= 4200)]:
            phase_actions[phase] = distribution(actions[mask, 0])
        total = 2 * env.wait + env.ride
        row = dict(arm=args.arm, seed=args.seed, scene=args.scene, evaluation_seed=78101,
            checkpoint=str(checkpoint) if checkpoint else None, groups=groups,
            holding=distribution(actions[:, 0]), phase_holding=phase_actions,
            holding_below_one_second_fraction=float((actions[:, 0] < 1).mean()),
            holding_above_59_seconds_fraction=float((actions[:, 0] > 59).mean()),
            plan_mae_seconds=float(actions[:, 2].mean()),
            imbalance_seconds=float(actions[:, 3].mean()),
            station_waiting=distribution(actions[:, 4]),
            headway_reward=distribution(transitions[:, 0]),
            passenger_reward=distribution(transitions[:, 1]),
            transition_seconds=distribution(transitions[:, 2]),
            passenger_reward_interval_correlation=float(np.corrcoef(
                -transitions[:, 1], transitions[:, 2])[0, 1]),
            summed_vehicle_rewards_to_system_cost=float(-transitions[:, 1].sum() / total),
            global_integral_cost=total, cost=groups["all"]["mean_generalized_seconds"],
            zero_passenger_reward_fraction=float((transitions[:, 1] == 0).mean()),
            config={k: getattr(config, k) for k in ["gamma", "use_context", "use_nau_actor",
                "jac_weight", "weight_sym", "enable_reward_ema"]} if config else None,
            wall_seconds=time.monotonic() - started)
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps({k: row[k] for k in ["arm", "seed", "scene", "cost", "wall_seconds"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["headway", "passenger", "fixed0", "fixed15", "fixed30", "fixed60"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scene", choices=["normal", "burst50"], required=True)
    evaluate(parser.parse_args())
