"""Isolated CS reward-interface training; original learner and bus dynamics frozen."""

import argparse
from collections import deque
from contextlib import redirect_stdout
from dataclasses import asdict
from io import StringIO
import json
import os
from pathlib import Path
import random
import resource
import shutil
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "cs_resac_passenger_cohorts_v1"
RUNTIME = PRIOR / "inputs/runtime"
P = json.loads((HERE / "protocol.json").read_text())
runtime = json.loads((HERE / "node_runtime.json").read_text())
os.environ.update(runtime["env"])
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", OFFLINE_SUMO_ENV=str(RUNTIME / "offline-sumo/env"))
sys.path.insert(0, str(RUNTIME / "CS-BAPR"))
sys.path.insert(0, str(RUNTIME / "offline-sumo/env"))
sys.path.insert(0, str(PRIOR))
import numpy as np
import torch
from csbapr.agent import CSBAPRAgent
from csbapr.config import CSBAPRConfig
from csbapr.networks.critic import EnsembleQNet
from scripts.test_multiline_convergence import normalize_obs
from sim_core.passenger import Passenger
from sim_core.sim import MultiLineEnv
from cohorts import install_arrivals, passenger_class, summarize_passengers
from contract import ArrivalFlow, EventCredit, observation


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CostEnv(MultiLineEnv):
    def reset(self):
        self.wait = self.ride = self.legacy_cost = 0.
        self.marks, self.passenger_rewards = {}, {}
        result = super().reset()
        tagged = passenger_class(Passenger)
        for line in self.line_map.values():
            install_arrivals(line, tagged, ArrivalFlow(self.day, line.line_idx, self.schedule))
        self.lines = sorted(self.line_map.values(), key=lambda line: line.line_idx)
        assert [line.line_idx for line in self.lines] == list(range(12))
        self.measure()
        return result

    def measure(self):
        self.line_stats = []
        self.bus_lookup = {}
        for line in self.lines:
            waiting = sum(len(s.waiting_passengers) for s in line.stations)
            onboard = sum(len(b.passengers) for b in line.bus_all)
            active = [b for b in line.bus_all if b.on_route]
            dwell = sum(float(b.holding_time or 0.) + float(b.dwelling_time or 0.)
                        for b in active) / max(1, len(active))
            self.line_stats.append((waiting, onboard, len(active), dwell))
            for bus in line.bus_all:
                self.bus_lookup[line.line_idx, bus.bus_id] = bus
        self.waiting = sum(s[0] for s in self.line_stats)
        self.onboard = sum(s[1] for s in self.line_stats)

    @property
    def cost(self):
        return 2 * self.wait + self.ride

    def step(self, actions, **kwargs):
        if self.current_time >= self.horizon:
            self.done = True
            return self._aggregate_state(), self._aggregate_reward(), True
        before = self.current_time
        old_waiting, old_onboard = self.waiting, self.onboard
        states, rewards, done = super().step(actions, **kwargs)
        elapsed = self.current_time - before
        self.measure()
        # Passenger exchange is timestamped at the start of this simulation tick.
        self.wait += self.waiting * elapsed
        self.ride += self.onboard * elapsed
        self.legacy_cost += (2 * old_waiting + old_onboard) * elapsed
        self.passenger_rewards = {}
        for line, buses in states.items():
            for bus, values in buses.items():
                if values:
                    key = (line, bus)
                    previous = self.marks.get(key, self.legacy_cost)
                    self.passenger_rewards[key] = -(self.legacy_cost - previous)
                    self.marks[key] = self.legacy_cost
        return states, rewards, done or self.current_time >= self.horizon


def events(states):
    return [(line, bus, values[-1]) for line, buses in sorted(states.items())
            for bus, values in sorted(buses.items()) if values]


def feature(env, raw, arm, committed, pending):
    bus = env.bus_lookup[int(raw[0]), int(raw[1])]
    return observation(normalize_obs(raw), len(bus.passengers), bus.capacity,
                       env.current_time, env.horizon, env.line_stats, committed, pending,
                       local_only=arm == "local_overlap")


def new_agent(arm, seed):
    payload = torch.load(HERE.parent / "cs_reward_diagnosis_v1/policies" /
                         f"passenger_seed{seed}.pt", map_location="cpu", weights_only=False)
    config = CSBAPRConfig(**payload["config"])
    config.state_dim, config.action_dim, config.device = 92, 1, "cpu"
    config.buffer_size, config.max_episodes = P["buffer_size"], P["episodes"]
    config.gamma = 1. if arm.startswith("global") else .99
    config.enable_reward_ema = not arm.startswith("global")
    seed_all(seed)
    agent = CSBAPRAgent(92, 1, config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 100000)
        critic = EnsembleQNet(92, 1, config.hidden_dim, config.num_critics)
    agent.critic.load_state_dict(critic.state_dict())
    agent.target_critic.load_state_dict(critic.state_dict())
    assert not config.use_nau_actor and not config.use_context
    assert config.jac_weight == config.weight_sym == config.beta_bc == 0.
    return agent


def rollout(env, agent, arm, day, train=False, updates=400):
    seed_all(day)
    env.day = day
    env.reset()
    states, _, done = env.initialize_state()
    aligned = arm.startswith("global")
    credit = EventCredit(env.horizon, P["reward_scale"], arm == "global_shaped")
    recent, pending = deque(maxlen=256), {}
    decisions = transitions = update_count = 0
    holds, headway_rewards, plan_errors, imbalances = [], [], [], []
    raw_rewards = []
    learning_sums, last_update = {}, {}
    commands = {line: {bus: 0. for bus in range(le.max_agent_num)}
                for line, le in env.line_map.items()}

    def store(transition):
        nonlocal transitions
        if transition is None:
            return
        transitions += 1
        raw_rewards.append(float(transition[2]))
        if train:
            agent.store_transition(*transition)
            recent.append(transition)

    def learn(target):
        nonlocal update_count, last_update
        if not train or not recent or agent.replay_buffer.size < agent.config.batch_size:
            return
        while update_count < target:
            rolling = dict(state=np.stack([r[0] for r in recent]),
                           action=np.stack([r[1] for r in recent]),
                           reward=np.array([r[2] for r in recent], dtype=np.float32))
            result = agent.update(recent_rollout=rolling)
            assert result is not None
            last_update = {k: float(v) for k, v in result.items()}
            assert all(np.isfinite(v) for v in last_update.values())
            for key, value in last_update.items():
                learning_sums[key] = learning_sums.get(key, 0.) + value
            update_count += 1

    while not done:
        wave, committed = events(states), {}
        for index, (line, bus, raw) in enumerate(wave):
            state = feature(env, raw, arm, committed, len(wave) - index)
            if aligned:
                store(credit.finish(state, env.current_time, env.cost, env.waiting, env.onboard))
            elif (line, bus) in pending:
                old_state, old_action = pending[line, bus]
                if int(old_state[2] * 55) != int(state[2] * 55):
                    store((old_state, old_action, env.passenger_rewards[line, bus], state, 0.))
            with torch.no_grad():
                action = np.asarray(agent.select_action(state, deterministic=not train), dtype=np.float32)
            hold = 30. * (float(np.clip(action[0], -1., 1.)) + 1.)
            assert np.isfinite(hold) and 0 <= hold <= 60
            commands[line][bus] = np.array([hold, 1.], dtype=np.float32)
            if aligned:
                credit.begin(state, action, env.current_time, env.cost, env.waiting, env.onboard)
            else:
                pending[line, bus] = (state, action)
            committed.setdefault(int(raw[0]), []).append(hold)
            holds.append(hold)
            plan_errors.append(.5 * (abs(float(raw[5] - raw[8])) + abs(float(raw[6] - raw[8]))))
            imbalances.append(abs(float(raw[5] - raw[6])))
            decisions += 1
        learn(int(updates * env.current_time / env.horizon))
        states, rewards, done = env.step_to_event(commands)
        for line, bus, _ in events(states):
            headway_rewards.append(float(rewards.get(line, {}).get(bus, 0.)))
        commands = {line: {bus: 0. for bus in buses} for line, buses in commands.items()}
    if aligned:
        store(credit.finish(np.zeros(92, dtype=np.float32), env.current_time, env.cost,
                            env.waiting, env.onboard, terminal=True))
        identity = credit.verify(env.cost)
    else:
        identity = dict(cost_identity_passed=None, terminal_transitions=0,
                        unsettled_vehicle_transitions=len(pending))
    learn(updates)
    assert env.current_time == env.horizon
    if train:
        assert update_count == updates, (update_count, updates, agent.replay_buffer.size)
    people = (p for line in env.lines for station in line.stations for p in station.total_passenger)
    groups = summarize_passengers(people, env.horizon, 3600, 4200)
    timestamp_cost = 2 * groups["all"]["waiting_seconds"] + groups["all"]["onboard_seconds"]
    assert np.isclose(env.cost, timestamp_cost, rtol=0., atol=.01), (env.cost, timestamp_cost)
    return dict(day=day, horizon_seconds=env.current_time, groups=groups,
                cost=groups["all"]["mean_generalized_seconds"], exact_cost=env.cost,
                legacy_cost=env.legacy_cost, timestamp_cost_identity_passed=True,
                decisions=decisions, transitions=transitions, updates=update_count,
                mean_holding_seconds=float(np.mean(holds)),
                plan_mae_seconds=float(np.mean(plan_errors)),
                imbalance_seconds=float(np.mean(imbalances)),
                native_headway_penalty=float(-np.mean(headway_rewards)),
                replay_reward_mean=float(np.mean(raw_rewards)),
                replay_reward_std=float(np.std(raw_rewards)), credit=identity,
                update_mean={k: v / update_count for k, v in learning_sums.items()},
                update_last=last_update,
                reward_ema_std=float(getattr(agent, "_reward_ema_var", 1.) ** .5))


def make_env(directory, horizon, scene):
    source = RUNTIME / "offline-sumo/env/calibrated_env"
    shutil.copy2(source / "config.json", directory / "config.json")
    (directory / "data").symlink_to(source / "data", target_is_directory=True)
    env = CostEnv(str(directory))
    env.horizon, env.schedule, env.day = horizon, P["scenes"][scene], 78101
    for line in env.line_map.values():
        line.max_time = max(line.max_time, horizon + line.time_step)
    return env


def training(args):
    out = HERE / ("preflight" if args.smoke else "runs") / args.arm / f"seed{args.seed}"
    if (out / "ready.json").exists():
        print("Already complete: " + str(out), flush=True)
        return
    torch.set_num_threads(1)
    agent = new_agent(args.arm, args.seed)
    started, history = time.monotonic(), []
    horizon = 3600 if args.smoke else P["horizon_seconds"]
    quota = 40 if args.smoke else P["updates_per_episode"]
    count = 2 if args.smoke else P["episodes"]
    write(out / "settings.json", dict(protocol=P, arm=args.arm, config=asdict(agent.config),
          episodes=count, updates_per_episode=quota, horizon_seconds=horizon,
          matched_initialization=True, checkpoint_selection="final episode only"))
    with tempfile.TemporaryDirectory(prefix="thesis-cs-alignment-") as tmp:
        with redirect_stdout(StringIO()):
            env = make_env(Path(tmp), horizon, "normal")
            for episode in range(count):
                row = rollout(env, agent, args.arm, args.seed * 1000 + episode, train=True, updates=quota)
                row.update(episode=episode + 1, wall_seconds=time.monotonic() - started,
                           peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
                history.append(row)
                write(out / "progress.json", row)
                sys.__stdout__.write(json.dumps(dict(arm=args.arm, seed=args.seed,
                    episode=episode + 1, cost=row["cost"], updates=row["updates"],
                    alpha=row["update_last"]["alpha"], wall_seconds=row["wall_seconds"])) + "\n")
                sys.__stdout__.flush()
    torch.save(dict(actor=agent.actor.state_dict(), config=asdict(agent.config), arm=args.arm,
                    training_seed=args.seed, episodes=count, updates=agent.training_steps,
                    objective="passenger", action_space="holding"), out / "policy.pt")
    write(out / "train_summary.json", dict(history=history, training_wall_seconds=time.monotonic()-started,
          training_steps=agent.training_steps, peak_rss_mb=history[-1]["peak_rss_mb"]))
    write(out / "ready.json", dict(arm=args.arm, seed=args.seed, episodes=count,
          updates=agent.training_steps, timestamp_cost_identity_passed=True))


def evaluate(args):
    out = HERE / "results" / args.arm / f"seed{args.seed}" / f"day{args.day}_{args.scene}.json"
    if out.exists():
        return
    torch.set_num_threads(1)
    if args.arm in P["arms"]:
        directory = "preflight" if args.smoke else "runs"
        payload = torch.load(HERE / directory / args.arm / f"seed{args.seed}" / "policy.pt",
                             map_location="cpu", weights_only=False)
        config = CSBAPRConfig(**payload["config"])
        config.device = "cpu"
        agent = CSBAPRAgent(92, 1, config)
        agent.actor.load_state_dict(payload["actor"])
        agent.actor.eval()
        arm = args.arm
    elif args.arm == "fixed0":
        class Fixed:
            def select_action(self, state, deterministic):
                return np.array([-1.], dtype=np.float32)
        agent, arm = Fixed(), "local_overlap"
    else:
        objective = args.arm.removeprefix("frozen_")
        payload = torch.load(HERE.parent / "cs_reward_diagnosis_v1/policies" /
                             f"{objective}_seed{args.seed}.pt", map_location="cpu", weights_only=False)
        config = CSBAPRConfig(**payload["config"])
        config.device = "cpu"
        frozen = CSBAPRAgent(15, 1, config)
        frozen.actor.load_state_dict(payload["actor"])
        frozen.actor.eval()
        class Frozen:
            def select_action(self, state, deterministic):
                return frozen.select_action(state[:15], deterministic=True)
        agent, arm = Frozen(), "local_overlap"
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="thesis-cs-alignment-eval-") as tmp:
        with redirect_stdout(StringIO()):
            env = make_env(Path(tmp), P["horizon_seconds"], args.scene)
            row = rollout(env, agent, arm, args.day)
    row.update(arm=args.arm, seed=args.seed, scene=args.scene, evaluation_seed=args.day,
               policy_phase="preflight" if args.smoke else "formal",
               wall_seconds=time.monotonic()-started,
               peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    if args.smoke:
        out = HERE / "preflight_results" / args.arm / f"{args.scene}.json"
    write(out, row)
    print(json.dumps({k: row[k] for k in ["arm", "seed", "scene", "cost", "wall_seconds", "peak_rss_mb"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval", "preflight"])
    parser.add_argument("--arm", choices=P["arms"] + ["fixed0", "frozen_passenger", "frozen_headway"], required=True)
    parser.add_argument("--seed", type=int, default=64001)
    parser.add_argument("--day", type=int, default=78101)
    parser.add_argument("--scene", choices=list(P["scenes"]), default="normal")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.mode == "preflight":
        args.smoke = True
        training(args)
        for args.scene in ["normal", "burst50"]:
            evaluate(args)
    else:
        (training if args.mode == "train" else evaluate)(args)
