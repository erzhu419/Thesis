"""Frozen legacy RE-SAC/BAPR bus policies under paired operating regimes."""

import argparse
import ast
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

ROOT = Path(__file__).resolve().parent / "runtime/BAPR"
WEIGHTS = Path(__file__).resolve().parent / "weights"
sys.path.insert(0, str(ROOT))
from env.sim import env_bus
from mode_profiles import MODE_PROFILES

SCENES = ("normal", "congestion_severe", "demand_surge", "slow_switch", "fast_switch")
SEEDS = (15001, 15002, 15003)
HORIZON = 24000


def load_policy(method):
    if method in ("no_hold", "headway"):
        return RulePolicy(method)
    script = "sac_ensemble_bapr.py" if method == "bapr" else "sac_ensemble_baseline_stress.py"
    # Import only the original network definitions, without executing training globals.
    tree = ast.parse((ROOT / script).read_text())
    names = {"EmbeddingLayer", "ContextNetwork", "PolicyNetwork"}
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    namespace = dict(torch=torch, nn=nn, F=F, Normal=Normal, np=np, device=torch.device("cpu"))
    exec(compile(tree, str(ROOT / script), "exec"), namespace)
    label = "bapr_mode_s2" if method == "bapr" else "sac_mode_s2"
    state = torch.load(WEIGHTS / label / "final_policy", map_location="cpu", weights_only=True)
    categories = ["bus_id", "station_id", "time_period", "direction"]
    codes = {key: dict.fromkeys(range(state[f"embedding_layer.embeddings.{key}.weight"].shape[0]))
             for key in categories}
    embedding = namespace["EmbeddingLayer"](codes, categories)
    hidden = state["linear1.weight"].shape[0]
    input_dim = state["linear1.weight"].shape[1]
    kwargs = dict(action_range=60.)
    if method == "bapr":
        ep_dim = state["context_net.net.3.weight"].shape[0]
        input_dim -= ep_dim
        kwargs.update(ep_dim=ep_dim, context_net=namespace["ContextNetwork"](input_dim, ep_dim))
    policy = namespace["PolicyNetwork"](input_dim, 1, hidden, embedding, **kwargs)
    policy.load_state_dict(state)
    policy.eval()
    policy._warmup_active = False
    return policy


class RulePolicy:
    def __init__(self, method):
        self.method = method

    def get_action(self, state, deterministic=True):
        hold = max(0., min(60., float(state[5]-state[4])/2)) if self.method == "headway" else 0.
        return np.asarray([hold], np.float32)


class ObservedEnv(env_bus):
    def __init__(self, scene):
        self.scene = scene
        self.windows = {}
        fixed = scene if scene in MODE_PROFILES else None
        super().__init__(str(ROOT / "env"), fixed_mode=fixed)

    def _maybe_switch_mode(self):
        if self.scene not in ("slow_switch", "fast_switch"):
            return
        interval = 7200 if self.scene == "slow_switch" else 1800
        mode = list(MODE_PROFILES)[int(self.current_time)//interval % len(MODE_PROFILES)]
        if mode != self.current_mode_name:
            self._apply_mode(mode)
            self.mode_switch_count += 1
            self.mode_history.append((mode, self.current_time))

    def step(self, action, **kwargs):
        row = self.windows.setdefault(int(self.current_time)//300,
                  dict(wait_seconds=0., ride_seconds=0., samples=0,
                       hold_sum=0., imbalance_sum=0., bunching_count=0, decisions=0))
        row["wait_seconds"] += sum(len(s.waiting_passengers) for s in self.stations)*self.time_step
        row["ride_seconds"] += sum(len(b.passengers) for b in self.bus_all)*self.time_step
        row["samples"] += self.time_step
        state, reward, done = super().step(action, **kwargs)
        if done and self.current_time < HORIZON:
            raise RuntimeError("Service ended and cleared queues before the fixed observation horizon")
        return state, reward, done or self.current_time >= HORIZON


def evaluate(policy, method, scene, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = ObservedEnv(scene)
    env.reset()
    states, rewards, done = env.initialize_state(render=False)
    actions = {key: None for key in range(env.max_agent_num)}
    reward_sum = 0.
    started = time.monotonic()
    while not done:
        for key, observations in states.items():
            select = len(observations) == 1 and actions[key] is None
            if len(observations) == 2:
                if observations[0][1] != observations[1][1]:
                    reward_sum += float(rewards[key])
                states[key] = observations = observations[1:]
                select = True
            if select:
                obs = np.asarray(observations[0], np.float32)
                with torch.inference_mode():
                    actions[key] = policy.get_action(torch.from_numpy(obs), deterministic=True)
                row = env.windows.setdefault(int(env.current_time)//300,
                      dict(wait_seconds=0., ride_seconds=0., samples=0,
                           hold_sum=0., imbalance_sum=0., bunching_count=0, decisions=0))
                row["hold_sum"] += float(actions[key][0])
                row["imbalance_sum"] += abs(float(obs[4]-obs[5]))
                row["bunching_count"] += int(float(obs[4]) < 30.)
                row["decisions"] += 1
        states, rewards, done = env.step(actions)
    people = [p for s in env.stations for p in s.total_passenger]
    wait = sum(row["wait_seconds"] for row in env.windows.values())
    ride = sum(row["ride_seconds"] for row in env.windows.values())
    return dict(method=method, version="legacy_context_bus", training_seed=2,
                evaluation_seed=seed, scene=scene, horizon_seconds=env.current_time,
                generated_passengers=len(people), arrived_passengers=sum(p.arrived for p in people),
                waiting_tail=sum(len(s.waiting_passengers) for s in env.stations),
                onboard_tail=sum(len(b.passengers) for b in env.bus_all),
                wait_passenger_seconds=wait, ride_passenger_seconds=ride,
                generalized_seconds_per_passenger=(2*wait+ride)/len(people),
                reward=reward_sum, mode_history=env.mode_history,
                windows=[dict(start=k*300, end=(k+1)*300, **v) for k, v in sorted(env.windows.items())],
                elapsed_seconds=time.monotonic()-started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["resac", "bapr", "no_hold", "headway"], required=True)
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--load-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    policy = load_policy(args.method)
    if args.load_only:
        print(f"{args.method}: frozen policy loaded", flush=True)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        path = args.out / f"{args.method}_{args.scene}_seed{seed}.json"
        if path.exists():
            continue
        result = evaluate(policy, args.method, args.scene, seed)
        path.write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps({k:v for k,v in result.items() if k != "windows"}), flush=True)


if __name__ == "__main__":
    main()
