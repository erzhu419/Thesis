"""Reconcile the original transit recipe and a context-dependent equilibrium."""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
from io import StringIO
import json
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from followup_common import HERE, EXPERIMENTS, P, folder, parent, write
sys.path.insert(0, str(EXPERIMENTS / "bus_legacy_dual_v1"))
import cs_runner as previous


def context_policy(dimension):
    from csbapr.networks.policy import GaussianPolicy
    from adaptation import equilibrium_states

    class ContextPolicy(GaussianPolicy):
        def __init__(self):
            super().__init__(15, dimension, 64, activation="tanh")
            self.context_mean = nn.Sequential(nn.Linear(15, 32), nn.Tanh(), nn.Linear(32, dimension))
            nn.init.zeros_(self.context_mean[-1].weight)
            nn.init.zeros_(self.context_mean[-1].bias)

        def forward(self, states):
            mean, log_std = super().forward(states)
            projected = equilibrium_states(states)
            equilibrium_mean, _ = super().forward(projected)
            return self.context_mean(projected) + mean - equilibrium_mean, log_std

        def compute_L_eff(self):
            return float("inf")

    return ContextPolicy()


def setup(item):
    kind = "cs_native" if item["family"] == "cs_recipe" else "cs_symbolic"
    return kind, previous.setup(kind)


def build(item, kind, runtime):
    dimension = 2 if item["mode"] == "dual" else 1
    arm = item["arm"]
    if item["family"] == "cs_recipe":
        base = runtime["base"]
        runtime["seed_all"](item["seed"])
        config = base.configuration("nau_bc0" if arm == "stabilized_nau" else "mlp_bc0", "cpu")
        config.action_dim, config.max_episodes = dimension, P["cs_episodes"]
        if arm == "pristine_mlp":
            config.actor_weight_decay, config.beta_bc = 0., .001
            config.enable_min_q_target = False
            config.enable_reward_ema = False
            config.enable_entropy_floor = False
            config.enable_rollout_surprise = False
            config.bapr_warmup_iters, config.penalty_scale = 0, 5.
        agent = base.CSBAPRAgent(15, dimension, config)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(item["seed"] + 100000)
            from csbapr.networks.critic import EnsembleQNet
            critic = EnsembleQNet(15, dimension, 64, 3)
        agent.critic.load_state_dict(critic.state_dict())
        agent.target_critic.load_state_dict(critic.state_dict())
    else:
        symbolic = arm.endswith("symbolic")
        original = SimpleNamespace(kind=kind, mode=item["mode"], objective=item["objective"],
            arm="smooth_soft" if symbolic else "soft_no_symbolic", seed=item["seed"])
        agent = previous.build_agent(original, runtime)
        if arm.startswith("context_"):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(item["seed"] + 50000)
                agent.actor = context_policy(dimension)
            agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=agent.config.lr_actor,
                weight_decay=agent.config.actor_weight_decay)
    return agent


def contracts(item):
    kind, runtime = setup(item)
    results = []
    arms = ["pristine_mlp"] if item["family"] == "cs_recipe" else ["context_symbolic", "context_control"]
    for arm in arms:
        value = dict(item, arm=arm, mode="dual")
        agent = build(value, kind, runtime)
        agent.config.batch_size = 16
        rng = np.random.default_rng(731)
        states = rng.uniform(.1, .9, (64, 15)).astype(np.float32)
        states[:, 0] = np.arange(64) % 12 / 12
        actions = rng.uniform(-.8, .8, (64, 2)).astype(np.float32)
        rewards = -np.arange(1, 65, dtype=np.float32)
        for s, a, r in zip(states, actions, rewards):
            agent.replay_buffer.push(s, a, float(r), s.copy(), 0.)
        if item["family"] == "cs_context":
            from adaptation import equilibrium_states
            eq = equilibrium_states(torch.from_numpy(states))
            mean, _ = agent.actor(eq)
            assert torch.allclose(mean, agent.actor.context_mean(eq), atol=1e-6)
            with torch.no_grad():
                agent.actor.context_mean[-1].weight.fill_(.02)
            eq.requires_grad_(True)
            jac = torch.autograd.grad(agent.actor(eq)[0][:, 0].sum(), eq)[0]
            assert float(jac[:, 7].abs().sum()) > 0
        metrics = agent.update(recent_rollout=dict(state=list(states), action=list(actions), reward=list(rewards)))
        assert all(np.isfinite(v) for k, v in metrics.items() if not k.startswith("L_eff"))
        results.append(dict(arm=arm, q_loss=metrics["q_loss"], alpha=metrics["alpha"]))
    write(HERE / "contracts" / (item["family"] + ".json"), dict(passed=True, results=results))


def run(args):
    item = vars(args).copy()
    item.pop("phase"); item.pop("case"); item.pop("layer")
    if args.phase == "contracts":
        contracts(item)
        return
    out = folder(item)
    path = out / ("ready.json" if args.phase == "train" else f"eval_{args.case:02d}.json")
    if path.exists():
        return
    kind, runtime = setup(item)
    if args.arm.startswith("zero_"):
        action = np.array([-1., 1. if args.arm == "zero_fast" else 0.], np.float32)
        agent = SimpleNamespace(select_action=lambda *a, **k: action.copy())
    else:
        agent = build(item, kind, runtime)
        if args.phase == "evaluate":
            source = parent(item) / "policy.pt"
            payload = torch.load(source, map_location="cpu", weights_only=False)
            agent.actor.load_state_dict(payload["actor"])
            agent.actor.eval()
    horizon = 3600 if args.smoke else P["cs_horizon_seconds"]
    runtime["observed_protocol"]["horizon_seconds"] = horizon
    runtime["Env"].objective = args.objective
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="thesis-cs-followup-") as directory:
        directory = Path(directory)
        shutil.copy2(runtime["source"] / "config.json", directory / "config.json")
        (directory / "data").symlink_to(runtime["source"] / "data", target_is_directory=True)
        with redirect_stdout(StringIO()):
            env = runtime["Env"](str(directory))
        for line in env.line_map.values():
            line.max_time = max(line.max_time, horizon + line.time_step)
        if args.phase == "train":
            write(out / "protocol.json", dict(item=item, protocol=P, config=asdict(agent.config)))
            history = []
            for index in range(1 if args.smoke else P["cs_episodes"]):
                runtime["seed_all"](args.seed * 1000 + index)
                with redirect_stdout(StringIO()):
                    row = previous.episode(env, agent, runtime["normalize"], kind,
                        train=True, deterministic=False, config=agent.config)
                row.update(episode=index+1, horizon_seconds=env.current_time, wall_seconds=time.monotonic()-started)
                history.append(row)
                write(out / "progress.json", row)
                print(row, flush=True)
            torch.save(dict(actor=agent.actor.state_dict(), config=asdict(agent.config)), out / "policy.pt")
            write(out / "train_summary.json", dict(history=history))
            write(path, dict(item=item, episodes=len(history), action_dim=2 if args.mode == "dual" else 1,
                horizon_seconds=horizon, wall_seconds=time.monotonic()-started,
                peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
        else:
            scene = list(P["cs_scenes"])[args.case // 3]
            evaluation_seed = P["cs_evaluation_seeds"][args.case % 3]
            runtime["seed_all"](evaluation_seed)
            env.set_ood_schedule(P["cs_scenes"][scene])
            with redirect_stdout(StringIO()):
                row = previous.episode(env, agent, runtime["normalize"], kind, train=False, deterministic=True)
            if env.current_time != horizon:
                raise RuntimeError("Incomplete paired CS evaluation")
            ledger = getattr(env, runtime["ledger"])
            people = [p for line in env.line_map.values() for station in line.stations for p in station.total_passenger]
            row.update(item=item, case=args.case, scene=scene, evaluation_seed=evaluation_seed,
                horizon_seconds=horizon, cost=(2*ledger.wait+ledger.ride)/len(people),
                completion_rate=sum(p.arrived for p in people)/len(people), generated_passengers=len(people),
                wall_seconds=time.monotonic()-started,
                peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            write(path, row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("cs_recipe", "cs_context"), required=True)
    parser.add_argument("--phase", choices=("contracts", "train", "evaluate"), required=True)
    parser.add_argument("--mode", choices=("holding", "dual"), default="dual")
    parser.add_argument("--objective", choices=P["cs_objectives"], default="headway")
    parser.add_argument("--arm", default="pristine_mlp")
    parser.add_argument("--seed", type=int, default=P["cs_seeds"][0])
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--layer", default="native")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args)
    print("DONE CS followup", vars(args), flush=True)
