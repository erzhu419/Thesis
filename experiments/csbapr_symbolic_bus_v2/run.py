"""Frozen-v8 symbolic bus training and disjoint fixed-horizon evaluations."""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
from io import StringIO
from types import SimpleNamespace
import time

import numpy as np
import torch

from settings import P, model_folder, run_folder, write
from adaptation import ResidualPolicy, RuleReference, load_reference
from bus_runtime import (CSBAPRAgent, CSBAPRConfig, environment, run_episode,
                         seed_all, MeasuredPolicy, service_metrics)


class MeasuredAgent(CSBAPRAgent):
    last_metrics = None

    def update(self, **kwargs):
        metrics = super().update(**kwargs)
        if not all(np.isfinite(v) for k, v in metrics.items() if not k.startswith("L_eff")):
            raise RuntimeError("Nonfinite learner update")
        self.last_metrics = {k:v for k,v in metrics.items() if not k.startswith("L_eff")}
        return metrics


def build_agent(args):
    symbolic = args.arm in ("centered_symbolic", "uncentered_symbolic")
    config = CSBAPRConfig(env_name="MultiLineBus", state_dim=15, action_dim=1,
        max_episodes=P["episodes"], hidden_dim=64, num_critics=3, batch_size=256,
        warmup_steps=500, use_nau_actor=False, actor_type="tanh-mlp", beta_bc=0.,
        weight_reg=0., jac_weight=.01 if symbolic else 0.,
        weight_sym=.01 if symbolic else 0., nau_reg_weight=0.,
        actor_weight_decay=1e-4, track_L_eff=False, device="cpu")
    seed_all(args.seed)
    agent = MeasuredAgent(15, 1, config)
    reference = (load_reference(model_folder(args.seed, args.smoke), args.objective)
                 if symbolic else RuleReference(args.objective))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed+50000)
        agent.actor = ResidualPolicy(reference, args.arm != "uncentered_symbolic")
        torch.manual_seed(args.seed+100000)
        from csbapr.networks.critic import EnsembleQNet
        critic = EnsembleQNet(15, 1, 64, 3)
    agent.critic.load_state_dict(critic.state_dict())
    agent.target_critic.load_state_dict(critic.state_dict())
    agent.actor_optimizer = torch.optim.Adam((p for p in agent.actor.parameters() if p.requires_grad),
        lr=config.lr_actor, weight_decay=config.actor_weight_decay)
    agent.pi_sym_torch = reference if symbolic else None
    return agent


def train(args):
    out = run_folder(args.objective, args.arm, args.seed, args.smoke)
    if (out / "results/ready.json").exists():
        return
    agent = build_agent(args)
    episodes = 1 if args.smoke else P["episodes"]
    write(out / "results/protocol.json", dict(protocol=P, config=asdict(agent.config),
        objective=args.objective, arm=args.arm, seed=args.seed,
        source_snapshot="CS-BAPR completed validation v8", identification_is_extra=True))
    history, started = [], time.monotonic()
    with environment(args.objective) as env:
        for episode in range(episodes):
            seed_all(args.seed*1000+episode)
            with redirect_stdout(StringIO()):
                rewards, decisions, updates = run_episode(env, agent, config=agent.config)
            row = dict(episode=episode+1, native_reward=sum(rewards.values()), decisions=decisions,
                updates=updates, total_updates=agent.training_steps,
                global_passenger_seconds=2*env.ledger.wait+env.ledger.ride,
                metrics=agent.last_metrics, elapsed_seconds=time.monotonic()-started)
            history.append(row)
            write(out / "results/progress.json", row)
            print(row, flush=True)
    torch.save(dict(actor=agent.actor.state_dict(), config=asdict(agent.config),
                    objective=args.objective, arm=args.arm, training_seed=args.seed), out / "policy.pt")
    write(out / "results/train_summary.json", dict(history=history))
    write(out / "results/ready.json", dict(arm=args.arm, seed=args.seed, objective=args.objective,
        episodes=episodes, total_updates=agent.training_steps, wall_seconds=time.monotonic()-started))


def evaluate(args):
    out = run_folder(args.objective, args.arm, args.seed, args.smoke)
    path = out / "results" / f"eval_{args.case:02d}.json"
    if path.exists():
        return
    if args.arm in ("fitted_reference", "daganzo", "zero"):
        reference = (load_reference(model_folder(args.seed, args.smoke), args.objective)
                     if args.arm == "fitted_reference" else RuleReference("passenger" if args.arm == "zero" else "headway"))
        # Non-learning controls use exact endpoint actions, unlike Gaussian anchors.
        def action(states, deterministic=False):
            with torch.no_grad():
                value = reference(torch.as_tensor(states, dtype=torch.float32).reshape(1, 15)).numpy()[0]
                if args.arm == "zero":
                    value[:] = -1.
                elif args.arm == "daganzo":
                    hold = np.clip((states[6]-states[5])*300, 0, 60)
                    value[0] = hold/30-1
                return value
        agent = SimpleNamespace(select_action=action)
    else:
        agent = build_agent(args)
        payload = torch.load(out / "policy.pt", map_location="cpu", weights_only=False)
        agent.actor.load_state_dict(payload["actor"])
        agent.actor.eval()
    scene = list(P["scenes"])[args.case//3]
    eval_seed = P["evaluation_seeds"][args.case % 3]
    seed_all(eval_seed)
    started = time.monotonic()
    with environment(args.objective) as env:
        env.set_ood_schedule(P["scenes"][scene])
        policy = MeasuredPolicy(agent, env)
        with redirect_stdout(StringIO()):
            rewards, decisions, _ = run_episode(env, policy, deterministic=True, train=False)
        row = dict(arm=args.arm, training_seed=args.seed, objective=args.objective,
            case=args.case, scene=scene, evaluation_seed=eval_seed,
            native_reward=sum(rewards.values()), wall_seconds=time.monotonic()-started,
            **service_metrics(env, policy, decisions))
    write(path, row)
    print(row, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("collect", "train", "evaluate"), required=True)
    parser.add_argument("--objective", choices=P["objectives"], default="headway")
    parser.add_argument("--arm", default="centered_symbolic")
    parser.add_argument("--seed", type=int, default=P["training_seeds"][0])
    parser.add_argument("--day", type=int, default=0)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == "collect":
        from identify import collect_day
        collect_day(args.seed, args.day)
    else:
        globals()[args.phase](args)
