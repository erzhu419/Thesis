"""Source-only anchored training and independent frozen-policy evaluation."""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import importlib.util
from io import StringIO
import json
from pathlib import Path
import resource
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
PRIOR = HERE.parent / "cs_input_training_v1"
sys.path.insert(0, str(PRIOR))
spec = importlib.util.spec_from_file_location("source_anchor_runtime", PRIOR / "run.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
base, np, torch = old.base, old.np, old.torch
from anchor import SourceAnchor


class AnchoredAgent(base.CSBAPRAgent):
    def update(self, batch_size=None, recent_rollout=None):
        if self.replay_buffer.size < (batch_size or self.config.batch_size):
            return super().update(batch_size=batch_size, recent_rollout=recent_rollout)
        loss, samples = self.source_anchor.prepare()
        try:
            result = super().update(batch_size=batch_size, recent_rollout=recent_rollout)
        finally:
            self.source_anchor.clear()
        assert result is not None
        result.update(source_anchor_loss=loss, source_anchor_samples=samples)
        return result


def teacher(seed):
    return old.shared.actor_for(seed)


def new_agent(seed, weight, cap):
    reference = teacher(seed)
    native = base.CSBAPRAgent
    base.CSBAPRAgent = AnchoredAgent
    try:
        agent = base.new_agent("global_coverage", seed)
    finally:
        base.CSBAPRAgent = native
    agent.source_anchor = SourceAnchor(agent.actor, reference, cap, weight, seed,
                                       P["reservoir_size"], P["anchor_batch_size"])
    settings = json.loads((old.REFERENCE / "runs/global_coverage" / f"seed{seed}/settings.json").read_text())
    assert asdict(agent.config) == settings["config"]
    return agent


def gradient_check():
    torch.manual_seed(1123)
    factory = old.shared.GaussianPolicy
    actor, copy, reference = (factory(92, 1, 64) for _ in range(3))
    copy.load_state_dict(actor.state_dict())
    with torch.no_grad():
        reference.mean_linear.bias.fill_(-1.)
    anchor = SourceAnchor(actor, reference, 2.35, 3., 123, batch_size=8)
    state = torch.full((8, 92), .1)
    for s in state.numpy():
        anchor.observe(s)
    value, n = anchor.prepare()
    assert value > 0 and n == 8
    mean, _ = actor(state)
    torch.tanh(mean).square().mean().backward()
    anchor.clear()
    raw = state.clone()
    from anchor import WAITING
    raw[:, WAITING] *= 2.35
    with torch.no_grad():
        target = torch.tanh(reference(raw)[0])
    mean, _ = copy(state)
    action = torch.tanh(mean)
    (action.square().mean() + 3. * torch.relu(action - target).square().mean()).backward()
    for a, b in zip(actor.parameters(), copy.parameters()):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            assert torch.allclose(a.grad, b.grad, atol=1e-6, rtol=1e-5)
    assert all(p.grad is None for p in reference.parameters())
    assert not bool(torch.relu(torch.tensor([-1.]) - target).square().sum())
    return dict(actor_gradient_matches_sum_loss=True, teacher_gradient_absent=True,
                shorter_action_has_no_penalty=True)


def native_check(folder):
    original = PRIOR / "preflight/source_clip/seed64001"
    before = torch.load(original / "policy.pt", map_location="cpu", weights_only=False)
    after = torch.load(folder / "policy.pt", map_location="cpu", weights_only=False)
    assert before["config"] == after["config"]
    assert all(torch.equal(before["actor"][key], after["actor"][key]) for key in before["actor"])
    history = json.loads((original / "train_summary.json").read_text())["history"]
    current = json.loads((folder / "train_summary.json").read_text())["history"]
    for a, b in zip(history, current):
        for key in ["cost", "exact_cost", "decisions", "transitions", "updates", "mean_holding_seconds"]:
            assert a[key] == b[key], (key, a[key], b[key])
    return dict(passed=True, final_actor_identical=True, two_episode_execution_identical=True)


def train(args):
    weight = P["arms"].get(args.arm, 0.)
    folder = HERE / ("preflight" if args.smoke else "runs") / args.arm / f"seed{args.seed}"
    if (folder / "ready.json").exists():
        return
    torch.set_num_threads(1)
    checks = gradient_check() if args.smoke and args.arm == "identity" else None
    cap = json.loads((old.THRESHOLDS / "inputs/envelope.json").read_text())["daily_caps"][0]
    agent = new_agent(args.seed, weight, cap)
    count, horizon, quota = (2, 3600, 40) if args.smoke else (P["episodes"], P["horizon_seconds"], P["updates_per_episode"])
    base.write(folder / "settings.json", dict(config=asdict(agent.config), protocol=P, source_cap=cap,
               anchor_weight=weight, teacher="frozen same-seed source raw policy", reward_change="none",
               checkpoint_selection="final episode only", anchor_scope="source-normal reservoir; actor gradient before native clipping"))
    history, started = [], time.monotonic()
    with tempfile.TemporaryDirectory(prefix="thesis-cs-source-anchor-") as tmp:
        with redirect_stdout(StringIO()):
            env = base.make_env(Path(tmp), horizon, "normal")
            for episode in range(count):
                regime = base.training_regime("global_coverage", episode, old.P["source_training_cycle"], args.smoke)
                if args.smoke and weight > 0:
                    regime = ["normal", "pulse3"][episode]
                env.schedule = base.source_schedule(regime, horizon)
                stats = old.InputStats("source_clip", cap)
                original = base.feature

                def feature(env, raw, arm, committed, pending):
                    state = stats.apply(original(env, raw, arm, committed, pending))
                    if regime == "normal":
                        agent.source_anchor.observe(state)
                    return state

                base.feature = feature
                try:
                    row = base.rollout(env, agent, "global_coverage", args.seed * 1000 + episode,
                                       train=True, updates=quota)
                finally:
                    base.feature = original
                assert stats.n == row["decisions"]
                row.update(episode=episode + 1, source_regime=regime, source_schedule=env.schedule,
                           input_stats=stats.summary(), anchor=agent.source_anchor.summary(),
                           wall_seconds=time.monotonic() - started,
                           peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.)
                history.append(row)
                base.write(folder / "progress.json", row)
                sys.__stdout__.write(json.dumps(dict(arm=args.arm, seed=args.seed, episode=episode + 1,
                    cost=row["cost"], updates=row["updates"], regime=regime, anchor=row["anchor"],
                    alpha=row["update_last"]["alpha"], wall_seconds=row["wall_seconds"])) + "\n")
                sys.__stdout__.flush()
    torch.save(dict(actor=agent.actor.state_dict(), config=asdict(agent.config), arm=args.arm,
               source_cap=cap, training_seed=args.seed, episodes=count, updates=agent.training_steps,
               objective="passenger", action_space="holding", source_anchor_weight=weight), folder / "policy.pt")
    base.write(folder / "train_summary.json", dict(history=history, training_steps=agent.training_steps,
               training_wall_seconds=time.monotonic() - started, anchor=agent.source_anchor.summary(),
               peak_rss_mb=history[-1]["peak_rss_mb"]))
    if args.arm == "identity":
        checks.update(native_check(folder))
        base.write(folder / "native_equivalence.json", checks)
    base.write(folder / "ready.json", dict(arm=args.arm, seed=args.seed, episodes=count,
               updates=agent.training_steps, anchor=agent.source_anchor.summary()))


def evaluate(args):
    out = HERE / "results" / args.domain / args.arm / f"seed{args.seed}" / f"day{args.day}_{args.scene}.json"
    if out.exists():
        return
    assert args.day in P[args.domain + "_days"] and args.scene in P[args.domain + "_scenes"]
    torch.set_num_threads(1)
    payload = torch.load(HERE / "runs" / args.arm / f"seed{args.seed}/policy.pt", map_location="cpu", weights_only=False)
    assert payload["episodes"] == 100 and payload["updates"] == 40000 and payload["training_seed"] == args.seed
    assert payload["source_anchor_weight"] == P["arms"][args.arm]
    actor = old.shared.GaussianPolicy(92, 1, payload["config"]["hidden_dim"])
    actor.load_state_dict(payload["actor"])
    actor.eval()

    class Policy:
        def select_action(self, state, deterministic):
            assert deterministic
            return actor.get_action(torch.as_tensor(state), deterministic=True)

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="thesis-cs-source-anchor-eval-") as tmp:
        with redirect_stdout(StringIO()):
            env = base.make_env(Path(tmp), P["horizon_seconds"], "normal")
            env.schedule = (base.source_schedule(args.scene, P["horizon_seconds"]) if args.domain == "source"
                            else old.P["scenes"][args.scene])
            with old.representation("source_clip", payload["source_cap"]) as stats:
                row = base.rollout(env, Policy(), "global_event", args.day)
    ref = json.loads((HERE / "inputs/reference.json").read_text())[f"{args.domain}/{args.seed}/{args.day}/{args.scene}"]
    for group, passengers in ref["passengers"].items():
        assert row["groups"][group]["passengers"] == passengers
    assert row["updates"] == 0 and row["timestamp_cost_identity_passed"] and stats.n == row["decisions"]
    row.update(arm=args.arm, seed=args.seed, scene=args.scene, evaluation_seed=args.day, domain=args.domain,
               input_stats=stats.summary(), cost_reduction_vs_unanchored=ref["cost"] - row["cost"],
               wall_seconds=time.monotonic() - started,
               peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.)
    base.write(out, row)
    print(json.dumps({k:row[k] for k in ["arm", "seed", "scene", "evaluation_seed", "domain", "cost",
                                       "mean_holding_seconds", "cost_reduction_vs_unanchored", "wall_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "preflight", "eval"])
    parser.add_argument("--arm", choices=list(P["arms"]) + ["identity"], required=True)
    parser.add_argument("--seed", type=int, choices=P["training_seeds"], default=64001)
    parser.add_argument("--domain", choices=["source", "target"])
    parser.add_argument("--day", type=int)
    parser.add_argument("--scene", choices=sorted(set(P["source_scenes"] + P["target_scenes"])))
    args = parser.parse_args()
    args.smoke = args.mode == "preflight"
    (evaluate if args.mode == "eval" else train)(args)
