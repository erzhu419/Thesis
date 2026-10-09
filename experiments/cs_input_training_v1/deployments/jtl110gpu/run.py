"""Matched source-only training and lightweight distributed evaluation."""

import argparse
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict
import importlib.util
from io import StringIO
import json
from pathlib import Path
import resource
import sys
import tempfile
import time

from representation import WAITING, transform

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
REFERENCE = HERE.parent / P["reference_campaign"]
THRESHOLDS = HERE.parent / P["threshold_campaign"]
sys.path.insert(0, str(THRESHOLDS))
spec = importlib.util.spec_from_file_location("frozen_input_runtime", THRESHOLDS / "run.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
base, np, torch = shared.base, shared.np, shared.torch
for key in ["episodes", "updates_per_episode", "horizon_seconds", "buffer_size", "reward_scale", "source_training_cycle"]:
    assert P[key] == base.P[key], key


class InputStats:
    def __init__(self, arm, cap):
        self.arm, self.cap = arm, cap
        self.n = self.changed = self.exceeded = 0
        self.max_raw = self.max_transformed = self.factor_sum = 0.

    def apply(self, state):
        changed, factor = transform(state, self.arm, self.cap)
        assert np.isfinite(changed).all()
        assert np.isclose(float(changed[17]) * 10., np.sum(changed[WAITING[1:]], dtype=np.float64), atol=2e-5)
        if self.arm == "source_clip":
            assert 0 <= changed[17] <= 1.000001
        self.n += 1
        self.changed += int(np.any(changed != state))
        self.exceeded += int(state[17] > self.cap)
        self.max_raw = max(self.max_raw, float(state[17]))
        self.max_transformed = max(self.max_transformed, float(changed[17]))
        self.factor_sum += factor
        return changed

    def summary(self):
        return dict(feature_calls=self.n, changed_fraction=self.changed / self.n,
            source_cap_exceeded_fraction=self.exceeded / self.n, raw_total_maximum=self.max_raw,
            transformed_total_maximum=self.max_transformed, mean_factor=self.factor_sum / self.n)


@contextmanager
def representation(arm, cap):
    stats = InputStats(arm, cap)
    original = base.feature

    def feature(env, raw, rollout_arm, committed, pending):
        return stats.apply(original(env, raw, rollout_arm, committed, pending))

    base.feature = feature
    try:
        yield stats
    finally:
        base.feature = original


def check_identity(out):
    original = REFERENCE / "preflight/global_coverage/seed64001"
    before = torch.load(original / "policy.pt", map_location="cpu", weights_only=False)
    after = torch.load(out / "policy.pt", map_location="cpu", weights_only=False)
    assert before["config"] == after["config"]
    assert before["actor"].keys() == after["actor"].keys()
    assert all(torch.equal(before["actor"][key], after["actor"][key]) for key in before["actor"]), "Native short-training actor changed"
    old = json.loads((original / "train_summary.json").read_text())["history"]
    new = json.loads((out / "train_summary.json").read_text())["history"]
    for a, b in zip(old, new):
        for key in ["cost", "exact_cost", "decisions", "transitions", "updates", "mean_holding_seconds"]:
            assert a[key] == b[key], (key, a[key], b[key])
    base.write(out / "native_equivalence.json", dict(passed=True, final_actor_identical=True,
        short_training_costs_and_actions_identical=True, episodes=len(new)))


def train(args):
    out = HERE / ("preflight" if args.smoke else "runs") / args.arm / f"seed{args.seed}"
    if (out / "ready.json").exists():
        print("Already complete: " + str(out), flush=True)
        return
    torch.set_num_threads(1)
    limits = json.loads((THRESHOLDS / "inputs/envelope.json").read_text())
    cap = limits["daily_caps"][0]
    agent = base.new_agent("global_coverage", args.seed)
    reference_config = json.loads((REFERENCE / "runs/global_coverage" / f"seed{args.seed}/settings.json").read_text())["config"]
    assert asdict(agent.config) == reference_config
    count, horizon, quota = (2, 3600, 40) if args.smoke else (P["episodes"], P["horizon_seconds"], P["updates_per_episode"])
    base.write(out / "settings.json", dict(protocol=P, arm=args.arm, config=asdict(agent.config),
        episodes=count, horizon_seconds=horizon, updates_per_episode=quota, source_cap=cap,
        source_probe_days=limits["source_days"], learner_change="none", checkpoint_selection="final episode only",
        input_transform_scope="actor state and both replay states; train and evaluation", smoke=args.smoke))
    history, started = [], time.monotonic()
    with tempfile.TemporaryDirectory(prefix="thesis-cs-input-train-") as tmp:
        with redirect_stdout(StringIO()):
            env = base.make_env(Path(tmp), horizon, "normal")
            for episode in range(count):
                regime = base.training_regime("global_coverage", episode, P["source_training_cycle"], args.smoke)
                env.schedule = base.source_schedule(regime, horizon)
                with representation(args.arm, cap) as stats:
                    row = base.rollout(env, agent, "global_coverage", args.seed * 1000 + episode, train=True, updates=quota)
                assert stats.n == row["decisions"]
                row.update(episode=episode + 1, source_regime=regime, source_schedule=env.schedule,
                    input_stats=stats.summary(), wall_seconds=time.monotonic() - started,
                    peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.)
                history.append(row)
                base.write(out / "progress.json", row)
                sys.__stdout__.write(json.dumps(dict(arm=args.arm, seed=args.seed, episode=episode + 1,
                    cost=row["cost"], updates=row["updates"], regime=regime, input_stats=row["input_stats"],
                    q_mean=row["update_last"]["q_mean"], alpha=row["update_last"]["alpha"],
                    wall_seconds=row["wall_seconds"])) + "\n")
                sys.__stdout__.flush()
    torch.save(dict(actor=agent.actor.state_dict(), config=asdict(agent.config), arm=args.arm,
        source_cap=cap, training_seed=args.seed, episodes=count, updates=agent.training_steps,
        objective="passenger", action_space="holding"), out / "policy.pt")
    base.write(out / "train_summary.json", dict(history=history, training_steps=agent.training_steps,
        training_wall_seconds=time.monotonic() - started, peak_rss_mb=history[-1]["peak_rss_mb"]))
    if args.arm == "identity":
        check_identity(out)
    base.write(out / "ready.json", dict(arm=args.arm, seed=args.seed, episodes=count,
        updates=agent.training_steps, train_eval_representation_consistent=True))


def evaluate(args):
    out = HERE / ("preflight_results" if args.smoke else "results") / args.arm / f"seed{args.seed}" / f"day{args.day}_{args.scene}.json"
    if out.exists():
        return
    torch.set_num_threads(1)
    folder = HERE / ("preflight" if args.smoke else "runs") / args.arm / f"seed{args.seed}"
    payload = torch.load(folder / "policy.pt", map_location="cpu", weights_only=False)
    assert payload["arm"] == args.arm and payload["training_seed"] == args.seed
    config = payload["config"]
    kind = config.get("actor_type")
    activation = kind.split("-")[0] if kind in ("tanh-mlp", "gelu-mlp", "softplus-mlp") else "relu"
    actor = shared.GaussianPolicy(92, 1, config["hidden_dim"], activation=activation)
    actor.load_state_dict(payload["actor"])
    actor.eval()

    class Policy:
        def select_action(self, state, deterministic):
            assert deterministic
            return actor.get_action(torch.as_tensor(state), deterministic=True)

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="thesis-cs-input-eval-") as tmp:
        with redirect_stdout(StringIO()):
            env = base.make_env(Path(tmp), P["horizon_seconds"], "normal")
            env.schedule = P["scenes"][args.scene]
            with representation(args.arm, payload["source_cap"]) as stats:
                row = base.rollout(env, Policy(), "global_event", args.day)
    assert stats.n == row["decisions"] and row["updates"] == 0
    reference = json.loads((THRESHOLDS / "inputs/reference.json").read_text())[f"{args.seed}/{args.day}/{args.scene}"]
    for group in ["all", "baseline", "extra", "baseline_during"]:
        assert row["groups"][group]["passengers"] == reference["groups"][group]["passengers"]
    row.update(arm=args.arm, seed=args.seed, scene=args.scene, evaluation_seed=args.day,
        input_stats=stats.summary(), policy_phase="preflight" if args.smoke else "formal",
        wall_seconds=time.monotonic() - started, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.)
    base.write(out, row)
    print(json.dumps({k: row[k] for k in ["arm", "seed", "scene", "cost", "input_stats", "wall_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval", "preflight"])
    parser.add_argument("--arm", required=True, choices=P["arms"] + ["identity"])
    parser.add_argument("--seed", type=int, choices=P["training_seeds"], default=64001)
    parser.add_argument("--day", type=int, choices=P["evaluation_days"], default=78101)
    parser.add_argument("--scene", choices=P["scenes"], default="normal")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.mode == "preflight":
        args.smoke = True
        train(args)
        for args.scene in P["scenes"]:
            evaluate(args)
    else:
        (train if args.mode == "train" else evaluate)(args)
