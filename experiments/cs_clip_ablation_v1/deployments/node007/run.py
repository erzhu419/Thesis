"""Remove test-time clipping while retaining the same learned actor and scale."""

import argparse
from contextlib import redirect_stdout
import importlib.util
from io import StringIO
import json
from pathlib import Path
import resource
import sys
import tempfile
import time

from linear import WAITING, normalize_linear

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
PRIOR = HERE.parent / P["policy_campaign"]
sys.path.insert(0, str(PRIOR))
spec = importlib.util.spec_from_file_location("frozen_clip_learning", PRIOR / "run.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
base, np, torch = old.base, old.np, old.torch
assert P["horizon_seconds"] == old.P["horizon_seconds"] and P["scenes"] == old.P["scenes"]


class LinearStats(old.InputStats):
    def apply(self, state):
        changed, factor = normalize_linear(state, self.cap)
        assert np.isfinite(changed).all()
        assert np.isclose(float(changed[17]) * 10., np.sum(changed[WAITING[1:]], dtype=np.float64), atol=2e-5)
        self.n += 1
        self.changed += int(np.any(changed != state))
        self.exceeded += int(state[17] > self.cap)
        self.max_raw = max(self.max_raw, float(state[17]))
        self.max_transformed = max(self.max_transformed, float(changed[17]))
        self.factor_sum += factor
        return changed


def run(args):
    out = HERE / ("smoke" if args.smoke else "results") / args.variant / f"seed{args.seed}" / f"day{args.day}_{args.scene}.json"
    if out.exists():
        print("Already complete: " + str(out), flush=True)
        return
    torch.set_num_threads(1)
    payload = torch.load(PRIOR / "runs" / P["policy_arm"] / f"seed{args.seed}/policy.pt", map_location="cpu", weights_only=False)
    assert payload["arm"] == P["policy_arm"] and payload["training_seed"] == args.seed
    assert payload["episodes"] == 100 and payload["updates"] == 40000
    assert payload["objective"] == "passenger" and payload["action_space"] == "holding"
    config = payload["config"]
    kind = config.get("actor_type")
    activation = kind.split("-")[0] if kind in ("tanh-mlp", "gelu-mlp", "softplus-mlp") else "relu"
    actor = old.shared.GaussianPolicy(92, 1, config["hidden_dim"], activation=activation)
    actor.load_state_dict(payload["actor"])
    actor.eval()

    class Policy:
        def select_action(self, state, deterministic):
            assert deterministic
            return actor.get_action(torch.as_tensor(state), deterministic=True)

    reference = json.loads((HERE / "inputs/reference.json").read_text())[f"{args.seed}/{args.day}/{args.scene}"]
    stats = (old.InputStats("source_clip", payload["source_cap"]) if args.variant == "hard_clip" else LinearStats("linear_total", payload["source_cap"]))
    original_feature = base.feature

    def capture(env, raw, arm, committed, pending):
        return stats.apply(original_feature(env, raw, arm, committed, pending))

    started = time.monotonic()
    base.feature = capture
    try:
        with tempfile.TemporaryDirectory(prefix="thesis-cs-clip-ablation-") as tmp:
            with redirect_stdout(StringIO()):
                env = base.make_env(Path(tmp), P["horizon_seconds"], "normal")
                env.schedule = P["scenes"][args.scene]
                row = base.rollout(env, Policy(), "global_event", args.day)
    finally:
        base.feature = original_feature
    assert row["updates"] == 0 and row["timestamp_cost_identity_passed"] and stats.n == row["decisions"]
    for group in ["all", "baseline", "extra", "baseline_during"]:
        assert row["groups"][group]["passengers"] == reference["groups"][group]["passengers"]
    expected_same = args.variant == "hard_clip" or stats.exceeded == 0
    if expected_same:
        for key in ["cost", "exact_cost", "decisions", "mean_holding_seconds"]:
            assert np.isclose(row[key], reference[key], rtol=0., atol=1e-5), (key, row[key], reference[key])
    row.update(variant=args.variant, seed=args.seed, scene=args.scene, evaluation_seed=args.day,
        input_stats=stats.summary(), source_cap=payload["source_cap"], frozen_actor=True,
        reference_rollout_matched=expected_same, reduction_vs_clip=reference["cost"] - row["cost"],
        wall_seconds=time.monotonic() - started, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.)
    base.write(out, row)
    print(json.dumps({k: row[k] for k in ["variant", "seed", "scene", "cost", "mean_holding_seconds", "reference_rollout_matched", "input_stats", "wall_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["hard_clip", "linear_total"], default="linear_total")
    parser.add_argument("--seed", type=int, choices=P["training_seeds"], required=True)
    parser.add_argument("--day", type=int, choices=P["evaluation_days"], required=True)
    parser.add_argument("--scene", choices=P["scenes"], required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
