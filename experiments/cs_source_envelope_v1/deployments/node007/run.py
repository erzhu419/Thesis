"""Independent source probes and frozen-policy closed-loop evaluations."""

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

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
PRIOR = HERE.parent / "cs_terminal_coverage_v1"
sys.path.insert(0, str(PRIOR))
spec = importlib.util.spec_from_file_location("frozen_rollout", PRIOR / "run.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
np, torch = base.np, base.torch
from csbapr.networks.policy import GaussianPolicy
from envelope import WAITING, compress


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def actor_for(seed):
    payload = torch.load(HERE.parent / P["actors"][str(seed)], map_location="cpu", weights_only=False)
    config = payload["config"]
    assert not config["use_nau_actor"] and not config.get("use_context", False)
    kind = config.get("actor_type")
    activation = kind.split("-")[0] if kind in ("tanh-mlp", "gelu-mlp", "softplus-mlp") else "relu"
    actor = GaussianPolicy(92, 1, config["hidden_dim"], activation=activation)
    actor.load_state_dict(payload["actor"])
    actor.eval()
    return actor


class Driver:
    def __init__(self, actor, variant, limits):
        self.actor, self.variant, self.limits = actor, variant, limits
        self.current_time = 0
        self.n = self.changed = self.above59 = 0
        self.factor_sum, self.max_ratio = 0., 0.
        self.bins = np.zeros((P["horizon_seconds"] // P["time_bin_seconds"], 5))

    def select_action(self, state, deterministic):
        assert deterministic
        if self.actor is None:
            return np.array([-1.], dtype=np.float32)
        transformed, ratio, factor = compress(state, self.variant, self.current_time, self.limits)
        assert np.isfinite(transformed).all()
        assert np.isclose(float(transformed[17]) * 10., np.sum(transformed[WAITING[1:]], dtype=np.float64), atol=2e-5)
        action = self.actor.get_action(torch.as_tensor(transformed), deterministic=True)
        hold = 30. * (float(np.clip(action[0], -1., 1.)) + 1.)
        self.n += 1
        self.changed += int(factor < 1.)
        self.above59 += int(hold >= 59.)
        self.factor_sum += factor
        self.max_ratio = max(self.max_ratio, ratio)
        phase = min(int(self.current_time // P["time_bin_seconds"]), len(self.bins) - 1)
        self.bins[phase] += [1., float(factor < 1.), factor, hold, float(hold >= 59.)]
        return action

    def summary(self):
        return dict(decisions=self.n, changed_decisions=self.changed,
                    changed_fraction=self.changed / self.n, mean_factor=self.factor_sum / self.n,
                    above59_fraction=self.above59 / self.n, maximum_ratio=self.max_ratio,
                    time_bins=[dict(start=i * P["time_bin_seconds"], decisions=int(v[0]),
                        changed_fraction=v[1] / v[0], mean_factor=v[2] / v[0],
                        holding_seconds=v[3] / v[0], above59_fraction=v[4] / v[0])
                        for i, v in enumerate(self.bins) if v[0]])


def run(args):
    probe = args.mode == "probe"
    out = (HERE / "probes" / f"{args.scene}_day{args.day}.json" if probe else
           HERE / ("smoke" if args.smoke else "results") / args.variant / f"seed{args.seed}" / f"day{args.day}_{args.scene}.json")
    if out.exists():
        print("Already complete: " + str(out), flush=True)
        return
    torch.set_num_threads(1)
    started = time.monotonic()
    limits = None if probe or args.variant == "identity" else json.loads((HERE / "inputs/envelope.json").read_text())
    driver = Driver(None if probe else actor_for(args.seed), args.variant, limits)
    peaks = np.zeros((P["horizon_seconds"] // P["time_bin_seconds"], len(WAITING)))
    seen = np.zeros(len(peaks), dtype=int)
    original_feature = base.feature

    def capture(env, raw, arm, committed, pending):
        state = original_feature(env, raw, arm, committed, pending)
        driver.current_time = env.current_time
        if probe:
            phase = min(int(env.current_time // P["time_bin_seconds"]), len(peaks) - 1)
            peaks[phase] = np.maximum(peaks[phase], state[WAITING])
            seen[phase] += 1
        return state

    base.feature = capture
    try:
        with tempfile.TemporaryDirectory(prefix="thesis-cs-envelope-") as tmp:
            with redirect_stdout(StringIO()):
                env = base.make_env(Path(tmp), P["horizon_seconds"], "normal")
                env.schedule = (P["source_scenes"] if probe else P["scenes"])[args.scene]
                row = base.rollout(env, driver, "global_event", args.day)
    finally:
        base.feature = original_feature
    assert row["updates"] == 0 and row["timestamp_cost_identity_passed"]
    result = dict(scene=args.scene, day=args.day, rollout=row, learner_updates=0,
        wall_seconds=time.monotonic() - started, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.)
    if probe:
        assert (seen > 0).all()
        result.update(peaks=peaks.tolist(), seen=seen.tolist(), driver="fixed0")
    else:
        original = json.loads((HERE / "inputs/reference.json").read_text())[f"{args.seed}/{args.day}/{args.scene}"]
        for group in ["all", "baseline", "extra", "baseline_during"]:
            assert row["groups"][group]["passengers"] == original["groups"][group]["passengers"]
        unchanged = driver.changed == 0
        if unchanged:
            for key in ["cost", "exact_cost", "decisions", "mean_holding_seconds"]:
                assert np.isclose(row[key], original[key], rtol=0., atol=1e-5), (key, row[key], original[key])
        result.update(seed=args.seed, variant=args.variant, adaptation=driver.summary(),
                      unchanged_rollout_matched=unchanged, cost_reduction_seconds=original["cost"] - row["cost"])
    write(out, result)
    print(json.dumps(dict(mode=args.mode, scene=args.scene, day=args.day, seed=args.seed,
        variant=args.variant, cost=row["cost"], holding=row["mean_holding_seconds"],
        changed_fraction=None if probe else driver.changed / driver.n,
        wall_seconds=result["wall_seconds"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["probe", "eval"])
    parser.add_argument("--scene", required=True, choices=sorted(set(P["source_scenes"]) | set(P["scenes"])))
    parser.add_argument("--day", type=int, required=True)
    parser.add_argument("--variant", choices=["identity"] + P["variants"], default="identity")
    parser.add_argument("--seed", type=int, choices=[64001, 64002], default=64002)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
