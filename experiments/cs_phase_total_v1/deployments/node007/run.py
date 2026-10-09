"""Frozen-policy evaluation with a time-bin global waiting bound."""

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

from phase_total import WAITING, compress_total

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
PRIOR = HERE.parent / P["frozen_campaign"]
sys.path.insert(0, str(PRIOR))
spec = importlib.util.spec_from_file_location("source_envelope_frozen", PRIOR / "run.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
base, np, torch = shared.base, shared.np, shared.torch
assert P["horizon_seconds"] == shared.P["horizon_seconds"]
assert P["time_bin_seconds"] == shared.P["time_bin_seconds"]


class Driver(shared.Driver):
    def select_action(self, state, deterministic):
        assert deterministic
        transformed, ratio, factor = compress_total(state, self.current_time, self.limits)
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


def run(args):
    out = HERE / ("smoke" if args.smoke else "results") / f"seed{args.seed}" / f"day{args.day}_{args.scene}.json"
    if out.exists():
        print("Already complete: " + str(out), flush=True)
        return
    torch.set_num_threads(1)
    started = time.monotonic()
    limits = json.loads((PRIOR / "inputs/envelope.json").read_text())
    driver = Driver(shared.actor_for(args.seed), P["variant"], limits)
    original_feature = base.feature

    def capture(env, raw, arm, committed, pending):
        state = original_feature(env, raw, arm, committed, pending)
        driver.current_time = env.current_time
        return state

    base.feature = capture
    try:
        with tempfile.TemporaryDirectory(prefix="thesis-cs-phase-total-") as tmp:
            with redirect_stdout(StringIO()):
                env = base.make_env(Path(tmp), P["horizon_seconds"], "normal")
                env.schedule = P["scenes"][args.scene]
                row = base.rollout(env, driver, "global_event", args.day)
    finally:
        base.feature = original_feature
    assert row["updates"] == 0 and row["timestamp_cost_identity_passed"]
    original = json.loads((PRIOR / "inputs/reference.json").read_text())[f"{args.seed}/{args.day}/{args.scene}"]
    for group in ["all", "baseline", "extra", "baseline_during"]:
        assert row["groups"][group]["passengers"] == original["groups"][group]["passengers"]
    unchanged = driver.changed == 0
    if unchanged:
        for key in ["cost", "exact_cost", "decisions", "mean_holding_seconds"]:
            assert np.isclose(row[key], original[key], rtol=0., atol=1e-5), (key, row[key], original[key])
    result = dict(seed=args.seed, day=args.day, scene=args.scene, variant=P["variant"], rollout=row,
        learner_updates=0, adaptation=driver.summary(), unchanged_rollout_matched=unchanged,
        cost_reduction_seconds=original["cost"] - row["cost"], wall_seconds=time.monotonic() - started,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.)
    shared.write(out, result)
    print(json.dumps(dict(seed=args.seed, day=args.day, scene=args.scene, cost=row["cost"],
        holding=row["mean_holding_seconds"], changed_fraction=driver.changed / driver.n,
        wall_seconds=result["wall_seconds"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, choices=P["scenes"])
    parser.add_argument("--day", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int, choices=P["seeds"])
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
