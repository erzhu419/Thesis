"""Frozen source-range validation and same-state action comparison."""

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

from response import Response

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
PRIOR = HERE.parent / "cs_input_training_v1"
sys.path.insert(0, str(PRIOR))
spec = importlib.util.spec_from_file_location("source_validation_runtime", PRIOR / "run.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
base, np, torch = old.base, old.np, old.torch


def models(seed):
    raw = torch.load(old.REFERENCE / "runs/global_coverage" / f"seed{seed}/policy.pt",
                     map_location="cpu", weights_only=False)
    normalized = torch.load(PRIOR / "runs/source_clip" / f"seed{seed}/policy.pt",
                            map_location="cpu", weights_only=False)
    assert raw["config"] == normalized["config"]
    for item in [raw, normalized]:
        assert item["training_seed"] == seed and item["episodes"] == 100 and item["updates"] == 40000
        assert item["objective"] == "passenger" and item["action_space"] == "holding"
    assert raw["arm"] == "global_coverage" and normalized["arm"] == "source_clip"
    config = raw["config"]
    kind = config.get("actor_type")
    activation = kind.split("-")[0] if kind in ("tanh-mlp", "gelu-mlp", "softplus-mlp") else "relu"
    actors = []
    for item in [raw, normalized]:
        actor = old.shared.GaussianPolicy(92, 1, config["hidden_dim"], activation=activation)
        actor.load_state_dict(item["actor"])
        actor.eval()
        actors.append(actor)
    return actors, normalized["source_cap"]


class Driver:
    def __init__(self, arm, seed):
        self.arm, self.current_time = arm, 0.
        self.response = Response(P["time_bin_seconds"])
        if arm != "fixed0":
            self.actors, cap = models(seed)
            self.input_stats = old.InputStats("source_clip", cap)

    def select_action(self, state, deterministic):
        assert deterministic
        if self.arm == "fixed0":
            return np.array([-1.], dtype=np.float32)
        normalized = self.input_stats.apply(state)
        actions = [actor.get_action(torch.as_tensor(values), deterministic=True)
                   for actor, values in zip(self.actors, [state, normalized])]
        holds = [30. * (float(np.clip(a[0], -1., 1.)) + 1.) for a in actions]
        chosen = 0 if self.arm == "old_raw" else 1
        self.response.record(self.current_time, float(state[17]) / self.input_stats.cap,
                             float(state[15]), *holds, holds[chosen])
        return actions[chosen]


def run(args):
    assert (args.arm == "fixed0") == (args.seed == 0)
    if not args.smoke:
        assert args.day in P["validation_days"]
        excluded = {seed * 1000 + episode for seed in P["training_seeds"]
                    for episode in range(old.P["episodes"])}
        excluded.update(old.P["evaluation_days"])
        excluded.update(json.loads((old.THRESHOLDS / "protocol.json").read_text())["source_probe_days"])
        assert not excluded.intersection(P["validation_days"])
    out = HERE / ("smoke" if args.smoke else "results") / args.arm / f"seed{args.seed}" / f"day{args.day}_{args.scene}.json"
    if out.exists():
        print("Already complete: " + str(out), flush=True)
        return
    assert P["horizon_seconds"] == old.P["horizon_seconds"]
    torch.set_num_threads(1)
    driver = Driver(args.arm, args.seed)
    original_feature = base.feature

    def capture(env, raw, arm, committed, pending):
        driver.current_time = env.current_time
        return original_feature(env, raw, arm, committed, pending)

    started = time.monotonic()
    base.feature = capture
    try:
        with tempfile.TemporaryDirectory(prefix="thesis-cs-source-validation-") as tmp:
            with redirect_stdout(StringIO()):
                env = base.make_env(Path(tmp), P["horizon_seconds"], "normal")
                env.schedule = base.source_schedule(args.scene, P["horizon_seconds"])
                row = base.rollout(env, driver, "global_event", args.day)
    finally:
        base.feature = original_feature
    assert row["updates"] == 0 and row["timestamp_cost_identity_passed"]
    if args.arm != "fixed0":
        response = driver.response.summary()
        total = response["total"]
        assert total["decisions"] == row["decisions"] == driver.input_stats.n
        assert np.isclose(total["actual_hold_sum"] / total["decisions"], row["mean_holding_seconds"], atol=1e-6)
        row.update(response=response, input_stats=driver.input_stats.summary(), source_cap=driver.input_stats.cap)
    native_matched = False
    if args.smoke:
        assert args.day == 78101 and args.scene == "normal"
        ref = json.loads((HERE / "inputs/smoke_references.json").read_text())[args.arm]
        for key in ["cost", "exact_cost", "decisions", "mean_holding_seconds"]:
            assert np.isclose(row[key], ref[key], rtol=0., atol=1e-5), (key, row[key], ref[key])
        native_matched = True
    row.update(arm=args.arm, seed=args.seed, scene=args.scene, evaluation_seed=args.day,
               source_schedule=env.schedule, frozen_actor=True, native_rollout_matched=native_matched,
               wall_seconds=time.monotonic() - started,
               peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.)
    base.write(out, row)
    print(json.dumps({k:row[k] for k in ["arm", "seed", "scene", "evaluation_seed", "cost",
                                      "mean_holding_seconds", "native_rollout_matched", "wall_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["old_raw", "source_clip", "fixed0"], required=True)
    parser.add_argument("--seed", type=int, choices=[0] + P["training_seeds"], required=True)
    parser.add_argument("--day", type=int, required=True)
    parser.add_argument("--scene", choices=P["scenes"], required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
