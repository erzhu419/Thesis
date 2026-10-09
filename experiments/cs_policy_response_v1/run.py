"""Frozen-actor responses on common observations; no learner updates."""

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

GROUPS = {
    "station_waiting": [7],
    "network_waiting": [17] + list(range(19, 91, 6)),
    "all_waiting": [7, 17] + list(range(19, 91, 6)),
    "onboard": [15, 18] + list(range(20, 91, 6)),
    "dwell": [9] + list(range(22, 91, 6)),
    "same_tick_actions": [91] + list(range(23, 91, 6)) + list(range(24, 91, 6)),
    "headway": [5, 6, 8, 11, 12, 13],
    "clock": [10, 16],
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_actors():
    actors, payloads = {}, {}
    for name, relative in P["actors"].items():
        payload = torch.load(HERE.parent / relative, map_location="cpu", weights_only=False)
        config = payload["config"]
        assert not config["use_nau_actor"] and not config.get("use_context", False)
        kind = config.get("actor_type")
        activation = kind.split("-")[0] if kind in ("tanh-mlp", "gelu-mlp", "softplus-mlp") else "relu"
        actor = GaussianPolicy(92, 1, config["hidden_dim"], activation=activation)
        actor.load_state_dict(payload["actor"])
        actor.eval()
        actors[name], payloads[name] = actor, payload
    return actors, payloads


def response(actor, states):
    with torch.no_grad():
        logit, _ = actor(torch.as_tensor(states, dtype=torch.float32))
        holding = 30. * (torch.tanh(logit[:, 0]) + 1.)
    return logit[:, 0].numpy(), holding.numpy()


def moments(values):
    return dict(n=len(values), sum=float(np.sum(values, dtype=np.float64)),
                sum_squares=float(np.sum(np.square(values.astype(np.float64)))),
                minimum=float(np.min(values)), maximum=float(np.max(values)))


def action_summary(logit, holding):
    return dict(logit=moments(logit), holding=moments(holding),
                above55=int(np.count_nonzero(holding >= 55.)),
                above59=int(np.count_nonzero(holding >= 59.)),
                below1=int(np.count_nonzero(holding <= 1.)))


def analyze_actor(actor, states, times, sampled):
    logit, holding = response(actor, states)
    result = action_summary(logit, holding)
    result["time_bins"] = []
    for start in range(0, 18000, P["time_bin_seconds"]):
        mask = (times >= start) & (times < start + P["time_bin_seconds"])
        if mask.any():
            result["time_bins"].append(dict(start=start, **action_summary(logit[mask], holding[mask])))
    x = torch.tensor(sampled, requires_grad=True)
    mean, _ = actor(x)
    gradient = torch.autograd.grad(mean.sum(), x)[0].detach().numpy()
    result["feature_logit_sensitivity"] = np.mean(gradient * sampled, axis=0).tolist()
    result["feature_absolute_logit_sensitivity"] = np.mean(np.abs(gradient * sampled), axis=0).tolist()
    initial_logit, initial_holding = response(actor, sampled)
    result["sampled_original"] = action_summary(initial_logit, initial_holding)
    result["group_probes"] = {}
    for group, indices in GROUPS.items():
        # These off-trajectory inputs diagnose the actor, not a physically executed policy.
        probes = {}
        for factor in [0., .3]:
            changed = sampled.copy()
            changed[:, indices] *= factor
            new_logit, new_holding = response(actor, changed)
            probes[str(factor)] = dict(**action_summary(new_logit, new_holding),
                mean_logit_change=float(np.mean(new_logit - initial_logit)),
                mean_holding_change=float(np.mean(new_holding - initial_holding)))
        result["group_probes"][group] = probes
    return result


class Driver:
    def __init__(self, actor):
        self.actor = actor

    def select_action(self, state, deterministic):
        assert deterministic
        if self.actor is None:
            return np.array([-1.], dtype=np.float32)
        return self.actor.get_action(torch.as_tensor(state), deterministic=True)


def run(args):
    out = HERE / ("smoke" if args.smoke else "results") / args.driver / f"{args.scene}_day{args.day}.json"
    if out.exists():
        print("Already complete: " + str(out), flush=True)
        return
    torch.set_num_threads(1)
    started = time.monotonic()
    actors, payloads = load_actors()
    states, times = [], []
    original_feature = base.feature

    def record(env, raw, arm, committed, pending):
        state = original_feature(env, raw, arm, committed, pending)
        states.append(state.copy())
        times.append(env.current_time)
        return state

    base.feature = record
    try:
        with tempfile.TemporaryDirectory(prefix="thesis-cs-response-") as tmp:
            with redirect_stdout(StringIO()):
                env = base.make_env(Path(tmp), 18000, "normal")
                env.schedule = P["scenes"][args.scene]
                row = base.rollout(env, Driver(actors.get(args.driver)), "global_event", args.day)
    finally:
        base.feature = original_feature
    states = np.asarray(states, dtype=np.float32)
    times = np.asarray(times)
    assert len(states) == row["decisions"] and row["updates"] == 0
    # Check that observation instrumentation preserves the completed closed-loop result.
    checked = args.scene != "static3" and args.driver != "fixed0"
    if checked:
        seed = int(args.driver.rsplit("_", 1)[1])
        references = json.loads((HERE / "reference.json").read_text())
        old = references[f"{seed}/{args.day}/{args.scene}"]
        for key in ["cost", "mean_holding_seconds", "decisions", "exact_cost"]:
            assert np.isclose(row[key], old[key], rtol=0., atol=1e-5), (key, row[key], old[key])
    indices = np.unique(np.linspace(0, len(states) - 1, min(len(states), P["sensitivity_sample_limit"]), dtype=int))
    sampled = states[indices]
    native_checked = False
    if args.smoke:
        for name, payload in payloads.items():
            config = base.CSBAPRConfig(**payload["config"])
            config.device = "cpu"
            native = base.CSBAPRAgent(92, 1, config)
            native.actor.load_state_dict(payload["actor"])
            native.actor.eval()
            for state in sampled[::max(1, len(sampled) // 20)]:
                assert np.array_equal(native.select_action(state, True),
                                      Driver(actors[name]).select_action(state, True))
            _, batched = response(actors[name], sampled)
            individual = np.array([30. * (float(native.select_action(state, True)[0]) + 1.) for state in sampled])
            assert np.allclose(batched, individual, rtol=0., atol=2e-4)
        native_checked = True
    responses = {name: analyze_actor(actor, states, times, sampled) for name, actor in actors.items()}
    feature_stats = dict(n=len(states), sum=states.sum(axis=0, dtype=np.float64).tolist(),
        minimum=states.min(axis=0).tolist(), maximum=states.max(axis=0).tolist(),
        quantiles=np.quantile(states, [.05, .5, .95], axis=0).tolist())
    result = dict(driver=args.driver, scene=args.scene, day=args.day, rollout=row,
        sampled_states=len(sampled), features=feature_stats, responses=responses,
        same_state_comparison=True, previous_rollout_matched=checked,
        native_actor_equivalence_checked=native_checked, learner_updates=0,
        wall_seconds=time.monotonic() - started,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.)
    write(out, result)
    print(json.dumps(dict(driver=args.driver, scene=args.scene, day=args.day, cost=row["cost"],
        holding=row["mean_holding_seconds"], wall_seconds=result["wall_seconds"],
        native_actor_equivalence_checked=native_checked, previous_rollout_matched=checked)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", choices=P["drivers"], required=True)
    parser.add_argument("--scene", choices=list(P["scenes"]), required=True)
    parser.add_argument("--day", type=int, default=78101)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
