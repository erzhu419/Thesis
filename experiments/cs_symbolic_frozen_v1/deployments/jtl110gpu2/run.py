"""Frozen controller interventions using the preceding campaign's exact runtime."""

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
PRIOR = HERE.parent / P["reference_campaign"]
sys.path.insert(0, str(PRIOR))
spec = importlib.util.spec_from_file_location("frozen_symbolic_runtime", PRIOR / "run.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
torch, np, base = runtime.torch, runtime.old.np, runtime.base


def load_actor(arm, seed):
    payload = torch.load(PRIOR / "runs" / arm / f"seed{seed}/policy.pt",
                         map_location="cpu", weights_only=False)
    assert payload["episodes"] == 100 and payload["updates"] == 40000
    assert payload["training_seed"] == seed
    assert payload["source_anchor_weight"] == runtime.P["arms"][arm]
    actor = runtime.CenteredPolicy(92, 1, payload["config"]["hidden_dim"]).eval()
    actor.load_state_dict(payload["actor"])
    actor.requires_grad_(False)
    return actor, payload["source_cap"]


def contracts():
    torch.set_num_threads(1)
    records = []
    states = torch.full((12, 92), .2)
    states[:, 0] = torch.arange(12) / 12
    states[:, 5], states[:, 6], states[:, 8] = .3, .7, .5
    caps = []
    for seed in P["training_seeds"]:
        ref = runtime.StageReference(seed).eval().requires_grad_(False)
        with torch.no_grad():
            actions = ref(states)
            assert torch.equal(actions, ref(states[:, :15]))
            assert bool(torch.isfinite(actions).all()) and bool((actions.abs() <= 1).all())
        for arm in P["arms"]:
            actor, cap = load_actor(arm, seed)
            caps.append(cap)
            with torch.no_grad():
                projected = torch.tanh(actor(runtime.equilibrium_states(states))[0])
                constant = torch.tanh(actor.equilibrium_mean).expand_as(projected)
                assert torch.allclose(projected, constant, atol=1e-6, rtol=0)
            records.append(dict(arm=arm, seed=seed, bias_holding_seconds=float(30 * (constant[0, 0] + 1))))
    assert len(set(caps)) == 1
    assert (base.RUNTIME / "offline-sumo/env/calibrated_env/config.json").is_file()
    assert (base.RUNTIME / "offline-sumo/env/calibrated_env/data").is_dir()
    result = dict(passed=True, records=records, source_cap=caps[0], reference_15_and_92_identical=True)
    base.write(HERE / "contracts.json", result)
    print(json.dumps(result), flush=True)


def evaluate(args):
    out = HERE / ("smoke" if args.mode == "smoke" else "results") / args.domain / args.variant / args.arm / f"seed{args.seed}" / f"day{args.day}_{args.scene}.json"
    if out.exists():
        return
    assert args.day in P[args.domain + "_days"] and args.scene in P[args.domain + "_scenes"]
    if args.variant == "sindy":
        assert args.arm == "reference"
        controller = runtime.StageReference(args.seed).eval().requires_grad_(False)
        _, cap = load_actor("symbolic0", args.seed)
        action = lambda state: controller(torch.as_tensor(state)[None])[0].numpy()
    else:
        assert args.arm in P["arms"]
        actor, cap = load_actor(args.arm, args.seed)
        bias = torch.tanh(actor.equilibrium_mean).detach().numpy()
        action = (lambda state: bias.copy()) if args.variant == "bias" else (
            lambda state: actor.get_action(torch.as_tensor(state), deterministic=True))

    class Policy:
        def select_action(self, state, deterministic):
            assert deterministic
            return action(state)

    started = time.monotonic()
    horizon = 7200 if args.mode == "smoke" and args.variant != "native" else P["horizon_seconds"]
    with tempfile.TemporaryDirectory(prefix="thesis-cs-symbolic-frozen-") as tmp:
        with redirect_stdout(StringIO()):
            env = base.make_env(Path(tmp), horizon, "normal")
            env.schedule = (base.source_schedule(args.scene, horizon) if args.domain == "source"
                            else runtime.old.old.P["scenes"][args.scene])
            with runtime.old.old.representation("source_clip", cap) as stats:
                row = base.rollout(env, Policy(), "global_event", args.day)
    assert row["updates"] == 0 and row["timestamp_cost_identity_passed"] and stats.n == row["decisions"]
    if horizon == P["horizon_seconds"]:
        counts = json.loads((PRIOR / "inputs/reference.json").read_text())[f"{args.domain}/{args.seed}/{args.day}/{args.scene}"]["passengers"]
        assert all(row["groups"][name]["passengers"] == count for name, count in counts.items())
    row.update(variant=args.variant, arm=args.arm, seed=args.seed, scene=args.scene,
               evaluation_seed=args.day, domain=args.domain, input_stats=stats.summary(),
               wall_seconds=time.monotonic() - started,
               peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.)
    base.write(out, row)
    print(json.dumps({k: row[k] for k in ["variant", "arm", "seed", "domain", "scene", "evaluation_seed", "cost", "mean_holding_seconds", "wall_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["contracts", "smoke", "eval"])
    parser.add_argument("--variant", choices=["native", "bias", "sindy"])
    parser.add_argument("--arm", choices=P["arms"] + ["reference"])
    parser.add_argument("--seed", type=int, choices=P["training_seeds"], default=64001)
    parser.add_argument("--domain", choices=["source", "target"])
    parser.add_argument("--day", type=int)
    parser.add_argument("--scene", choices=sorted(set(P["source_scenes"] + P["target_scenes"])))
    args = parser.parse_args()
    torch.set_num_threads(1)
    contracts() if args.mode == "contracts" else evaluate(args)
