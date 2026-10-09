"""Change frozen feedback temperature or selector, not learning or dynamics."""

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
spec = importlib.util.spec_from_file_location("temperature_frozen_runtime", PRIOR / "run.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
runtime, torch, np, base = previous.runtime, previous.torch, previous.np, previous.base


class Reference(runtime.StageReference):
    def __init__(self, seed, variant):
        super().__init__(seed)
        self.variant = variant
        self.temperature = P["original_temperature"] if variant == "identity" else P["variants"][variant]

    def costs(self, states):
        n, width = len(states), len(self.grid)
        tiled = states[:, None, :15].expand(n, width, 15).reshape(-1, 15)
        lines = (tiled[:, 0] * 12).round().long().clamp(0, 11)
        x = torch.cat((tiled[:, self.indices], self.grid.repeat(n)[:, None]), -1)
        x = torch.cat((x, torch.ones(len(x), 1)), -1)
        features = torch.ones(len(x), len(self.factors))
        for ids in self.factors.T:
            features *= x[:, ids]
        return (self.coefficients[lines] * features).sum(-1).reshape(n, width)

    def forward(self, states):
        costs = self.costs(states)
        if self.temperature is None:
            return self.grid[costs.argmin(-1)][:, None]
        actions = torch.softmax(-costs / self.temperature, -1) @ self.grid[:, None]
        return actions.clamp(-runtime.P["reference_action_clip"], runtime.P["reference_action_clip"])


def contracts():
    records = []
    states = torch.full((24, 92), .2)
    states[:, 0] = (torch.arange(24) % 12) / 12
    states[:, 5], states[:, 6], states[:, 8] = .3, .7, .5
    with torch.no_grad():
        for seed in P["training_seeds"]:
            _, cap = previous.load_actor("symbolic0", seed)
            original = runtime.StageReference(seed).eval().requires_grad_(False)
            identity = Reference(seed, "identity").eval().requires_grad_(False)
            assert torch.allclose(identity(states), original(states), atol=1e-6, rtol=0)
            for variant in P["variants"]:
                reference = Reference(seed, variant).eval().requires_grad_(False)
                assert torch.equal(reference.coefficients, original.coefficients)
                action = reference(states)
                assert action.shape == (24, 1) and bool(torch.isfinite(action).all())
                assert bool((action.abs() <= 1).all())
                assert torch.equal(action, reference(states[:, :15]))
                if variant == "minimum":
                    chosen = (action[:, 0] - reference.grid[0]) / (reference.grid[1] - reference.grid[0])
                    chosen = chosen.round().long()
                    costs = reference.costs(states)
                    assert torch.equal(costs.gather(1, chosen[:, None])[:, 0], costs.min(-1).values)
                records.append(dict(seed=seed, variant=variant, source_cap=cap,
                                    mean_holding_seconds=float((30 * (action + 1)).mean())))
    result = dict(passed=True, original_action_reproduced=True, coefficients_unchanged=True, records=records)
    base.write(HERE / "contracts.json", result)
    print(json.dumps(result), flush=True)


def evaluate(args):
    out = HERE / ("smoke" if args.mode == "smoke" else "results") / args.domain / args.variant / f"seed{args.seed}" / f"day{args.day}_{args.scene}.json"
    if out.exists():
        return
    assert args.day in P[args.domain + "_days"] and args.scene in P[args.domain + "_scenes"]
    assert args.mode == "smoke" or args.variant in P["variants"]
    _, cap = previous.load_actor("symbolic0", args.seed)
    controller = Reference(args.seed, args.variant).eval().requires_grad_(False)

    class Policy:
        def select_action(self, state, deterministic):
            assert deterministic
            return controller(torch.as_tensor(state)[None])[0].numpy()

    horizon = 7200 if args.mode == "smoke" and args.variant != "identity" else P["horizon_seconds"]
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="thesis-cs-symbolic-temperature-") as tmp:
        with redirect_stdout(StringIO()):
            env = base.make_env(Path(tmp), horizon, "normal")
            env.schedule = (base.source_schedule(args.scene, horizon) if args.domain == "source"
                            else runtime.old.old.P["scenes"][args.scene])
            with runtime.old.old.representation("source_clip", cap) as stats:
                row = base.rollout(env, Policy(), "global_event", args.day)
    assert row["updates"] == 0 and row["timestamp_cost_identity_passed"] and stats.n == row["decisions"]
    if horizon == P["horizon_seconds"]:
        counts = json.loads((previous.PRIOR / "inputs/reference.json").read_text())[f"{args.domain}/{args.seed}/{args.day}/{args.scene}"]["passengers"]
        assert all(row["groups"][name]["passengers"] == count for name, count in counts.items())
    row.update(variant=args.variant, seed=args.seed, scene=args.scene, evaluation_seed=args.day,
               domain=args.domain, temperature=controller.temperature, input_stats=stats.summary(),
               wall_seconds=time.monotonic() - started,
               peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.)
    base.write(out, row)
    print(json.dumps({k: row[k] for k in ["variant", "seed", "domain", "scene", "evaluation_seed", "cost", "mean_holding_seconds", "wall_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["contracts", "smoke", "eval"])
    parser.add_argument("--variant", choices=list(P["variants"]) + ["identity"])
    parser.add_argument("--seed", type=int, choices=P["training_seeds"], default=64001)
    parser.add_argument("--domain", choices=["source", "target"])
    parser.add_argument("--day", type=int)
    parser.add_argument("--scene", choices=sorted(set(P["source_scenes"] + P["target_scenes"])))
    args = parser.parse_args()
    torch.set_num_threads(1)
    contracts() if args.mode == "contracts" else evaluate(args)
