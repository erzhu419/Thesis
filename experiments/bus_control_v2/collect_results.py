"""Analyze on shared storage and publish only a compact JSON handoff."""

from collections import defaultdict
import json
import statistics

from station_common import HERE, P, write


def collect():
    ready, smoke_eval, formal_eval, policies = [], [], [], []
    for smoke in (False, True):
        prefix = "smoke/" if smoke else ""
        data = HERE / ("smoke_data" if smoke else "data")
        for mode in ("holding", "dual"):
            if (data / mode / "ready.json").exists():
                ready.append(prefix+"data/"+mode)
        for shard in range(2 if smoke else P["offline_dual_shards"]):
            if (data / "dual" / f"shard{shard:02d}/ready.json").exists():
                ready.append(prefix+f"shard/{shard}")
        inputs = HERE / ("smoke_inputs" if smoke else "inputs")
        seeds = P["training_seeds"][:1] if smoke else P["training_seeds"]
        scenes = ["calibrated"] if smoke else [s["name"] for s in P["scenarios"]]
        for mode, objective in P["cells"]:
            for seed in seeds:
                if (inputs / mode / objective / f"seed{seed}/ready.json").exists():
                    ready.append(prefix+f"init/{mode}/{objective}/{seed}")
                for scene in scenes:
                    for arm in P["arms"]:
                        label = f"{mode}/{objective}/{scene}/{arm}_seed{seed}"
                        folder = HERE / ("smoke" if smoke else "runs") / label
                        if (folder / "ready.json").exists():
                            ready.append(prefix+"policy/"+label)
                            policies.append(dict(smoke=smoke, **json.loads((folder / "ready.json").read_text())))
                        for case in ([0] if smoke else range(10)):
                            path = folder / "evaluation" / f"case{case:02d}.json"
                            if path.exists():
                                row = json.loads(path.read_text())
                                (smoke_eval if smoke else formal_eval).append(row)
        for rule in ("zero", "daganzo_duet"):
            for case in ([0] if smoke else range(10)):
                path = HERE / ("rule_smoke" if smoke else "rules") / rule / "evaluation" / f"case{case:02d}.json"
                if path.exists():
                    (smoke_eval if smoke else formal_eval).append(json.loads(path.read_text()))
    if len(smoke_eval) == 8:
        ready.append("smoke-pass")
    grouped = defaultdict(list)
    for row in formal_eval:
        key = (row["mode"], row["objective"], row["source_scene"], row["arm"], row["training_seed"])
        grouped[key].append(row)
    means = []
    fields = ("generalized_seconds_per_departed_passenger", "headway_ridge_mean", "headway_plan_mae_seconds", "headway_imbalance_seconds")
    for key, rows in grouped.items():
        means.append(dict(zip(("mode", "objective", "scene", "arm", "seed"), key), cases=len(rows),
            **{k:statistics.mean(r[k] for r in rows) for k in fields},
            completion_rate=statistics.mean(r["passengers"]["completion_rate"] for r in rows)))
    result = dict(experiment=P["name"], analysis_location="server", ready=ready,
        smoke_evaluations=len(smoke_eval), formal_evaluations=len(formal_eval),
        expected_formal_evaluations=560, policies=policies, seed_means=means,
        complete=len(formal_eval) == 560)
    write(HERE / "summary.json", result)
    print(json.dumps({k:result[k] for k in ("smoke_evaluations", "formal_evaluations", "complete")}))


if __name__ == "__main__":
    collect()
