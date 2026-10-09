"""Analyze small result records on shared storage; no trajectory transfer."""

from collections import defaultdict
import argparse
import json
import math
from statistics import mean
from cross_common import HERE, P, folder, initializer_folder, write


def placements(smoke=False):
    seeds = P["training_seeds"][:1] if smoke else P["training_seeds"]
    for kind in ("legacy", "cs", "mc"):
        scenes = P["mc_scenes"][-1:] if smoke and kind == "mc" else P["mc_scenes"] if kind == "mc" else ["normal"]
        for objective in P["objectives"]:
            for seed in seeds:
                for scene in scenes:
                    for arm in P[kind+"_arms"]:
                        yield dict(kind=kind, objective=objective, arm=arm, seed=seed, scene=scene, smoke=smoke)


def metrics(row, kind):
    passenger = row.get("generalized_seconds_per_passenger", row.get("generalized_seconds_per_departed_passenger"))
    completion = row.get("completion_rate", row.get("passengers", {}).get("completion_rate"))
    return dict(passenger_seconds=passenger, completion=completion,
        plan_mae_seconds=row["headway_plan_mae_seconds"], imbalance_seconds=row["headway_imbalance_seconds"])


def collect(smoke=False):
    initializers = []
    seeds = P["training_seeds"][:1] if smoke else P["training_seeds"]
    for objective in P["objectives"]:
        for seed in seeds:
            path = initializer_folder(objective, seed, smoke) / "ready.json"
            if path.exists():
                row = json.loads(path.read_text())
                assert row["objective"] == objective and row["seed"] == seed and row["online_events"] == 0
                initializers.append(dict(objective=objective, seed=seed, offline_updates=row["offline_updates"]))
    policies, rows, training = [], [], []
    items = list(placements(smoke))
    expected_evaluations = 0
    for item in items:
        out = folder(item["kind"], item["objective"], item["arm"], item["seed"], item["scene"], smoke)
        ready = out / "results/ready.json"
        if ready.exists():
            policies.append(item)
            row = json.loads(ready.read_text())
            training.append(dict(item, **{k:row[k] for k in ("training_wall_seconds", "peak_host_rss_mb") if k in row}))
        cases = [0] if smoke else range(10 if item["kind"] == "mc" else 9)
        expected_evaluations += len(cases)
        for case in cases:
            path = out / "results" / f"eval_{case:02d}.json"
            if path.exists():
                try:
                    row = json.loads(path.read_text())
                except json.JSONDecodeError:
                    continue
                # Inherited evaluators publish an intermediate record before
                # the crossover adds the independent headway measurements.
                if row.get("objective") != item["objective"]:
                    continue
                value = metrics(row, item["kind"])
                if not all(v is not None and math.isfinite(v) for v in value.values()):
                    raise ValueError("Incomplete evaluation metrics: "+str(path))
                rows.append(dict(item, case=case, evaluation_scene=row.get("scene", row.get("demand_scale")), **value))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["kind"], row["objective"], row["scene"], row["evaluation_scene"], row["arm"], row["seed"])].append(row)
    seed_means = []
    for key, values in sorted(grouped.items(), key=lambda pair:str(pair[0])):
        kind, objective, source_scene, evaluation_scene, arm, seed = key
        seed_means.append(dict(kind=kind, objective=objective, source_scene=source_scene,
            evaluation_scene=evaluation_scene, arm=arm, seed=seed, cases=len(values),
            **{name:mean(v[name] for v in values) for name in
               ("passenger_seconds", "completion", "plan_mae_seconds", "imbalance_seconds")}))
    index = {(r["kind"], r["objective"], r["source_scene"], r["evaluation_scene"], r["arm"], r["seed"]):r for r in seed_means}
    paired = defaultdict(list)
    controls = {"legacy": "ensemble_sac_control", "cs": "csbapr_no_nau", "mc": "h2o_matched"}
    for row in seed_means:
        control = controls[row["kind"]]
        key = (row["kind"], row["objective"], row["source_scene"], row["evaluation_scene"])
        reference = index.get((*key, control, row["seed"]))
        expected_cases = 5 if row["kind"] == "mc" else len(P["evaluation_seeds"])
        if reference and row["arm"] != control and row["cases"] == reference["cases"] == expected_cases:
            paired[(*key, row["arm"], control)].append(dict(seed=row["seed"], **{k:row[k]-reference[k] for k in
                ("passenger_seconds", "completion", "plan_mae_seconds", "imbalance_seconds")}))
    contrasts = []
    for key, values in sorted(paired.items(), key=lambda pair:str(pair[0])):
        kind, objective, source, target, arm, control = key
        contrasts.append(dict(kind=kind, objective=objective, source_scene=source, evaluation_scene=target,
            arm=arm, control=control, seeds=len(values), per_seed=values,
            mean_difference={k:mean(v[k] for v in values) for k in ("passenger_seconds", "completion", "plan_mae_seconds", "imbalance_seconds")}))
    result = dict(experiment=P["name"], analysis_location="server", smoke=smoke,
        initializers=initializers, policies=policies, initialization_ready=len(initializers),
        training_ready=len(policies), evaluation_ready=len(rows), expected_training=len(items),
        expected_evaluation=expected_evaluations, complete=len(initializers) == 2*len(seeds)
        and len(policies) == len(items) and len(rows) == expected_evaluations,
        seed_means=seed_means, paired_contrasts=contrasts, training=training)
    write(HERE / ("smoke_summary.json" if smoke else "summary.json"), result)
    print(json.dumps({k:result[k] for k in ("initialization_ready", "training_ready", "evaluation_ready", "complete")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    collect(parser.parse_args().smoke)
