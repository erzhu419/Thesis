"""Summarize paired test days within seeds, then compare independent training seeds."""

from study_common import HERE, P, write
import json
import math
import statistics


def interval(values):
    mean = statistics.mean(values)
    error = 2.776445105 * statistics.stdev(values) / math.sqrt(5)
    return dict(mean=mean, ci95=[mean-error, mean+error], by_training_seed=dict(zip(P["training_seeds"], values)))


def measures(kind, row):
    if kind == "mc":
        return dict(cost=row["generalized_seconds_per_departed_passenger"],
                    completion=row["passengers"]["completion_rate"], hold=row["mean_hold_seconds"])
    windows = list(row["windows"].values()) if kind == "cs" else row["windows"]
    decisions = sum(w["decisions"] for w in windows)
    result = dict(reward=row["reward"],
                  headway_imbalance=sum(w["imbalance_sum"] for w in windows)/max(decisions, 1),
                  mean_hold=sum(w["hold_sum"] for w in windows)/max(decisions, 1))
    if kind == "legacy":
        result["cost"] = row["generalized_seconds_per_passenger"]
    else:
        result["wait_passenger_seconds"] = sum(w["waiting"] for w in windows)
        result["ride_passenger_seconds"] = sum(w["onboard"] for w in windows)
    return result


def summarize_kind(kind, folder_for=None):
    arms, seeds = P[kind+"_arms"], P["training_seeds"]
    cases = 10 if kind == "mc" else 9
    missing, records, costs = [], {}, []
    for arm in arms:
        for seed in seeds:
            folder = (folder_for(arm, seed) if folder_for else
                      HERE / "runs" / kind / f"{arm}_seed{seed}" / "results")
            ready = folder / "ready.json"
            if not ready.exists():
                missing.append(str(ready.relative_to(HERE) if ready.is_relative_to(HERE) else ready))
            for case in range(cases):
                path = folder / f"eval_{case:02d}.json"
                if not path.exists():
                    missing.append(str(path.relative_to(HERE) if path.is_relative_to(HERE) else path))
                    continue
                row = json.loads(path.read_text())
                if (row["arm"], row["training_seed"], row["case"]) != (arm, seed, case):
                    raise ValueError("Result identity mismatch: "+str(path))
                records[arm, seed, case] = measures(kind, row)
            if kind == "mc" and (folder / "train_summary.json").exists():
                cost = json.loads((folder / "train_summary.json").read_text())
                if (cost["policy_updates"], cost["total_critic_updates"]) != (16144, 17680):
                    raise ValueError("Unexpected learner budget: "+str(folder))
                costs.append(cost)
    if missing:
        return dict(complete=False, available_tests=len(records), expected_tests=len(arms)*len(seeds)*cases,
                    missing=missing, cost_records=len(costs))
    metrics = list(next(iter(records.values())))
    scenes = ([("demand1", range(5)), ("demand1.25", range(5, 10))] if kind == "mc" else
              [(name, range(i*3, (i+1)*3)) for i, name in enumerate(
                  P["legacy_scenes"] if kind == "legacy" else P["cs_scenes"])])
    results = {}
    for scene, indices in scenes:
        means = {(arm, seed, metric): statistics.mean(records[arm, seed, c][metric] for c in indices)
                 for arm in arms for seed in seeds for metric in metrics}
        comparisons = {}
        reference = "mixed_nomodel" if kind == "mc" else "learned_full" if kind == "legacy" else "mlp_bc0"
        for arm in arms:
            if arm != reference:
                comparisons[arm+"_minus_"+reference] = {
                    metric: interval([means[arm, s, metric]-means[reference, s, metric] for s in seeds])
                    for metric in metrics}
        if kind in ("legacy", "cs"):
            if kind == "legacy":
                contrasts = dict(
                    critic_on_minus_off={"learned_full": .5, "actor_off": .5, "critic_off": -.5, "both_off": -.5},
                    actor_on_minus_off={"learned_full": .5, "critic_off": .5, "actor_off": -.5, "both_off": -.5},
                    interaction={"learned_full": 1., "both_off": 1., "critic_off": -1., "actor_off": -1.})
            else:
                contrasts = dict(
                    nau_minus_mlp={"nau_bc0": .5, "nau_bc1": .5, "mlp_bc0": -.5, "mlp_bc1": -.5},
                    bc_on_minus_off={"nau_bc1": .5, "mlp_bc1": .5, "nau_bc0": -.5, "mlp_bc0": -.5},
                    interaction={"nau_bc1": 1., "mlp_bc0": 1., "nau_bc0": -1., "mlp_bc1": -1.})
            for name, weights in contrasts.items():
                comparisons[name] = {m: interval([sum(w*means[a, s, m] for a, w in weights.items()) for s in seeds])
                                     for m in metrics}
        results[scene] = dict(means={arm: {m: interval([means[arm, s, m] for s in seeds]) for m in metrics}
                                    for arm in arms}, paired_differences=comparisons)
    return dict(complete=True, training_seeds=seeds, results=results, costs=costs,
                interval="Student t, 4 df; unit is training seed after within-seed test-day averaging")


if __name__ == "__main__":
    summary = {kind: summarize_kind(kind) for kind in ("legacy", "cs", "mc")}
    write(HERE / "summary.json", summary)
    print(json.dumps({kind: {k: v for k, v in row.items() if k in ("complete", "available_tests", "expected_tests")}
                      for kind, row in summary.items()}, indent=2))
