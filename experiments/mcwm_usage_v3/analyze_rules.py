"""Compare the frozen actors with unchanged rules on the same evaluation days."""

import json
from statistics import mean

from usage_common import HERE, P, write


def read(path):
    return json.loads(path.read_text())


def analyze():
    rules = {}
    for method in ("zero", "headway"):
        for case in range(10):
            row = read(HERE / f"rules/{method}/case{case:02d}/results/case{case:02d}.json")
            block = P["evaluation_blocks"][case % 5]
            assert row["method"] == method and row["execution_layer"] == "early60"
            assert row["horizon_seconds"] == 24000 and row["deployment_hold_scale"] == 1.
            assert row["sumo_version"][1] == "SUMO 1.25.0"
            assert (row["demand_seed"], row["sumo_seed"]) == (block["demand_seed"], block["sumo_seed"])
            assert row["demand_scale"] == P["evaluation_demand_scales"][case // 5]
            rules[method, case] = row
    baselines, comparisons = [], []
    for index, demand in enumerate(P["evaluation_demand_scales"]):
        cases = range(index*5, index*5+5)
        for method in ("zero", "headway"):
            rows = [rules[method, case] for case in cases]
            baselines.append(dict(method=method, demand_scale=demand,
                cost=mean(r["generalized_seconds_per_departed_passenger"] for r in rows),
                completion_percent=100*mean(r["passengers"]["completion_rate"] for r in rows)))
        for scene in [s["name"] for s in P["scenarios"]]:
            for arm in ("h2o_matched", "normalized"):
                for seed in P["training_seeds"]:
                    days = []
                    for case in cases:
                        actor = read(HERE / f"runs/{scene}/{arm}_seed{seed}/results/eval_{case:02d}.json")
                        reference = rules["headway", case]
                        for key in ("demand_seed", "sumo_seed", "demand_scale", "horizon_seconds", "deployment_hold_scale", "sumo_version"):
                            assert actor[key] == reference[key], (scene, arm, seed, case, key)
                        assert actor["passengers"]["departed_passengers"] == reference["passengers"]["departed_passengers"]
                        days.append(dict(case=case,
                            cost=actor["generalized_seconds_per_departed_passenger"],
                            cost_reduction=reference["generalized_seconds_per_departed_passenger"]-actor["generalized_seconds_per_departed_passenger"],
                            completion_pp_gain=100*(actor["passengers"]["completion_rate"]-reference["passengers"]["completion_rate"])))
                    comparisons.append(dict(scenario=scene, arm=arm, seed=seed, demand_scale=demand,
                        cost=mean(r["cost"] for r in days),
                        cost_reduction=mean(r["cost_reduction"] for r in days),
                        completion_pp_gain=mean(r["completion_pp_gain"] for r in days),
                        days=days))
    result = dict(rule_evaluations=20, baselines=baselines, comparisons=comparisons,
        scope="Existing five evaluation days per demand; two trained actors reported separately; additional references, not independent confirmation")
    write(HERE / "rule_analysis.json", result)
    return result


if __name__ == "__main__":
    result = analyze()
    print(json.dumps(result["baselines"], indent=2))
    for row in result["comparisons"]:
        print(row["scenario"], row["demand_scale"], row["arm"], row["seed"],
              "cost_reduction", round(row["cost_reduction"], 3),
              "completion_pp_gain", round(row["completion_pp_gain"], 3),
              "better_days", sum(d["cost_reduction"] > 0 for d in row["days"]))
