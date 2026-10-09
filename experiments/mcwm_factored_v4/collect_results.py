"""Keep every mechanism treatment and compare against frozen v3 actors."""

from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"mcwm_usage_v3"))
from usage_common import P as PREVIOUS, write
P = dict(PREVIOUS, **json.loads((HERE/"protocol.json").read_text()))


def read(path):
    return json.loads(path.read_text())


def summarize():
    old = read(HERE.parent/"mcwm_usage_v3/summary.json")
    assert old["complete"]
    references = {(r["scenario"], r["arm"], r["seed"], r["demand_scale"]):r for r in old["outcomes"]}
    training, evaluations, missing, outcomes = [], [], [], []
    for scene in [s["name"] for s in P["scenarios"]]:
        for structure in P["structures"]:
            for arm in P["arms"]:
                for seed in P["training_seeds"]:
                    folder = HERE/"runs"/scene/f"{structure}_{arm}_seed{seed}"/"results"
                    path = folder/"ready.json"
                    if path.exists():
                        row = read(path)
                        assert (row["structure"], row["arm"], row["scenario"], row["seed"]) == (structure, arm, scene, seed)
                        assert row["continuation_updates"] == 20000 and row["total_policy_updates"] == 70000
                        assert row["total_critic_updates"] == 70000 and row["target_online_training_events"] == 0
                        assert 25000 <= row["source_events"] <= 25002
                        training.append(row)
                    else:
                        missing.append(str(path.relative_to(HERE)))
                    cells = defaultdict(list)
                    for case in range(10):
                        path = folder/f"eval_{case:02d}.json"
                        if not path.exists():
                            missing.append(str(path.relative_to(HERE)))
                            continue
                        row = read(path)
                        assert (row["structure"], row["arm"], row["scenario"], row["training_seed"], row["case"]) == (structure, arm, scene, seed, case)
                        assert row["horizon_seconds"] == 24000 and row["deployment_hold_scale"] == 1.
                        assert row["sumo_version"][1] == "SUMO 1.25.0"
                        block = P["evaluation_blocks"][case % 5]
                        assert (row["demand_seed"], row["sumo_seed"]) == (block["demand_seed"], block["sumo_seed"])
                        demand = P["evaluation_demand_scales"][case // 5]
                        assert row["demand_scale"] == demand
                        previous = read(HERE.parent/f"mcwm_usage_v3/runs/{scene}/normalized_seed{seed}/results/eval_{case:02d}.json")
                        assert row["passengers"]["departed_passengers"] == previous["passengers"]["departed_passengers"]
                        cells[demand].append(row)
                        evaluations.append(path)
                    for demand, rows in cells.items():
                        if len(rows) != 5:
                            continue
                        cost = mean(r["generalized_seconds_per_departed_passenger"] for r in rows)
                        completion = mean(r["passengers"]["completion_rate"] for r in rows)
                        result = dict(scenario=scene, structure=structure, arm=arm, seed=seed,
                            demand_scale=demand, cost=cost, completion=completion)
                        for ref in ("normalized", "source_repair", "h2o_matched"):
                            previous = references[scene, ref, seed, demand]
                            result["cost_minus_"+ref] = cost-previous["cost"]
                            result["completion_pp_minus_"+ref] = 100*(completion-previous["completion"])
                        outcomes.append(result)
    report = dict(complete=not missing, training_ready=len(training), training_expected=24,
        evaluation_ready=len(evaluations), evaluation_expected=240, missing=missing,
        training=training, outcomes=outcomes)
    if not missing:
        differences, index = [], {(r["scenario"],r["structure"],r["arm"],r["seed"],r["demand_scale"]):r for r in outcomes}
        for row in outcomes:
            if row["structure"] != "factored":
                continue
            ref = index[row["scenario"], "unmasked", row["arm"], row["seed"], row["demand_scale"]]
            differences.append(dict(scenario=row["scenario"], arm=row["arm"], seed=row["seed"], demand_scale=row["demand_scale"],
                cost_minus_unmasked=row["cost"]-ref["cost"],
                completion_pp_minus_unmasked=100*(row["completion"]-ref["completion"])))
        report["parent_restriction_effect"] = differences
    return report


if __name__ == "__main__":
    result = summarize()
    write(HERE/"summary.json", result)
    print(json.dumps({k:result[k] for k in ("complete", "training_ready", "evaluation_ready")}, indent=2))
