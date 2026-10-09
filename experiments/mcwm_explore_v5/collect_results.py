"""Compact paired outcomes against the fresh normalized control."""

import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"mcwm_usage_v3"))
from usage_common import P as PREVIOUS, write
P = dict(PREVIOUS, **json.loads((HERE/"protocol.json").read_text()))


def summarize():
    cells = [(scene["name"], arm, seed) for seed in P["training_seeds"]
        for scene in P["scenarios"] for arm in P["arms"]]
    training, results, missing = [], {}, []
    for scene, arm, seed in cells:
        folder = HERE/"runs"/scene/f"{arm}_seed{seed}"/"results"
        path = folder/"ready.json"
        if path.exists():
            row = json.loads(path.read_text())
            assert (row["arm"], row["scenario"], row["seed"]) == (arm, scene, seed), path
            assert row["continuation_updates"] == 20000 and row["total_policy_updates"] == 70000, path
            assert row["total_critic_updates"] == 70000 and row["target_online_training_events"] == 0, path
            training.append(row)
        else:
            missing.append(str(path.relative_to(HERE)))
        for case in range(10):
            path = folder/f"eval_{case:02d}.json"
            if not path.exists():
                missing.append(str(path.relative_to(HERE)))
                continue
            row = json.loads(path.read_text())
            block = P["evaluation_blocks"][case % 5]
            assert (row["arm"], row["scenario"], row["training_seed"], row["case"]) == (arm, scene, seed, case), path
            assert row["horizon_seconds"] == 24000 and row["deployment_hold_scale"] == 1., path
            assert row["demand_scale"] == P["evaluation_demand_scales"][case//5], path
            assert (row["demand_seed"], row["sumo_seed"]) == (block["demand_seed"], block["sumo_seed"]), path
            assert row["sumo_version"][1] == "SUMO 1.25.0" and "execution_actions" in row, path
            results[scene, arm, seed, case] = row
    report = dict(complete=not missing, training_ready=len(training), training_expected=len(cells),
        evaluation_ready=len(results), evaluation_expected=len(cells)*10, missing=missing)
    if missing:
        return report
    outcomes = []
    for scene, arm, seed in cells:
        for scale_index, scale in enumerate(P["evaluation_demand_scales"]):
            rows = [results[scene, arm, seed, c] for c in range(5*scale_index, 5*scale_index+5)]
            reference = [results[scene, "normalized", seed, c] for c in range(5*scale_index, 5*scale_index+5)]
            assert [r["passengers"]["departed_passengers"] for r in rows] == [r["passengers"]["departed_passengers"] for r in reference]
            cost = statistics.mean(r["generalized_seconds_per_departed_passenger"] for r in rows)
            completion = statistics.mean(r["passengers"]["completion_rate"] for r in rows)
            deltas = [r["generalized_seconds_per_departed_passenger"]-b["generalized_seconds_per_departed_passenger"]
                for r, b in zip(rows, reference)]
            outcomes.append(dict(scenario=scene, arm=arm, seed=seed, demand_scale=scale,
                cost=cost, completion=completion, cost_minus_normalized=statistics.mean(deltas),
                paired_days_better=sum(d < 0 for d in deltas), paired_day_deltas=deltas,
                completion_pp_minus_normalized=100*(completion-statistics.mean(r["passengers"]["completion_rate"] for r in reference))))
    report.update(training=training, outcomes=outcomes, inference=P["interpretation"])
    return report


if __name__ == "__main__":
    result = summarize()
    write(HERE/"summary.json", result)
    print(json.dumps({k: result[k] for k in ("complete", "training_ready", "evaluation_ready")}))
