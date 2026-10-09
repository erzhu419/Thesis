"""Summarize the frozen model-use trial without selecting evaluation days."""

from collections import defaultdict
import json
from statistics import mean

from usage_common import HERE, MODEL_ARMS, write
from collect_results import summarize


def read(path):
    return json.loads(path.read_text())


def analyze():
    summary = summarize()
    if not summary["complete"]:
        raise ValueError("The frozen comparison is incomplete")
    index = {(r["scenario"], r["demand_scale"], r["arm"], r["seed"]): r
             for r in summary["outcomes"]}
    groups = defaultdict(list)
    for row in summary["outcomes"]:
        groups[row["scenario"], row["demand_scale"], row["arm"]].append(row)
    cells = []
    for (scenario, demand, arm), rows in groups.items():
        cell = dict(scenario=scenario, demand_scale=demand, arm=arm,
                    cost=mean(r["cost"] for r in rows),
                    completion_percent=100*mean(r["completion"] for r in rows))
        for reference in ("h2o_matched", "mcwm", "normalized", "offline_only"):
            differences = [r["cost"]-index[scenario, demand, reference, r["seed"]]["cost"] for r in rows]
            cell["cost_minus_"+reference] = mean(differences)
            cell["paired_cost_minus_"+reference] = differences
        h2o_cost = mean(index[scenario, demand, "h2o_matched", r["seed"]]["cost"] for r in rows)
        cell["cost_reduction_percent_vs_h2o"] = -100*cell["cost_minus_h2o_matched"]/h2o_cost
        cell["completion_pp_minus_h2o"] = mean(r["completion_pp_minus_h2o"] for r in rows)
        cells.append(cell)

    diagnostics = []
    for train in summary["training"]:
        arm, seed, scenario = train["arm"], train["seed"], train["scenario"]
        if arm not in MODEL_ARMS:
            continue
        folder = HERE/"runs"/scenario/f"{arm}_seed{seed}"/"results/diagnostics"
        last = read(folder/"update070000.json")
        model = last["model"]["calibrate"]
        row = dict(scenario=scenario, arm=arm, seed=seed,
            final_model_weight=model["actual_weights"]["mean"],
            final_scaled_gradient_ratio=last["scaled_model_gradient_norm"]/last["scaled_real_gradient_norm"],
            final_prediction_loss=model["unweighted_prediction_loss"]["mean"],
            final_slot_identity_accuracy=model["components"]["slot_identity_accuracy"])
        if arm == "source_repair":
            intervals = [read(p)["source_repair"] for p in sorted(folder.glob("update*.json"))]
            corrected = sum(r["corrected_rows"] for r in intervals)
            row["corrected_source_fraction"] = corrected/sum(r["source_rows"] for r in intervals)
            row["corrected_eligible_fraction"] = corrected/sum(r["eligible_rows"] for r in intervals)
            row["mean_absolute_target_shift_corrected"] = sum(
                r["mean_absolute_target_shift_corrected"]*r["corrected_rows"] for r in intervals)/corrected
        diagnostics.append(row)

    service = []
    for cell in cells:
        if cell["arm"] != "normalized":
            continue
        scene, demand = cell["scenario"], cell["demand_scale"]
        cases = range(5) if demand == 1. else range(5, 10)
        differences = defaultdict(list)
        for seed in (6201, 6202):
            for case in cases:
                rows = [read(HERE/"runs"/scene/f"{arm}_seed{seed}"/"results"/f"eval_{case:02d}.json")
                        for arm in ("h2o_matched", "normalized")]
                for key in ("accrued_wait_seconds_per_departed_passenger", "accrued_in_vehicle_seconds_per_departed_passenger"):
                    differences[key].append(rows[1]["passengers"][key]-rows[0]["passengers"][key])
                differences["headway_absolute_error"].append(
                    rows[1]["headway"]["normalized_absolute_error"]["mean"]-
                    rows[0]["headway"]["normalized_absolute_error"]["mean"])
                differences["executed_hold_seconds"].append(rows[1]["mean_hold_seconds"]-rows[0]["mean_hold_seconds"])
        service.append(dict(scenario=scene, demand_scale=demand,
                            normalized_minus_h2o={k: mean(v) for k, v in differences.items()}))
    result = dict(training_count=summary["training_ready"], evaluation_count=summary["evaluation_ready"],
                  cells=cells, diagnostics=diagnostics, service_decomposition=service,
                  scope="Two development training seeds; paired days are not independent training repetitions")
    write(HERE/"analysis.json", result)
    return result


if __name__ == "__main__":
    result = analyze()
    for cell in result["cells"]:
        if cell["arm"] == "normalized":
            print(json.dumps(cell))
