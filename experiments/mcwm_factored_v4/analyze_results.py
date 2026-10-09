"""Summarize every frozen treatment, paired seed and model-use diagnostic."""

from collections import Counter, defaultdict
from statistics import mean

from collect_results import HERE, P, read, summarize, write


def analyze():
    summary = summarize()
    if not summary["complete"]:
        raise ValueError("The mechanism comparison is incomplete")
    groups = defaultdict(list)
    for row in summary["outcomes"]:
        groups[row["scenario"], row["demand_scale"], row["structure"], row["arm"]].append(row)
    cells = []
    for (scene, demand, structure, arm), rows in groups.items():
        cells.append(dict(scenario=scene, demand_scale=demand, structure=structure, arm=arm,
            cost=mean(r["cost"] for r in rows),
            completion_percent=100*mean(r["completion"] for r in rows),
            cost_minus_normalized=mean(r["cost_minus_normalized"] for r in rows),
            completion_pp_minus_normalized=mean(r["completion_pp_minus_normalized"] for r in rows),
            seed_cost_differences={r["seed"]:r["cost_minus_normalized"] for r in rows}))

    candidates = []
    for structure in P["structures"]:
        for arm in P["arms"]:
            rows = [r for r in summary["outcomes"] if r["structure"] == structure and r["arm"] == arm]
            high = [r for r in rows if r["scenario"] != "calibrated"]
            candidates.append(dict(structure=structure, arm=arm,
                high_mismatch_cost_increase=mean(r["cost_minus_normalized"] for r in high),
                high_mismatch_completion_pp_change=mean(r["completion_pp_minus_normalized"] for r in high),
                high_mismatch_cost_minus_h2o=mean(r["cost_minus_h2o_matched"] for r in high),
                seed_high_mismatch_cost_increases={seed:mean(r["cost_minus_normalized"] for r in high if r["seed"] == seed)
                                                   for seed in P["training_seeds"]},
                better_seed_cells=sum(r["cost_minus_normalized"] < 0 for r in rows), total_seed_cells=len(rows)))

    diagnostics = []
    variants = [("v3", "normalized"), ("v3", "source_repair")]+[
        (structure, arm) for structure in P["structures"] for arm in P["arms"]]
    for structure, arm in variants:
        root = HERE.parent/"mcwm_usage_v3" if structure == "v3" else HERE
        for scene in [s["name"] for s in P["scenarios"]]:
            for seed in P["training_seeds"]:
                name = f"{arm if structure == 'v3' else structure+'_'+arm}_seed{seed}"
                folder = root/"runs"/scene/name/"results"
                if structure != "v3":
                    current = read(folder/"protocol.json")
                    reference = read(HERE.parent/f"mcwm_usage_v3/runs/{scene}/{arm}_seed{seed}/results/protocol.json")
                    assert current["config"] == reference["config"]
                    assert current["offline_rows"] == reference["offline_rows"]
                    for key, value in reference["protocol"].items():
                        if key not in ("name", "arms"):
                            assert current["protocol"][key] == value, (structure, arm, scene, seed, key)
                last = read(folder/"diagnostics/update070000.json")
                train = read(folder/"ready.json")
                model = last["model"]["calibrate"]
                diagnostic = dict(structure=structure, arm=arm, scenario=scene, seed=seed,
                    final_weight=model["actual_weights"]["mean"],
                    final_gradient_ratio_percent=100*last["scaled_model_gradient_norm"]/last["scaled_real_gradient_norm"],
                    prediction=model["components"],
                    training_minutes=train["training_wall_seconds"]/60,
                    peak_host_rss_mb=train["peak_host_rss_mb"], peak_gpu_allocated_mb=train["peak_gpu_allocated_mb"],
                    family_choices=dict(Counter(h["selected_family"] for h in train["model_history"])))
                if arm == "source_repair":
                    intervals = [read(p)["source_repair"] for p in sorted((folder/"diagnostics").glob("update*.json"))]
                    diagnostic["corrected_source_fraction"] = sum(r["corrected_rows"] for r in intervals)/sum(r["source_rows"] for r in intervals)
                diagnostics.append(diagnostic)

    # The execution decomposition uses all paired days, not selected rollouts.
    service = []
    for cell in cells:
        rows = defaultdict(list)
        scene, arm, structure, demand = (cell[k] for k in ("scenario", "arm", "structure", "demand_scale"))
        for seed in P["training_seeds"]:
            for case in (range(5) if demand == 1. else range(5, 10)):
                current = read(HERE/f"runs/{scene}/{structure}_{arm}_seed{seed}/results/eval_{case:02d}.json")
                previous = read(HERE.parent/f"mcwm_usage_v3/runs/{scene}/normalized_seed{seed}/results/eval_{case:02d}.json")
                for key in ("accrued_wait_seconds_per_departed_passenger", "accrued_in_vehicle_seconds_per_departed_passenger"):
                    rows[key].append(current["passengers"][key]-previous["passengers"][key])
                rows["mean_hold_seconds"].append(current["mean_hold_seconds"]-previous["mean_hold_seconds"])
                rows["reserve_changed_fraction"].append(current["execution_actions"]["reserve_changed_fraction"]-
                                                        previous["execution_actions"]["reserve_changed_fraction"])
        differences = {k:mean(v) for k,v in rows.items()}
        assert abs(2*differences["accrued_wait_seconds_per_departed_passenger"]+
                   differences["accrued_in_vehicle_seconds_per_departed_passenger"]-
                   cell["cost_minus_normalized"]) < 1e-8
        service.append(dict(scenario=scene, demand_scale=demand, structure=structure, arm=arm,
                            differences=differences))

    result = dict(complete=True, training_count=24, evaluation_count=240, cells=cells,
        candidates=candidates, parent_restriction_effect=summary["parent_restriction_effect"],
        diagnostics=diagnostics, service_decomposition=service,
        scope="Two development training seeds; shared five days per demand are not independent training repeats")
    write(HERE/"analysis.json", result)
    return result


if __name__ == "__main__":
    result = analyze()
    for row in result["candidates"]:
        print(row)
