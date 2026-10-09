"""Validate all stages and compare only the frozen final test stream."""

from collections import Counter
import json
from statistics import mean

from context_common import HERE, P, FIXED, CONTEXTS, cases, bundle_path, write


def summarize():
    training, results, missing = [], {}, []
    for seed in P["training_seeds"]:
        for scene in CONTEXTS:
            for arm in P["arms"]:
                folder = HERE / "runs" / scene / f"{arm}_seed{seed}" / "results"
                path = folder / "ready.json"
                if path.exists():
                    row = json.loads(path.read_text())
                    assert (row["arm"], row["seed"], row["scenario"]) == (arm, seed, scene), path
                    assert row["continuation_updates"] == 20000 and row["total_policy_updates"] == row["total_critic_updates"] == 70000, path
                    assert row["target_online_training_events"] == 0 and row["real_td_coefficient"] == .75, path
                    assert 25000 <= row["source_events"] < 25016, path
                    if arm not in FIXED:
                        assert row["context_provider_seed"] != seed, path
                    training.append(row)
                else:
                    missing.append(str(path.relative_to(HERE)))
                for case in cases(arm):
                    path = folder / f"eval_{case:02d}.json"
                    if not path.exists():
                        missing.append(str(path.relative_to(HERE)))
                        continue
                    row = json.loads(path.read_text())
                    split = "development" if case < 10 else "test"
                    block = P[split+"_blocks"][case % 5]
                    assert (row["arm"], row["training_seed"], row["scenario"], row["case"], row["split"]) == (arm, seed, scene, case, split), path
                    assert row["horizon_seconds"] == 24000 and row["deployment_hold_scale"] == 1., path
                    assert row["sumo_version"][1] == "SUMO 1.25.0" and "execution_actions" in row, path
                    assert row["demand_scale"] == P["evaluation_demand_scales"][(case % 10) // 5], path
                    assert all(row[k] == v for k, v in block.items()), path
                    results[scene, arm, seed, case] = row
    report = dict(complete=not missing, training_ready=len(training), training_expected=30,
        evaluation_ready=len(results), evaluation_expected=480, missing=missing)
    if missing:
        return report
    outcomes, recognition, mappings = [], [], []
    for seed in P["training_seeds"]:
        bundle = json.loads(bundle_path(seed).read_text())
        mappings.append(dict(seed=seed, provider_seed=bundle["provider_seed"],
            coefficients=bundle["coefficients"], best_global_fixed_arm=bundle["best_global_fixed_arm"],
            mapping_has_context_effect=bundle["mapping_has_context_effect"]))
        for scene in CONTEXTS:
            for arm in P["arms"]:
                for scale_index, scale in enumerate(P["evaluation_demand_scales"]):
                    case_ids = range(10+5*scale_index, 15+5*scale_index)
                    rows = [results[scene, arm, seed, c] for c in case_ids]
                    for role, reference in (("fixed_original", "fixed_original"),
                            ("best_global_fixed", bundle["best_global_fixed_arm"]), ("oracle_context", "oracle_context")):
                        controls = [results[scene, reference, seed, c] for c in case_ids]
                        assert [r["passengers"]["departed_passengers"] for r in rows] == [r["passengers"]["departed_passengers"] for r in controls]
                        deltas = [r["generalized_seconds_per_departed_passenger"]-c["generalized_seconds_per_departed_passenger"] for r, c in zip(rows, controls)]
                        outcomes.append(dict(scenario=scene, arm=arm, seed=seed, demand_scale=scale,
                            reference=reference, reference_role=role,
                            delta_seconds=mean(deltas), paired_day_deltas=deltas,
                            completion_delta_pp=100*mean(r["passengers"]["completion_rate"]-c["passengers"]["completion_rate"] for r, c in zip(rows, controls))))
            trace = json.loads((HERE / "runs" / scene / f"recognized_context_seed{seed}" / "results/context_trace.json").read_text())
            classes = bundle["recognizer"]["classes"]
            selected = Counter(classes[max(range(len(classes)), key=lambda i: r["belief"][i])] for r in trace["trace"])
            recognition.append(dict(scenario=scene, seed=seed, blocks=len(trace["trace"]),
                predicted_counts=dict(selected), block_accuracy=selected[scene]/len(trace["trace"]) if trace["trace"] else None,
                mean_coefficient=mean(r["coefficient"] for r in trace["trace"]),
                oracle_coefficient=bundle["coefficients"][scene]))
    report.update(training=training, outcomes=outcomes, recognition=recognition,
        mappings=mappings, interpretation=P["interpretation"])
    return report


if __name__ == "__main__":
    report = summarize()
    write(HERE / "summary.json", report)
    print(json.dumps({k: report[k] for k in ("complete", "training_ready", "evaluation_ready")}))
