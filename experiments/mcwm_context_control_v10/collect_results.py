"""Validate budgets and summarize only this branch's fresh test days."""

from statistics import mean
import json

from control_common import HERE, P, FIXED, CONTEXTS, bundle_path, write


def summarize():
    training, results, missing = [], {}, []
    for seed in P["training_seeds"]:
        for scene in CONTEXTS:
            for arm in P["arms"]:
                folder = HERE / "runs" / scene / f"{arm}_seed{seed}" / "results"
                path = folder / "ready.json"
                if path.exists():
                    row = json.loads(path.read_text())
                    assert (row["arm"], row["seed"], row["scenario"]) == (arm, seed, scene)
                    assert row["continuation_updates"] == 20000 and row["total_policy_updates"] == row["total_critic_updates"] == 70000
                    assert row["target_online_training_events"] == 0 and row["real_td_coefficient"] == .75
                    assert 25000 <= row["source_events"] < 25016
                    runtime = json.loads((folder / "runtime.json").read_text())
                    assert runtime["device"] == P["training_device"] and runtime["gpu_model"] is None
                    if arm not in FIXED:
                        assert row["context_provider_seed"] != seed
                    training.append(dict(row, runtime=runtime))
                else:
                    missing.append(str(path.relative_to(HERE)))
                for case in range(10, 20):
                    path = folder / f"eval_{case:02d}.json"
                    if not path.exists():
                        missing.append(str(path.relative_to(HERE)))
                        continue
                    row = json.loads(path.read_text())
                    assert (row["arm"], row["training_seed"], row["scenario"], row["case"], row["split"]) == (arm, seed, scene, case, "test")
                    assert row["horizon_seconds"] == 24000 and row["deployment_hold_scale"] == 1.
                    assert row["sumo_version"][1] == "SUMO 1.25.0" and "execution_actions" in row
                    assert row["demand_scale"] == P["evaluation_demand_scales"][(case-10)//5]
                    assert all(row[k] == v for k, v in P["test_blocks"][case % 5].items())
                    results[scene, arm, seed, case] = row
    count = len(P["training_seeds"])*len(CONTEXTS)*len(P["arms"])
    report = dict(complete=not missing, training_ready=len(training), training_expected=count,
        evaluation_ready=len(results), evaluation_expected=count*10, missing=missing)
    if missing:
        return report
    outcomes, recognition, mappings = [], [], []
    for seed in P["training_seeds"]:
        bundle = json.loads(bundle_path(seed).read_text())
        mappings.append(dict(seed=seed, provider_seed=bundle["provider_seed"], coefficients=bundle["coefficients"],
            global_coefficient=bundle["global_coefficient"], selection=bundle["selection_scope"]))
        for scene in CONTEXTS:
            for arm in P["arms"]:
                for scale_index, scale in enumerate(P["evaluation_demand_scales"]):
                    cases = range(10+5*scale_index, 15+5*scale_index)
                    rows = [results[scene, arm, seed, c] for c in cases]
                    for reference in ("fixed0250", "recognized_full"):
                        controls = [results[scene, reference, seed, c] for c in cases]
                        assert [r["passengers"]["departed_passengers"] for r in rows] == [r["passengers"]["departed_passengers"] for r in controls]
                        deltas = [r["generalized_seconds_per_departed_passenger"]-c["generalized_seconds_per_departed_passenger"] for r,c in zip(rows,controls)]
                        outcomes.append(dict(scenario=scene, arm=arm, seed=seed, demand_scale=scale,
                            reference=reference, delta_seconds=mean(deltas), paired_day_deltas=deltas,
                            completion_delta_pp=100*mean(r["passengers"]["completion_rate"]-c["passengers"]["completion_rate"] for r,c in zip(rows,controls))))
                if arm not in FIXED:
                    trace = json.loads((HERE / "runs" / scene / f"{arm}_seed{seed}" / "results/context_trace.json").read_text())["trace"]
                    classes = bundle["recognizer"]["classes"]
                    recognition.append(dict(scenario=scene, arm=arm, seed=seed, blocks=len(trace),
                        block_accuracy=mean(classes[max(range(len(classes)),key=lambda i:r["belief"][i])]==scene for r in trace),
                        accepted_fraction=mean(r["context_accepted"] for r in trace),
                        wrong_high_score_fraction=mean(max(r["belief"])>.9 and classes[max(range(len(classes)),key=lambda i:r["belief"][i])]!=scene for r in trace),
                        mean_coefficient=mean(r["coefficient"] for r in trace),
                        episode_indices=sorted({r["episode_index"] for r in trace})))
    report.update(training=training, outcomes=outcomes, recognition=recognition, mappings=mappings,
        interpretation=P["interpretation"])
    return report


if __name__ == "__main__":
    result = summarize()
    write(HERE / "summary.json", result)
    print(json.dumps({k:result[k] for k in ("complete", "training_ready", "evaluation_ready")}))
