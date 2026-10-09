"""Keep every arm and average paired test days within each development seed."""

from usage_common import HERE, P, write
import json
import statistics


def summarize():
    scenarios = [s["name"] for s in P["scenarios"]]
    cells = [(s, a, seed) for seed in P["training_seeds"] for s in scenarios
             for a in P["arms"]+(["offline_only"] if s == "calibrated" else [])]
    training, results, missing = [], {}, []
    for scene, arm, seed in cells:
        folder = HERE/"runs"/scene/f"{arm}_seed{seed}"/"results"
        path = folder/"ready.json"
        if path.exists():
            row = json.loads(path.read_text())
            assert (row["arm"], row["seed"], row["scenario"]) == (arm, seed, scene), path
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
            assert (row["arm"], row["training_seed"], row["scenario"], row["case"]) == (arm, seed, scene, case), path
            assert row["horizon_seconds"] == 24000 and row["deployment_hold_scale"] == 1., path
            assert (row["demand_seed"], row["sumo_seed"]) == (block["demand_seed"], block["sumo_seed"]), path
            assert row["demand_scale"] == P["evaluation_demand_scales"][case // 5], path
            assert row["sumo_version"][1] == "SUMO 1.25.0", path
            assert "execution_actions" in row, path
            results[scene, arm, seed, case] = row
    report = dict(complete=not missing, training_ready=len(training), training_expected=len(cells),
                  evaluation_ready=len(results), evaluation_expected=len(cells)*10, missing=missing)
    if missing:
        return report
    outcomes = []
    for scene in scenarios:
        for scale_index, scale in enumerate(P["evaluation_demand_scales"]):
            for arm in P["arms"]+["offline_only"]:
                source_scene = "calibrated" if arm == "offline_only" else scene
                for seed in P["training_seeds"]:
                    rows = [results[source_scene, arm, seed, c] for c in range(5*scale_index,5*scale_index+5)]
                    reference = [results[scene, "h2o_matched", seed, c] for c in range(5*scale_index,5*scale_index+5)]
                    assert [r["passengers"]["departed_passengers"] for r in rows] == [r["passengers"]["departed_passengers"] for r in reference]
                    cost = statistics.mean(r["generalized_seconds_per_departed_passenger"] for r in rows)
                    completion = statistics.mean(r["passengers"]["completion_rate"] for r in rows)
                    outcomes.append(dict(scenario=scene, demand_scale=scale, arm=arm, seed=seed,
                        cost=cost, completion=completion,
                        cost_minus_h2o=cost-statistics.mean(r["generalized_seconds_per_departed_passenger"] for r in reference),
                        completion_pp_minus_h2o=100*(completion-statistics.mean(r["passengers"]["completion_rate"] for r in reference)),
                        reserve_changed_fraction=statistics.mean(r["execution_actions"]["reserve_changed_fraction"] for r in rows)))
    report.update(training=training, outcomes=outcomes,
                  inference="Two development training seeds; no formal significance or independent confirmation claim")
    return report


if __name__ == "__main__":
    result = summarize()
    write(HERE/"summary.json", result)
    print(json.dumps({k:result[k] for k in ("complete", "training_ready", "evaluation_ready")}, indent=2))
