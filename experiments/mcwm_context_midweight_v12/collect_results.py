"""Paired fixed-weight and gated-policy comparisons, kept separate by seed."""

import json
from statistics import mean

from midweight_common import HERE, P, FIXED, CONTEXTS, write


def summarize():
    training, results, missing, recognition = [], {}, [], []
    updates = P["source_event_budget"]-P["warmup_events"]
    total_updates = P["offline_updates_in_initializer"]+updates
    for seed in P["training_seeds"]:
        for scene in CONTEXTS:
            for arm in P["arms"]:
                folder = HERE / "runs" / scene / f"{arm}_seed{seed}" / "results"
                paths = [folder / name for name in ("ready.json", "runtime.json", "coefficient_usage.json")]
                if all(path.exists() for path in paths):
                    row, runtime, usage = [json.loads(path.read_text()) for path in paths]
                    assert (row["arm"], row["seed"], row["scenario"]) == (arm, seed, scene)
                    assert row["continuation_updates"] == usage["updates"] == updates
                    assert row["total_policy_updates"] == row["total_critic_updates"] == total_updates
                    assert row["target_online_training_events"] == 0 and row["real_td_coefficient"] == P["real_td_coefficient"]
                    assert P["source_event_budget"] <= row["source_events"] < P["source_event_budget"]+16
                    assert runtime["device"] == "cpu" and runtime["gpu_model"] is None
                    if arm in FIXED:
                        assert usage["minimum_coefficient"] == usage["maximum_coefficient"] == FIXED[arm]
                        assert abs(usage["update_mean_coefficient"]-FIXED[arm]) < 1e-10
                    else:
                        assert row["context_provider_seed"] != seed
                        trace = json.loads((folder / "context_trace.json").read_text())["trace"]
                        recognition.append(dict(seed=seed, scenario=scene, arm=arm, blocks=len(trace),
                            block_accuracy=mean(r["predicted_context"] == scene for r in trace),
                            accepted_fraction=mean(r["context_accepted"] for r in trace),
                            update_mean_coefficient=usage["update_mean_coefficient"]))
                    training.append(dict(row, runtime=runtime, coefficient_usage=usage))
                else:
                    missing.extend(str(path.relative_to(HERE)) for path in paths if not path.exists())
                for case in range(10, 20):
                    path = folder / f"eval_{case:02d}.json"
                    if not path.exists():
                        missing.append(str(path.relative_to(HERE)))
                        continue
                    row = json.loads(path.read_text())
                    assert (row["arm"], row["training_seed"], row["scenario"], row["case"], row["split"]) == (arm, seed, scene, case, "test")
                    assert row["horizon_seconds"] == P["evaluation_horizon_seconds"] and row["deployment_hold_scale"] == 1.
                    assert row["sumo_version"][1] == "SUMO 1.25.0"
                    assert row["demand_scale"] == P["evaluation_demand_scales"][(case-10)//5]
                    assert all(row[k] == v for k, v in P["test_blocks"][case % 5].items())
                    results[scene, arm, seed, case] = row
    count = len(P["training_seeds"])*len(CONTEXTS)*len(P["arms"])
    report = dict(complete=not missing, training_ready=len(training), training_expected=count,
        evaluation_ready=len(results), evaluation_expected=count*10, missing=missing,
        training=training, recognition=recognition, interpretation=P["interpretation"])
    if missing:
        return report
    outcomes = []
    for seed in P["training_seeds"]:
        for scene in CONTEXTS:
            for arm in P["arms"]:
                for reference in P["arms"]:
                    if arm == reference:
                        continue
                    for scale_index, scale in enumerate(P["evaluation_demand_scales"]):
                        cases = range(10+5*scale_index, 15+5*scale_index)
                        rows = [results[scene, arm, seed, c] for c in cases]
                        controls = [results[scene, reference, seed, c] for c in cases]
                        assert [r["passengers"]["departed_passengers"] for r in rows] == [r["passengers"]["departed_passengers"] for r in controls]
                        deltas = [r["generalized_seconds_per_departed_passenger"]-c["generalized_seconds_per_departed_passenger"] for r, c in zip(rows, controls)]
                        outcomes.append(dict(seed=seed, scenario=scene, arm=arm, reference=reference,
                            demand_scale=scale, mean_cost_seconds=mean(r["generalized_seconds_per_departed_passenger"] for r in rows),
                            delta_seconds=mean(deltas), paired_day_deltas=deltas, better_days=sum(d < 0 for d in deltas),
                            completion_delta_pp=100*mean(r["passengers"]["completion_rate"]-c["passengers"]["completion_rate"] for r, c in zip(rows, controls)),
                            wait_delta_seconds=mean(r["passengers"]["accrued_wait_seconds_per_departed_passenger"]-c["passengers"]["accrued_wait_seconds_per_departed_passenger"] for r, c in zip(rows, controls)),
                            in_vehicle_delta_seconds=mean(r["passengers"]["accrued_in_vehicle_seconds_per_departed_passenger"]-c["passengers"]["accrued_in_vehicle_seconds_per_departed_passenger"] for r, c in zip(rows, controls))))
    report["outcomes"] = outcomes
    return report


if __name__ == "__main__":
    report = summarize()
    write(HERE / "summary.json", report)
    print(json.dumps({k:report[k] for k in ("complete", "training_ready", "evaluation_ready")}))
