"""Freeze leave-one-seed-out choices on development days; report test only."""

import json
from statistics import mean

from td_common import HERE, P, FIXED, CONTEXTS, write


def choose(development):
    mappings = []
    for consumer in P["training_seeds"]:
        providers = [s for s in P["training_seeds"] if s != consumer]
        cost = {(scene, arm): mean(development[scene, arm, seed, case]["generalized_seconds_per_departed_passenger"]
            for seed in providers for case in range(10)) for scene in CONTEXTS for arm in FIXED}
        best = lambda scene: min(FIXED, key=lambda a: (cost[scene, a], FIXED[a]))
        global_arm = min(FIXED, key=lambda a: (mean(cost[s, a] for s in CONTEXTS), FIXED[a]))
        mappings.append(dict(consumer_seed=consumer, provider_seeds=providers, global_arm=global_arm,
            context_arms={s: best(s) for s in CONTEXTS},
            development_costs={s: {a: cost[s, a] for a in FIXED} for s in CONTEXTS}, test_outcomes_used=False))
    return mappings


def summarize():
    trains, results, missing = [], {}, []
    for seed in P["training_seeds"]:
        for scene in CONTEXTS:
            for arm in FIXED:
                folder = HERE / "runs" / scene / f"{arm}_seed{seed}" / "results"
                path = folder / "ready.json"
                if path.exists():
                    r = json.loads(path.read_text())
                    runtime = json.loads((folder / "runtime.json").read_text())
                    assert (r["arm"], r["seed"], r["scenario"]) == (arm, seed, scene)
                    assert r["continuation_updates"] == 20000 and r["total_policy_updates"] == r["total_critic_updates"] == 70000
                    assert 25000 <= r["source_events"] < 25016 and r["target_online_training_events"] == 0
                    assert r["real_td_coefficient"] == .75 and r["final_model_coefficient"] == FIXED[arm]
                    assert P["expected_gpu_model"] in runtime["gpu_model"] and runtime["device"] == "cuda"
                    trains.append(dict(r, runtime=runtime))
                else:
                    missing.append(str(path.relative_to(HERE)))
                for case in range(20):
                    path = folder / f"eval_{case:02d}.json"
                    if not path.exists():
                        missing.append(str(path.relative_to(HERE)))
                        continue
                    r = json.loads(path.read_text())
                    split = "development" if case < 10 else "test"
                    assert (r["arm"], r["training_seed"], r["scenario"], r["case"], r["split"]) == (arm, seed, scene, case, split)
                    assert r["horizon_seconds"] == 24000 and r["deployment_hold_scale"] == 1.
                    assert r["demand_scale"] == P["evaluation_demand_scales"][(case % 10)//5]
                    assert all(r[k] == v for k, v in P[split+"_blocks"][case % 5].items())
                    assert r["sumo_version"][1] == "SUMO 1.25.0" and "execution_actions" in r
                    results[scene, arm, seed, case] = r
    report = dict(complete=not missing, training_ready=len(trains), training_expected=36,
        evaluation_ready=len(results), evaluation_expected=720, missing=missing)
    development = {k: r for k, r in results.items() if k[-1] < 10}
    mapping_path = HERE / "selection.json"
    if len(development) == 360 and not mapping_path.exists():
        write(mapping_path, choose(development))
    if missing:
        return report
    mappings = json.loads(mapping_path.read_text())
    outcomes = []
    for mapping in mappings:
        seed = mapping["consumer_seed"]
        for scene in CONTEXTS:
            arms = [(a, a) for a in FIXED] + [("selected_global", mapping["global_arm"]),
                ("selected_context", mapping["context_arms"][scene])]
            for role, arm in arms:
                for scale_index, scale in enumerate(P["evaluation_demand_scales"]):
                    case_ids = range(10+5*scale_index, 15+5*scale_index)
                    rows = [results[scene, arm, seed, c] for c in case_ids]
                    for reference_role, reference in [("fixed0250", "td0250"), ("selected_global", mapping["global_arm"])]:
                        controls = [results[scene, reference, seed, c] for c in case_ids]
                        assert [r["passengers"]["departed_passengers"] for r in rows] == [r["passengers"]["departed_passengers"] for r in controls]
                        deltas = [r["generalized_seconds_per_departed_passenger"]-b["generalized_seconds_per_departed_passenger"] for r, b in zip(rows, controls)]
                        outcomes.append(dict(seed=seed, scenario=scene, demand_scale=scale, role=role, arm=arm,
                            reference_role=reference_role, reference=reference, cost=mean(r["generalized_seconds_per_departed_passenger"] for r in rows),
                            delta_seconds=mean(deltas), paired_day_deltas=deltas,
                            completion_delta_pp=100*mean(r["passengers"]["completion_rate"]-b["passengers"]["completion_rate"] for r, b in zip(rows, controls))))
    report.update(training=trains, mappings=mappings, outcomes=outcomes, interpretation=P["interpretation"])
    return report


if __name__ == "__main__":
    r = summarize()
    write(HERE / "summary.json", r)
    print(json.dumps({k:r[k] for k in ["complete", "training_ready", "evaluation_ready"]}))
