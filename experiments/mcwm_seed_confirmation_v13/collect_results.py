"""Keep all training seeds; freeze deployment selection using development days only."""

import json
from collections import defaultdict
from statistics import mean, stdev

from confirmation_common import HERE, P, CONTEXTS, write, initializer_folder, policy_folder


def result_path(scene, arm, seed, case, smoke=False):
    return HERE / ("smoke" if smoke else "runs") / scene / f"{arm}_seed{seed}" / "results" / f"eval_{case:02d}.json"


def select_development():
    path = HERE / "selection.json"
    if path.exists():
        return json.loads(path.read_text())
    chosen = []
    for scene in CONTEXTS:
        for arm in P["arms"]:
            candidates = []
            for seed in P["training_seeds"]:
                paths = [result_path(scene, arm, seed, case) for case in range(10)]
                if not all(p.exists() for p in paths):
                    return None
                rows = [json.loads(p.read_text()) for p in paths]
                assert all(r["split"] == "development" for r in rows)
                score = mean(r["generalized_seconds_per_departed_passenger"] for r in rows)
                candidates.append((score, seed))
            score, seed = min(candidates)
            chosen.append(dict(scenario=scene, arm=arm, seed=seed, development_cost_seconds=score))
    selection = dict(selected=chosen, candidates_per_method=len(P["training_seeds"]),
        cases=list(range(10)), criterion="Mean development generalized passenger time; lowest seed breaks ties",
        test_results_used=False)
    write(path, selection)
    return selection


def validate_evaluation(row, *, scene, arm, seed, case, smoke=False):
    assert (row["arm"], row["training_seed"], row["scenario"], row["case"]) == (arm, seed, scene, case)
    assert row["split"] == ("development" if case < 10 else "test")
    assert row["horizon_seconds"] == (1200 if smoke else P["evaluation_horizon_seconds"])
    assert row["deployment_hold_scale"] == 1. and row["sumo_version"][1] == "SUMO 1.25.0"
    local = case % 10
    assert row["demand_scale"] == P["evaluation_demand_scales"][local // 5]
    block = P["development_blocks" if case < 10 else "test_blocks"][local % 5]
    assert all(row[k] == v for k, v in block.items())


def paired(rows, controls):
    assert [r["passengers"]["departed_passengers"] for r in rows] == [r["passengers"]["departed_passengers"] for r in controls]
    deltas = [r["generalized_seconds_per_departed_passenger"]-c["generalized_seconds_per_departed_passenger"] for r, c in zip(rows, controls)]
    return dict(delta_seconds=mean(deltas), paired_day_deltas=deltas, better_days=sum(d < 0 for d in deltas),
        completion_delta_pp=100*mean(r["passengers"]["completion_rate"]-c["passengers"]["completion_rate"] for r, c in zip(rows, controls)))


def summarize(smoke=False):
    seeds = P["training_seeds"][:1] if smoke else P["training_seeds"]
    scenes = CONTEXTS[-1:] if smoke else CONTEXTS
    cases = [0, 10] if smoke else range(20)
    missing, initializers, training, references, evaluations, rules = [], [], [], [], {}, {}
    for seed in seeds:
        path = initializer_folder(seed, smoke) / "ready.json"
        if path.exists():
            row = json.loads(path.read_text())
            assert row["fresh_initialization"] and row["seed"] == seed and row["online_events"] == 0
            assert row["offline_updates"] == (64 if smoke else P["offline_updates_in_initializer"])
            initializers.append(row)
        else:
            missing.append(str(path))
    for seed in seeds:
        for scene in scenes:
            for arm in P["arms"]:
                folder = policy_folder(scene, arm, seed, smoke) / "results"
                paths = [folder / name for name in ("ready.json", "runtime.json", "coefficient_usage.json")]
                if all(path.exists() for path in paths):
                    row, runtime, usage = [json.loads(path.read_text()) for path in paths]
                    assert (row["arm"], row["seed"], row["scenario"], row["initializer_seed"]) == (arm, seed, scene, seed)
                    assert row["fresh_initialization"] and row["target_online_training_events"] == 0
                    expected = 64 if smoke else 20000
                    assert row["continuation_updates"] == usage["updates"] == expected
                    assert row["total_policy_updates"] == row["total_critic_updates"] == expected+(64 if smoke else 50000)
                    budget = 192 if smoke else P["source_event_budget"]
                    assert budget <= row["source_events"] < budget+16
                    assert row["real_td_coefficient"] == (1. if arm == "h2o_matched" else .75)
                    assert runtime["device"] == "cpu" and runtime["gpu_model"] is None
                    if arm in P["fixed_coefficients"]:
                        coefficient = P["fixed_coefficients"][arm]
                        assert usage["minimum_coefficient"] == usage["maximum_coefficient"] == coefficient
                        assert abs(usage["update_mean_coefficient"]-coefficient) < 1e-10
                    entry = dict(row, runtime=runtime, coefficient_usage=usage,
                        origin_experiment=P.get("reference_experiment") if arm not in P["training_arms"] else HERE.name)
                    (training if arm in P["training_arms"] else references).append(entry)
                else:
                    missing.extend(str(p) for p in paths if not p.exists())
                for case in cases:
                    path = result_path(scene, arm, seed, case, smoke)
                    if path.exists():
                        row = json.loads(path.read_text())
                        validate_evaluation(row, scene=scene, arm=arm, seed=seed, case=case, smoke=smoke)
                        evaluations[scene, arm, seed, case] = row
                    else:
                        missing.append(str(path))
    for rule in P["rules"]:
        for case in cases:
            path = HERE / ("rule_smoke" if smoke else "rules") / rule / f"case{case:02d}/results/case{case:02d}.json"
            if path.exists():
                row = json.loads(path.read_text())
                assert row["comparison_protocol"] == P["name"] and row["case"] == case
                assert row["horizon_seconds"] == (1200 if smoke else P["evaluation_horizon_seconds"])
                assert row["split"] == ("development" if case < 10 else "test")
                assert row["deployment_hold_scale"] == 1. and row["sumo_version"][1] == "SUMO 1.25.0"
                block = P["development_blocks" if case < 10 else "test_blocks"][case % 5]
                assert all(row[k] == v for k, v in block.items())
                assert row["demand_scale"] == P["evaluation_demand_scales"][(case % 10)//5]
                rules[rule, case] = row
            else:
                missing.append(str(path))
    report = dict(complete=not missing, initialization_ready=len(initializers), initialization_expected=len(seeds),
        training_ready=len(training), training_expected=len(seeds)*len(scenes)*len(P["training_arms"]),
        reference_ready=len(references), reference_expected=len(seeds)*len(scenes)*(len(P["arms"])-len(P["training_arms"])),
        evaluation_ready=len(evaluations), evaluation_expected=len(seeds)*len(scenes)*len(P["arms"])*len(cases),
        rule_ready=len(rules), rule_expected=len(P["rules"])*len(cases), missing=missing,
        initializers=initializers, training=training, references=references, interpretation=P["interpretation"])
    if smoke:
        return report
    selection = select_development()
    report["selection"] = selection
    if missing:
        return report
    comparisons = []
    controls = P.get("reference_arms", ["h2o_matched", "fixed0250", "fixed0300"])
    for scene in scenes:
        for arm in P["arms"]:
            for reference in controls:
                if arm == reference:
                    continue
                for seed in seeds:
                    for scale_index, scale in enumerate(P["evaluation_demand_scales"]):
                        day_cases = range(10+5*scale_index, 15+5*scale_index)
                        rows = [evaluations[scene, arm, seed, c] for c in day_cases]
                        controls_rows = [evaluations[scene, reference, seed, c] for c in day_cases]
                        comparisons.append(dict(scenario=scene, arm=arm, reference=reference, seed=seed,
                            demand_scale=scale, mean_cost_seconds=mean(r["generalized_seconds_per_departed_passenger"] for r in rows),
                            **paired(rows, controls_rows)))
    aggregates = []
    for scene in scenes:
        for arm in P["arms"]:
            for scale_index, scale in enumerate(P["evaluation_demand_scales"]):
                costs = [mean(evaluations[scene, arm, seed, case]["generalized_seconds_per_departed_passenger"]
                    for case in range(10+5*scale_index, 15+5*scale_index)) for seed in seeds]
                aggregates.append(dict(scenario=scene, arm=arm, demand_scale=scale, training_seeds=len(seeds),
                    mean_cost_seconds=mean(costs), sd_cost_seconds=stdev(costs), seed_costs=costs))
    selected_tests = []
    chosen = {(r["scenario"], r["arm"]):r["seed"] for r in selection["selected"]}
    for scene in scenes:
        for arm in P["arms"]:
            seed = chosen[scene, arm]
            for reference in controls+P["rules"]:
                if arm == reference:
                    continue
                reference_seed = chosen.get((scene, reference))
                for scale_index, scale in enumerate(P["evaluation_demand_scales"]):
                    day_cases = range(10+5*scale_index, 15+5*scale_index)
                    rows = [evaluations[scene, arm, seed, c] for c in day_cases]
                    controls_rows = [rules[reference, c] if reference in P["rules"] else
                        evaluations[scene, reference, reference_seed, c] for c in day_cases]
                    selected_tests.append(dict(scenario=scene, arm=arm, selected_seed=seed,
                        reference=reference, reference_selected_seed=reference_seed, demand_scale=scale,
                        **paired(rows, controls_rows)))
    report.update(paired_seed_outcomes=comparisons, all_seed_test_outcomes=aggregates,
        development_selected_test_outcomes=selected_tests)
    return report


def compact_report(report, smoke=False):
    fields = ("complete", "initialization_ready", "initialization_expected", "training_ready",
        "training_expected", "reference_ready", "reference_expected", "evaluation_ready",
        "evaluation_expected", "rule_ready", "rule_expected", "interpretation", "selection",
        "all_seed_test_outcomes", "development_selected_test_outcomes")
    result = {k:report[k] for k in fields if k in report}
    result.update(experiment=P["name"], smoke=smoke, analysis_location="server",
        missing_count=len(report["missing"]), missing_examples=report["missing"][:5])
    progress = []
    for seed in (P["training_seeds"][:1] if smoke else P["training_seeds"]):
        path = initializer_folder(seed, smoke) / "progress.json"
        if path.exists():
            row = json.loads(path.read_text())
            progress.append({k:row[k] for k in ("seed", "offline_updates", "elapsed_seconds")})
    result["initialization_progress"] = progress
    groups = defaultdict(list)
    for row in report.get("paired_seed_outcomes", []):
        groups[tuple(row[k] for k in ("scenario", "arm", "reference", "demand_scale"))].append(row)
    result["paired_seed_summary"] = [dict(zip(("scenario", "arm", "reference", "demand_scale"), key),
        training_seeds=len(rows), mean_delta_seconds=mean(r["delta_seconds"] for r in rows),
        sd_delta_seconds=stdev(r["delta_seconds"] for r in rows) if len(rows) > 1 else 0.,
        better_seeds=sum(r["delta_seconds"] < 0 for r in rows),
        seed_deltas=[dict(seed=r["seed"], delta_seconds=r["delta_seconds"],
            completion_delta_pp=r["completion_delta_pp"]) for r in rows]) for key, rows in groups.items()]
    return result


def server_analysis(smoke=False):
    report = summarize(smoke)
    prefix = "smoke_" if smoke else ""
    write(HERE / (prefix+"summary.json"), report)
    init_fields = ("seed", "fresh_initialization", "offline_updates", "online_events")
    policy_fields = ("arm", "seed", "scenario", "initializer_seed", "fresh_initialization",
        "continuation_updates")
    handoff = dict(experiment=P["name"], smoke=smoke,
        initializers=[{k:r[k] for k in init_fields} for r in report["initializers"]],
        policies=[{k:r[k] for k in policy_fields} for r in report["training"]+report["references"]])
    write(HERE / (prefix+"handoff_status.json"), handoff)
    compact = compact_report(report, smoke)
    write(HERE / (prefix+"analysis_summary.json"), compact)
    print(json.dumps({k:compact[k] for k in ("complete", "training_ready", "evaluation_ready")}))
    return compact


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    server_analysis(parser.parse_args().smoke)
