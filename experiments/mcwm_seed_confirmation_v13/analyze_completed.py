"""Aggregate completed resident evaluations without transferring rollout files."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean, median, stdev

from scipy.stats import t


METRICS = (
    "cost_seconds", "wait_seconds", "in_vehicle_seconds", "completion_percent",
    "bus_completion_percent", "headway_error", "bunching_percent", "large_gap_percent",
    "hold_seconds", "reserve_percent", "inference_ms", "evaluation_wall_seconds",
)


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def stats(values):
    n = len(values)
    average = mean(values)
    sd = stdev(values) if n > 1 else 0.
    radius = float(t.ppf(.975, n - 1)) * sd / n**.5 if n > 1 else None
    return dict(n=n, mean=average, sd=sd, median=median(values), minimum=min(values),
        maximum=max(values), ci95=None if radius is None else [average-radius, average+radius])


def metrics(row):
    passengers, headway = row["passengers"], row["headway"]
    return dict(cost_seconds=row["generalized_seconds_per_departed_passenger"],
        wait_seconds=passengers["accrued_wait_seconds_per_departed_passenger"],
        in_vehicle_seconds=passengers["accrued_in_vehicle_seconds_per_departed_passenger"],
        completion_percent=100*passengers["completion_rate"],
        bus_completion_percent=100*row["service"]["completion_rate"],
        headway_error=headway["normalized_absolute_error"]["mean"],
        bunching_percent=100*headway["bunching_rate"],
        large_gap_percent=100*headway["large_gap_rate"], hold_seconds=row["mean_hold_seconds"],
        reserve_percent=(100*row["execution_actions"]["reserve_changed_fraction"]
            if "execution_actions" in row else None),
        inference_ms=row.get("policy_inference_ms", {}).get("mean"),
        evaluation_wall_seconds=row["elapsed_seconds"])


def average(rows):
    return {key: mean(r[key] for r in rows if r[key] is not None)
        for key in METRICS if any(r[key] is not None for r in rows)}


def paired(first, second):
    return {key: first[key]-second[key] for key in first.keys() & second.keys()}


def aggregate_pairs(rows):
    costs = [row["cost_seconds"] for row in rows]
    return dict(cost_delta_seconds=stats(costs), better_seeds=sum(v < -1e-9 for v in costs),
        equal_seeds=sum(abs(v) <= 1e-9 for v in costs),
        completion_delta_pp=stats([r["completion_percent"] for r in rows]),
        wait_delta_seconds=stats([r["wait_seconds"] for r in rows]),
        in_vehicle_delta_seconds=stats([r["in_vehicle_seconds"] for r in rows]),
        seed_deltas=rows)


def policy_folder(root, protocol, scene, arm, seed):
    origin = root if arm in protocol["training_arms"] else root.parent/protocol["reference_experiment"]
    return origin/"runs"/scene/f"{arm}_seed{seed}"/"results"


def analyze(root):
    compact = load(root/"analysis_summary.json")
    if not compact["complete"]:
        raise ValueError("Only complete, protocol-validated campaigns can be analyzed")
    protocol = load(next((root/"runs").glob("*/*/results/protocol.json")))["protocol"]
    # The full protocol is already attached to every resident training result.
    seeds, arms = protocol["training_seeds"], protocol["arms"]
    scenes = [r["name"] for r in protocol["scenarios"]]
    cases, evaluation_rows, evaluation_metrics = range(20), {}, {}
    for scene in scenes:
        for arm in arms:
            for seed in seeds:
                for case in cases:
                    row = load(root/"runs"/scene/f"{arm}_seed{seed}"/"results"/f"eval_{case:02d}.json")
                    key = (scene, arm, seed, case)
                    evaluation_rows[key] = row["passengers"]["departed_passengers"]
                    evaluation_metrics[key] = metrics(row)

    rule_metrics, rule_rows = {}, {}
    for rule in protocol["rules"]:
        for case in cases:
            row = load(root/"rules"/rule/f"case{case:02d}"/"results"/f"case{case:02d}.json")
            rule_rows[rule, case] = row["passengers"]["departed_passengers"]
            rule_metrics[rule, case] = metrics(row)

    test = {(scene, arm, seed): average([evaluation_metrics[scene, arm, seed, case]
        for case in range(10, 20)]) for scene in scenes for arm in arms for seed in seeds}
    aggregate, comparisons = [], []
    controls = protocol.get("reference_arms", ["h2o_matched", "fixed0250", "fixed0300"])
    for scene in scenes:
        for arm in arms:
            rows = [dict(seed=seed, **test[scene, arm, seed]) for seed in seeds]
            aggregate.append(dict(scenario=scene, arm=arm, metrics={key:stats([r[key] for r in rows])
                for key in METRICS if key in rows[0]}, seeds=rows))
            for reference in controls + protocol["rules"]:
                if arm == reference:
                    continue
                differences = []
                for seed in seeds:
                    for case in range(10, 20):
                        control_count = (rule_rows[reference, case] if reference in protocol["rules"]
                            else evaluation_rows[scene, reference, seed, case])
                        if evaluation_rows[scene, arm, seed, case] != control_count:
                            raise ValueError("Compared methods must have the same departed passengers")
                    control = (average([rule_metrics[reference, case] for case in range(10, 20)])
                        if reference in protocol["rules"] else test[scene, reference, seed])
                    differences.append(dict(seed=seed, **paired(test[scene, arm, seed], control)))
                comparisons.append(dict(scenario=scene, arm=arm, reference=reference,
                    **aggregate_pairs(differences)))

    selected = compact["selection"]["selected"]
    selection_errors, selected_metrics = [], []
    choices = {(r["scenario"], r["arm"]):r["seed"] for r in selected}
    for scene in scenes:
        for arm in arms:
            scores = [(average([evaluation_metrics[scene, arm, seed, c] for c in range(10)])["cost_seconds"], seed)
                for seed in seeds]
            expected_cost, expected_seed = min(scores)
            seed = choices[scene, arm]
            if expected_seed != seed:
                selection_errors.append(dict(scenario=scene, arm=arm, frozen_seed=seed, expected_seed=expected_seed))
            selected_metrics.append(dict(scenario=scene, arm=arm, selected_seed=seed,
                development_cost_seconds=expected_cost, test=test[scene, arm, seed],
                test_seed_rank=1+sum(test[scene, arm, s]["cost_seconds"] < test[scene, arm, seed]["cost_seconds"]
                    for s in seeds),
                comparisons=[dict(reference=ref, **paired(test[scene, arm, seed],
                    average([rule_metrics[ref, c] for c in range(10, 20)]) if ref in protocol["rules"]
                    else test[scene, ref, choices[scene, ref]])) for ref in controls+protocol["rules"] if ref != arm]))

    diagnostics, timings = [], []
    for scene in scenes:
        for arm in arms:
            for seed in seeds:
                folder = policy_folder(root, protocol, scene, arm, seed)
                row, usage = load(folder/"ready.json"), load(folder/"coefficient_usage.json")
                timings.append(dict(scenario=scene, arm=arm, seed=seed,
                    origin_experiment=row.get("origin_experiment", root.name if arm in protocol["training_arms"]
                        else protocol["reference_experiment"]),
                    training_wall_seconds=row["training_wall_seconds"],
                    peak_host_rss_mb=row["peak_host_rss_mb"], phases=row["phases"],
                    model_families=[x["selected_family"] for x in row["model_history"]]))
                trace = load(folder/"context_trace.json")["trace"]
                entry = dict(scenario=scene, arm=arm, seed=seed, usage=usage,
                    blocks=len(trace), predicted_context_counts=dict(Counter(x["predicted_context"] for x in trace)),
                    accepted_blocks=sum(x["context_accepted"] for x in trace))
                path = folder/"gradient_gate_trace.json"
                if path.exists():
                    probes = load(path)["trace"]
                    up = [x for x in probes if x["requested_coefficient"] > protocol["global_coefficient"]]
                    cosines = [x["gradient_cosine"] for x in probes if x["gradient_cosine"] is not None]
                    entry["gradient_probes"] = dict(count=len(probes), upward_requested=len(up),
                        positive=sum(x["next_update_allowed"] for x in probes),
                        positive_when_upward=sum(x["next_update_allowed"] for x in up),
                        cosine=stats(cosines) if cosines else None)
                diagnostics.append(entry)

    report = dict(experiment=protocol["name"], analysis_location="server", complete=True,
        initialization_count=compact["initialization_ready"], training_count=compact["training_ready"],
        reused_control_count=compact["reference_ready"], evaluation_count=compact["evaluation_ready"],
        rule_count=compact["rule_ready"], protocol=dict(source_event_budget=protocol["source_event_budget"],
            offline_updates=protocol["offline_updates_in_initializer"], continuation_updates=20000,
            evaluation_horizon_seconds=protocol["evaluation_horizon_seconds"],
            demand_scales=protocol["evaluation_demand_scales"], scenarios=protocol["scenarios"]),
        statistic_scope="Paired training seeds, averaged over the ten fixed test cases; CI describes seed variability conditional on these days, not independent daily replicas or city generalization.",
        selection_errors=selection_errors, all_seed_metrics=aggregate, comparisons=comparisons,
        selected_metrics=selected_metrics,
        rule_metrics=[dict(rule=rule, metrics=average([rule_metrics[rule, c] for c in range(10, 20)]),
            by_demand=[dict(demand_scale=scale, **average([rule_metrics[rule, c]
                for c in range(10+5*i, 15+5*i)])) for i,scale in enumerate(protocol["evaluation_demand_scales"])])
            for rule in protocol["rules"]],
        diagnostics=diagnostics, timing_summary=[dict(arm=arm,
            wall_seconds=stats([r["training_wall_seconds"] for r in timings if r["arm"] == arm]),
            peak_rss_mb=stats([r["peak_host_rss_mb"] for r in timings if r["arm"] == arm]),
            model_family_counts=dict(Counter(f for r in timings if r["arm"] == arm for f in r["model_families"])))
            for arm in arms])
    write(root/"completed_analysis.json", report)
    write(root/"completed_timing_details.json", dict(records=timings))
    print(json.dumps(dict(experiment=protocol["name"], evaluation_count=len(evaluation_metrics),
        selection_errors=len(selection_errors), compact_bytes=(root/"completed_analysis.json").stat().st_size)))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    analyze(parser.parse_args().root.resolve())
