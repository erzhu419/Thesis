"""Small-file analysis of paired formal controls and common-stream predictions.

Run while the campaign is in progress to inspect coverage.  Missing and failed
runs remain explicit, and aggregate estimates carry the number of paired seeds.
Only protocol/summary/metrics/prequential JSON files are read; no checkpoint or
trajectory downloads are needed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


ARMS = ("replay", "online", "context")
MODELS = ("frozen_model_reference", "online", "context")
SCENARIOS = ("stable", "local_speed_drop")
PERIODS = {"before": (6000., 9000.), "early_after": (9000., 12000.),
           "late_after": (15000., 18000.), "all_after": (9000., 18000.)}
PHYSICAL_ERRORS = {
    "duration_absolute_error": "duration_absolute_error_seconds",
    "cost_absolute_error": "cost_absolute_error_generalized_passenger_seconds",
}


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def describe(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return dict(n=0, mean=None, sample_std=None, min=None, max=None)
    return dict(n=len(values), mean=statistics.mean(values),
        sample_std=statistics.stdev(values) if len(values) > 1 else None,
        min=min(values), max=max(values))


def _cost(point, component="generalized"):
    if component == "wait":
        return float(point["accrued_wait_seconds"])
    if component == "ride":
        return float(point["accrued_in_vehicle_seconds"])
    return 2*float(point["accrued_wait_seconds"])+float(point["accrued_in_vehicle_seconds"])


def paired_control(stable, shock, *, shock_at=9000., end=18000., step=300., tolerance=.1):
    """Compare fixed clock intervals with the same arm/seed's stable day."""
    stable_points = {float(row["time_seconds"]): row for row in stable}
    shock_points = {float(row["time_seconds"]): row for row in shock}
    bins = []
    for start in range(int(shock_at), int(end), int(step)):
        stop = start+step
        points = (stable_points.get(start), stable_points.get(stop),
                  shock_points.get(start), shock_points.get(stop))
        if any(point is None for point in points):
            bins.append(dict(start=start, end=stop, status="missing_endpoint",
                stable_rate=None, shock_rate=None, excess_rate=None,
                tolerance_rate=None, within_tolerance=None))
            continue
        stable_rate = (_cost(points[1])-_cost(points[0]))/step
        shock_rate = (_cost(points[3])-_cost(points[2]))/step
        excess = shock_rate-stable_rate
        bins.append(dict(start=start, end=stop, status="complete",
            stable_rate=stable_rate, shock_rate=shock_rate, excess_rate=excess,
            tolerance_rate=tolerance*stable_rate,
            within_tolerance=excess <= tolerance*stable_rate))

    result = dict(bins=bins, excess_generalized_passenger_seconds=None,
        excess_wait_seconds=None, excess_in_vehicle_seconds=None,
        recovery_status="missing_metrics", recovery_onset_seconds_after_shock=None,
        recovery_confirmed_seconds_after_shock=None,
        recovery_observation_limit_seconds=end-shock_at)
    if any(row["status"] != "complete" for row in bins):
        return result
    for component, field in (("generalized", "excess_generalized_passenger_seconds"),
            ("wait", "excess_wait_seconds"), ("ride", "excess_in_vehicle_seconds")):
        result[field] = ((_cost(shock_points[end], component)-_cost(shock_points[shock_at], component))
            - (_cost(stable_points[end], component)-_cost(stable_points[shock_at], component)))
    # First establish a visible early degradation.  An unchanged trajectory has
    # no recovery latency, rather than a fabricated successful zero-second one.
    initial = next((i for i, row in enumerate(bins)
        if row["start"] < shock_at+3000. and not row["within_tolerance"]), None)
    if initial is None:
        result["recovery_status"] = "no_detectable_early_degradation"
        return result
    result["recovery_status"] = "right_censored"
    for index in range(initial+1, len(bins)-2):
        if all(row["within_tolerance"] for row in bins[index:index+3]):
            result.update(recovery_status="recovered",
                recovery_onset_seconds_after_shock=bins[index]["start"]-shock_at,
                recovery_confirmed_seconds_after_shock=bins[index+2]["end"]-shock_at)
            break
    return result


def _prediction_means(rows):
    scores = [row["selected_score"] for row in rows]
    result = dict(n=len(scores), mean_score=statistics.mean(scores) if scores else None)
    for field, measure in PHYSICAL_ERRORS.items():
        values = [row[field] for row in rows if row.get(field) is not None]
        result[f"n_{measure}"] = len(values)
        result[f"mean_{measure}"] = statistics.mean(values) if values else None
    return result


def prediction_summary(rows):
    """Use the chosen family's saved prediction, selected before its label."""
    result = {}
    result["by_300_second_issue_bin"] = []
    usable = []
    for row in rows:
        # No fallback to the best family after observing the label.
        score = row.get("selected_score")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("A prediction record lacks a finite selected_score saved at issue")
        if row["completed_at"] > row["available_at"] or row["available_at"] > row["scored_at"]:
            raise ValueError("A prediction label was scored before completion/availability")
        usable.append(row)
    for name, (start, end) in PERIODS.items():
        result[name] = _prediction_means([row for row in usable if start <= row["issued_at"] < end])
    for start in range(6000, 18000, 300):
        values = [row for row in usable if start <= row["issued_at"] < start+300]
        result["by_300_second_issue_bin"].append(dict(start=start, end=start+300,
            **_prediction_means(values)))
    return result


def analyze_campaign(root, expected_seeds=5):
    root = Path(root)
    runs, controls, predictions, prediction_sequences = [], {}, {}, {}
    prediction_pair_issues = []
    for protocol_path in sorted(root.rglob("protocol.json")):
        protocol = json.loads(protocol_path.read_text())
        args = protocol.get("arguments", {})
        if args.get("mode") not in ("control", "prediction"):
            continue
        directory = protocol_path.parent
        summary_path, failure_path = directory/"summary.json", directory/"failure.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        failure = json.loads(failure_path.read_text()) if failure_path.exists() else None
        complete = (summary is not None and summary.get("status") == "complete"
                    and summary.get("full_horizon_completed") is True
                    and summary.get("target_time_seconds") == 24000.)
        entry = dict(path=str(directory), mode=args["mode"], seed=args.get("seed"),
            scenario=args.get("scenario"), arm=args.get("arm"),
            status="failed" if failure else "complete" if complete else "incomplete",
            failure=failure,
            target_time_seconds=summary.get("target_time_seconds") if summary else None,
            target_online_training_events=summary.get("target_online_training_events") if summary else None,
            policy_updates=summary.get("policy_updates") if summary else None,
            generalized_seconds_per_departed_passenger=(
                summary.get("generalized_seconds_per_departed_passenger") if summary else None))
        runs.append(entry)
        if not complete or failure:
            continue
        key = (entry["seed"], entry["scenario"])
        if entry["mode"] == "control":
            cell = (*key, entry["arm"])
            if cell in controls:
                raise ValueError(f"Duplicate complete control cell: {cell}")
            controls[cell] = dict(entry=entry, metrics=read_jsonl(directory/"metrics.jsonl"))
        else:
            logs = read_jsonl(directory/"prequential.jsonl")
            for model in MODELS:
                rows = [row for row in logs if row.get("model_arm") == model]
                cell = (*key, model)
                if cell in predictions:
                    raise ValueError(f"Duplicate complete prediction cell: {cell}")
                sequences = [row["sequence"] for row in rows]
                if not rows or len(sequences) != len(set(sequences)):
                    prediction_pair_issues.append(dict(seed=key[0], scenario=key[1],
                        model_arm=model, reason="empty or duplicate prediction sequences"))
                    continue
                predictions[cell] = prediction_summary(rows)
                prediction_sequences[cell] = set(sequences)
            available = [prediction_sequences.get((*key, model)) for model in MODELS]
            if any(value is None for value in available) or any(value != available[0] for value in available[1:]):
                prediction_pair_issues.append(dict(seed=key[0], scenario=key[1],
                    reason="the three models did not score identical common-stream sequences"))

    seeds = sorted({run["seed"] for run in runs if run["seed"] is not None})
    control_pairs = []
    missing_pairs = []
    for seed in seeds:
        for arm in ARMS:
            stable = controls.get((seed, "stable", arm))
            shock = controls.get((seed, "local_speed_drop", arm))
            if stable is None or shock is None:
                missing_pairs.append(dict(seed=seed, arm=arm,
                    missing=[name for name, value in (("stable", stable), ("local_speed_drop", shock))
                             if value is None]))
                continue
            control_pairs.append(dict(seed=seed, arm=arm,
                **paired_control(stable["metrics"], shock["metrics"])))
    grouped_control = []
    for arm in ARMS:
        pairs = [pair for pair in control_pairs if pair["arm"] == arm]
        grouped_control.append(dict(arm=arm,
            paired_seeds=len(pairs), expected_seeds=expected_seeds,
            excess_generalized_passenger_seconds=describe(
                [pair["excess_generalized_passenger_seconds"] for pair in pairs]),
            excess_wait_seconds=describe([pair["excess_wait_seconds"] for pair in pairs]),
            excess_in_vehicle_seconds=describe([pair["excess_in_vehicle_seconds"] for pair in pairs]),
            recovery_status_counts={status: sum(pair["recovery_status"] == status for pair in pairs)
                for status in ("recovered", "right_censored", "no_detectable_early_degradation", "missing_metrics")},
            recovery_confirmed_seconds_conditional_on_recovery=describe(
                [pair["recovery_confirmed_seconds_after_shock"] for pair in pairs])))

    control_effects = []
    for candidate, reference in (("online", "replay"), ("context", "replay"), ("context", "online")):
        per_seed = []
        for seed in seeds:
            rows = {pair["arm"]: pair for pair in control_pairs if pair["seed"] == seed}
            values = [rows.get(arm, {}).get("excess_generalized_passenger_seconds")
                      for arm in (candidate, reference)]
            per_seed.append(dict(seed=seed, delta_excess_generalized_passenger_seconds=(
                values[0]-values[1] if all(value is not None for value in values) else None)))
        control_effects.append(dict(candidate=candidate, reference=reference, per_seed=per_seed,
            aggregate=describe([row["delta_excess_generalized_passenger_seconds"] for row in per_seed])))

    prediction_cells = [dict(seed=seed, scenario=scenario, model_arm=model, **value)
        for (seed, scenario, model), value in sorted(predictions.items())]
    prediction_effects = []
    for scenario in SCENARIOS:
        for candidate, reference in (("online", "frozen_model_reference"),
                ("context", "frozen_model_reference"), ("context", "online")):
            for period in PERIODS:
                per_seed = []
                for seed in seeds:
                    cand, ref = predictions.get((seed, scenario, candidate)), predictions.get((seed, scenario, reference))
                    paired = (prediction_sequences.get((seed, scenario, candidate))
                              == prediction_sequences.get((seed, scenario, reference)))
                    values = [value[period]["mean_score"] if value else None for value in (cand, ref)]
                    delta = (values[0]-values[1] if paired and all(value is not None for value in values) else None)
                    effect = dict(seed=seed, delta_mean_score=delta,
                        relative_change_percent=100*delta/values[1] if delta is not None and values[1] > 0 else None)
                    for measure in PHYSICAL_ERRORS.values():
                        physical = [value[period][f"mean_{measure}"] if value else None
                                    for value in (cand, ref)]
                        observed_all = all(value and value[period][f"n_{measure}"] == value[period]["n"]
                                           for value in (cand, ref))
                        effect[f"delta_mean_{measure}"] = (
                            physical[0]-physical[1] if paired and observed_all
                            and all(value is not None for value in physical) else None)
                    per_seed.append(effect)
                prediction_effects.append(dict(scenario=scenario, candidate=candidate,
                    reference=reference, period=period, per_seed=per_seed,
                    delta_mean_score=describe([row["delta_mean_score"] for row in per_seed]),
                    relative_change_percent=describe([row["relative_change_percent"] for row in per_seed]),
                    **{f"delta_mean_{measure}": describe([row[f"delta_mean_{measure}"] for row in per_seed])
                       for measure in PHYSICAL_ERRORS.values()}))

    ready = (len(seeds) == expected_seeds and len(controls) == 6*expected_seeds
        and len(predictions) == 6*expected_seeds and not prediction_pair_issues
        and all(pair["recovery_status"] != "missing_metrics" for pair in control_pairs))
    return dict(status="complete" if ready else "incomplete", expected_seeds=expected_seeds,
        observed_seeds=seeds, runs=runs, complete_control_cells=len(controls),
        complete_prediction_cells=len(predictions), missing_control_pairs=missing_pairs,
        prediction_pair_issues=prediction_pair_issues,
        recovery_definition=dict(comparison="same arm and seed: local_speed_drop minus stable",
            window_seconds=[9000, 18000], bin_seconds=300, tolerance_fraction_of_stable_rate=.1,
            consecutive_bins=3, require_early_degradation_window=[9000, 12000],
            reported_latency="third consecutive qualifying bin end minus9000; null if no early degradation or censored"),
        control_pairs=control_pairs, control_aggregates=grouped_control,
        control_candidate_effects=control_effects, prediction_cells=prediction_cells,
        prediction_candidate_effects=prediction_effects,
        physical_prediction_semantics=dict(duration="absolute error in global control-transition duration, seconds",
            cost="absolute error in global control-transition 2*wait+ride, generalized passenger seconds",
            missing="absent observations remain null with an explicit count; paired physical deltas require complete period observations"),
        interpretation="Lower error and lower shock-minus-stable excess cost are favorable; scientific conclusions require complete paired seeds. The frozen model is an A1 diagnostic reference, not an A1 world-model component.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = analyze_campaign(args.root, args.expected_seeds)
    destination = args.out or args.root/"analysis.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: result[key] for key in
        ("status", "observed_seeds", "complete_control_cells", "complete_prediction_cells")}))


if __name__ == "__main__":
    main()
