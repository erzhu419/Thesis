"""Fixed eight-day capacity-feedback trial: passenger costs decide admission."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026 import metrics as old
from experiments.mcwm_closed_loop_recent30_cql_v69_20261026.learning import recency_checks
from experiments.mcwm_closed_loop_tail_policy_exchange_v71_20261026.analyze import waiting_comparison

MAX_BYTES = 48000
ARMS = ("fresh", "no_fresh")
PERIODS = old.PERIODS
COMPARISONS = {
    "matched": ("fresh", "no_fresh"),
    "previous_no_fresh": ("fresh", "v70_no_fresh"),
    "previous_fresh": ("fresh", "v70_fresh"),
    "no_fresh_shift": ("no_fresh", "v70_no_fresh"),
    "original_no_fresh": ("fresh", "original_no_fresh"),
}


def statistics(cells, protocol):
    expected = len(protocol["development_seeds"])*len(protocol["scenarios"])
    return {comparison: {period: dict(overall=old.group(cells, comparison, period, expected),
        by_scenario={scene: old.group([c for c in cells if c["scenario"] == scene],
            comparison, period, len(protocol["development_seeds"]))
            for scene in protocol["scenarios"]}) for period, _ in PERIODS}
        for comparison in COMPARISONS}


def scientific_gates(cells, stats, protocol):
    """Keep per-cell stable tails and both absolute/matched worst cases binding."""
    expected = len(protocol["development_seeds"])
    stable = [cell for cell in cells if cell["scenario"] == "stable"]
    tails = {comparison: len(stable) == expected and all(
        row["comparisons"][comparison]["post_budget"]["status"] == "complete"
        and row["comparisons"][comparison]["post_budget"]["candidate_minus_baseline"]
            ["generalized_seconds_per_departed_passenger"] <= 0 for row in stable)
        for comparison in ("matched", "previous_no_fresh")}
    cap = protocol["worst_full_day_cost_difference_cap"]
    worst = {name: stats[name]["full_day"]["overall"]["worst_cost_difference"]
             for name in ("matched", "previous_no_fresh")}
    previous_fresh = stats["previous_fresh"]["full_day"]["overall"]
    return dict(matched_full_day=old.cost_gate(stats["matched"]),
        previous_no_fresh_full_day=old.cost_gate(stats["previous_no_fresh"]),
        stable_tail_matched=tails["matched"],
        stable_tail_previous_no_fresh=tails["previous_no_fresh"],
        preserves_previous_fresh=previous_fresh["complete_cells"] == previous_fresh["expected_cells"]
            and previous_fresh["complete_cells"] > 0 and previous_fresh["mean_cost_difference"] <= 0,
        matched_worst_cell=worst["matched"] is not None and worst["matched"] <= cap,
        previous_no_fresh_worst_cell=worst["previous_no_fresh"] is not None
            and worst["previous_no_fresh"] <= cap)


def native_checks(summary, config, *, seed, scene, arm, current, historical=False):
    result = old._run_checks(summary, arm)
    result.update(old.target_checks(summary, arm, historical=historical))
    if not historical:
        result.update(old.schedule_checks(summary))
        result.update(old.cql_checks(summary, config.get("policy_config", {})))
        result.update({"recency/"+key: flag for key, flag in
            recency_checks(summary.get("replay", {}), arm, 456, seed).items()})
    cell = summary.get("cell", {})
    result.update(correct_cell=cell.get("seed") == seed and cell.get("scenario") == scene
            and cell.get("arm") == arm and cell.get("candidate") ==
                ("fresh_value" if historical else "executed_cql"),
        all_native_checks=bool(summary.get("integrity", {}).get("checks")) and all(
            flag is True for flag in summary.get("integrity", {}).get("checks", {}).values()))
    if current:
        from experiments.mcwm_closed_loop_capacity_feedback_v72_20261026.selection import (
            selection_checks, summarize_selection)
        receipts = summary.get("receipts", [])
        selection_present = bool(receipts) and all(isinstance(row.get("capacity_selection"), dict)
            for row in receipts)
        result["selection/receipts_recorded"] = selection_present
        if selection_present:
            result.update({"selection/"+key: flag for key, flag in selection_checks(receipts).items()})
            result["selection/reported_aggregate_exact"] = summary.get("capacity_selection") == summarize_selection(receipts)
    return result


def coverage_view(summary):
    """Aggregate all observed lines/windows, never choose a line from outcomes."""
    groups = {}
    for receipt in summary.get("receipts", []):
        record = receipt.get("capacity_selection", {})
        deadline = receipt.get("deadline")
        window = "pre9000" if deadline < 9000 else "9000-15000" if deadline < 15000 else "15000-18000"
        for field in ("candidate_line_counts", "risk_candidate_line_counts", "selected_line_counts", "risk_selected_line_counts"):
            for line, count in record.get(field, {}).items():
                key = (line, window)
                group = groups.setdefault(key, Counter())
                group[field] += count
    return dict(aggregate=summary.get("capacity_selection"),
        columns=["line", "window", "candidates", "risk_candidates", "selected", "risk_selected"],
        rows=[[line, window, *[groups[line, window][name] for name in
            ("candidate_line_counts", "risk_candidate_line_counts", "selected_line_counts", "risk_selected_line_counts")]]
            for line, window in sorted(groups)])


def service_view(directory):
    """Read existing resident observations; unavailable service data remain missing."""
    value = old.read(directory/"service_summary.json")
    if value is None:
        return dict(status="service_summary_missing"), None
    final = value.get("final", {})
    people = final.get("waiting_passengers", [])
    metadata = [row for row in people if row.get("metadata_observed") is True]
    last_groups = value.get("final_waiting_groups", [])
    final_summary = dict(waiting_unique=final.get("waiting_unique"),
        ledger_waiting=final.get("ledger_waiting"),
        inventory_complete=final.get("waiting_inventory_complete"),
        inventory_matches_ledger=final.get("waiting_matches_ledger"),
        metadata_missing=len(people)-len(metadata),
        completed_legs_counts=dict(Counter(str(row.get("completed_legs")) for row in metadata)),
        at_recorded_transfer_stop=sum(row.get("station_id") == row.get("transfer_stop") for row in metadata),
        compatible_last_departure_missing=sum(group.get("last_compatible_service", {}).get("departure") is None
            for group in last_groups))
    path = directory/"service_trace.jsonl"
    counts = Counter()
    if path.is_file():
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("record") != "service" or row.get("kind") != "departure":
                    continue
                counts["departure_observations"] += 1
                observed = all(isinstance(row.get(key), (int, float)) for key in
                    ("onboard", "capacity", "eligible_waiting", "native_stranded_count"))
                if not observed:
                    counts["departure_inventory_missing"] += 1
                    continue
                counts["full_and_queue_departures"] += row["onboard"] >= row["capacity"] and row["eligible_waiting"] > 0
                counts["native_stranded_count_sum"] += row["native_stranded_count"]
    result = dict(status="complete", final=final_summary,
        service_event_counts=value.get("service_event_counts"), missing_observations=value.get("missing_observations"),
        departures=dict(counts) if path.is_file() else None,
        interpretation="Departure stranding is an event-count sum, not unique people or per-person welfare. Completed legs and recorded transfer stop describe the final waiting cohort only.")
    return result, value


def passenger_comparison(left, right):
    if left is None or right is None:
        return dict(status="service_summary_missing", membership_difference=None)
    result = waiting_comparison(left.get("final"), right.get("final"))
    result.pop("clock_note", None)
    # Keep complete population counts; station details remain resident when there are many groups.
    result.pop("left_only_station_groups", None)
    result.pop("right_only_station_groups", None)
    return result


def analyze(root=HERE):
    root = Path(root)
    protocol = old.read(root/"protocol.json")
    cells, evidence, extra, checks = [], [], [], {}
    missing, failures, failed_work = [], [], []
    for seed in protocol["development_seeds"]:
        for scene in protocol["scenarios"]:
            key = f"seed{seed}/{scene}"
            current = root/"runs/executed_cql/development"/f"seed{seed}"/scene
            ref = protocol["previous_reference"]
            previous = root.parent/ref["directory"]/ref["relative"].format(seed=seed, scenario=scene)
            ref = protocol["original_reference_by_seed"][str(seed)]
            original = root.parent/ref["directory"]/ref["relative"].format(seed=seed, scenario=scene)
            dirs = {"fresh": current/"fresh", "no_fresh": current/"no_fresh",
                "v70_fresh": previous/"fresh", "v70_no_fresh": previous/"no_fresh",
                "original_no_fresh": original/"no_fresh"}
            runs, configs, points = {}, {}, {}
            for label, directory in dirs.items():
                summary = old.read(directory/"summary.json")
                if summary is None:
                    failure = old.read(directory/"failure.json")
                    (failures if failure else missing).append(key+"/"+label)
                    if failure:
                        failed_work.append(dict(cell=key+"/"+label, **{name: failure.get(name) for name in
                            ("type", "native_time", "consumed_budget", "wall_seconds")}))
                    continue
                arm = "no_fresh" if label.endswith("no_fresh") else "fresh"
                runs[label] = summary
                configs[label] = old.read(directory/"protocol.json") or {}
                points[label] = old.read_metrics(directory, summary)
                audit = native_checks(summary, configs[label], seed=seed, scene=scene, arm=arm,
                    current=label in ARMS, historical=label.startswith("original_"))
                audit["cost_endpoints"] = all(old.window(summary, points[label], *bounds)["status"] == "complete"
                    for _, bounds in PERIODS)
                audit["external_seed"] = summary.get("cell", {}).get("external_seed") == seed+protocol["external_seed_offset"]
                audit["fixed_departed_cohort"] = summary.get("cost", {}).get("departed_passengers") == protocol["full_day_departed_passengers"]
                final = {row["time_seconds"]: row for row in points[label]}.get(24000, {})
                audit["final_metrics_match_ledger"] = all(final.get(field) == summary.get("cost", {}).get(field)
                    for field in ("accrued_wait_seconds", "accrued_in_vehicle_seconds"))
                if label in ARMS:
                    audit["recipe"] = summary.get("cell", {}).get("recipe") == protocol["recipe"]
                checks.update({key+"/"+label+"/"+name: flag is True for name, flag in audit.items()})
            if any(label not in runs for label in dirs):
                continue
            fresh, control = runs["fresh"], runs["no_fresh"]
            first = [row.get("clock_audit", {}).get("first_feedback_applied_at") for row in runs.values()]
            valid_first = all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in first)
            prefix = {label: old.compare_prefix(dirs["fresh"]/"prefix_trace.jsonl", dirs[label]/"prefix_trace.jsonl",
                min(first) if valid_first else 0.) for label in ("no_fresh", "v70_fresh", "v70_no_fresh")}
            parallel = old._overlap(fresh, control)
            checks.update({key+"/first_feedback_exact": valid_first and len(set(first)) == 1,
                key+"/common_prefix_exact": all(row["all_pass"] for row in prefix.values()),
                key+"/actual_parallel": parallel["positive_overlap"],
                key+"/same_initializer": all(fresh.get("initializer") == row.get("initializer") for row in runs.values()),
                key+"/unchanged_F_and_runtime": all(old._runtime(fresh) == old._runtime(row) for row in runs.values()),
                key+"/same_v70_learner_config": all(configs[label].get("policy_config") == configs["fresh"].get("policy_config")
                    for label in ("no_fresh", "v70_fresh", "v70_no_fresh"))})
            periods, comparisons = {}, {name: {} for name in COMPARISONS}
            for period, bounds in PERIODS:
                periods[period] = {label: old.window(row, points[label], *bounds) for label, row in runs.items()}
                for comparison, (left, right) in COMPARISONS.items():
                    comparisons[comparison][period] = old.difference(periods[period][left], periods[period][right])
            cells.append(dict(identity=seed, scenario=scene, periods=periods, comparisons=comparisons))
            evidence.append(dict(seed=seed, scenario=scene, prefix=prefix, actual_parallel=parallel,
                policy_drift={arm: old.drift_view(runs[arm].get("policy_drift", {})) for arm in ARMS},
                executed_cql={arm: old.cql_view(runs[arm].get("cql_audit", {})) for arm in ARMS},
                additional_CQL_wall_seconds={arm: runs[arm].get("cql_audit", {}).get("regularizer_wall_seconds") for arm in ARMS},
                native_wall_seconds={arm: runs[arm].get("wall_seconds") for arm in ARMS}))
            service, resident = {}, {}
            for arm in ARMS:
                service[arm], resident[arm] = service_view(dirs[arm])
            passenger_pairs = dict(matched=passenger_comparison(resident["fresh"], resident["no_fresh"]))
            if scene == "stable":
                v71 = root.parent/"mcwm_closed_loop_tail_policy_exchange_v71_20261026"/"runs"/f"seed{seed}"
                for arm, branch in (("fresh", "low"), ("no_fresh", "high")):
                    previous_service = old.read(v71/arm/"suffixes"/branch/"service_summary.json")
                    passenger_pairs["fresh_minus_v70_"+arm] = passenger_comparison(resident["fresh"], previous_service)
            extra.append(dict(seed=seed, scenario=scene,
                capacity_coverage={arm: coverage_view(runs[arm]) for arm in ARMS},
                recency={arm: runs[arm].get("replay") for arm in ARMS},
                passengers={arm: runs[arm].get("passengers") for arm in ARMS},
                services=service, final_waiting_membership=passenger_pairs))
    stats = statistics(cells, protocol)
    expected = len(protocol["development_seeds"])*len(protocol["scenarios"])
    technical = len(cells) == expected and not missing and not failures and bool(checks) and all(checks.values())
    conditions = scientific_gates(cells, stats, protocol)
    passed = technical and all(conditions.values())
    status = "complete" if technical else "integrity_failed" if failures or any(not x for x in checks.values()) else "pending"
    decision = dict(scientific_gate_pass=bool(passed), advance_to_confirmation=False,
        eligible_for_root_review=bool(passed), automatic_confirmation=False, ordinary_dripping_goal_achieved=False,
        matched_cost_gate_pass=bool(technical and conditions["matched_full_day"]),
        previous_reference_gate_pass=bool(technical and conditions["previous_no_fresh_full_day"]),
        reason="development_pass_requires_root_review" if passed else "incomplete_or_integrity_failed" if not technical
            else "candidate_closed_cost_gate_failed")
    main = dict(status=status, complete=bool(technical), candidate="executed_cql", recipe=protocol["recipe"],
        stage="development", expected_cells=expected, available_cells=len(cells), missing=missing, failures=failures,
        failed_run_work=failed_work,
        integrity=dict(all_pass=bool(technical), checked_conditions=len(checks), failed_checks=[name for name, flag in checks.items() if not flag]),
        aggregates=stats,
        cells=[dict(identity=cell["identity"], scenario=cell["scenario"], periods={period: dict(
            arms={label: old.cost_view(value) for label, value in cell["periods"][period].items()},
            comparisons={label: old.compact_comparison(value[period]) for label, value in cell["comparisons"].items()})
            for period, _ in PERIODS}) for cell in cells], native_evidence=evidence,
        policy_drift_columns=old.DRIFT_COLUMNS, policy_drift_precision="Six significant digits for passive diagnostics only; exact costs retained.",
        cql_head_columns=old.CQL_HEAD_COLUMNS,
        optimization_assessment=dict(conditions=conditions,
            worst_full_day_cost_difference_cap=protocol["worst_full_day_cost_difference_cap"]),
        decision=decision, recency_summary_file="recency.json",
        acceptance_protocol=dict(early_window="descriptive_only", old_v64_v65="descriptive_only",
            old_results_unchanged=True, no_fresh_full_trajectory_v70_exactness_required=False),
        budget=dict(planned_native_days=8, planned_native_seconds=192000, planned_feedback_slots=2280,
            planned_updates_per_optimizer=3648, new_initialization_training=0, native_diagnostic_branches=0),
        units="Passenger seconds W/R; J=(2W+R)/full-day departed cohort including unfinished journeys.",
        interpretation="One fixed capacity-state selection rule. No_fresh uses the same rule and can change relative to v70. New matching and immutable v70 controls both bind. Exposed development only; no automatic confirmation or tuning.")
    auxiliary = dict(status=status, recipe=protocol["recipe"], cells=extra,
        interpretation="All-population passenger costs are primary. Final-waiting exclusive IDs show changed membership, not per-person cost beneficiaries or harms. Missing observations stay missing. Old22 identities never select or train this model.",
        prior_code_and_results_modified=False, new_native_branches=0)
    return main, auxiliary


def export(root, out):
    main, auxiliary = analyze(root)
    cost, policy = old.split_summary(main)
    out = Path(out)
    payloads = {out: old.encoded(cost), out.with_name("policy_drift.json"): old.encoded(policy),
        out.with_name("recency.json"): old.encoded(auxiliary)}
    for path, raw in payloads.items():
        if len(raw) > MAX_BYTES:
            raise ValueError(f"{path.name} exceeds48KB: {len(raw)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    for path, raw in payloads.items():
        with path.open("xb") as handle:
            handle.write(raw)
    return dict(status=cost["status"], decision=cost["decision"], files={path.name: len(raw) for path, raw in payloads.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(HERE, args.out), allow_nan=False))
