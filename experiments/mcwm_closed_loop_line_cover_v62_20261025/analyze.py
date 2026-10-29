"""Resident v62 actual-cost gates against matched and original no-fresh controls."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_control_campaign_v60_20261025.control_analyze import (
    _overlap, _runtime, aggregate, compare_prefix, difference, read, read_metrics, window)
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.analyze import _run_checks

ROLES = ("fresh", "no_fresh", "original_no_fresh")
PERIODS = (("full_day", (0, 24000)), ("early", (9000, 12000)), ("post_budget", (18300, 24000)))


def run_directory(root, protocol, stage, seed, scenario, role):
    if stage == "development" and role == "original_no_fresh":
        original = root.parent/protocol["original_no_fresh_development_directory"]
        return original/"runs"/"recent30"/"development"/f"seed{seed}"/scenario/"no_fresh"
    return root/"runs"/"line_cover"/stage/f"seed{seed}"/scenario/role


def coverage_checks(summary, focus_lines):
    receipts = summary.get("receipts", [])
    complete = len(receipts) == 57
    for receipt in receipts:
        coverage = receipt.get("line_cover", {})
        latest = coverage.get("focus_latest_ids", {})
        counts = coverage.get("selected_line_counts", {})
        selected = receipt.get("sequence_ids", [])
        complete &= (isinstance(latest, dict) and isinstance(counts, dict)
                     and set(latest).issubset(focus_lines)
                     and len(selected) == 5 and sum(counts.values()) == 5
                     and coverage.get("eligible_records", 0) >= 5
                     and all(value in selected and counts.get(line, 0) >= 1
                             for line, value in latest.items()))
    report = summary.get("feedback_selection", {})
    selected_counts = report.get("selected_record_counts_by_line", {})
    return dict(line_selection_native_audit=summary.get("integrity", {}).get("checks", {}).get(
                    "line_selection_complete") is True,
                coverage_receipts_complete=bool(complete),
                selected_feedback_counts285=sum(selected_counts.values()) == 285,
                present_focus_lines_covered=bool(report) and report.get("focus_available_packets")
                    == report.get("focus_covered_packets"))


def group(cells, comparator, period="full_day", *, rates=False):
    value = aggregate(cells, comparator, period)
    if not rates:
        return value
    outcomes = [cell["comparisons"][comparator][period] for cell in cells
                if cell["comparisons"][comparator][period]["status"] == "complete"]
    for outcome, name in (("win", "winning"), ("loss", "losing")):
        deltas = [row["candidate_minus_baseline"]["generalized_seconds_per_departed_passenger"]
                  for row in outcomes if row["outcome"] == outcome]
        value[name+"_fraction"] = len(deltas)/len(outcomes) if outcomes else None
        value[name+"_mean_cost_difference"] = sum(deltas)/len(deltas) if deltas else None
    return value


def coverage_evidence(summary):
    value = summary.get("feedback_selection", {})
    return {key: value.get(key) for key in ("eligible_record_counts_by_line", "selected_record_counts_by_line",
        "focus_available_packets", "focus_covered_packets", "label_age_seconds")}


def analyze(root, stage):
    root = Path(root)
    protocol = read(root/"protocol.json")
    seeds, scenarios = protocol[f"{stage}_seeds"], protocol["scenarios"]
    cells, missing, failures, failed_work, checks = [], [], [], [], {}
    for seed in seeds:
        for scenario in scenarios:
            key = f"seed{seed}/{scenario}"
            runs, configurations, points, sources, directories = {}, {}, {}, {}, {}
            for role in ROLES:
                directory = run_directory(root, protocol, stage, seed, scenario, role)
                reused = stage == "development" and role == "original_no_fresh"
                summary = read(directory/"summary.json")
                if summary is None:
                    failure = read(directory/"failure.json")
                    (failures if failure else missing).append(f"{key}/{role}")
                    if failure:
                        failed_work.append(dict(cell=f"{key}/{role}", reused=reused,
                            **{field: failure.get(field) for field in
                               ("type", "native_time", "consumed_budget", "wall_seconds")}))
                    continue
                actual_arm = "fresh" if role == "fresh" else "no_fresh"
                runs[role] = summary
                directories[role] = directory
                configurations[role] = read(directory/"protocol.json") or {}
                sources[role] = dict(reused=reused, recorded_arm=summary.get("cell", {}).get("arm"),
                    recorded_candidate=summary.get("cell", {}).get("candidate"))
                run_checks = _run_checks(summary, actual_arm)
                run_checks.update(correct_recorded_role=summary.get("cell", {}).get("arm") == actual_arm
                        and summary.get("cell", {}).get("seed") == seed
                        and summary.get("cell", {}).get("scenario") == scenario,
                    exact_external_seed=summary.get("cell", {}).get("external_seed")
                        == seed+protocol["external_seed_offset"],
                    native_all_checks=bool(summary.get("integrity", {}).get("checks")) and all(
                        value is True for value in summary.get("integrity", {}).get("checks", {}).values()),
                    learner_config_recorded=isinstance(configurations[role].get("policy_config"), dict))
                if role != "original_no_fresh":
                    run_checks.update(coverage_checks(summary, protocol["focus_lines"]))
                points[role] = read_metrics(directory, summary)
                final = {row["time_seconds"]: row for row in points[role]}.get(24000, {})
                run_checks["final_metrics_match_ledger"] = all(final.get(field) == summary.get("cost", {}).get(field)
                    for field in ("accrued_wait_seconds", "accrued_in_vehicle_seconds"))
                run_checks["full_primary_ledger_valid"] = window(summary, points[role], 0, 24000)["status"] == "complete"
                checks.update({f"{key}/{role}/{name}": value is True for name, value in run_checks.items()})
            if len(runs) != len(ROLES):
                continue
            comparisons, prefixes, parallel = {}, {}, {}
            for left_role, right_role in itertools.combinations(ROLES, 2):
                left, right = runs[left_role], runs[right_role]
                label = left_role+"_vs_"+right_role
                if left_role == "fresh":
                    first = [value.get("clock_audit", {}).get("first_feedback_applied_at") for value in (left, right)]
                    valid_first = all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0 for value in first)
                    checks[f"{key}/{label}/first_feedback_exact"] = valid_first and first[0] == first[1]
                    left_dir, right_dir = directories[left_role], directories[right_role]
                    prefixes[label] = compare_prefix(left_dir/"prefix_trace.jsonl", right_dir/"prefix_trace.jsonl",
                        min(first) if valid_first else 0.)
                    checks[f"{key}/{label}/prefix_exact"] = prefixes[label]["all_pass"]
                    checks[f"{key}/{label}/initializer_exact"] = left.get("initializer") == right.get("initializer")
                    checks[f"{key}/{label}/learner_config_exact"] = (
                        configurations[left_role].get("policy_config") == configurations[right_role].get("policy_config"))
                    checks[f"{key}/{label}/same_runtime_F"] = _runtime(left) == _runtime(right)
                require_parallel = not (sources[left_role]["reused"] or sources[right_role]["reused"])
                parallel[label] = dict(required=require_parallel, reused_baseline=not require_parallel,
                    **(_overlap(left, right) if require_parallel else dict(positive_overlap=None, seconds=None)))
                if require_parallel:
                    checks[f"{key}/{label}/actual_parallel"] = parallel[label]["positive_overlap"]
            for comparator in ROLES[1:]:
                comparisons[comparator] = {name: difference(window(runs["fresh"], points["fresh"], *bounds),
                    window(runs[comparator], points[comparator], *bounds)) for name, bounds in PERIODS}
                for comparison in comparisons[comparator].values():
                    if comparison["status"] == "complete":
                        # The full-day cohort denominators are recorded once
                        # in the cell ledgers and reused by every period.
                        for role in ("candidate", "baseline"):
                            comparison[role].pop("departed_passengers")
            cells.append(dict(identity=seed, scenario=scenario, comparisons=comparisons,
                sources=sources, pre_feedback_prefix=prefixes, actual_parallel=parallel,
                initializer=runs["fresh"].get("initializer"),
                cost_ledgers={role: value.get("cost") for role, value in runs.items()},
                feedback_selection={role: coverage_evidence(value) for role, value in runs.items()
                                    if role != "original_no_fresh"}))
    expected = len(seeds)*len(scenarios)
    intact = not missing and not failures and len(cells) == expected and bool(checks) and all(checks.values())
    aggregates = {comparator: dict(overall=group(cells, comparator, rates=True),
        by_scenario={scenario: group([cell for cell in cells if cell["scenario"] == scenario], comparator)
                     for scenario in scenarios},
        by_identity={str(seed): group([cell for cell in cells if cell["identity"] == seed], comparator)
                     for seed in seeds},
        early=group(cells, comparator, "early"), post_budget=group(cells, comparator, "post_budget"))
        for comparator in ROLES[1:]}
    passed = intact and all(value["overall"]["mean_cost_difference"] < 0
        and value["by_scenario"]["stable"]["mean_cost_difference"] <= 0
        and value["by_scenario"]["local_speed_drop"]["mean_cost_difference"] < 0 for value in aggregates.values())
    return dict(status="complete" if intact else "integrity_failed" if failures or any(
        value is not True for value in checks.values()) else "pending", candidate="line_cover", stage=stage,
        complete=intact, expected_cells=expected, available_cells=len(cells),
        integrity=dict(all_pass=intact, checked_conditions=len(checks),
            failed_checks=[key for key, value in checks.items() if value is not True]),
        missing=missing, failures=failures, failed_run_work=failed_work, cells=cells, aggregates=aggregates,
        data_sources=dict(current_root=str(root), layout="runs/line_cover/{stage}/seed{seed}/{scenario}/{role}",
            reused_original_root=str(root.parent/protocol["original_no_fresh_development_directory"]),
            reused_original_layout="runs/recent30/development/seed{seed}/{scenario}/no_fresh"),
        reused_original_baseline_days=len(cells) if stage == "development" else 0,
        decision=dict(advance_to_confirmation=bool(passed and stage == "development"),
            confirmed_control_improvement=bool(passed and stage == "confirmation"),
            ordinary_dripping_goal_ready_for_root_review=bool(passed and stage == "confirmation"),
            ordinary_dripping_goal_achieved=False, scientific_gate_pass=bool(passed),
            reason="matched_and_original_cost_gates_pass" if passed else
                   "incomplete_or_integrity_failed" if not intact else "candidate_closed_cost_gate_failed"),
        metric="Full-day(2W+R)/all_departed_passengers including unfinished-trip time; negative is better",
        interpretation="Both fixed comparator gates are required. Reused development baselines need no cross-batch overlap. Confirmation is evidence for root review, not an automatic goal completion.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("development", "confirmation"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    value = analyze(HERE, args.stage)
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(raw.encode()) > 48000:
        raise ValueError("Control summary exceeds48000 bytes")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        handle.write(raw+"\n")
    print(json.dumps({key: value[key] for key in ("status", "stage", "decision")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
