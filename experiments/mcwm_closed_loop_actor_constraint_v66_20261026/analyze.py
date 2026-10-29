"""Fixed actor-KL development costs against matched and original controls."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_control_campaign_v60_20261025.control_analyze import (
    _overlap, _runtime, aggregate, compare_prefix, difference, read, read_metrics, window)
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.analyze import _run_checks
from experiments.mcwm_closed_loop_fresh_value_v64_20261025.analyze import target_checks

ARMS = ("fresh", "no_fresh")
PERIODS = (("full_day", (0, 24000)), ("early", (9000, 12000)), ("post_budget", (18300, 24000)))
COMPARISONS = {
    "matched": ("fresh", "no_fresh"),
    "original_no_fresh": ("fresh", "original_no_fresh"),
    "original_fresh": ("fresh", "original_fresh"),
    "no_fresh_shift": ("no_fresh", "original_no_fresh"),
}
DRIFT_SUMS = ("gate_kl_sum", "weighted_positive_kl_sum", "total_kl_sum",
              "holding_delta_sum_seconds", "holding_abs_delta_sum_seconds")
DRIFT_COLUMNS = ("line", "window", "states", "gate_crossings", "gate_crossing_fraction",
                 "mean_gate_kl", "mean_weighted_positive_kl", "mean_total_kl",
                 "mean_holding_delta_seconds", "mean_abs_holding_delta_seconds", "max_abs_holding_delta_seconds")


def cost_view(value):
    if value.get("status") != "complete":
        return dict(status=value.get("status", "missing"))
    return dict(wait=value["accrued_wait_seconds"], ride=value["accrued_in_vehicle_seconds"],
                cost=value["generalized_passenger_seconds"], departed=value["departed_passengers"],
                cost_per_passenger=value["generalized_seconds_per_departed_passenger"])


def compact_comparison(value):
    if value.get("status") != "complete":
        return dict(status=value.get("status", "missing"))
    return dict(wait_per_passenger=value["wait_seconds_per_passenger_difference"],
                ride_per_passenger=value["ride_seconds_per_passenger_difference"],
                cost_per_passenger=value["candidate_minus_baseline"]["generalized_seconds_per_departed_passenger"])


def group(cells, comparator, period, expected):
    result = aggregate(cells, comparator, period)
    valid = [cell["comparisons"][comparator][period] for cell in cells
             if cell["comparisons"][comparator][period]["status"] == "complete"]
    deltas = [row["candidate_minus_baseline"]["generalized_seconds_per_departed_passenger"] for row in valid]
    wins, losses = [x for x in deltas if x < 0], [x for x in deltas if x > 0]
    result.update(expected_cells=expected, complete_cells=len(valid), missing_cells=expected-len(valid),
                  win_fraction=len(wins)/len(valid) if valid else None,
                  loss_fraction=len(losses)/len(valid) if valid else None,
                  winning_mean_reduction=-sum(wins)/len(wins) if wins else None,
                  losing_mean_increase=sum(losses)/len(losses) if losses else None,
                  worst_cost_difference=max(deltas) if deltas else None)
    # Absolute four-arm W/R/J/N values are retained once per cell/window.
    # Repeating their derived means for four comparisons consumes the pull cap.
    for key in ("status", "cells", "raw_weighted_passenger_seconds_difference",
                "candidate_mean_cost", "baseline_mean_cost"):
        result.pop(key, None)
    return result


def statistics(cells, protocol):
    expected = len(protocol["development_seeds"])*len(protocol["scenarios"])
    return {comparator: {name: dict(overall=group(cells, comparator, name, expected),
                by_scenario={scene: group([c for c in cells if c["scenario"] == scene], comparator,
                    name, len(protocol["development_seeds"])) for scene in protocol["scenarios"]})
                for name, _ in PERIODS} for comparator in COMPARISONS}


def cost_gate(value):
    full = value["full_day"]
    groups = [full["overall"], *full["by_scenario"].values()]
    if not all(x["complete_cells"] == x["expected_cells"] and x["complete_cells"] > 0 for x in groups):
        return False
    return (full["overall"]["mean_cost_difference"] < 0
            and full["by_scenario"]["stable"]["mean_cost_difference"] <= 0
            and full["by_scenario"]["local_speed_drop"]["mean_cost_difference"] < 0)


def constraint_checks(summary, coefficient):
    value, drift = summary.get("actor_constraint_audit", {}), summary.get("policy_drift", {})
    groups = drift.get("groups", [])
    return dict(constraint_coefficient=value.get("coefficient") == coefficient,
        packet_reference_and_update_budget=value.get("packets") == value.get("reference_refreshes") == 57
            and value.get("updates") == value.get("actor_updates") == value.get("alpha_updates") == 456
            and value.get("packet_step_counts") == [8]*57,
        fixed_reference_complete_hurdle=value.get("packet_reference_fixed") is True
            and value.get("full_hurdle_kl_scope") is True
            and value.get("kl_orientation") == "new||packet_reference",
        no_added_random_draws=value.get("additional_random_draws") == 0
            and drift.get("additional_random_draws") == 0,
        same_state_same_F_diagnostics=drift.get("same_states") is True and drift.get("same_F") is True,
        complete_policy_drift_coverage=drift.get("packets") == 57 and drift.get("states") == 285
            and sum(g.get("count", 0) for g in groups) == 285,
        finite_policy_drift=bool(groups) and all(g.get("count", 0) > 0
            and 0 <= g.get("gate_crossings", -1) <= g["count"]
            and all(isinstance(g.get(k), (int, float)) and math.isfinite(g[k])
                    for k in (*DRIFT_SUMS, "holding_abs_delta_max_seconds")) for g in groups))


def drift_view(value):
    # Only diagnostic means are compacted; passenger ledgers and cost gates
    # retain their original numeric precision.
    compact = lambda number: float(format(number, ".6g"))
    rows = []
    for row in value.get("groups", []):
        n = row["count"]
        rows.append([row["line"], row["window"], n, row["gate_crossings"], compact(row["gate_crossings"]/n),
                     *[compact(row[key]/n) for key in DRIFT_SUMS], compact(row["holding_abs_delta_max_seconds"])])
    return dict(states=value.get("states"), packets=value.get("packets"), rows=rows)


def analyze(root=HERE):
    root = Path(root); protocol = read(root/"protocol.json")
    seeds, scenarios = protocol["development_seeds"], protocol["scenarios"]
    cells, evidence, checks, missing, failures, failed_work = [], [], {}, [], [], []
    coefficient = protocol["actor_kl_coefficient"]
    for seed in seeds:
        for scene in scenarios:
            key = f"seed{seed}/{scene}"
            runs, points, dirs, configs = {}, {}, {}, {}
            reference = protocol["original_reference_by_seed"][str(seed)]
            old_root = root.parent/reference["directory"]/reference["relative"].format(seed=seed, scenario=scene)
            for arm in (*ARMS, "original_fresh", "original_no_fresh"):
                original = arm.startswith("original_")
                native_arm = arm.removeprefix("original_")
                directory = old_root/native_arm if original else root/"runs/actor_constrained/development"/f"seed{seed}"/scene/arm
                summary = read(directory/"summary.json")
                if summary is None:
                    failure = read(directory/"failure.json")
                    (failures if failure else missing).append(key+"/"+arm)
                    if failure: failed_work.append(dict(cell=key+"/"+arm, **{field: failure.get(field)
                        for field in ("type", "native_time", "consumed_budget", "wall_seconds")}))
                    continue
                runs[arm], dirs[arm] = summary, directory
                configs[arm] = read(directory/"protocol.json") or {}
                points[arm] = read_metrics(directory, summary)
                audit = _run_checks(summary, native_arm); audit.update(target_checks(summary, native_arm))
                audit.update(correct_cell=summary.get("cell", {}).get("seed") == seed
                    and summary.get("cell", {}).get("scenario") == scene
                    and summary.get("cell", {}).get("arm") == native_arm
                    and summary.get("cell", {}).get("candidate") == ("fresh_value" if original else "actor_constrained"),
                    external_seed=summary.get("cell", {}).get("external_seed") == seed+protocol["external_seed_offset"],
                    all_native_checks=bool(summary.get("integrity", {}).get("checks")) and all(
                        x is True for x in summary.get("integrity", {}).get("checks", {}).values()),
                    learner_config_recorded=isinstance(configs[arm].get("policy_config"), dict),
                    all_required_cost_endpoints=all(window(summary, points[arm], *bounds)["status"] == "complete" for _, bounds in PERIODS))
                final = {row["time_seconds"]: row for row in points[arm]}.get(24000, {})
                audit["final_metrics_match_ledger"] = all(final.get(field) == summary.get("cost", {}).get(field)
                    for field in ("accrued_wait_seconds", "accrued_in_vehicle_seconds"))
                if not original: audit.update(constraint_checks(summary, coefficient))
                checks.update({key+"/"+arm+"/"+name: value is True for name, value in audit.items()})
            if len(runs) != 4: continue
            fresh, control = runs["fresh"], runs["no_fresh"]
            first = [runs[arm].get("clock_audit", {}).get("first_feedback_applied_at") for arm in runs]
            valid_first = all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in first)
            stop = min(first) if valid_first else 0.
            prefix = {arm: compare_prefix(dirs["fresh"]/"prefix_trace.jsonl", dirs[arm]/"prefix_trace.jsonl", stop)
                      for arm in ("no_fresh", "original_no_fresh")}
            parallel = _overlap(fresh, control)
            checks.update({key+"/first_feedback_exact": valid_first and len(set(first)) == 1,
                key+"/common_prefix_exact": all(x["all_pass"] for x in prefix.values()),
                key+"/actual_parallel": parallel["positive_overlap"],
                key+"/same_initializer": all(fresh.get("initializer") == x.get("initializer") for x in runs.values()),
                key+"/same_policy_config": all(configs["fresh"].get("policy_config") == x.get("policy_config") for x in configs.values()),
                key+"/unchanged_F_and_runtime": all(_runtime(fresh) == _runtime(x) for x in runs.values())})
            periods, comparisons = {}, {name: {} for name in COMPARISONS}
            for period, bounds in PERIODS:
                values = {arm: window(summary, points[arm], *bounds) for arm, summary in runs.items()}
                periods[period] = values
                for name, (left, right) in COMPARISONS.items(): comparisons[name][period] = difference(values[left], values[right])
            cells.append(dict(identity=seed, scenario=scene, periods=periods, comparisons=comparisons))
            evidence.append(dict(seed=seed, scenario=scene, original_reference=str(old_root), prefix=prefix,
                actual_parallel=parallel, policy_drift={arm: drift_view(runs[arm].get("policy_drift", {})) for arm in ARMS},
                actor_constraint={arm: {name: runs[arm].get("actor_constraint_audit", {}).get(name) for name in
                    ("coefficient", "packets", "reference_refreshes", "updates", "actor_updates", "alpha_updates",
                     "packet_reference_fixed", "full_hurdle_kl_scope", "kl_orientation", "additional_random_draws")} for arm in ARMS}))
    expected = len(seeds)*len(scenarios)
    intact = not missing and not failures and len(cells) == expected and bool(checks) and all(checks.values())
    stats = statistics(cells, protocol)
    matched, absolute = cost_gate(stats["matched"]), cost_gate(stats["original_no_fresh"])
    old_fresh = stats["original_fresh"]
    full_old = old_fresh["full_day"]["overall"]["mean_cost_difference"]
    early_old = old_fresh["early"]["overall"]["mean_cost_difference"]
    actual_worst = stats["original_no_fresh"]["full_day"]["overall"]["worst_cost_difference"]
    old_deltas = [cell["periods"]["full_day"]["original_fresh"]["generalized_seconds_per_departed_passenger"]
        - cell["periods"]["full_day"]["original_no_fresh"]["generalized_seconds_per_departed_passenger"] for cell in cells
        if all(cell["periods"]["full_day"][arm]["status"] == "complete" for arm in ("original_fresh", "original_no_fresh"))]
    original_worst = max(old_deltas) if old_deltas else None
    stable_tail = stats["original_no_fresh"]["post_budget"]["by_scenario"]["stable"]["mean_cost_difference"]
    optimization = dict(full_day_not_worse_than_original_fresh=full_old is not None and full_old <= 0,
        early_lower_than_original_fresh=early_old is not None and early_old < 0,
        worst_cell_not_worse=actual_worst is not None and original_worst is not None and actual_worst <= original_worst,
        stable_tail_not_higher_than_original_no_fresh=stable_tail is not None and stable_tail <= 0)
    passed = intact and matched and absolute and all(optimization.values())
    return dict(status="complete" if intact else "integrity_failed" if failures or any(x is not True for x in checks.values()) else "pending",
        candidate="actor_constrained", stage="development", complete=intact, expected_cells=expected, available_cells=len(cells),
        integrity=dict(all_pass=intact, checked_conditions=len(checks), failed_checks=[k for k, v in checks.items() if v is not True]),
        missing=missing, failures=failures, failed_run_work=failed_work, aggregates=stats,
        cells=[dict(identity=c["identity"], scenario=c["scenario"], periods={name: dict(
            arms={arm: cost_view(value) for arm, value in c["periods"][name].items()},
            comparisons={comparator: compact_comparison(c["comparisons"][comparator][name]) for comparator in COMPARISONS})
            for name, _ in PERIODS}) for c in cells], native_evidence=evidence, policy_drift_columns=DRIFT_COLUMNS,
        policy_drift_precision="Six significant digits for diagnostic means/maxima only; all passenger costs retain source precision.",
        optimization_assessment=dict(conditions=optimization, original_worst_full_day_difference=original_worst,
            candidate_worst_full_day_difference=actual_worst, stable_tail_difference=stable_tail),
        decision=dict(advance_to_confirmation=bool(passed), scientific_gate_pass=bool(passed),
            matched_cost_gate_pass=bool(intact and matched), original_reference_gate_pass=bool(intact and absolute),
            optimization_gate_pass=bool(intact and all(optimization.values())), ordinary_dripping_goal_achieved=False,
            reason="fixed_cost_gates_pass" if passed else "incomplete_or_integrity_failed" if not intact else "candidate_closed_cost_gate_failed"),
        units="Raw waiting/riding passenger-seconds; total=2W+R; per-passenger windows divide by the full-day departed cohort.",
        interpretation="6201/6203 are mechanism-development seeds, not independent confirmation. New no_fresh is rerun with the same KL constraint; original controls are read-only absolute references. KL or action movement alone cannot pass actual-cost gates.")


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); value = analyze(HERE)
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(raw.encode()) > 48000: raise ValueError("Actor-constraint summary exceeds48000 bytes")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle: handle.write(raw+"\n")
    print(json.dumps({key: value[key] for key in ("status", "stage", "decision")}, ensure_ascii=False))


if __name__ == "__main__": main()
