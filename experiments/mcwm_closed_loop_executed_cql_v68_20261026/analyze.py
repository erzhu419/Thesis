"""Executed-action finite-candidate CQL: matched and original actual costs."""
from __future__ import annotations

import argparse
from copy import deepcopy
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
from experiments.mcwm_closed_loop_executed_cql_v68_20261026.learning import cql_checks as native_cql_checks

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


SCHEDULE = dict(online_actor_update_interval=1, online_actor_batch_mode="latest",
                online_update_order="actor-before-critic")
CQL_COEFFICIENT = .2
EXPECTED_COUNTS = dict(critic=456, value=456, actor=456, alpha=456,
                       policy=456, world=0, trust=0)


def schedule_checks(summary):
    value, drift = summary.get("actor_schedule_audit", {}), summary.get("policy_drift", {})
    budget, post = summary.get("budget", {}), summary.get("post_budget_counts", {})
    groups = drift.get("groups", [])
    return dict(original_critic_value_budget=value.get("packets") == 57
            and value.get("critic_updates") == value.get("value_updates") == 456
            and value.get("packet_critic_steps") == value.get("packet_value_steps") == [8]*57,
        original_actor_alpha_budget=value.get("actor_updates") == value.get("alpha_updates") == 456
            and value.get("packet_actor_steps") == value.get("packet_alpha_steps") == [8]*57,
        original_optimizer_order=value.get("optimizer_order_exact") is True and value.get("schedule") == SCHEDULE,
        no_training_kl=value.get("kl_penalty") is False
            and value.get("reference_for_diagnostics_only") is True,
        exact_update_horizon=budget.get("policy_updates") == budget.get("actor_updates") == budget.get("alpha_updates") == 456
            and budget.get("critic_updates") == budget.get("value_updates") == 456
            and budget.get("native_seconds") == 24000,
        post_budget_zero_training=post.get("before") == post.get("after") == EXPECTED_COUNTS,
        no_added_diagnostic_random_draws=drift.get("additional_random_draws") == 0,
        same_state_same_F_diagnostics=drift.get("same_states") is True and drift.get("same_F") is True,
        complete_policy_drift_coverage=drift.get("packets") == 57 and drift.get("states") == 285
            and sum(g.get("count", 0) for g in groups) == 285,
        finite_policy_drift=bool(groups) and all(g.get("count", 0) > 0
            and 0 <= g.get("gate_crossings", -1) <= g["count"]
            and all(isinstance(g.get(k), (int, float)) and math.isfinite(g[k])
                    for k in (*DRIFT_SUMS, "holding_abs_delta_max_seconds")) for g in groups))


CQL_HEAD_COLUMNS = ("head", "factual_q_mean", "candidate_q_mean", "logmeanexp_mean",
    "penalty_mean", "candidate_advantage_mean", "candidate_advantage_abs_mean",
    "deterministic_positive_minus_zero_mean")


def cql_checks(summary, config):
    value = summary.get("cql_audit", {})
    unique = value.get("unique_candidates_per_state", {})
    rows = value.get("factual_rows", 0)
    after = value.get("candidate_rows_after_dedup", 0)
    heads = value.get("by_head", [])
    finite = lambda x: isinstance(x, (int, float)) and math.isfinite(x)
    finite_heads = bool(heads) and all(all(finite(row.get(key))
        for key in CQL_HEAD_COLUMNS) for row in heads)
    finite_stats = all(all(finite(value.get(name, {}).get(key)) for key in ("mean", "min", "max"))
        for name in ("penalty", "weighted_penalty"))
    checks = {"cql_"+key: flag is True for key, flag in native_cql_checks(
        value, expected_calls=456, rows_per_call=128).items()} if value else {"cql_audit_present": False}
    checks.update(cql_fixed_coefficient=value.get("coefficient") == CQL_COEFFICIENT
            and value.get("temperature") == 1.,
        cql_actual_calls_and_covariate_rows=value.get("penalty_calls") == 456 and rows == 456*128,
        cql_F_candidates_deduplicated=value.get("proposals_per_state") == 22
            and value.get("candidate_rows_before_dedup") == rows*22
            and isinstance(after, int) and rows <= after <= rows*22
            and all(finite(unique.get(key)) for key in ("min", "max", "mean"))
            and 1 <= unique.get("min", 0) <= unique.get("mean", 0) <= unique.get("max", 0) <= 22
            and math.isclose(unique.get("mean", 0), after/rows if rows else 0., rel_tol=1e-9),
        cql_executed_action_scope=value.get("factual_actions_transformed") is False
            and value.get("candidate_actions_transformed") is True and value.get("outcome_fields_read") is False,
        cql_all_heads=value.get("all_heads") is True and finite_heads
            and [row.get("head") for row in heads] == list(range(config.get("ensemble_size", 0))),
        cql_finite_loss=value.get("finite_penalty") is True and finite_stats and finite_heads,
        cql_weight_applied=finite_stats and math.isclose(value["weighted_penalty"]["mean"],
            CQL_COEFFICIENT*value["penalty"]["mean"], rel_tol=1e-6, abs_tol=1e-7),
        cql_isolated_proposals_preserve_main_rng=value.get("main_rng_preserved") is True
            and value.get("numpy_draws") == 0)
    return checks


def cql_view(value):
    compact = lambda number: float(format(number, ".6g"))
    scalar_fields = ("coefficient", "temperature", "penalty_calls", "factual_rows", "proposals_per_state", "regularizer_wall_seconds",
        "candidate_rows_before_dedup", "candidate_rows_after_dedup", "main_rng_preserved", "numpy_draws",
        "isolated_uniform_proposals", "isolated_positive_proposals", "exact_deduplication",
        "all_heads", "outcome_fields_read", "factual_actions_transformed", "candidate_actions_transformed", "finite_penalty")
    result = {key: value.get(key) for key in scalar_fields}
    for key in ("unique_candidates_per_state", "penalty", "weighted_penalty"):
        result[key] = {name: compact(number) for name, number in value.get(key, {}).items()}
    result["head_rows"] = [[row.get("head"), *[compact(row[key]) for key in CQL_HEAD_COLUMNS[1:]]]
        for row in value.get("by_head", [])]
    return result


def same_policy_configs(configs):
    current = [configs[arm].get("policy_config") for arm in ARMS]
    if not all(isinstance(value, dict) for value in current) or current[0] != current[1]:
        return False
    if current[0].get("online_cql_alpha") != CQL_COEFFICIENT:
        return False
    if any(current[0].get(key) != value for key, value in SCHEDULE.items()):
        return False
    expected_old = dict(current[0], online_cql_alpha=0.)
    return all(configs[arm].get("policy_config") == expected_old
               for arm in ("original_fresh", "original_no_fresh"))


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
    for seed in seeds:
        for scene in scenarios:
            key = f"seed{seed}/{scene}"
            runs, points, dirs, configs = {}, {}, {}, {}
            reference = protocol["original_reference_by_seed"][str(seed)]
            old_root = root.parent/reference["directory"]/reference["relative"].format(seed=seed, scenario=scene)
            for arm in (*ARMS, "original_fresh", "original_no_fresh"):
                original = arm.startswith("original_")
                native_arm = arm.removeprefix("original_")
                directory = old_root/native_arm if original else root/"runs/executed_cql/development"/f"seed{seed}"/scene/arm
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
                    and summary.get("cell", {}).get("candidate") == ("fresh_value" if original else "executed_cql"),
                    external_seed=summary.get("cell", {}).get("external_seed") == seed+protocol["external_seed_offset"],
                    all_native_checks=bool(summary.get("integrity", {}).get("checks")) and all(
                        x is True for x in summary.get("integrity", {}).get("checks", {}).values()),
                    learner_config_recorded=isinstance(configs[arm].get("policy_config"), dict),
                    all_required_cost_endpoints=all(window(summary, points[arm], *bounds)["status"] == "complete" for _, bounds in PERIODS))
                final = {row["time_seconds"]: row for row in points[arm]}.get(24000, {})
                audit["final_metrics_match_ledger"] = all(final.get(field) == summary.get("cost", {}).get(field)
                    for field in ("accrued_wait_seconds", "accrued_in_vehicle_seconds"))
                if not original:
                    audit.update(schedule_checks(summary))
                    audit.update(cql_checks(summary, configs[arm].get("policy_config", {})))
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
                key+"/same_policy_config_except_online_cql_alpha": same_policy_configs(configs),
                key+"/unchanged_F_and_runtime": all(_runtime(fresh) == _runtime(x) for x in runs.values())})
            periods, comparisons = {}, {name: {} for name in COMPARISONS}
            for period, bounds in PERIODS:
                values = {arm: window(summary, points[arm], *bounds) for arm, summary in runs.items()}
                periods[period] = values
                for name, (left, right) in COMPARISONS.items(): comparisons[name][period] = difference(values[left], values[right])
            cells.append(dict(identity=seed, scenario=scene, periods=periods, comparisons=comparisons))
            evidence.append(dict(seed=seed, scenario=scene, original_reference=str(old_root), prefix=prefix,
                actual_parallel=parallel, policy_drift={arm: drift_view(runs[arm].get("policy_drift", {})) for arm in ARMS},
                actor_schedule={arm: {name: runs[arm].get("actor_schedule_audit", {}).get(name) for name in
                    ("packets", "critic_updates", "value_updates", "actor_updates", "alpha_updates", "schedule",
                     "optimizer_order_exact", "kl_penalty", "reference_for_diagnostics_only")} for arm in ARMS},
                executed_cql={arm: cql_view(runs[arm].get("cql_audit", {})) for arm in ARMS}))
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
        candidate="executed_cql", stage="development", complete=intact, expected_cells=expected, available_cells=len(cells),
        integrity=dict(all_pass=intact, checked_conditions=len(checks), failed_checks=[k for k, v in checks.items() if v is not True]),
        missing=missing, failures=failures, failed_run_work=failed_work, aggregates=stats,
        cql_head_columns=CQL_HEAD_COLUMNS,
        cql_diagnostics_scope="Running pre-update ensemble Q levels/advantages on legal finite candidates. Original critic TD diagnostics exclude this regularizer; they are not total critic gradients.",
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
        interpretation="6201/6203 are mechanism-development seeds, not independent confirmation. New no_fresh is rerun with the same executed-action CQL and original456 critic/V/actor/temperature updates; original controls are read-only absolute references. Finite-candidate conservative loss is a regularizer, not a normalized executed-action density objective. Recorded original TD diagnostics exclude the CQL gradient. Passive KL or action movement alone cannot pass actual-cost gates.")


MAX_BYTES = 48000


def split_summary(value):
    """Keep complete costs and move every line/window policy row to its own file."""
    main = deepcopy(value)
    policy = dict(candidate=value["candidate"], stage=value["stage"],
        policy_drift_columns=deepcopy(value["policy_drift_columns"]),
        policy_drift_precision=value["policy_drift_precision"], native_evidence=[])
    for row in main["native_evidence"]:
        policy["native_evidence"].append(dict(seed=row["seed"], scenario=row["scenario"],
                                             policy_drift=row.pop("policy_drift")))
    main["policy_drift_file"] = "policy_drift.json"
    return main, policy


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode()+b"\n"


def export(root, out):
    out = Path(out)
    cost, policy = split_summary(analyze(root))
    payloads = {out: encoded(cost), out.with_name("policy_drift.json"): encoded(policy)}
    for path, raw in payloads.items():
        if len(raw) > MAX_BYTES:
            raise ValueError(f"{path.name} exceeds {MAX_BYTES} bytes: {len(raw)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    for path, raw in payloads.items():
        with path.open("xb") as handle:
            handle.write(raw)
    return dict(status=cost["status"], stage=cost["stage"], decision=cost["decision"],
                files={path.name: len(raw) for path, raw in payloads.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(HERE, args.out), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__": main()
