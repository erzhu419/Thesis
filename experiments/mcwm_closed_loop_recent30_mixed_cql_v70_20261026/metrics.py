"""V70 actual-cost orchestration with explicit current/historical target audits."""
from __future__ import annotations

import math
from pathlib import Path

from experiments.mcwm_closed_loop_executed_cql_v68_20261026.analyze import (
    ARMS, COMPARISONS, CQL_HEAD_COLUMNS, DRIFT_COLUMNS, PERIODS,
    _overlap, _run_checks, _runtime, compare_prefix, compact_comparison,
    cost_gate, cost_view, cql_checks, cql_view, difference, drift_view, encoded,
    group, read, read_metrics, same_policy_configs, schedule_checks, split_summary,
    statistics, target_checks as historical_target_checks, window)
from experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026.learning import (
    target_checks as mixed_target_checks)

HERE = Path(__file__).resolve().parent


def target_checks(summary, arm, *, historical=False):
    if historical:
        return historical_target_checks(summary, arm)
    checks = mixed_target_checks(summary.get("target_audit", {}), arm, 456)
    checks["native_mixed_target_scope"] = summary.get("integrity", {}).get("checks", {}).get(
        "fresh_mixed_target_scope") is True
    return checks


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
                audit = _run_checks(summary, native_arm); audit.update(target_checks(summary, native_arm, historical=original))
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
        interpretation="6201/6203 are exposed development seeds, not independent confirmation. Current fresh uses original mixed backup; historical fresh-V references retain their original audits. New no_fresh is rerun with the same executed-action CQL and original456 critic/V/actor/temperature updates; original controls are read-only absolute references. Finite-candidate conservative loss is a regularizer, not a normalized executed-action density objective. Recorded original TD diagnostics exclude the CQL gradient. Passive KL or action movement alone cannot pass actual-cost gates.")

