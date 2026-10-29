"""Fixed decoupled-pool trial; cached control costs and full no-fresh regression."""
from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import zip_longest
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026 import metrics as old
from experiments.mcwm_closed_loop_capacity_feedback_v72_20261026.analyze import (
    coverage_view, passenger_comparison, service_view)
from experiments.mcwm_closed_loop_recent30_cql_v69_20261026.learning import recency_checks

MAX_BYTES = 48000
PERIODS = old.PERIODS
COMPARISONS = dict(matched=("fresh", "v70_no_fresh"),
    previous_fresh=("fresh", "v70_fresh"), capacity_fresh=("fresh", "v72_fresh"))


def run_dir(root, seed, scene, arm):
    return Path(root)/"runs/executed_cql/development"/f"seed{seed}"/scene/arm


def reference_dir(root, protocol, key, seed, scene, arm):
    value = protocol[key]
    return Path(root).parent/value["directory"]/value["relative"].format(seed=seed, scenario=scene)/arm


def native_checks(summary, config, *, seed, scene, arm, current=False):
    checks = old._run_checks(summary, arm)
    checks.update(old.target_checks(summary, arm))
    checks.update(old.schedule_checks(summary))
    checks.update(old.cql_checks(summary, config.get("policy_config", {})))
    checks.update({"recency/"+key: value for key, value in
        recency_checks(summary.get("replay", {}), arm, 456, seed).items()})
    cell = summary.get("cell", {})
    checks.update(correct_cell=cell.get("seed") == seed and cell.get("scenario") == scene
        and cell.get("arm") == arm and cell.get("candidate") == "executed_cql",
        all_native_checks=bool(summary.get("integrity", {}).get("checks")) and all(
            value is True for value in summary.get("integrity", {}).get("checks", {}).values()))
    if current:
        from experiments.mcwm_closed_loop_decoupled_feedback_v73_20261026.selection import (
            selection_checks, summarize_selection)
        receipts = summary.get("receipts", [])
        checks.update({"selection/"+key: value for key, value in selection_checks(receipts).items()})
        checks["selection/reported_aggregate_exact"] = summary.get("capacity_selection") == summarize_selection(receipts)
        cov = {identity for row in receipts for identity in row["covariate_sequence_ids"]}
        outcomes = {identity for row in receipts for identity in row["outcome_sequence_ids"]}
        audit = summary.get("replay", {})
        fresh = arm == "fresh"
        runtime = summary.get("runtime_contract", {})
        checks.update(actor_value_state_rule=audit.get("fresh_current_s_a_use") ==
                ["IQL value", "SAC actor", "temperature", "executed CQL"]
                and runtime.get("actor_value_covariates") == "fresh_current_s_a"
                and runtime.get("temperature_states") == "current",
            pool_covariates_exact=len(cov) == audit.get("covariate_unique_records") == 285,
            pool_outcomes_exact=audit.get("outcome_unique_records") == (len(outcomes) if fresh else 0)
                and len(outcomes) == 285,
            pool_received_exact=audit.get("covariate_records_received") == 285
                and audit.get("outcome_labels_received") == (285 if fresh else 0),
            pool_actual_union_exact=audit.get("distinct_target_records_used") ==
                (len(cov | outcomes) if fresh else len(cov)),
            pool_selection_union_exact=audit.get("selected_target_record_union") == len(cov | outcomes),
            pool_union_bound=285 <= len(cov | outcomes) <= 342)
    return checks


def covariate_receipts(current, cached):
    """Compare the paid state path, retaining every historical receipt field."""
    if len(current) != len(cached):
        return False
    for left, right in zip(current, cached):
        normalized = {key: left.get(key) for key in right}
        normalized["sequence_ids"] = left.get("covariate_sequence_ids")
        normalized["max_available_at"] = left.get("covariate_max_available_at")
        if normalized != right:
            return False
    return True


def compare_update_logs(current, cached):
    """Exact first/eighth-step logs; CQL profiler time is not a learning value."""
    ignored = {"regularizer_wall_seconds"}
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items() if key not in ignored}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value
    def differing(left, right, prefix=""):
        if isinstance(left, dict) and isinstance(right, dict):
            result = []
            for key in sorted(left.keys() | right.keys()):
                path = f"{prefix}.{key}" if prefix else key
                if key not in left or key not in right:
                    result.append(path)
                else:
                    result.extend(differing(left[key], right[key], path))
            return result
        if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
            return [path for index, (x, y) in enumerate(zip(left, right))
                for path in differing(x, y, f"{prefix}[{index}]")]
        return [] if left == right else [prefix]
    if not current.is_file() or not cached.is_file():
        return dict(all_pass=False, status="missing_log", compared_rows=0,
            missing=[name for name, path in (("new", current), ("cached", cached)) if not path.is_file()])
    count, differences, schedule = 0, [], True
    with current.open() as left, cached.open() as right:
        for index, pair in enumerate(zip_longest((json.loads(line) for line in left if line.strip()),
                (json.loads(line) for line in right if line.strip()))):
            count += 1
            x, y = pair
            expected = (index//2+1, 1 if index % 2 == 0 else 8)
            schedule &= bool(x is not None and y is not None and
                (x.get("packet"), x.get("step_in_packet")) == expected ==
                (y.get("packet"), y.get("step_in_packet")))
            fields = differing(clean(x), clean(y))
            if fields:
                differences.append(dict(row=index, fields=fields[:20]))
    return dict(all_pass=count == 114 and schedule and not differences, compared_rows=count,
        expected_rows=114, schedule_exact=schedule, differing_rows=len(differences),
        first_differences=differences[:5], ignored_profiler_fields=sorted(ignored),
        scope="First and eighth update of every57packets; every logged non-profiler field remains exact.")


def no_fresh_regression(root=HERE):
    root = Path(root)
    protocol = old.read(root/"protocol.json")
    seed, scene = 6201, "stable"
    current = run_dir(root, seed, scene, "no_fresh")
    cached = reference_dir(root, protocol, "previous_reference", seed, scene, "no_fresh")
    new, previous = old.read(current/"summary.json"), old.read(cached/"summary.json")
    if new is None or previous is None:
        failure = old.read(current/"failure.json")
        return dict(status="integrity_failed" if failure else "pending", all_pass=False,
            integrity=dict(all_pass=False, checked_conditions=0, failed_checks=[]),
            missing=[name for name, value in (("new_no_fresh", new), ("v70_no_fresh", previous)) if value is None],
            failure=failure)
    config, previous_config = old.read(current/"protocol.json") or {}, old.read(cached/"protocol.json") or {}
    checks = native_checks(new, config, seed=seed, scene=scene, arm="no_fresh", current=True)
    trace = old.compare_prefix(current/"prefix_trace.jsonl", cached/"prefix_trace.jsonl", math.inf)
    trace.pop("stop_exclusive_seconds", None)
    trace["scope"] = "complete issued trace; no cutoff"
    cql = deepcopy(new.get("cql_audit", {}))
    old_cql = deepcopy(previous.get("cql_audit", {}))
    for value in (cql, old_cql):
        value.pop("regularizer_wall_seconds", None)
    updates = compare_update_logs(current/"updates.jsonl", cached/"updates.jsonl")
    exact = dict(ledger=new.get("cost") == previous.get("cost"),
        metrics=old.read_metrics(current, new) == old.read_metrics(cached, previous),
        full_trace=trace["all_pass"] and trace["compared_rows"] ==
            new.get("factual_trace_rows") == previous.get("factual_trace_rows"),
        covariate_receipts=covariate_receipts(new.get("receipts", []), previous.get("receipts", [])),
        initializer=new.get("initializer") == previous.get("initializer"),
        F_and_runtime=old._runtime(new) == old._runtime(previous),
        optimizer_schedule=new.get("actor_schedule_audit") == previous.get("actor_schedule_audit"),
        target_path=new.get("target_audit") == previous.get("target_audit"),
        logged_update_values=updates["all_pass"],
        main_stream_CQL_audit=cql == old_cql,
        policy_config=config.get("policy_config") == previous_config.get("policy_config"))
    rng_fields = [key for key in previous if "rng" in key.lower() or "adam" in key.lower()]
    exact.update({"logged_"+key: new.get(key) == previous[key] for key in rng_fields})
    checks.update({"exact/"+key: value for key, value in exact.items()})
    failed = [name for name, value in checks.items() if value is not True]
    return dict(status="complete" if not failed else "integrity_failed", all_pass=not failed,
        integrity=dict(all_pass=not failed, checked_conditions=len(checks), failed_checks=failed),
        seed=seed, scenario=scene, exact=exact, full_trace=trace, update_log=updates,
        RNG_scope=dict(common_logged_state_fields=rng_fields,
            CQL_main_rng_preserved=new.get("cql_audit", {}).get("main_rng_preserved"),
            full_rng_states_logged=bool(rng_fields)),
        no_new_baseline_cost_cell=True)


def statistics(cells, protocol):
    expected = len(protocol["development_seeds"])*len(protocol["scenarios"])
    return {name: {period: dict(overall=old.group(cells, name, period, expected),
        by_scenario={scene: old.group([row for row in cells if row["scenario"] == scene],
            name, period, len(protocol["development_seeds"])) for scene in protocol["scenarios"]})
        for period, _ in PERIODS} for name in COMPARISONS}


def scientific_gates(cells, aggregates, protocol):
    stable = [row for row in cells if row["scenario"] == "stable"]
    tail = len(stable) == len(protocol["development_seeds"]) and all(
        row["comparisons"]["matched"]["post_budget"]["status"] == "complete"
        and row["comparisons"]["matched"]["post_budget"]["candidate_minus_baseline"]
            ["generalized_seconds_per_departed_passenger"] <= 0 for row in stable)
    previous = aggregates["previous_fresh"]["full_day"]["overall"]
    worst = aggregates["matched"]["full_day"]["overall"]["worst_cost_difference"]
    return dict(cached_no_fresh_full_day=old.cost_gate(aggregates["matched"]),
        both_stable_tails=bool(tail), preserves_v70_fresh=previous["complete_cells"] ==
            previous["expected_cells"] and previous["complete_cells"] > 0
            and previous["mean_cost_difference"] <= 0,
        worst_cell=worst is not None and worst <= protocol["worst_full_day_cost_difference_cap"])


def analyze(root=HERE):
    root = Path(root)
    protocol = old.read(root/"protocol.json")
    checks, missing, failures, failed_work = {}, [], [], []
    cells, policy, pools = [], [], []
    regression = no_fresh_regression(root)
    if regression["status"] != "pending":
        checks["no_fresh_full_regression"] = regression["all_pass"]
    else:
        missing.append("no_fresh_regression")
    for seed in protocol["development_seeds"]:
        for scene in protocol["scenarios"]:
            name = f"seed{seed}/{scene}"
            dirs = dict(fresh=run_dir(root, seed, scene, "fresh"),
                v70_no_fresh=reference_dir(root, protocol, "previous_reference", seed, scene, "no_fresh"),
                v70_fresh=reference_dir(root, protocol, "previous_reference", seed, scene, "fresh"),
                v72_fresh=reference_dir(root, protocol, "capacity_reference", seed, scene, "fresh"))
            runs, points, configs = {}, {}, {}
            for label, directory in dirs.items():
                value = old.read(directory/"summary.json")
                if value is None:
                    failure = old.read(directory/"failure.json")
                    (failures if failure else missing).append(name+"/"+label)
                    if failure:
                        failed_work.append(dict(cell=name+"/"+label, **{key: failure.get(key) for key in
                            ("type", "native_time", "consumed_budget", "wall_seconds")}))
                    continue
                runs[label], points[label] = value, old.read_metrics(directory, value)
                configs[label] = old.read(directory/"protocol.json") or {}
                arm = "no_fresh" if label.endswith("no_fresh") else "fresh"
                audit = native_checks(value, configs[label], seed=seed, scene=scene, arm=arm,
                    current=label == "fresh")
                audit.update(cost_endpoints=all(old.window(value, points[label], *bounds)["status"] == "complete"
                    for _, bounds in PERIODS), external_seed=value.get("cell", {}).get("external_seed") ==
                    seed+protocol["external_seed_offset"], departed_cohort=value.get("cost", {}).get("departed_passengers") ==
                    protocol["full_day_departed_passengers"])
                final = {row["time_seconds"]: row for row in points[label]}.get(24000, {})
                audit["final_metrics_match_ledger"] = all(final.get(field) == value.get("cost", {}).get(field)
                    for field in ("accrued_wait_seconds", "accrued_in_vehicle_seconds"))
                if label == "fresh":
                    audit["recipe"] = value.get("cell", {}).get("recipe") == protocol["recipe"]
                checks.update({name+"/"+label+"/"+key: flag is True for key, flag in audit.items()})
            if len(runs) != len(dirs):
                continue
            fresh = runs["fresh"]
            first = [row.get("clock_audit", {}).get("first_feedback_applied_at") for row in runs.values()]
            valid = all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in first)
            prefixes = {label: old.compare_prefix(dirs["fresh"]/"prefix_trace.jsonl",
                dirs[label]/"prefix_trace.jsonl", min(first) if valid else 0.)
                for label in ("v70_fresh", "v70_no_fresh", "v72_fresh")}
            checks.update({name+"/common_preupdate_prefix": valid and len(set(first)) == 1 and all(
                value["all_pass"] for value in prefixes.values()),
                name+"/same_initializer": all(fresh.get("initializer") == value.get("initializer") for value in runs.values()),
                name+"/same_F_runtime": all(old._runtime(fresh) == old._runtime(value) for value in runs.values()),
                name+"/same_policy_config": all(configs["fresh"].get("policy_config") == value.get("policy_config")
                    for value in configs.values())})
            comparisons = {key: {} for key in COMPARISONS}
            periods = {}
            for period, bounds in PERIODS:
                periods[period] = {label: old.window(value, points[label], *bounds) for label, value in runs.items()}
                for key, (left, right) in COMPARISONS.items():
                    comparisons[key][period] = old.difference(periods[period][left], periods[period][right])
            cells.append(dict(identity=seed, scenario=scene, periods=periods, comparisons=comparisons))
            policy.append(dict(seed=seed, scenario=scene, prefix=prefixes,
                policy_drift=old.drift_view(fresh.get("policy_drift", {})),
                executed_cql=old.cql_view(fresh.get("cql_audit", {})), native_wall_seconds=fresh.get("wall_seconds"),
                extra_CQL_wall_seconds=fresh.get("cql_audit", {}).get("regularizer_wall_seconds")))
            services, resident = service_view(dirs["fresh"])
            population = dict(versus_v72_fresh=passenger_comparison(resident,
                old.read(dirs["v72_fresh"]/"service_summary.json")))
            if scene == "stable":
                prior = root.parent/"mcwm_closed_loop_tail_policy_exchange_v71_20261026"/"runs"/f"seed{seed}"
                for arm, branch in (("fresh", "low"), ("no_fresh", "high")):
                    population["versus_v70_"+arm] = passenger_comparison(resident,
                        old.read(prior/arm/"suffixes"/branch/"service_summary.json"))
            pools.append(dict(seed=seed, scenario=scene, pools=fresh.get("replay"),
                selection=coverage_view(fresh), services=services, final_waiting_membership=population))
    regression_summary = old.read(run_dir(root, 6201, "stable", "no_fresh")/"summary.json")
    fresh_summary = old.read(run_dir(root, 6201, "stable", "fresh")/"summary.json")
    overlap = old._overlap(fresh_summary, regression_summary) if fresh_summary and regression_summary else None
    if overlap is not None:
        checks["first_pair_actually_parallel"] = overlap["positive_overlap"]
    expected = len(protocol["development_seeds"])*len(protocol["scenarios"])
    aggregates = statistics(cells, protocol)
    conditions = scientific_gates(cells, aggregates, protocol)
    technical = len(cells) == expected and not missing and not failures and bool(checks) and all(checks.values())
    passed = technical and all(conditions.values())
    status = "complete" if technical else "integrity_failed" if failures or any(not flag for flag in checks.values()) else "pending"
    decision = dict(scientific_gate_pass=bool(passed), eligible_for_root_review=bool(passed),
        advance_to_confirmation=False, automatic_confirmation=False, ordinary_dripping_goal_achieved=False,
        reason="development_pass_requires_root_review" if passed else "incomplete_or_integrity_failed" if not technical
            else "candidate_closed_cost_gate_failed")
    main = dict(status=status, complete=bool(technical), candidate="executed_cql", recipe=protocol["recipe"],
        expected_cells=expected, available_cells=len(cells), missing=missing, failures=failures,
        failed_run_work=failed_work, integrity=dict(all_pass=bool(technical), checked_conditions=len(checks),
            failed_checks=[name for name, flag in checks.items() if not flag]),
        aggregates={"matched": aggregates["matched"]}, cells=[dict(identity=row["identity"], scenario=row["scenario"],
            periods={period: dict(arms={label: old.cost_view(row["periods"][period][label]) for label in
                ("fresh", "v70_no_fresh")}, matched=old.compact_comparison(row["comparisons"]["matched"][period]))
                for period, _ in PERIODS}) for row in cells],
        optimization_assessment=dict(conditions=conditions, worst_full_day_cost_difference_cap=
            protocol["worst_full_day_cost_difference_cap"]), decision=decision,
        budget=dict(planned_new_native_days=5, planned_native_seconds=120000, planned_fresh_outcome_labels=1140,
            planned_covariate_slots=1425, planned_updates_per_optimizer=2280, cached_baseline_days=4,
            native_branches=0, new_initialization_training=0,
            actual_fresh_outcome_labels=sum(row["pools"]["outcome_labels_received"] for row in pools)
                if len(pools) == 4 else None,
            actual_covariate_slots=sum(row["pools"]["covariate_records_received"] for row in pools)
                +regression_summary["replay"]["covariate_records_received"]
                if len(pools) == 4 and regression_summary is not None else None,
            distinct_target_records_used=sum(row["pools"]["distinct_target_records_used"] for row in pools)
                +regression_summary["replay"]["distinct_target_records_used"]
                if len(pools) == 4 and regression_summary is not None else None,
            actual_distinct_records_complete=len(pools) == 4 and regression_summary is not None),
        units="J=(2W+R)/15079; all windows use full-day departed cohort and unfinished-trip accrued costs.",
        acceptance_protocol=dict(primary_control="cached_v70_no_fresh", regression_excluded_from_cost_groups=True,
            v72_fresh="descriptive_only", early_window="descriptive_only", old_results_read_only=True,
            information_contract="285covariates plus285complete outcomes; actual learner union reported, at most342/fresh day"))
    references = dict(status=status, aggregates={key: aggregates[key] for key in ("previous_fresh", "capacity_fresh")},
        cells=[dict(identity=row["identity"], scenario=row["scenario"], periods={period: dict(
            arms={label: old.cost_view(row["periods"][period][label]) for label in ("v70_fresh", "v72_fresh")},
            comparisons={key: old.compact_comparison(row["comparisons"][key][period]) for key in
                ("previous_fresh", "capacity_fresh")}) for period, _ in PERIODS}) for row in cells])
    diagnostics = dict(status=status, policy_drift_columns=old.DRIFT_COLUMNS, cql_head_columns=old.CQL_HEAD_COLUMNS,
        precision="Six significant digits for passive diagnostics only; exact passenger costs in cost files.", cells=policy)
    recency = dict(status=status, cells=pools, no_fresh_regression=regression,
        first_pair_actual_parallel=overlap, interpretation="Own-trajectory latest5 covariates; risk outcomes stay complete. Cached controls are not new runs. Missing observations remain missing.")
    return main, references, diagnostics, recency


def export(root, out, *, regression_gate=False):
    out = Path(out)
    if regression_gate:
        result = no_fresh_regression(root)
        payloads = {out: old.encoded(result)}
    else:
        main, refs, policy, recency = analyze(root)
        result = main
        payloads = {out: old.encoded(main), out.with_name("reference_summary.json"): old.encoded(refs),
            out.with_name("policy_drift.json"): old.encoded(policy), out.with_name("recency.json"): old.encoded(recency)}
    for path, raw in payloads.items():
        if len(raw) > MAX_BYTES:
            raise ValueError(f"{path.name} exceeds48KB: {len(raw)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    for path, raw in payloads.items():
        with path.open("xb") as handle:
            handle.write(raw)
    return dict(status=result["status"], all_pass=result.get("all_pass"), decision=result.get("decision"),
        files={path.name: len(raw) for path, raw in payloads.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--regression-gate", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export(HERE, args.out, regression_gate=args.regression_gate), allow_nan=False))
