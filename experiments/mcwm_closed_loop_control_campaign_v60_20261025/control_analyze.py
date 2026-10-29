"""Small server-side full-day control decision; missing runs never become zeros."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCENARIOS = ("stable", "local_speed_drop")
COST_KEYS = ("departed_passengers", "accrued_wait_seconds", "accrued_in_vehicle_seconds",
             "generalized_passenger_seconds", "generalized_seconds_per_departed_passenger")


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def read_metrics(directory, summary):
    path = directory / "metrics.jsonl"
    if not path.exists() and summary.get("metrics_path"):
        path = Path(summary["metrics_path"])
        if not path.is_absolute():
            path = directory / path
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def compare_prefix(left, right, stop):
    """Stream only the common, already executed pre-intervention factual prefix."""
    if not left.exists() or not right.exists():
        return dict(all_pass=False, compared_rows=0, reason="missing_prefix_trace")

    def rows(path):
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if float(row["started_at"]) >= stop:
                        break
                    yield row

    count, differences, first = 0, 0, None
    for index, pair in enumerate(itertools.zip_longest(rows(left), rows(right))):
        count += 1
        if pair[0] != pair[1]:
            differences += 1
            if first is None:
                first = index
    return dict(all_pass=count > 0 and differences == 0, compared_rows=count,
                differing_rows=differences, first_difference_index=first,
                stop_exclusive_seconds=stop)


def _cost_valid(cost):
    if not all(isinstance(cost.get(key), (int, float)) and math.isfinite(cost[key])
               for key in COST_KEYS) or cost["departed_passengers"] <= 0:
        return False
    total = 2 * cost["accrued_wait_seconds"] + cost["accrued_in_vehicle_seconds"]
    return (math.isclose(cost["generalized_passenger_seconds"], total, rel_tol=1e-12, abs_tol=1e-7)
            and math.isclose(cost["generalized_seconds_per_departed_passenger"],
                             total / cost["departed_passengers"], rel_tol=1e-12, abs_tol=1e-9))


def window(summary, points, start, stop):
    cost = summary.get("cost", {})
    if not _cost_valid(cost):
        return dict(status="invalid_final_ledger", start=start, end=stop)
    if start == 0 and stop == 24000:
        return dict(status="complete", start=start, end=stop,
                    **{key: cost[key] for key in COST_KEYS})
    index = {row["time_seconds"]: row for row in points}
    first, last = index.get(start), index.get(stop)
    if first is None or last is None:
        return dict(status="missing_endpoint", start=start, end=stop)
    wait = last["accrued_wait_seconds"] - first["accrued_wait_seconds"]
    ride = last["accrued_in_vehicle_seconds"] - first["accrued_in_vehicle_seconds"]
    total = 2 * wait + ride
    return dict(status="complete", start=start, end=stop,
                departed_passengers=cost["departed_passengers"],
                accrued_wait_seconds=wait, accrued_in_vehicle_seconds=ride,
                generalized_passenger_seconds=total,
                generalized_seconds_per_departed_passenger=total / cost["departed_passengers"])


def difference(candidate, baseline):
    if candidate.get("status") != "complete" or baseline.get("status") != "complete":
        return dict(status="missing_or_invalid_endpoint", candidate=candidate, baseline=baseline)
    delta = {key: candidate[key] - baseline[key] for key in COST_KEYS if key != "departed_passengers"}
    baseline_primary = baseline["generalized_seconds_per_departed_passenger"]
    primary = delta["generalized_seconds_per_departed_passenger"]
    # Absolute W/R ledgers are reported once per arm/cell, not repeated for
    # each comparator and period in the bounded download.
    primary_fields = ("departed_passengers", "generalized_seconds_per_departed_passenger")
    return dict(status="complete", candidate={key: candidate[key] for key in primary_fields},
                baseline={key: baseline[key] for key in primary_fields},
                candidate_minus_baseline=delta,
                wait_seconds_per_passenger_difference=(candidate["accrued_wait_seconds"] /
                    candidate["departed_passengers"] - baseline["accrued_wait_seconds"] /
                    baseline["departed_passengers"]),
                ride_seconds_per_passenger_difference=(candidate["accrued_in_vehicle_seconds"] /
                    candidate["departed_passengers"] - baseline["accrued_in_vehicle_seconds"] /
                    baseline["departed_passengers"]),
                percent_change=100 * primary / baseline_primary if baseline_primary else None,
                outcome="win" if primary < 0 else "loss" if primary > 0 else "tie")


def aggregate(cells, comparator, period="full_day"):
    values = [cell["comparisons"][comparator][period] for cell in cells
              if cell["comparisons"][comparator][period]["status"] == "complete"]
    if not values:
        return dict(status="missing", cells=0, mean_cost_difference=None,
                    wins=0, losses=0, ties=0)
    size = len(values)
    candidate_mean = sum(row["candidate"]["generalized_seconds_per_departed_passenger"]
                         for row in values) / size
    baseline_mean = sum(row["baseline"]["generalized_seconds_per_departed_passenger"]
                        for row in values) / size
    return dict(status="complete", cells=size, candidate_mean_cost=candidate_mean,
                baseline_mean_cost=baseline_mean,
                mean_cost_difference=candidate_mean-baseline_mean,
                percent_change=100*(candidate_mean-baseline_mean)/baseline_mean if baseline_mean else None,
                mean_wait_seconds_per_passenger_difference=sum(
                    row["wait_seconds_per_passenger_difference"] for row in values)/size,
                mean_ride_seconds_per_passenger_difference=sum(
                    row["ride_seconds_per_passenger_difference"] for row in values)/size,
                raw_weighted_passenger_seconds_difference=sum(
                    row["candidate_minus_baseline"]["generalized_passenger_seconds"] for row in values),
                wins=sum(row["outcome"] == "win" for row in values),
                losses=sum(row["outcome"] == "loss" for row in values),
                ties=sum(row["outcome"] == "tie" for row in values))


def _runtime(summary):
    # TD quotas differ intentionally between fresh and historical replacement.
    return {key: value for key, value in summary.get("runtime_contract", {}).items()
            if key != "td_slots"}


def _run_checks(summary, arm, candidate, identity, scenario):
    budget, initializer = summary.get("budget", {}), summary.get("initializer", {})
    checks = dict(native_complete=summary.get("status") == "complete"
        and summary.get("integrity", {}).get("all_pass") is True,
        native_audits=bool(summary.get("integrity", {}).get("checks")) and all(
            value is True for value in summary.get("integrity", {}).get("checks", {}).values()),
        full_native_horizon=budget.get("native_seconds") == 24000,
        complete_cost_ledger=_cost_valid(summary.get("cost", {})),
        correct_arm_scenario=summary.get("cell", {}).get("arm") == arm
            and summary.get("cell", {}).get("scenario") == scenario,
        no_deployment_branches=budget.get("counterfactual_queries") == 0)
    if candidate == "recent30":
        replay = summary.get("replay", {})
        post = summary.get("post_budget_counts", {})
        checks.update(correct_seed=summary.get("cell", {}).get("seed") == identity,
            initializer_50000=initializer.get("initial_updates") == 50000,
            archived_source_history=initializer.get("history_rows") == 32768
                and initializer.get("source_rows") == 4096,
            feedback285_updates456=budget.get("target_feedback_slots") == 285
                and budget.get("policy_updates") == 456,
            no_world_or_trust=budget.get("world_updates") == budget.get("trust_updates") == 0,
            current_covariates285=replay.get("covariate_window") == 285
                and replay.get("recent_covariates") == 285,
            outcome_window30=replay.get("critic_fresh_outcome_window") == (30 if arm == "fresh" else 0),
            isolated_outcome_sampling456=replay.get("independent_outcome_sampler_calls") == 456,
            actor_real_clock=summary.get("clock_audit", {}).get("actor_real_clock") is True,
            unified_F=summary.get("action_audit", {}).get("shared_rule_exact") is True,
            post_budget_zero_training=post.get("before") == post.get("after")
                == dict(policy=456, world=0, trust=0))
        if arm == "no_fresh":
            checks["no_fresh_outcomes_retained"] = (
                summary.get("integrity", {}).get("checks", {}).get("no_fresh_outcomes_retained") is True
                and replay.get("fresh_outcomes_stored") == 0)
    else:
        checks.update(correct_external_seed=summary.get("cell", {}).get("external_seed") == identity,
            inherited285_456=budget.get("inherited_feedback_slots") == 285
                and budget.get("inherited_policy_updates") == 456,
            zero_new_training=budget.get("new_feedback_slots") == budget.get("new_policy_updates") == 0)
    return checks


def _overlap(left, right):
    stamps = [value.get("execution_wall", value.get("wall", {}).get("execution_wall", {}))
              for value in (left, right)]
    keys = ("native_started_wall", "native_finished_wall")
    if not all(all(isinstance(row.get(key), (int, float)) and math.isfinite(row[key])
                   for key in keys) and row[keys[1]] >= row[keys[0]] for row in stamps):
        return dict(positive_overlap=False, seconds=None)
    seconds = min(row[keys[1]] for row in stamps) - max(row[keys[0]] for row in stamps)
    return dict(positive_overlap=seconds > 0, seconds=max(0., seconds),
                elapsed_seconds=[row[keys[1]]-row[keys[0]] for row in stamps])


def intervention_evidence(summary):
    value = summary.get("intervention", {})
    keys = ("selected", "selected_high", "selected_zero", "training_station_unseen", "headway_extrapolation")
    return dict(counts={key: value.get("counts", {}).get(key, 0) for key in keys},
                by_line={line: {key: counts.get(key, 0) for key in keys}
                         for line, counts in value.get("by_line", {}).items()},
                used_bins=len(value.get("used_bins", [])))


def analyze(root, candidate, stage):
    root = Path(root)
    protocol = read(root / "protocol.json")
    seeds = protocol[f"{stage}_seeds"] if candidate == "recent30" else protocol[candidate][f"{stage}_external_seeds"]
    arms = ("fresh", "no_fresh") if candidate == "recent30" else ("readout", "q_selector", "actor_mcwm")
    candidate_arm, comparators = arms[0], arms[1:]
    cells, missing, failures, failure_work, checks = [], [], [], [], {}
    for identity in seeds:
        identity_dir = f"seed{identity}" if candidate == "recent30" else f"external{identity}"
        for scenario in protocol["scenarios"]:
            key = f"{identity_dir}/{scenario}"
            runs, points, configurations = {}, {}, {}
            for arm in arms:
                directory = root/"runs"/candidate/stage/identity_dir/scenario/arm
                summary = read(directory/"summary.json")
                if summary is None:
                    failure = read(directory/"failure.json")
                    (failures if failure else missing).append(f"{key}/{arm}")
                    if failure:
                        failure_work.append(dict(cell=f"{key}/{arm}", **{
                            field: failure.get(field) for field in
                            ("type", "native_time", "consumed_budget", "wall_seconds")}))
                    continue
                runs[arm] = (directory, summary)
                configurations[arm] = read(directory/"protocol.json") or {}
                run_checks = _run_checks(summary, arm, candidate, identity, scenario)
                if candidate == "recent30":
                    run_checks["learner_config_recorded"] = isinstance(
                        configurations[arm].get("policy_config"), dict)
                for name, value in run_checks.items():
                    checks[f"{key}/{arm}/{name}"] = value
                points[arm] = read_metrics(directory, summary)
                final = {row["time_seconds"]: row for row in points[arm]}.get(24000, {})
                checks[f"{key}/{arm}/final_metrics_match_ledger"] = all(
                    final.get(field) == summary.get("cost", {}).get(field)
                    for field in ("accrued_wait_seconds", "accrued_in_vehicle_seconds"))
            if len(runs) != len(arms):
                continue
            prefix, parallel, comparisons = {}, {}, {}
            left_dir, left = runs[candidate_arm]
            for comparator in comparators:
                right_dir, right = runs[comparator]
                first = [row.get("clock_audit", {}).get("first_feedback_applied_at")
                         for row in (left, right)]
                valid_first = all(isinstance(value, (int, float)) and math.isfinite(value)
                                  and value > 0 for value in first)
                stop = min(first) if candidate == "recent30" and valid_first else 18300.
                if candidate == "recent30":
                    checks[f"{key}/{comparator}/first_feedback_exact"] = valid_first and first[0] == first[1]
                prefix[comparator] = compare_prefix(left_dir/"prefix_trace.jsonl", right_dir/"prefix_trace.jsonl", stop)
                parallel[comparator] = _overlap(left, right)
                checks[f"{key}/{comparator}/prefix_exact"] = prefix[comparator]["all_pass"]
                checks[f"{key}/{comparator}/actual_parallel"] = parallel[comparator]["positive_overlap"]
                if candidate == "recent30":
                    checks[f"{key}/{comparator}/initializer_exact"] = left.get("initializer") == right.get("initializer")
                    checks[f"{key}/{comparator}/learner_config_exact"] = (
                        configurations[candidate_arm].get("policy_config") == configurations[comparator].get("policy_config"))
                    checks[f"{key}/{comparator}/same_runtime_F"] = _runtime(left) == _runtime(right)
                comparisons[comparator] = {name: difference(window(left, points[candidate_arm], *bounds),
                        window(right, points[comparator], *bounds))
                    for name, bounds in (("full_day", (0, 24000)), ("early", (9000, 12000)),
                                         ("post_budget", (18300, 24000)))}
            if candidate == "paired_readout":
                name = "q_selector_vs_actor_mcwm"
                parallel[name] = _overlap(runs["q_selector"][1], runs["actor_mcwm"][1])
                checks[f"{key}/{name}/actual_parallel"] = parallel[name]["positive_overlap"]
            cells.append(dict(identity=identity, scenario=scenario, comparisons=comparisons,
                pre_intervention_prefix=prefix, actual_parallel=parallel,
                initializers={arm: value.get("initializer") for arm, (_, value) in runs.items()},
                cost_ledgers={arm: value.get("cost") for arm, (_, value) in runs.items()},
                intervention={arm: intervention_evidence(value) for arm, (_, value) in runs.items()}
                    if candidate == "paired_readout" else None))
    expected = len(seeds)*len(protocol["scenarios"])
    intact = not missing and not failures and len(cells) == expected and bool(checks) and all(checks.values())
    aggregates = {comparator: dict(overall=aggregate(cells, comparator),
        by_scenario={scenario: aggregate([cell for cell in cells if cell["scenario"] == scenario], comparator)
                     for scenario in SCENARIOS},
        by_identity={str(identity): aggregate([cell for cell in cells if cell["identity"] == identity], comparator)
                     for identity in seeds},
        early=aggregate(cells, comparator, "early"),
        post_budget=aggregate(cells, comparator, "post_budget")) for comparator in comparators}
    passed = intact and all(value["overall"]["mean_cost_difference"] < 0
        and value["by_scenario"]["stable"]["mean_cost_difference"] <= 0
        and value["by_scenario"]["local_speed_drop"]["mean_cost_difference"] < 0
        for value in aggregates.values())
    return dict(status="complete" if intact else "integrity_failed" if failures or any(
        value is not True for value in checks.values()) else "pending",
        candidate=candidate, stage=stage, complete=intact,
        integrity=dict(all_pass=intact, checked_conditions=len(checks),
            failed_checks=[key for key, value in checks.items() if value is not True]),
        missing=missing, failures=failures, failed_run_work=failure_work,
        expected_cells=expected, available_cells=len(cells), cells=cells, aggregates=aggregates,
        decision=dict(advance_to_confirmation=bool(passed and stage == "development"),
            confirmed_control_improvement=bool(passed and stage == "confirmation"),
            ordinary_dripping_goal_achieved=bool(passed and stage == "confirmation" and candidate == "recent30"),
            counterfactual_assisted_only=candidate == "paired_readout",
            scientific_gate_pass=bool(passed),
            reason="matched_full_day_cost_gate_pass" if passed else
                   "incomplete_or_integrity_failed" if not intact else "candidate_closed_cost_gate_failed"),
        metric="(2W+R)/all departed passengers; includes unfinished trips; negative difference is beneficial",
        interpretation="All fixed cells retained. Confirmation reports three independent seed differences; no confidence claim.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("recent30", "paired_readout"), required=True)
    parser.add_argument("--stage", choices=("development", "confirmation"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(HERE, args.candidate, args.stage)
    payload = json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(payload.encode()) > 48000:
        raise ValueError("Control summary exceeds the48KB small-pull budget")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        handle.write(payload+"\n")
    print(json.dumps({key: result[key] for key in ("status", "candidate", "stage", "decision")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
