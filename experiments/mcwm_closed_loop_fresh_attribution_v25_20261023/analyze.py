"""Small resident-file gates and factual-outcome attribution for the v25 pilot."""

import argparse
import itertools
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.mcwm_closed_loop_formal_20261020.analysis import paired_control


ARMS = ("fresh", "no_fresh")
SCENARIOS = ("stable", "local_speed_drop")
SEED = 6201


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compare_trace(left, right, stop):
    """Compare only the already executed prefix, without copying it into the gate."""
    def rows(path):
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if float(row["started_at"]) < stop:
                        yield row
    count, different, first = 0, 0, None
    for index, values in enumerate(itertools.zip_longest(rows(left), rows(right))):
        count += 1
        if values[0] != values[1]:
            different += 1
            if first is None:
                first = index
    return dict(all_pass=count > 0 and different == 0, compared_rows=count,
                differing_rows=different, first_difference_index=first, stop_exclusive_seconds=stop)


def _metric_path(directory, summary):
    path = Path(summary["metrics_path"])
    return path if path.is_absolute() else directory/path


def _run_checks(summary, arm):
    budget, replay = summary.get("budget", {}), summary.get("replay", {})
    runtime = summary.get("runtime_contract", {})
    quotas = dict(source=128, historical_target=64,
        fresh_target=64 if arm == "fresh" else 0,
        historical_replacement=64 if arm == "no_fresh" else 0)
    post = summary.get("post_budget_counts", {})
    before, after = post.get("before"), post.get("after")
    receipts = summary.get("receipts", [])
    ids = [sequence for receipt in receipts for sequence in receipt.get("sequence_ids", [])]
    return dict(native_complete=summary.get("status") == "complete"
            and summary.get("integrity", {}).get("all_pass") is True,
        initializer_50000=summary.get("initializer", {}).get("initial_updates") == 50000,
        history_source_rows=summary.get("initializer", {}).get("history_rows") == 32768
            and summary.get("initializer", {}).get("source_rows") == 4096,
        exact_feedback_budget=budget.get("target_feedback_slots") == 285
            and budget.get("packets") == 57 and len(receipts) == 57
            and replay.get("feedback_slots_received") == 285,
        unique_delayed_factual_feedback=len(ids) == len(set(ids)) == 285
            and all(receipt.get("records") == 5 and receipt["max_available_at"] <= receipt["deadline"]
                    <= receipt["processed_at"] for receipt in receipts),
        exact_update_horizon=budget.get("policy_updates") == 456 and budget.get("native_seconds") == 24000,
        no_world_trust_queries=all(budget.get(key) == 0
            for key in ("world_updates", "trust_updates", "counterfactual_queries")),
        fresh_supervision_rights=budget.get("training_fresh_outcomes") == (285 if arm == "fresh" else 0)
            and replay.get("fresh_outcomes_stored") == (285 if arm == "fresh" else 0)
            and replay.get("fresh_outcome_use") == (["critic factual Bellman target"] if arm == "fresh" else []),
        td_quota_exact=replay.get("td_slots") == quotas and runtime.get("td_slots") == quotas,
        actor_value_state_rule=replay.get("fresh_current_s_a_use") == ["IQL value", "SAC actor", "temperature"]
            and runtime.get("actor_value_covariates") == "fresh_current_s_a"
            and runtime.get("temperature_states") == "current",
        unified_f_real_td=runtime.get("action_transform") == "ExecutionAction"
            and runtime.get("real_td_coefficient") == 1
            and runtime.get("world_enabled") is False and runtime.get("ratio_correction") is False,
        actual_shared_f=summary.get("action_audit", {}).get("shared_rule_exact") is True,
        actor_real_clock=summary.get("clock_audit", {}).get("actor_real_clock") is True,
        post_budget_zero_training=isinstance(before, dict) and before == after
            and before == dict(policy=456, world=0, trust=0))


def _points_checks(points, summary):
    times = [row["time_seconds"] for row in points]
    cost = summary.get("cost", {})
    last = points[-1] if points else {}
    return dict(all_fixed_clock_metrics=times == list(range(0, 24001, 300)),
        final_ledger_matches=all(last.get(key) == cost.get(key) for key in (
            "accrued_wait_seconds", "accrued_in_vehicle_seconds")),
        complete_passenger_cost=cost.get("departed_passengers", 0) > 0
            and cost.get("generalized_passenger_seconds") == (
                2*cost.get("accrued_wait_seconds", float("nan"))
                + cost.get("accrued_in_vehicle_seconds", float("nan")))
            and cost.get("generalized_seconds_per_departed_passenger") == (
                cost.get("generalized_passenger_seconds", float("nan"))/cost["departed_passengers"]
                if cost.get("departed_passengers", 0) > 0 else None))


def cost_window(points, denominator, start, stop):
    by_time = {row["time_seconds"]: row for row in points}
    first, last = by_time.get(start), by_time.get(stop)
    if first is None or last is None:
        return dict(status="missing_endpoint", start=start, end=stop)
    if denominator <= 0:
        return dict(status="invalid_passenger_denominator", start=start, end=stop)
    wait = last["accrued_wait_seconds"]-first["accrued_wait_seconds"]
    ride = last["accrued_in_vehicle_seconds"]-first["accrued_in_vehicle_seconds"]
    cost = 2*wait+ride
    return dict(status="complete", start=start, end=stop, accrued_wait_seconds=wait,
        accrued_in_vehicle_seconds=ride, generalized_passenger_seconds=cost,
        generalized_seconds_per_departed_passenger=cost/denominator,
        departed_passengers_denominator=denominator,
        denominator_scope="Full-day departed cohort, including its unfinished passenger time.")


def cost_comparison(summaries, points):
    result = {}
    fields = ("accrued_wait_seconds", "accrued_in_vehicle_seconds",
              "generalized_passenger_seconds", "generalized_seconds_per_departed_passenger")
    for period, bounds in (("full_day", (0, 24000)), ("post_budget", (18300, 24000))):
        arms = {arm: cost_window(points[arm], summaries[arm]["cost"]["departed_passengers"], *bounds)
                for arm in ARMS}
        complete = all(row["status"] == "complete" for row in arms.values())
        delta = {field: arms["fresh"][field]-arms["no_fresh"][field] for field in fields} if complete else None
        result[period] = dict(arms=arms, fresh_minus_no_fresh=delta,
            favorable_direction="Negative differences mean lower fresh-arm cost.")
    return result


def analyze(root, stage):
    scenarios = SCENARIOS if stage == "pilot" else ("stable",)
    runs, pair_summaries, pending, failures = {}, {}, [], []
    for scenario in scenarios:
        directory = root/"runs"/scenario/f"seed{SEED}"
        pair_summaries[scenario] = read_json(directory/"pair_summary.json")
        for arm in ARMS:
            path = directory/arm
            summary = read_json(path/"summary.json")
            if summary is None:
                if (path/"failure.json").exists():
                    failures.append(f"{scenario}/{arm}: native failure.json")
                else:
                    pending.append(f"{scenario}/{arm}: summary")
            else:
                runs[scenario, arm] = (path, summary)
                if summary.get("status") != "complete" or summary.get("integrity", {}).get("all_pass") is not True:
                    failures.append(f"{scenario}/{arm}: native integrity failed")
        if pair_summaries[scenario] is None:
            pending.append(f"{scenario}: pair_summary")
        elif pair_summaries[scenario].get("status") != "complete":
            failures.append(f"{scenario}: paired execution failed")
    if failures or pending:
        return dict(status="integrity_failed" if failures else "pending", stage=stage,
            integrity=dict(all_pass=False), pending=pending, failures=failures)

    checks, run_evidence, points, scientific = {}, {}, {}, {}
    for (scenario, arm), (directory, summary) in runs.items():
        key = f"{scenario}/{arm}"
        run_checks = _run_checks(summary, arm)
        cell = summary.get("cell", {})
        run_checks["correct_cell"] = cell == dict(seed=SEED, scenario=scenario, arm=arm)
        metric_path = _metric_path(directory, summary)
        if not metric_path.exists() or not (directory/"prefix_trace.jsonl").exists():
            return dict(status="integrity_failed", stage=stage, integrity=dict(all_pass=False),
                        failures=[f"{key}: resident metrics/trace missing after complete summary"])
        points[scenario, arm] = read_jsonl(metric_path)
        run_checks.update(_points_checks(points[scenario, arm], summary))
        checks.update({f"{key}/{name}": value is True for name, value in run_checks.items()})
        run_evidence[key] = dict(status=summary.get("status"), budget=summary.get("budget"),
            replay=summary.get("replay"), post_budget_counts=summary.get("post_budget_counts"))

    prefix_evidence = {}
    parallel = {}
    for scenario in scenarios:
        left_path, left = runs[scenario, "fresh"]
        right_path, right = runs[scenario, "no_fresh"]
        checks[f"{scenario}/initializer_exact"] = left["initializer"] == right["initializer"]
        checks[f"{scenario}/feedback_deadlines_exact"] = [row["deadline"] for row in left["receipts"]] \
            == [row["deadline"] for row in right["receipts"]] == list(range(1200, 18001, 300))
        first = [value["clock_audit"].get("first_feedback_applied_at") for value in (left, right)]
        checks[f"{scenario}/first_feedback_clock_exact"] = all(
            isinstance(value, (int, float)) and math.isfinite(value) and value > 0 for value in first
        ) and first[0] == first[1]
        prefix = compare_trace(left_path/"prefix_trace.jsonl", right_path/"prefix_trace.jsonl", min(first)) \
            if all(isinstance(value, (int, float)) for value in first) else dict(all_pass=False)
        prefix_evidence[scenario] = prefix
        checks[f"{scenario}/pre_feedback_prefix_exact"] = prefix["all_pass"]
        pair = pair_summaries[scenario]
        parallel[scenario] = pair.get("parallel", {})
        checks[f"{scenario}/pair_integrity"] = pair.get("integrity", {}).get("all_pass") is True
        checks[f"{scenario}/actual_parallel"] = parallel[scenario].get("positive_overlap") is True
        scientific[scenario] = cost_comparison(dict(fresh=left, no_fresh=right),
            {arm: points[scenario, arm] for arm in ARMS})

    recovery, pre_shock = {}, {}
    if stage == "pilot":
        for arm in ARMS:
            stable_path = runs["stable", arm][0]
            shock_path = runs["local_speed_drop", arm][0]
            trace = compare_trace(stable_path/"prefix_trace.jsonl", shock_path/"prefix_trace.jsonl", 9000.)
            fields = ("time_seconds", "waiting_passengers", "in_vehicle_passengers",
                      "accrued_wait_seconds", "accrued_in_vehicle_seconds")
            ledgers = [[{key: row.get(key) for key in fields} for row in points[scenario, arm]
                        if row["time_seconds"] <= 9000]
                       for scenario in SCENARIOS]
            ledger_exact = ledgers[0] == ledgers[1]
            checks[f"{arm}/pre_shock_trace_exact"] = trace["all_pass"]
            checks[f"{arm}/pre_shock_ledger_exact"] = ledger_exact
            checks[f"{arm}/same_initializer_across_scenarios"] = runs["stable", arm][1]["initializer"] \
                == runs["local_speed_drop", arm][1]["initializer"]
            pre_shock[arm] = dict(trace=trace, ledger_exact=ledger_exact)
            recovery[arm] = paired_control(points["stable", arm], points["local_speed_drop", arm])
    passed = all(checks.values())
    result = dict(status="complete" if passed else "integrity_failed", stage=stage,
        integrity=dict(all_pass=passed, checks=checks), run_evidence=run_evidence,
        pre_feedback_prefix=prefix_evidence, pre_shock=pre_shock, parallel=parallel,
        absolute_cost=scientific, recovery=recovery,
        decision=dict(implementation_ready=passed, next_action=("run paired shock pilot" if stage == "stable"
                else "review paired absolute costs and stable harm before any next protocol") if passed
                else "stop and inspect failed implementation check",
            cost_used_as_dispatch_gate=False, service_reversal_used_as_dispatch_gate=False,
            independent_confirmation="not_authorized_by_this_gate", stage_b="closed"),
        interpretation="One paired seed: factual outcome supervision attribution, not independent confirmation or world-model benefit.")
    if len(json.dumps(result).encode()) > 30000:
        raise RuntimeError("v25 summary exceeds30KB; reduce resident diagnostic detail")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("stable", "pilot"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    while True:
        result = analyze(args.root, args.stage)
        if result["status"] != "pending" or not args.wait:
            break
        time.sleep(30)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.stage == "pilot" and result["status"] == "complete":
        from audit_summary import summarize
        audit = summarize(args.root)
        with (args.root/"issue_audit_summary.json").open("x") as handle:
            handle.write(json.dumps(audit, separators=(',', ':'), allow_nan=False)+"\n")
        result["issue_audit_summary"] = "issue_audit_summary.json"
    with args.out.open("x") as handle:
        handle.write(json.dumps(result)+"\n")
    print(json.dumps(dict(status=result["status"], integrity=result["integrity"])))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
