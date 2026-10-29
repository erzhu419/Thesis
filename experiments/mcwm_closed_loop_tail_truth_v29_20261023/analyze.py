"""Resident, bounded export of the two predeclared v29 native action pairs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time


HERE = Path(__file__).resolve().parent
SCENARIOS = ("stable", "local_speed_drop")
ARMS = ("low", "high")
MODELS = ("fresh", "no_fresh")
ENDPOINTS = ("next_event", "fixed300")
HOLDING = {"low": 0.0, "high": 0.9375}


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def audit(checks):
    return dict(all_pass=bool(checks) and all(checks.values()), checks=checks)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def direction(value):
    if not finite(value):
        return None
    return "high" if value > 0 else "low" if value < 0 else "tie"


def endpoint_pair(low, high, endpoint):
    """High-minus-low cost and reward have opposite signs; never fill missing truth."""
    arms = {"low": low or {}, "high": high or {}}
    checks = {}
    for arm, value in arms.items():
        complete = value.get("status") == "complete"
        ready = complete and all(finite(value.get(k)) for k in ("dt", "wait", "ride", "cost"))
        checks[arm+"/finite_complete"] = ready
        checks[arm+"/nonnegative_duration"] = ready and value["dt"] >= 0
        checks[arm+"/cost_definition"] = ready and math.isclose(
            value["cost"], 2*value["wait"]+value["ride"], rel_tol=1e-12, abs_tol=1e-8)
        if endpoint == "fixed300":
            checks[arm+"/fixed300_duration"] = ready and value["dt"] == 300
    ready = all(checks.values())
    delta = ({k: high[k]-low[k] for k in ("dt", "wait", "ride", "cost")} if ready else None)
    reward_delta = -delta["cost"] if delta is not None else None
    return dict(arms=arms, integrity=audit(checks), high_minus_low=delta,
        reward_high_minus_low=reward_delta, preferred_action=direction(reward_delta),
        zero_duration_arms=[arm for arm, value in arms.items()
                            if value.get("status") == "complete" and value.get("dt") == 0],
        missing_arms=[arm for arm, value in arms.items() if not value or value.get("status") == "missing"],
        truncated_arms=[arm for arm, value in arms.items() if value.get("status") == "truncated"],
        unequal_horizons=bool(ready and low["dt"] != high["dt"]))


def compare_prediction(prediction, endpoints):
    rows = {}
    for model in MODELS:
        values = prediction.get(model, {})
        qlow, qhigh = values.get("min_q_low"), values.get("min_q_high")
        supported = finite(qlow) and finite(qhigh)
        delta = qhigh-qlow if supported else None
        preference = direction(delta)
        comparisons = {}
        for endpoint, truth in endpoints.items():
            actual = truth["preferred_action"]
            comparisons[endpoint] = dict(
                same_direction=(preference == actual if preference is not None and actual is not None else None),
                strict_sign_agreement=(preference == actual if preference in ARMS and actual in ARMS else None))
        rows[model] = dict(supported=supported, min_q_low=qlow, min_q_high=qhigh,
            min_q_high_minus_low=delta, preferred_action=preference, endpoint_comparison=comparisons)
    return rows


def branch_checks(value, arm, selected, external_seed):
    counters = value.get("counters", {})
    delta = counters.get("delta")
    budget = value.get("budget", {})
    execution = value.get("execution", {})
    return dict(complete=value.get("status") == "complete",
        runner_integrity=value.get("integrity", {}).get("all_pass") is True,
        arm=value.get("arm") == arm, same_issue_state=value.get("selected") == selected,
        same_external_seed=value.get("external_seed") == external_seed,
        requested_holding=execution.get("requested_hold_seconds") == HOLDING[arm],
        actual_holding=execution.get("executed_hold_seconds") == HOLDING[arm],
        native_applied_holding=execution.get("native_applied_hold_seconds") == HOLDING[arm],
        zero_suffix_training=isinstance(delta, dict) and bool(delta) and all(x == 0 for x in delta.values()),
        counters_unchanged=counters.get("before") is not None and counters.get("before") == counters.get("after"),
        zero_replay_labels=budget.get("replay_target_labels") == 0,
        zero_replay_updates=budget.get("replay_policy_updates") == 0,
        one_diagnostic_truth=budget.get("new_truth_labels") == 1,
        suffix300=budget.get("suffix_sim_seconds") == 300)


def parallel_checks(fork, branches):
    clocks = {arm: value.get("execution_wall", {}) for arm, value in branches.items()}
    timed = len(clocks) == 2 and all(all(finite(clock.get(k)) for k in
        ("native_started_wall", "native_finished_wall")) for clock in clocks.values())
    overlap = (min(x["native_finished_wall"] for x in clocks.values())-
               max(x["native_started_wall"] for x in clocks.values())) if timed else None
    checks = dict(fork_integrity=fork.get("all_pass") is True,
        distinct_children=len(clocks) == 2 and all(clocks[a].get("pid") is not None for a in ARMS)
            and clocks["low"].get("pid") != clocks["high"].get("pid"),
        native_overlap=timed and overlap > 0,
        fork_clocks_exact=fork.get("children") == clocks,
        zero_child_exitcodes=fork.get("child_exitcodes") == {"low": 0, "high": 0},
        parent_clock_unchanged=fork.get("native_before") is not None and fork.get("native_before") == fork.get("native_after"),
        parent_training_unchanged=fork.get("counters_before") is not None and fork.get("counters_before") == fork.get("counters_after"),
        parent_rng_unchanged=fork.get("parent_rng_unchanged") is True,
        parent_trace_unchanged=fork.get("parent_trace_cookie_unchanged") is True,
        parent_routes_unchanged=fork.get("route_streams", {}).get("parent_offsets_unchanged") is True)
    return dict(integrity=audit(checks), native_overlap_seconds=overlap,
                children=clocks, child_exitcodes=fork.get("child_exitcodes"))


def analyze(root):
    root = Path(root)
    cases, pending, failures = [], [], []
    replay_budget = dict(parent_full_day_replay_seconds=0., prefix_native_seconds=0., replayed_feedback_slots=0,
                         replayed_main_updates=0, replayed_cumulative_shadow_updates=0)
    suffix_seconds, completed_branches = 0., 0
    for scenario in SCENARIOS:
        path = root/"runs"/"seed6202"/scenario
        main = read(path/"summary.json")
        if main is None:
            failure = read(path/"failure.json")
            (failures if failure is not None else pending).append(scenario)
            cases.append(dict(scenario=scenario, status="failed" if failure else "missing", failure=failure))
            continue
        selected = main.get("selected", {})
        external_seed = main.get("external_seed")
        branches = {arm: read(path/"branches"/arm/"summary.json") or {} for arm in ARMS}
        fork = read(path/"branches"/"live_fork.json") or {}
        checks = dict(main_complete=main.get("status") == "complete",
            main_integrity=main.get("integrity", {}).get("all_pass") is True,
            selected_7X=selected.get("line_id") == "7X",
            selected_tail=finite(selected.get("started_at")) and 18000 <= selected["started_at"] < 24000,
            external_seed_recorded=external_seed is not None)
        budget = main.get("budget", {})
        for field, expected in (("main_feedback_slots", 285), ("main_updates", 456),
                                ("cumulative_shadow_updates", 496)):
            checks[field] = budget.get(field) == expected
        for arm, value in branches.items():
            checks.update({arm+"/"+k:v for k,v in branch_checks(value, arm, selected, external_seed).items()})
            completed_branches += value.get("status") == "complete"
            suffix_seconds += value.get("budget", {}).get("suffix_sim_seconds", 0)
            if not value:
                branch_failure = read(path/"branches"/arm/"failure.json")
                (failures if branch_failure is not None else pending).append(scenario+"/"+arm)
        parallel = parallel_checks(fork, branches)
        checks["actual_parallel"] = parallel["integrity"]["all_pass"]
        other_actions = [branches[arm].get("native", {}).get("other_batch_actions") for arm in ARMS]
        checks["other_first_batch_controls_equal"] = other_actions[0] is not None and other_actions[0] == other_actions[1]
        endpoints = {endpoint:endpoint_pair(branches["low"].get("truth", {}).get(endpoint),
            branches["high"].get("truth", {}).get(endpoint), endpoint) for endpoint in ENDPOINTS}
        checks.update({endpoint+"/truth":value["integrity"]["all_pass"] for endpoint,value in endpoints.items()})
        prediction = compare_prediction(main.get("prediction", {}), endpoints)
        checks["predictions_supported"] = all(value["supported"] for value in prediction.values())
        replay_budget["prefix_native_seconds"] += budget.get("prefix_replay_seconds", 0)
        replay_budget["parent_full_day_replay_seconds"] += budget.get("native_replay_seconds", 0)
        replay_budget["replayed_feedback_slots"] += budget.get("main_feedback_slots", 0)
        replay_budget["replayed_main_updates"] += budget.get("main_updates", 0)
        replay_budget["replayed_cumulative_shadow_updates"] += budget.get("cumulative_shadow_updates", 0)
        cases.append(dict(scenario=scenario, status="complete" if all(checks.values()) else "integrity_failed",
            selected={k:selected.get(k) for k in ("sequence", "started_at", "line_id", "station_id", "vehicle_id", "trip_id")},
            external_seed=external_seed, integrity=audit(checks),
            prefix_integrity=main.get("integrity"), parallel=parallel,
            execution={arm:branches[arm].get("execution") for arm in ARMS},
            endpoints=endpoints, predictions=prediction))
    complete = len(cases) == 2 and all(case.get("status") == "complete" for case in cases)
    status = "complete" if complete else "failed" if failures else "incomplete" if pending else "integrity_failed"
    return dict(status=status, integrity=dict(all_pass=complete,
        failed=[case["scenario"] for case in cases if case.get("status") != "complete"]),
        pending=pending, failures=failures, cases=cases,
        budget=dict(**replay_budget, planned_prefixes=2, planned_control_branches=4,
            completed_control_branches=completed_branches, planned_suffix_seconds=1200,
            observed_suffix_seconds=suffix_seconds, new_training_labels=0,
            truth_labels_enter_training=0, suffix_optimizer_updates=0),
        interpretation=dict(prediction="Native min-Q(high)-min-Q(low); positive favors 0.9375s.",
            replay_budget="prefix_native_seconds is included in parent_full_day_replay_seconds; it is not extra replay.",
            cost="Waiting and riding are raw person-seconds; cost=2*wait+ride.",
            truth="High-minus-low cost is negated for reward high-minus-low. Positive reward favors 0.9375s.",
            next_event="Each arm's next control event may occur at a different time; this endpoint is not a common-duration cost comparison.",
            continuation="Only the first action differs; both suffixes use the same frozen fresh-origin controller.",
            scope="Two v28-selected 6202 tail 7X states are a local diagnostic, not independent policy confirmation."),
        next_decision="Review the two paired truth contrasts before proposing any new control experiment.")


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    while True:
        value = analyze(args.root)
        if value["status"] != "incomplete" or not args.wait:
            break
        time.sleep(20)
    data = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n"
    if len(data.encode()) > 20000:
        raise RuntimeError("v29 diagnostic summary exceeds 20KB")
    with args.out.open("x") as output:
        output.write(data)
    print(json.dumps(dict(status=value["status"], bytes=len(data.encode()), integrity=value["integrity"])))
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        failure = args.out.with_name(args.out.stem+"_failure.json")
        with failure.open("x") as output:
            json.dump(dict(status="failed", type=type(error).__name__, message=str(error)), output)
        raise


if __name__ == "__main__":
    main()
