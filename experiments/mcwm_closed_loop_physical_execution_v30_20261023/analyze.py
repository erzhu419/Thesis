"""Bounded native three-action physical-response and cost diagnosis for v30."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time


HERE = Path(__file__).resolve().parent
SCENARIOS = ("stable", "local_speed_drop")
ARMS = ("low", "mid", "high")
MODELS = ("fresh", "no_fresh")
PAIRS = (("low", "mid"), ("low", "high"), ("mid", "high"))
ENDPOINTS = ("next_event", "fixed300")
HOLDING = {"low": 0.0, "mid": 0.9375, "high": 1.875}
PHYSICAL_FIELDS = ("is_at_bus_stop", "stop_state", "speed_mps", "lane_id",
                   "lane_position_m", "position_xy")
STATE_FIELDS = ("native_clock_seconds", "ledger_clock_seconds", "action_step_length_seconds",
                "simulation_step_length_seconds", "last_action_time_seconds", "native_bus_state",
                "current_service_station_id", "next_stop_id", "registered_stop_duration_seconds") + PHYSICAL_FIELDS


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def audit(checks):
    return dict(all_pass=bool(checks) and all(checks.values()), checked_count=len(checks),
                failed=[key for key, value in checks.items() if not value])


def compact_audit(value):
    checks = value.get("checks")
    if isinstance(checks, dict):
        return dict(all_pass=value.get("all_pass") is True, checked_count=len(checks),
                    failed=[key for key, passed in checks.items() if not passed])
    return {key: value.get(key) for key in ("all_pass", "checked_count", "failed")}


def preferred(delta, low, high):
    return high if delta > 0 else low if delta < 0 else "tie"


def endpoint_set(branches, endpoint):
    fields = ("status", "dt", "wait", "ride", "cost", "next_vehicle_id", "terminal", "zero_duration")
    raw = {arm: branches[arm].get("truth", {}).get(endpoint) or {} for arm in ARMS}
    arms = {arm: {key: value[key] for key in fields if key in value} for arm, value in raw.items()}
    checks = {}
    for arm, value in arms.items():
        ready = value.get("status") == "complete" and all(finite(value.get(key)) for key in
                                                           ("dt", "wait", "ride", "cost"))
        checks[arm+"/finite_complete"] = ready
        checks[arm+"/nonnegative_duration"] = ready and value["dt"] >= 0
        checks[arm+"/cost_definition"] = ready and math.isclose(
            value["cost"], 2*value["wait"]+value["ride"], rel_tol=1e-12, abs_tol=1e-8)
        if endpoint == "fixed300":
            checks[arm+"/fixed300_duration"] = ready and value["dt"] == 300
    contrasts = {}
    for low, high in PAIRS:
        ready = all(passed for key, passed in checks.items() if key.startswith((low+"/", high+"/")))
        delta = ({key: arms[high][key]-arms[low][key] for key in ("dt", "wait", "ride", "cost")}
                 if ready else None)
        reward = -delta["cost"] if delta is not None else None
        contrasts[high+"_minus_"+low] = dict(delta=delta, reward_delta=reward,
            preferred_action=preferred(reward, low, high) if reward is not None else None,
            unequal_horizons=bool(ready and arms[low]["dt"] != arms[high]["dt"]))
    return dict(arms=arms, integrity=audit(checks), contrasts=contrasts,
        zero_duration_arms=[arm for arm, value in arms.items()
                            if value.get("status") == "complete" and value.get("dt") == 0],
        missing_arms=[arm for arm, value in arms.items() if not value or value.get("status") == "missing"],
        truncated_arms=[arm for arm, value in arms.items() if value.get("status") == "truncated"])


def observed_event(value):
    return value.get("status") in ("observed", "complete") and finite(value.get("clock_seconds"))


def event_contrast(low, high):
    ready = observed_event(low) and observed_event(high)
    same_station = ready and low.get("station_id") is not None and low.get("station_id") == high.get("station_id")
    result = dict(comparable=bool(same_station), delta_seconds=high["clock_seconds"]-low["clock_seconds"]
                  if same_station else None)
    if not same_station:
        result.update(statuses=[low.get("status", "missing"), high.get("status", "missing")],
                      same_station=bool(same_station))
    return result


def physical_states(value):
    states = [value.get("apply_before", {}), value.get("apply_after", {})]
    states.extend(value.get("short_tick_samples", []))
    # The post-application observation is the physical response at that clock.
    return {row["native_clock_seconds"]: row for row in states
            if isinstance(row, dict) and finite(row.get("native_clock_seconds"))}


def physical_contrast(low, high):
    departure = event_contrast(low.get("departure", {}), high.get("departure", {}))
    arrival = event_contrast(low.get("next_same_vehicle_arrival", {}), high.get("next_same_vehicle_arrival", {}))
    left, right = physical_states(low), physical_states(high)
    shared = sorted(left.keys() & right.keys())
    first = None
    compared = 0
    for clock in shared:
        differences = {}
        for key in PHYSICAL_FIELDS:
            a, b = left[clock].get(key), right[clock].get(key)
            if a is not None and b is not None:
                compared += 1
                if a != b:
                    differences[key] = [a, b]
        if differences and first is None:
            first = dict(native_clock_seconds=clock, fields=differences)
    sources = []
    if departure["comparable"] and departure["delta_seconds"] != 0:
        sources.append("actual_departure")
    if arrival["comparable"] and arrival["delta_seconds"] != 0:
        sources.append("next_same_vehicle_arrival")
    if first is not None:
        sources.append("matched_clock_physical_state")
    # Missing observations remain insufficient, never an inferred departure tie.
    status = ("separated" if sources else "no_separation_observed"
              if departure["comparable"] and compared > 0 else "insufficient")
    return dict(status=status, separation_sources=sources, actual_departure=departure,
        next_same_vehicle_arrival=arrival,
        trajectory=dict(common_clock_samples=len(shared), compared_field_values=compared, first_difference=first))


def physical_summary(branches):
    raw = {arm: branches[arm].get("physical", {}) for arm in ARMS}
    arms = {}
    for arm, value in raw.items():
        samples = value.get("short_tick_samples", [])
        samples = [row for row in samples if isinstance(row, dict)]
        clocks = [row["native_clock_seconds"] for row in samples if finite(row.get("native_clock_seconds"))]
        before, after = value.get("apply_before", {}), value.get("apply_after", {})
        coverage = value.get("coverage", {})
        registered = after.get("registered_stop_duration_seconds")
        arms[arm] = dict(
            apply_after_changed={key: [before.get(key), after.get(key)] for key in STATE_FIELDS
                                 if before.get(key) != after.get(key)},
            registered_hold_minus_command_seconds=registered-HOLDING[arm] if finite(registered) else None,
            short_tick_coverage=dict(count=len(samples), first_clock=min(clocks) if clocks else None,
                last_clock=max(clocks) if clocks else None,
                action_step_lengths_seconds=sorted({row["action_step_length_seconds"] for row in samples
                                                     if finite(row.get("action_step_length_seconds"))})),
            **{key: {field: value.get(key, {}).get(field) for field in
                     ("status", "station_id", "clock_seconds", "time_source")}
               for key in ("departure", "native_history_departure", "next_same_vehicle_arrival", "native_history_next_arrival")},
            coverage=dict(checked_fields=len(coverage), not_observed=[key for key, observed in coverage.items() if not observed]),
            missing={key: count for key, count in value.get("missing", {}).items() if count})
    contrasts = {high+"_minus_"+low: physical_contrast(raw[low], raw[high]) for low, high in PAIRS}
    return dict(issue_state={key: raw["low"].get("apply_before", {}).get(key) for key in STATE_FIELDS},
        arms=arms, contrasts=contrasts,
        three_distinct_observed_responses=all(row["status"] == "separated" for row in contrasts.values()),
        insufficient_contrasts=[key for key, row in contrasts.items() if row["status"] == "insufficient"],
        missing_departure_arms=[arm for arm in ARMS if not observed_event(raw[arm].get("departure", {}))],
        missing_next_arrival_arms=[arm for arm in ARMS
                                  if not observed_event(raw[arm].get("next_same_vehicle_arrival", {}))])


def compare_prediction(prediction, endpoints, physical):
    rows = {}
    truth = endpoints["fixed300"]
    truth_ready = truth["integrity"]["all_pass"]
    costs = {arm: truth["arms"][arm]["cost"] for arm in ARMS} if truth_ready else {}
    for model in MODELS:
        values = prediction.get(model, {})
        q = {arm: values.get("min_q_"+arm) for arm in ARMS}
        supported = all(finite(value) for value in q.values())
        contrasts = {}
        for low, high in PAIRS:
            key = high+"_minus_"+low
            delta = q[high]-q[low] if supported else None
            choice = preferred(delta, low, high) if delta is not None else None
            comparisons = {}
            for endpoint, actual in endpoints.items():
                true_choice = actual["contrasts"][key]["preferred_action"]
                comparisons[endpoint] = choice == true_choice if choice in (low, high) and true_choice in (low, high) else None
            contrasts[key] = dict(min_q_delta=delta, preferred_action=choice,
                physical_status=physical["contrasts"][key]["status"], strict_sign_agreement=comparisons)
        top = [arm for arm in ARMS if q[arm] == max(q.values())] if supported else []
        chosen = top[0] if top else None
        best = min(costs.values()) if costs else None
        rows[model] = dict(supported=supported, min_q=q,
            rank_high_to_low=sorted(ARMS, key=lambda arm: -q[arm]) if supported else None,
            top_actions=top, contrasts=contrasts,
            fixed300=dict(chosen_action=chosen, tie_break="first in low/mid/high order",
                oracle_actions=[arm for arm in ARMS if costs[arm] == best] if costs else None,
                realized_regret=costs[chosen]-best if chosen is not None and costs else None,
                top_action_regret_range=[min(costs[arm]-best for arm in top),
                                        max(costs[arm]-best for arm in top)] if top and costs else None))
    return rows


def branch_checks(value, arm, selected, external_seed):
    counters = value.get("counters", {})
    delta = counters.get("delta")
    budget = value.get("budget", {})
    execution = value.get("execution", {})
    before = value.get("physical", {}).get("apply_before", {})
    return dict(complete=value.get("status") == "complete",
        runner_integrity=value.get("integrity", {}).get("all_pass") is True,
        arm=value.get("arm") == arm, same_issue_state=value.get("selected") == selected,
        same_external_seed=value.get("external_seed") == external_seed,
        requested_holding=execution.get("requested_hold_seconds") == HOLDING[arm],
        actual_holding=execution.get("executed_hold_seconds") == HOLDING[arm],
        native_applied_holding=execution.get("native_applied_hold_seconds") == HOLDING[arm],
        real_clock_recorded=finite(before.get("native_clock_seconds")),
        action_step_length_recorded=finite(before.get("action_step_length_seconds")),
        zero_suffix_training=isinstance(delta, dict) and bool(delta) and all(x == 0 for x in delta.values()),
        counters_unchanged=counters.get("before") is not None and counters.get("before") == counters.get("after"),
        zero_replay_labels=budget.get("replay_target_labels") == 0,
        zero_replay_updates=budget.get("replay_policy_updates") == 0,
        one_diagnostic_truth=budget.get("new_truth_labels") == 1,
        suffix300=budget.get("suffix_sim_seconds") == 300)


def parallel_checks(fork, branches):
    clocks = {arm: value.get("execution_wall", {}) for arm, value in branches.items()}
    timed = len(clocks) == 3 and all(all(finite(clock.get(key)) for key in
        ("native_started_wall", "native_finished_wall")) for clock in clocks.values())
    overlap = (min(row["native_finished_wall"] for row in clocks.values())-
               max(row["native_started_wall"] for row in clocks.values())) if timed else None
    pids = [clocks[arm].get("pid") for arm in ARMS]
    checks = dict(fork_integrity=fork.get("all_pass") is True,
        distinct_children=all(pid is not None for pid in pids) and len(set(pids)) == 3,
        native_overlap=timed and overlap > 0,
        fork_clocks_exact=fork.get("children") == clocks,
        zero_child_exitcodes=fork.get("child_exitcodes") == dict.fromkeys(ARMS, 0),
        parent_clock_unchanged=fork.get("native_before") is not None and fork.get("native_before") == fork.get("native_after"),
        parent_training_unchanged=fork.get("counters_before") is not None and fork.get("counters_before") == fork.get("counters_after"),
        parent_rng_unchanged=fork.get("parent_rng_unchanged") is True,
        parent_trace_unchanged=fork.get("parent_trace_cookie_unchanged") is True,
        parent_routes_unchanged=fork.get("route_streams", {}).get("parent_offsets_unchanged") is True)
    return dict(integrity=audit(checks), common_native_overlap_seconds=overlap,
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
            same_external_seed=external_seed == 106202)
        budget = main.get("budget", {})
        for field, expected in (("main_feedback_slots", 285), ("main_updates", 456),
                                ("cumulative_shadow_updates", 496), ("native_replay_seconds", 24000)):
            checks[field] = budget.get(field) == expected
        for arm, value in branches.items():
            checks.update({arm+"/"+key: passed for key, passed in branch_checks(value, arm, selected, external_seed).items()})
            completed_branches += value.get("status") == "complete"
            suffix_seconds += value.get("budget", {}).get("suffix_sim_seconds", 0)
            if not value:
                branch_failure = read(path/"branches"/arm/"failure.json")
                (failures if branch_failure is not None else pending).append(scenario+"/"+arm)
        parallel = parallel_checks(fork, branches)
        checks["actual_three_arm_parallel"] = parallel["integrity"]["all_pass"]
        other_actions = [branches[arm].get("native", {}).get("other_batch_actions") for arm in ARMS]
        checks["other_first_batch_controls_equal"] = other_actions[0] is not None and all(
            value == other_actions[0] for value in other_actions)
        issue_states = [{key: branches[arm].get("physical", {}).get("apply_before", {}).get(key)
                         for key in STATE_FIELDS} for arm in ARMS]
        checks["same_physical_issue_state"] = all(value == issue_states[0] for value in issue_states)
        physical = physical_summary(branches)
        endpoints = {endpoint: endpoint_set(branches, endpoint) for endpoint in ENDPOINTS}
        checks.update({endpoint+"/truth": value["integrity"]["all_pass"] for endpoint, value in endpoints.items()})
        prediction = compare_prediction(main.get("prediction", {}), endpoints, physical)
        checks["predictions_supported"] = all(value["supported"] for value in prediction.values())
        for dest, source in (("prefix_native_seconds", "prefix_replay_seconds"),
                             ("parent_full_day_replay_seconds", "native_replay_seconds"),
                             ("replayed_feedback_slots", "main_feedback_slots"),
                             ("replayed_main_updates", "main_updates"),
                             ("replayed_cumulative_shadow_updates", "cumulative_shadow_updates")):
            replay_budget[dest] += budget.get(source, 0)
        cases.append(dict(scenario=scenario, status="complete" if all(checks.values()) else "integrity_failed",
            selected={key: selected.get(key) for key in ("sequence", "started_at", "line_id", "station_id", "vehicle_id", "trip_id")},
            external_seed=external_seed, integrity=audit(checks), prefix_integrity=compact_audit(main.get("integrity", {})),
            physical=physical, parallel=parallel,
            execution={arm: {key: branches[arm].get("execution", {}).get(key) for key in
                             ("requested_hold_seconds", "executed_hold_seconds", "native_applied_hold_seconds", "executed_action")}
                       for arm in ARMS}, endpoints=endpoints, predictions=prediction))
    complete = len(cases) == 2 and all(case.get("status") == "complete" for case in cases)
    status = "complete" if complete else "failed" if failures else "incomplete" if pending else "integrity_failed"
    return dict(status=status, integrity=dict(all_pass=complete,
        failed=[case["scenario"] for case in cases if case.get("status") != "complete"]),
        pending=pending, failures=failures, cases=cases,
        budget=dict(**replay_budget, planned_prefixes=2, planned_control_branches=6,
            completed_control_branches=completed_branches, planned_suffix_seconds=1800,
            observed_suffix_seconds=suffix_seconds, diagnostic_action_truths=completed_branches,
            new_training_labels=0, truth_labels_enter_training=0, suffix_optimizer_updates=0),
        interpretation=dict(physical="Physical separation uses observed departure/arrival or matched-clock physical state, never requested holding alone.",
            replay_budget="prefix_native_seconds is included in parent_full_day_replay_seconds, not extra replay.",
            cost="Waiting/riding are raw person-seconds; cost=2*wait+ride. Contrasts are higher-holding minus lower-holding.",
            prediction="min-Q delta is a reward direction; positive favors the higher holding. Q/cost amplitudes are not comparable.",
            next_event="Each next control event can have a different horizon; fixed300 has a common clock cutoff.",
            missing="Missing departure/next arrival is not a measured tie; zero-duration and missing/truncated truth remain separate.",
            continuation="Only the first action differs; all suffixes use the same frozen fresh-origin controller.",
            scope="Two selected 6202 tail 7X states are local action diagnostics, not independent policy confirmation."),
        next_decision="Confirm physical separation first; then inspect 300s directions and candidate regret before any new experiment.")


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
        raise RuntimeError("v30 diagnostic summary exceeds 20KB")
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
