"""Resident v31 Bellman/memory diagnosis; only a bounded summary leaves server."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_physical_execution_v30_20261023 import analyze as v30

ARMS, MODELS, PAIRS = v30.ARMS, v30.MODELS, v30.PAIRS
SCENARIOS = v30.SCENARIOS
BELLMAN_FIELDS = ("q_min", "next_value", "value_contribution", "sac_contribution",
                  "bootstrap_contribution", "target")
CANONICAL_FIELDS = ("s", "ns", "raw_s", "raw_ns", "r", "dt")


def read(path):
    return v30.read(Path(path))


def jsonl(path):
    if not path.is_file():
        return []
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n"


def delta(left, right):
    return right-left if v30.finite(left) and v30.finite(right) else None


def witness_difference(left, right):
    """Keep the first few changed coordinates, never emit whole native arrays."""
    if isinstance(left, list) and isinstance(right, list):
        changed = [index for index in range(max(len(left), len(right)))
                   if (left[index] if index < len(left) else None) !=
                      (right[index] if index < len(right) else None)]
        return dict(count=len(changed), first=[[index,
            left[index] if index < len(left) else None,
            right[index] if index < len(right) else None] for index in changed[:3]])
    return [left, right]


def event_rows(rows):
    return [row for row in rows if row.get("kind") in ("global_transition", "cutoff_actual_without_issue")]


def event_identity(row):
    transition = row.get("transition", {})
    fields = ("line_id", "vehicle_id", "trip_id", "station_id")
    values = tuple(transition.get(field) for field in fields)
    return values if all(value is not None for value in values) else None


def align_events(left, right):
    """Match logical bus/stop occurrences, so inserted events cannot shift a zip."""
    def index(rows):
        occurrences, indexed, unknown = defaultdict(int), {}, 0
        for row in event_rows(rows):
            key = event_identity(row)
            if key is None:
                unknown += 1
                continue
            occurrence = occurrences[key]
            occurrences[key] += 1
            indexed[key+(occurrence,)] = row
        return indexed, unknown
    a, unknown_a = index(left)
    b, unknown_b = index(right)
    shared = a.keys() & b.keys()
    pairs = [(key, a[key], b[key]) for key in shared]
    pairs.sort(key=lambda item: (max(item[1]["transition"]["started_at"],
                                     item[2]["transition"]["started_at"]), repr(item[0])))
    return pairs, dict(matched=len(pairs), unmatched=[len(a)-len(shared), len(b)-len(shared)],
                       identity_missing=[unknown_a, unknown_b])


def row_clock(row, endpoint="started_at"):
    return row.get("transition", {}).get(endpoint)


def event_witness(key, left, right):
    a, b = left["transition"], right["transition"]
    return dict(event=list(key), issue=[a.get("started_at"), b.get("started_at")],
                end=[a.get("completed_at"), b.get("completed_at")],
                strict_same_clock=all(a.get(field) == b.get(field) for field in ("started_at", "completed_at")),
                same_next_vehicle=a.get("next_vehicle_id") == b.get("next_vehicle_id"))


def first_ledger_difference(left, right, section, fields):
    a = {row["relative_seconds"]: row for row in left if v30.finite(row.get("relative_seconds"))}
    b = {row["relative_seconds"]: row for row in right if v30.finite(row.get("relative_seconds"))}
    common = sorted(a.keys() & b.keys())
    first = None
    for relative in common:
        av, bv = a[relative], b[relative]
        if section:
            av, bv = av.get(section) or {}, bv.get(section) or {}
        changes = {field: witness_difference(av.get(field), bv.get(field)) for field in fields
                   if av.get(field) is not None and bv.get(field) is not None and av.get(field) != bv.get(field)}
        if changes:
            first = dict(relative_seconds=relative, clock_seconds=[a[relative].get("clock_seconds"),
                         b[relative].get("clock_seconds")], fields=changes)
            break
    return dict(common_clocks=len(common), unmatched_clocks=[len(a)-len(common), len(b)-len(common)], first=first)


def model_values(row, model):
    return row.get("bellman", {}).get("models", {}).get(model, {})


def memory_contrast(left, right):
    """Separate the raw executed command from subsequent physical observations."""
    fields = ("executed_hold_seconds", "elapsed_since_apply_seconds",
              "departed_observed", "next_same_vehicle_arrived_observed")
    a, b = left.get("memory_s") or {}, right.get("memory_s") or {}
    changes = {field: [a.get(field), b.get(field)] for field in fields
               if a.get(field) != b.get(field)}
    for field in v30.PHYSICAL_FIELDS:
        av = (a.get("latest_physical") or {}).get(field)
        bv = (b.get("latest_physical") or {}).get(field)
        if av is not None and bv is not None and av != bv:
            changes["physical/"+field] = witness_difference(av, bv)
    return changes


def causal_memory(memory):
    if not memory:
        return False
    clock = memory.get("observed_at")
    if not v30.finite(clock):
        return False
    applied = memory.get("applied_real_clock")
    physical = (memory.get("latest_physical") or {}).get("native_clock_seconds")
    return ((applied is None or applied <= clock) and
            (physical is None or physical <= clock) and
            (memory.get("execution_applied") is True or memory.get("executed_hold_seconds") is None))


def visibility_contrast(left, right, left_ticks, right_ticks):
    pairs, alignment = align_events(left, right)
    first, witnesses, first_alignment, first_events, clock_different = {}, {}, {}, {}, 0
    common_ns = {model: dict(compared=0, bootstrap_violations=0, target_compared=0,
                             target_violations=0, first_violation=None) for model in MODELS}
    memory = dict(first_command=None, first_physical_flag=None, first_other_vehicle_issue=None,
                  first_physical_other_vehicle_issue=None)
    first_target = {}
    for key, a, b in pairs:
        ta, tb = a["transition"], b["transition"]
        base = event_witness(key, a, b)
        clock_different += any(ta.get(field) != tb.get(field) for field in ("started_at", "completed_at"))
        for field in CANONICAL_FIELDS:
            # A missing ns is a censored endpoint, never an observed state change.
            if ta.get(field) is not None and tb.get(field) is not None and ta[field] != tb[field] and field not in first:
                first[field] = base["end"] if field in ("r", "dt", "ns", "raw_ns") else base["issue"]
                first_alignment[field] = [base["strict_same_clock"], base["same_next_vehicle"]]
                first_events[field] = base
                if field in ("s", "ns", "r"):
                    witnesses[field] = dict(**base, difference=witness_difference(ta[field], tb[field]))
        changes = memory_contrast(a, b)
        if "executed_hold_seconds" in changes and memory["first_command"] is None:
            memory["first_command"] = dict(**base, fields={field: changes[field] for field in changes
                if field == "executed_hold_seconds"})
        if any(field in changes for field in ("departed_observed", "next_same_vehicle_arrived_observed")) and memory["first_physical_flag"] is None:
            memory["first_physical_flag"] = dict(**base, fields={field: changes[field] for field in changes
                if field in ("departed_observed", "next_same_vehicle_arrived_observed")})
        ma, mb = a.get("memory_s") or {}, b.get("memory_s") or {}
        if changes and ma.get("actor_is_selected_vehicle") is False and mb.get("actor_is_selected_vehicle") is False and memory["first_other_vehicle_issue"] is None:
            memory["first_other_vehicle_issue"] = dict(**base, changed_fields=list(changes),
                canonical_s_equal=ta.get("s") == tb.get("s"))
        physical_changes = {field: value for field, value in changes.items() if field.startswith("physical/")
                            or field in ("departed_observed", "next_same_vehicle_arrived_observed")}
        if physical_changes and ma.get("actor_is_selected_vehicle") is False and mb.get("actor_is_selected_vehicle") is False and memory["first_physical_other_vehicle_issue"] is None:
            memory["first_physical_other_vehicle_issue"] = dict(**base, fields=physical_changes,
                canonical_s_equal=ta.get("s") == tb.get("s"))
        for model in MODELS:
            av, bv = model_values(a, model), model_values(b, model)
            for field in BELLMAN_FIELDS:
                if av.get(field) is not None and bv.get(field) is not None and av[field] != bv[field]:
                    name = model+"/"+field
                    if name not in first:
                        first[name] = base["issue"] if field == "q_min" else base["end"]
                        first_alignment[name] = [base["strict_same_clock"], base["same_next_vehicle"]]
                        first_events[name] = base
                    if field == "target" and model not in first_target:
                        first_target[model] = dict(**base, delta={item: delta(av.get(item), bv.get(item))
                                                                for item in BELLMAN_FIELDS})
            if ta.get("ns") is not None and ta.get("ns") == tb.get("ns") and ta.get("terminal") == tb.get("terminal"):
                value = common_ns[model]
                if av.get("bootstrap_contribution") is not None and bv.get("bootstrap_contribution") is not None:
                    value["compared"] += 1
                    bad = av["bootstrap_contribution"] != bv["bootstrap_contribution"]
                    value["bootstrap_violations"] += bad
                    if bad and value["first_violation"] is None:
                        value["first_violation"] = base
                if ta.get("r") == tb.get("r") and av.get("target") is not None and bv.get("target") is not None:
                    value["target_compared"] += 1
                    bad = av["target"] != bv["target"]
                    value["target_violations"] += bad
                    if bad and value["first_violation"] is None:
                        value["first_violation"] = base
    alignment["different_event_clocks"] = clock_different
    physical_fields = ("is_at_bus_stop", "stop_state", "speed_mps", "lane_id", "lane_position_m", "position_xy")
    costs = first_ledger_difference(left_ticks, right_ticks, None, ("wait", "ride", "cost"))
    physical = first_ledger_difference(left_ticks, right_ticks, "physical", physical_fields)
    return dict(alignment=alignment, first_differences=first, first_onset_alignment=first_alignment,
        first_onset_events=first_events, first_witness=witnesses,
        first_target_difference=first_target, common_ns=common_ns, memory=memory,
        common_clock_cost=costs, common_clock_physical=physical)


def compact_endpoints(branches):
    result = {}
    for endpoint in v30.ENDPOINTS:
        value = v30.endpoint_set(branches, endpoint)
        result[endpoint] = dict(integrity=value["integrity"],
            arms={arm: [value["arms"][arm].get(key) for key in ("dt", "wait", "ride", "cost")] for arm in ARMS},
            deltas={key: [row["delta"].get(field) for field in ("dt", "wait", "ride", "cost")]
                    if row["delta"] else None for key, row in value["contrasts"].items()},
            zero_duration=value["zero_duration_arms"], missing=value["missing_arms"],
            truncated=value["truncated_arms"])
    return result


def compact_physical(branches):
    value = v30.physical_summary(branches)
    fields = ("native_clock_seconds", "ledger_clock_seconds", "action_step_length_seconds",
              "simulation_step_length_seconds", "is_at_bus_stop", "stop_state")
    return dict(issue={key: value["issue_state"].get(key) for key in fields},
        arms={arm: dict(departure=value["arms"][arm]["departure"],
            next_arrival=value["arms"][arm]["next_same_vehicle_arrival"],
            missing=value["arms"][arm]["missing"]) for arm in ARMS},
        contrasts={key: dict(status=row["status"], departure_delta=row["actual_departure"]["delta_seconds"],
            next_arrival_delta=row["next_same_vehicle_arrival"]["delta_seconds"])
            for key, row in value["contrasts"].items()})


def compact_predictions(main, branches):
    endpoints = {name: v30.endpoint_set(branches, name) for name in v30.ENDPOINTS}
    values = v30.compare_prediction(main.get("prediction", {}), endpoints, v30.physical_summary(branches))
    return {model: dict(min_q=row["min_q"], order=row["rank_high_to_low"],
        fixed300={key: row["fixed300"][key] for key in ("chosen_action", "oracle_actions", "realized_regret")},
        deltas={key: item["min_q_delta"] for key, item in row["contrasts"].items()},
        signs={key: item["strict_sign_agreement"] for key, item in row["contrasts"].items()})
        for model, row in values.items()}


def analyze(root):
    root = Path(root)
    cases, pending, failures = [], [], []
    budget = dict(parent_full_day_replay_seconds=0, prefix_native_seconds=0,
        replayed_feedback_slots=0, replayed_main_updates=0, replayed_cumulative_shadow_updates=0,
        completed_control_branches=0, observed_suffix_seconds=0, new_training_labels=0,
        truth_labels_enter_training=0, suffix_optimizer_updates=0, frozen_transition_probes=0)
    for scenario in SCENARIOS:
        path = root/"runs"/"seed6202"/scenario
        main = read(path/"summary.json")
        if main is None:
            failure = read(path/"failure.json")
            (failures if failure else pending).append(scenario)
            cases.append(dict(scenario=scenario, status="failed" if failure else "missing", failure=failure))
            continue
        selected = main.get("selected", {})
        branches = {arm: read(path/"branches"/arm/"summary.json") or {} for arm in ARMS}
        fork = read(path/"branches"/"live_fork.json") or {}
        parallel = v30.parallel_checks(fork, branches)
        checks = dict(main_complete=main.get("status") == "complete",
            prefix_and_old_references=main.get("integrity", {}).get("all_pass") is True,
            same_external_seed=main.get("external_seed") == 106202,
            actual_three_arm_parallel=parallel["integrity"]["all_pass"])
        b = main.get("budget", {})
        for key, expected in (("main_feedback_slots", 285), ("main_updates", 456),
                              ("cumulative_shadow_updates", 496), ("native_replay_seconds", 24000)):
            checks[key] = b.get(key) == expected
        rows, ticks = {}, {}
        counts = {}
        for arm, branch in branches.items():
            checks.update({arm+"/"+key: value for key, value in
                           v30.branch_checks(branch, arm, selected, main.get("external_seed")).items()})
            if not branch:
                failure = read(path/"branches"/arm/"failure.json")
                (failures if failure else pending).append(scenario+"/"+arm)
            arm_path = path/"branches"/arm
            rows[arm] = jsonl(arm_path/"visibility.jsonl")
            ticks[arm] = jsonl(arm_path/"ledger_ticks.jsonl")
            checks[arm+"/visibility_present"] = bool(rows[arm]) and bool(ticks[arm])
            evaluated = event_rows(rows[arm])
            summary = branch.get("visibility", {})
            budget["frozen_transition_probes"] += summary.get("evaluation_count", 0)
            checks[arm+"/causal_memory"] = summary.get("integrity", {}).get("all_pass") is True and all(
                causal_memory(row.get(field)) for row in rows[arm] for field in
                (("memory_s", "memory_ns") if row.get("kind") != "cutoff_partial" else ("memory_s",)))
            checks[arm+"/diagnostic_zero_training"] = summary.get("budget", {}).get("optimizer_updates") == 0 and summary.get("budget", {}).get("training_labels") == 0
            checks[arm+"/frozen_bellman_exact_rng"] = bool(evaluated) and all(
                row.get("bellman", {}).get("integrity", {}).get("rng_restored") is True and
                row.get("bellman", {}).get("integrity", {}).get("learner_rng_restored") is True and
                row.get("bellman", {}).get("integrity", {}).get("target_path_exact") is True for row in evaluated)
            counts[arm] = dict(global_transitions=len(evaluated),
                cutoff_actual_without_issue=sum(row.get("kind") == "cutoff_actual_without_issue" for row in rows[arm]),
                cutoff_partial=sum(row.get("kind") == "cutoff_partial" for row in rows[arm]),
                zero_duration=sum(row.get("transition", {}).get("dt") == 0 for row in evaluated),
                missing_next_state=sum(row.get("transition", {}).get("ns") is None for row in evaluated),
                ledger_ticks=len(ticks[arm]))
            budget["completed_control_branches"] += branch.get("status") == "complete"
            budget["observed_suffix_seconds"] += branch.get("budget", {}).get("suffix_sim_seconds", 0)
        visibility = {high+"_minus_"+low: visibility_contrast(rows[low], rows[high], ticks[low], ticks[high])
                      for low, high in PAIRS}
        checks["common_ns_exact_bootstrap_target"] = all(
            value["bootstrap_violations"] == value["target_violations"] == 0
            for contrast in visibility.values() for value in contrast["common_ns"].values())
        endpoints = compact_endpoints(branches)
        checks.update({endpoint+"/truth": row["integrity"]["all_pass"] for endpoint, row in endpoints.items()})
        for dest, source in (("parent_full_day_replay_seconds", "native_replay_seconds"),
            ("prefix_native_seconds", "prefix_replay_seconds"), ("replayed_feedback_slots", "main_feedback_slots"),
            ("replayed_main_updates", "main_updates"), ("replayed_cumulative_shadow_updates", "cumulative_shadow_updates")):
            budget[dest] += b.get(source, 0)
        physical = compact_physical(branches)
        memory_candidates, command_aliases = [], []
        for key, contrast in visibility.items():
            physical_memory = contrast["memory"]["first_physical_other_vehicle_issue"]
            if (physical["contrasts"][key]["status"] == "separated" and physical_memory is not None
                    and physical_memory["canonical_s_equal"]
                    and physical_memory["issue"][0] == physical_memory["issue"][1]):
                memory_candidates.append(key)
            if (physical["contrasts"][key]["status"] == "no_separation_observed"
                    and contrast["memory"]["first_command"] is not None and physical_memory is None):
                command_aliases.append(key)
        cases.append(dict(scenario=scenario, status="complete" if all(checks.values()) else "integrity_failed",
            selected={key: selected.get(key) for key in ("sequence", "started_at", "line_id", "vehicle_id", "trip_id", "station_id")},
            external_seed=main.get("external_seed"), integrity=v30.audit(checks),
            prefix_integrity=v30.compact_audit(main.get("integrity", {})),
            parallel=dict(integrity=parallel["integrity"], common_overlap_seconds=parallel["common_native_overlap_seconds"]),
            execution={arm: [branches[arm].get("execution", {}).get(field) for field in
                ("requested_hold_seconds", "executed_hold_seconds", "native_applied_hold_seconds")] for arm in ARMS},
            physical=physical, endpoints=endpoints, predictions=compact_predictions(main, branches),
            counts=counts, visibility=visibility, memory_gate=dict(
                hidden_physical_memory_at_matched_other_vehicle_issue=memory_candidates,
                command_difference_without_observed_physical_response=command_aliases)))
    complete = len(cases) == 2 and all(row.get("status") == "complete" for row in cases)
    status = "complete" if complete else "failed" if failures else "incomplete" if pending else "integrity_failed"
    return dict(status=status, integrity=dict(all_pass=complete,
        failed=[row["scenario"] for row in cases if row.get("status") != "complete"]), pending=pending,
        failures=failures, cases=cases, budget=dict(**budget, planned_control_branches=6,
        planned_suffix_seconds=1800, diagnostic_action_truths=budget["completed_control_branches"],
        frozen_model_probes=2*budget["frozen_transition_probes"],
        native_target_parity_probes=2*budget["completed_control_branches"]),
        columns=dict(endpoints=["dt", "wait", "ride", "cost"],
            execution=["requested_seconds", "executed_seconds", "native_applied_seconds"],
            event_identity=["line", "vehicle", "trip", "station", "occurrence"]),
        interpretation=dict(cost="Higher minus lower holding; person-seconds, C=2W+R; compare Q reward directions only.",
            alignment="Bus/trip/stop occurrences, never stream zip; retain different clocks/unmatched events.",
            bootstrap="Common-RNG frozen actual-target probes, not historical minibatch draws.",
            memory="Causal executed command can differ despite physical alias; gate uses same issue clock, not future end/agent.",
            partial="Censored costs without fabricated ns/target.",
            scope="Two states, frozen fresh continuation; diagnostic only, no critic training/truth feedback.",
            budget="Prefixes included in full parents; six suffixes additional."),
        next_decision="Use physical and reward/bootstrap onsets plus causal executed-memory visibility to choose at most one critic change; no automatic learning.")


def compact_export(value):
    """Columnize recurring observations, retaining every onset and failed key."""
    value = deepcopy(value)
    value["columns"].update(bellman_onsets=list(BELLMAN_FIELDS),
        target_deltas=list(BELLMAN_FIELDS),
        common_ns=["compared", "bootstrap_violations", "target_compared", "target_violations"],
        events=["identity", "issue_clock", "end_clock", "strict_same_issue_and_end_clocks", "same_next_vehicle"],
        onset_refs="Onsets/witness.event index case.events; s/raw_s/q_min use issue, ns/raw_ns/r/dt/value/bootstrap/target use end. Scalar clock means both arms.")
    value["columns"]["physical_event"] = ["status", "station_id", "clock_seconds", "time_source"]
    value["columns"]["visibility_counts"] = ["global_transitions", "cutoff_actual_without_issue", "cutoff_partial",
        "zero_duration", "missing_next_state", "ledger_ticks"]
    missing_fields = sorted({field for case in value["cases"] for row in case.get("physical", {}).get("arms", {}).values()
                             for field in row.get("missing", {})})
    value["columns"]["physical_missing"] = missing_fields
    def clock(value):
        return value[0] if isinstance(value, list) and len(value) == 2 and value[0] == value[1] else value
    for case in value["cases"]:
        for row in case.get("physical", {}).get("arms", {}).values():
            for field in ("departure", "next_arrival"):
                row[field] = [row[field].get(column) for column in value["columns"]["physical_event"]]
            row["missing"] = [row["missing"].get(field, 0) for field in missing_fields]
        case["counts"] = {arm: [row[field] for field in value["columns"]["visibility_counts"]]
                          for arm, row in case.get("counts", {}).items()}
        events = []
        def event_ref(item):
            record = [item["event"], clock(item["issue"]), clock(item["end"]),
                      item["strict_same_clock"], item["same_next_vehicle"]]
            if record not in events:
                events.append(record)
            return events.index(record)
        def project(item):
            if isinstance(item, list):
                return [project(child) for child in item]
            if not isinstance(item, dict):
                return item
            result = {key: project(child) for key, child in item.items()}
            if isinstance(item.get("event"), list) and "issue" in item and "end" in item:
                result["event"] = event_ref(item)
                for field in ("issue", "end", "strict_same_clock", "same_next_vehicle"):
                    result.pop(field)
            return result
        for key, contrast in case.get("visibility", {}).items():
            onset_events = contrast.pop("first_onset_events")
            onsets = {field: event_ref(row) for field, row in onset_events.items()}
            contrast = project(contrast)
            contrast["first_differences"] = {field: onsets[field] for field in CANONICAL_FIELDS if field in onsets}
            contrast["bellman_onsets"] = {model: [onsets.get(model+"/"+field) for field in BELLMAN_FIELDS]
                                            for model in MODELS}
            contrast.pop("first_onset_alignment")
            # Scalar onsets retain all fields. One changed state vector and
            # the first reward suffice as compact witnesses; full arrays stay
            # resident beside their recorded first-onset event.
            state_witnesses = [field for field in ("s", "ns") if field in contrast["first_witness"]]
            if len(state_witnesses) > 1:
                earliest = min(state_witnesses, key=lambda field: onset_events[field]["issue"]
                    if field == "s" else onset_events[field]["end"])
                for field in state_witnesses:
                    if field != earliest:
                        contrast["first_witness"].pop(field)
            for witness in contrast["first_target_difference"].values():
                witness["delta"] = [witness["delta"][field] for field in BELLMAN_FIELDS]
            for model, row in list(contrast["common_ns"].items()):
                result = dict(counts=[row[field] for field in value["columns"]["common_ns"]])
                if row["first_violation"] is not None:
                    result["first_violation"] = row["first_violation"]
                contrast["common_ns"][model] = result
            command = contrast["memory"]["first_command"]
            if command is not None:
                command.pop("fields")  # exact commands are in case.execution
            other = contrast["memory"]["first_other_vehicle_issue"]
            if other is not None:
                other.pop("changed_fields")
            for witness in (contrast["memory"]["first_physical_other_vehicle_issue"],
                            (contrast["common_clock_physical"].get("first"))):
                if witness is not None and len(witness.get("fields", {})) > 3:
                    witness["changed_field_count"] = len(witness["fields"])
                    witness["fields"] = dict(list(witness["fields"].items())[:3])
            case["visibility"][key] = contrast
        if events:
            case["events"] = events
    return value


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    while True:
        value = analyze(args.root)
        if value["status"] != "incomplete" or not args.wait:
            break
        time.sleep(20)
    value = compact_export(value)
    data = encoded(value)
    if len(data.encode()) > 20000:
        raise RuntimeError(f"v31 resident summary exceeds 20KB: {len(data.encode())}")
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
        with args.out.with_name(args.out.stem+"_failure.json").open("x") as output:
            json.dump(dict(status="failed", type=type(error).__name__, message=str(error)), output)
        raise


if __name__ == "__main__":
    main()
