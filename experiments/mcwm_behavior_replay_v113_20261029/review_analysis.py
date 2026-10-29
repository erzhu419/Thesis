"""Exact v111 replay checks and resident-only passenger/vehicle attribution."""
import argparse
import copy
import json
import math
import statistics
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RELATIVE = Path("experiments") / HERE.name
sys.path.insert(0, str(ROOT))
from experiments.mcwm_historical_coverage_v111_20261029 import analyze as original

read = original.read
finite = original.finite
write_once = original.write_once
expected_counts = original.expected_counts

# These are output locations, host placement and wall times, not simulated state.
# The observer receipt is new; it is checked separately against the exact ledger.
EXCLUDED_SUMMARY_PATHS = (
    ("execution_wall",), ("wall_seconds",), ("observer",),
    ("prefix", "trace_path"), ("prefix", "services_path"),
    ("packet_receipts_path",), ("metrics_path",),
    ("initialization", "runtime", "host"),
)


def semantic_summary(row):
    result = copy.deepcopy(row)
    for path in EXCLUDED_SUMMARY_PATHS:
        current = result
        for key in path[:-1]:
            current = current.get(key, {})
        current.pop(path[-1], None)
    return result


def summary_comparison(current, previous):
    left, right = semantic_summary(current), semantic_summary(previous)
    differing = [key for key in sorted(left.keys() | right.keys())
                 if key not in left or key not in right or left[key] != right[key]]
    return dict(all_pass=not differing, differing_top_level_fields=differing)


def prefix_comparison(current_folder, previous_folder, current, previous):
    a, b = current.get("prefix", {}), previous.get("prefix", {})
    checks = dict(prefix_metadata=all(a.get(key) == b.get(key)
        and a.get(key) is not None for key in ("clock", "service_events", "trace_rows", "service_rows")))
    logs = {name: original.compare_prefix(current_folder / name, previous_folder / name,
        a.get(key), b.get(key)) for name, key in (
            ("prefix_trace.jsonl", "trace_rows"), ("prefix_services.jsonl", "service_rows"))}
    return dict(all_pass=all(checks.values()) and all(row["all_pass"] for row in logs.values()),
                checks=checks, logs=logs)


def cell_reproduction(root, p, scene, arm, first_packet=False):
    root = Path(root)
    current_folder = root / RELATIVE / "runs" / scene / arm
    previous_folder = root / p["v111_root"] / "runs" / scene / arm
    name = "preflight.json" if first_packet else "summary.json"
    try:
        current, previous = read(current_folder / name), read(previous_folder / name)
    except (OSError, ValueError) as error:
        return dict(all_pass=False, error=str(error))
    updates = p["policy_updates_per_packet"] if first_packet else p["policy_update_budget"]
    checks = dict(current_complete=current.get("status") == "complete"
                  and current.get("integrity", {}).get("all_pass") is True,
        reference_complete=previous.get("status") == "complete"
                  and previous.get("integrity", {}).get("all_pass") is True,
        current_identity=original.identity(current, p, scene, arm),
        reference_identity=original.identity(previous, p, scene, arm),
        update_counts=current.get("counts") == original.expected_counts(updates),
        mechanism=original.mechanism_valid(current, arm, updates))
    summary = summary_comparison(current, previous)
    prefixes = prefix_comparison(current_folder, previous_folder, current, previous)
    logs, errors = {}, []
    if not first_packet:
        errors = original.run_errors(current, p, scene, arm) + original.period_values(current, p)[1]
        logs = {name: original.compare_json(current_folder / name, previous_folder / name)
                for name in ("metrics.json", "packet_receipts.json")}
        packets = len(range(p["feedback_first_deadline"], p["feedback_last_deadline"] + 1,
                            p["feedback_interval_seconds"]))
        logs["updates.jsonl"] = original.compare_prefix(current_folder / "updates.jsonl",
            previous_folder / "updates.jsonl", packets, packets)
    return dict(all_pass=all(checks.values()) and summary["all_pass"] and prefixes["all_pass"]
                and all(row["all_pass"] for row in logs.values()) and not errors,
        checks=checks, summary=summary, prefixes=prefixes, logs=logs, errors=errors)


def preflight(root, out_dir):
    root, out_dir = Path(root), Path(out_dir)
    destination = out_dir / "preflight_gate.json"
    if destination.exists():
        return read(destination)
    p = read(root / RELATIVE / "protocol.json")
    missing = []
    for arm in p["arms"]:
        folder = root / RELATIVE / "runs/stable" / arm
        if not (folder / "preflight.json").exists() and not (folder / "summary.json").exists():
            missing.append(arm)
    if missing:
        return dict(status="pending", missing=missing)
    cells = {arm: cell_reproduction(root, p, "stable", arm, first_packet=True)
             for arm in p["arms"]}
    passed = all(row["all_pass"] for row in cells.values())
    result = dict(status="complete" if passed else "failed", all_pass=passed,
        decision="PASS_LAUNCH_SHOCK_PAIR" if passed else "HOLD_TECHNICAL_FAILURE_NO_EXPANSION",
        cells=cells, interpretation="First-packet receipts and complete pre-update prefixes match v111. Full-day actions have no historical trace comparator.")
    return write_once(destination, result, p["summary_max_bytes"])


PHASES = ("service", "holding", "moving", "traffic_or_other")
WINDOWS = {"0_9000": (0, 9000), "9000_12000": (9000, 12000),
    "12000_18300": (12000, 18300), "18300_24000": (18300, 24000),
    "12600_12900": (12600, 12900)}
PARTITION = tuple(WINDOWS)[:4]


def phase_equal(a, b):
    # Fractional command ends are summed in different passenger/vehicle orders.
    # This rounding tolerance never applies to the integer native W/R ledger.
    return (set(a) == set(b) == set(PHASES)
        and all(finite(a[key]) and finite(b[key])
                and math.isclose(a[key], b[key], rel_tol=0., abs_tol=1e-6) for key in PHASES))


def zero():
    return dict(wait=0, ride=0, ride_phases={phase: 0 for phase in PHASES})


def add(total, row, sign=1, passenger=True):
    if passenger:
        total["wait"] += sign * row["wait"]
    total["ride"] += sign * row["ride"]
    for phase in PHASES:
        total["ride_phases"][phase] += sign * row["ride_phases"][phase]


def measured(row, passenger):
    keys = ("wait", "ride") if passenger else ("ride",)
    if not all(finite(row.get(key)) and row[key] >= 0 for key in keys):
        return False
    phases = row.get("ride_phases", {})
    return (set(phases) == set(PHASES)
        and all(finite(value) and value >= 0 for value in phases.values())
        and math.isclose(sum(phases.values()), row["ride"], rel_tol=0., abs_tol=1e-6))


def blocks():
    return {name: zero() for name in ("full_day", *WINDOWS)}


def row_blocks(row):
    return dict(full_day=row, **row["periods"])


def summed_rows(rows, passenger=True):
    total = blocks()
    for row in rows:
        for name, values in row_blocks(row).items():
            add(total[name], values, passenger=passenger)
    return total


def subtract(a, b):
    result = copy.deepcopy(a)
    add(result, b, -1)
    return result


def delta_blocks(a, b):
    return {name: subtract(a[name], b[name]) for name in a}


def report_block(row, denominator):
    return dict(row, cost=2 * row["wait"] + row["ride"],
        weighted_wait_per_passenger=2 * row["wait"] / denominator,
        ride_per_passenger=row["ride"] / denominator,
        cost_per_passenger=(2 * row["wait"] + row["ride"]) / denominator)


def report_blocks(rows, denominator):
    return {name: report_block(row, denominator) for name, row in rows.items()}


def load_records(path, passenger):
    """Read resident records; exact sums detect incomplete or mis-timed logging."""
    records, stop_occurrences = {}, {}
    with Path(path).open() as handle:
        for index, line in enumerate(handle, 1):
            row = json.loads(line)
            if passenger:
                key = row["passenger_id"]
            else:
                # A skipped preceding stop changes the trip-wide visit_index.
                # Match by stop identity, using only within-stop occurrence if
                # the resident record contains another visit to that same stop.
                base = (row["trip_id"], row["stop_id"])
                occurrence = stop_occurrences[base] = stop_occurrences.get(base, 0) + 1
                key = base if occurrence == 1 else (*base, occurrence)
                row["stop_occurrence"] = occurrence
            if key in records:
                raise ValueError(f"{path.name}: duplicate identity at row {index}")
            periods = row.get("periods", {})
            if set(periods) != set(WINDOWS) or not all(
                    measured(value, passenger) for value in (row, *periods.values())):
                raise ValueError(f"{path.name}: invalid or unbalanced phase record {index}")
            for field in (("wait", "ride") if passenger else ("ride",)):
                if sum(periods[name][field] for name in PARTITION) != row[field]:
                    raise ValueError(f"{path.name}: nonpartitioning {field} at row {index}")
            if not phase_equal({phase: sum(periods[name]["ride_phases"][phase]
                    for name in PARTITION) for phase in PHASES}, row["ride_phases"]):
                raise ValueError(f"{path.name}: nonpartitioning phases at row {index}")
            if passenger and type(row.get("completed")) is not bool:
                raise ValueError(f"{path.name}: missing completion state at row {index}")
            records[key] = row
    return records


def observed_cell(folder, run, p):
    receipt = read(folder / "observer_summary.json")
    passengers = load_records(folder / "passengers.jsonl", True)
    visits = load_records(folder / "visits.jsonl", False)
    totals, visit_totals = summed_rows(passengers.values()), summed_rows(visits.values(), False)
    metrics = {row["time_seconds"]: row for row in read(folder / "metrics.json")}
    expected = {"full_day": run["periods"]["full_day"]}
    for name, (start, end) in WINDOWS.items():
        a = metrics[start] if start else dict(accrued_wait_seconds=0, accrued_in_vehicle_seconds=0)
        b = metrics[end]
        expected[name] = dict(wait=b["accrued_wait_seconds"] - a["accrued_wait_seconds"],
            ride=b["accrued_in_vehicle_seconds"] - a["accrued_in_vehicle_seconds"])
    checks = dict(observer_complete=receipt.get("status") == "complete",
        observer_ledger_exact=receipt.get("ledger_exact") is True,
        observer_checks=bool(receipt.get("checks"))
            and all(value is True for value in receipt["checks"].values()),
        observer_endpoint=receipt.get("end") == p["horizon_seconds"],
        passenger_count=len(passengers) == p["full_day_departed_passengers"],
        completed_count=sum(row["completed"] for row in passengers.values())
            == run.get("passengers", {}).get("completed_passengers"),
        passenger_ledger_exact=all(totals[name][field] == expected[name][field]
            for name in expected for field in ("wait", "ride")),
        visit_ledger_exact=all(visit_totals[name]["ride"] == totals[name]["ride"]
            and phase_equal(visit_totals[name]["ride_phases"], totals[name]["ride_phases"]) for name in totals),
        receipt_totals_exact=all(receipt.get("passenger_totals", {}).get(key) == totals["full_day"][key]
            for key in ("wait", "ride")) and phase_equal(
                receipt.get("passenger_totals", {}).get("ride_phases", {}), totals["full_day"]["ride_phases"]))
    return dict(all_pass=all(checks.values()), checks=checks,
        receipt_errors=receipt.get("errors", []), totals=totals,
        traffic_or_other_reasons=receipt.get("traffic_or_other_reasons"),
        rejected_commands=receipt.get("rejected_commands"),
        runtime_evidence={key: receipt.get(key) for key in (
            "start", "end", "ticks", "issues", "commands", "maximum_count_gap", "maximum_ledger_gap")},
        passenger_records=len(passengers), visit_records=len(visits)), passengers, visits


def passenger_attribution(fresh, nofresh, denominator):
    common = fresh.keys() & nofresh.keys()
    groups = {"ride": {}, "completion": {}}
    for pid in common:
        a, b = fresh[pid], nofresh[pid]
        for kind, left, right in (("ride", a["ride"] > 0, b["ride"] > 0),
                                 ("completion", a["completed"], b["completed"])):
            name = "both" if left and right else "fresh_only" if left else "no_fresh_only" if right else "neither"
            group = groups[kind].setdefault(name, dict(passengers=0, delta=blocks()))
            group["passengers"] += 1
            for period in group["delta"]:
                add(group["delta"][period], row_blocks(a)[period])
                add(group["delta"][period], row_blocks(b)[period], -1)
    ride_deltas = [fresh[pid]["ride"] - nofresh[pid]["ride"] for pid in common
                   if fresh[pid]["ride"] > 0 and nofresh[pid]["ride"] > 0]
    journey_deltas = [fresh[pid]["completion_time"] - fresh[pid]["departure_time"]
                     - nofresh[pid]["completion_time"] + nofresh[pid]["departure_time"]
                     for pid in common if fresh[pid]["completed"] and nofresh[pid]["completed"]
                     and all(finite(row.get(key)) for row in (fresh[pid], nofresh[pid])
                             for key in ("completion_time", "departure_time"))]
    def distribution(values):
        return dict(n=len(values), mean=statistics.mean(values) if values else None,
                    median=statistics.median(values) if values else None)
    return dict(same_departed_ids=fresh.keys() == nofresh.keys(), common_passengers=len(common),
        only_fresh_ids=len(fresh.keys() - nofresh.keys()),
        only_no_fresh_ids=len(nofresh.keys() - fresh.keys()),
        by_ride_group=groups["ride"], by_completion_group=groups["completion"],
        both_ride_same_person_delta_seconds=distribution(ride_deltas),
        both_completed_same_person_journey_delta_seconds=distribution(journey_deltas))


def command_zero():
    return dict(paired=0, longer=0, shorter=0, equal=0, unknown=0,
                rejected_command_pairs=0,
                known_pairs=0, observed_delta_command_seconds=None,
                max_increase=None, max_decrease=None)


def add_command(total, a, b):
    total["paired"] += 1
    av, bv = a.get("command_hold_seconds"), b.get("command_hold_seconds")
    if a.get("command_accepted") is False or b.get("command_accepted") is False:
        total["rejected_command_pairs"] += 1
    if (not finite(av) or not finite(bv)
            or a.get("command_accepted") is not True or b.get("command_accepted") is not True):
        total["unknown"] += 1
        return
    delta = av - bv
    total["known_pairs"] += 1
    total["longer" if delta > 0 else "shorter" if delta < 0 else "equal"] += 1
    total["observed_delta_command_seconds"] = (total["observed_delta_command_seconds"] or 0) + delta
    detail = {key: b.get(key) for key in ("vehicle_id", "line_id", "trip_id", "stop_id", "stop_occurrence")}
    detail.update(fresh=av, no_fresh=bv, delta=delta,
        fresh_visit_index=a.get("visit_index"), no_fresh_visit_index=b.get("visit_index"))
    for key, eligible, condition in (("max_increase", delta > 0, lambda old: delta > old),
                                    ("max_decrease", delta < 0, lambda old: delta < old)):
        if eligible and (total[key] is None or condition(total[key]["delta"])):
            total[key] = detail


def visit_attribution(fresh, nofresh, denominator):
    common = fresh.keys() & nofresh.keys()
    commands = command_zero()
    command_periods = {name: command_zero() for name in WINDOWS}
    by_line, by_stop, visit_changes = {}, {}, []
    for key in fresh.keys() | nofresh.keys():
        a, b = fresh.get(key), nofresh.get(key)
        row = b or a
        delta = blocks()
        for sign, item in ((1, a), (-1, b)):
            if item is not None:
                for period, values in row_blocks(item).items():
                    add(delta[period], values, sign, passenger=False)
        for target, groupkey in ((by_line, str(row["line_id"])),
                (by_stop, str(row["line_id"]) + "/" + str(row["stop_id"]))):
            group = target.setdefault(groupkey, dict(visits=0, delta=blocks(), commands=command_zero()))
            group["visits"] += 1
            for period in delta:
                add(group["delta"][period], delta[period])
            if a is not None and b is not None:
                add_command(group["commands"], a, b)
        if a is not None and b is not None:
            add_command(commands, a, b)
            for name, (start, end) in WINDOWS.items():
                if finite(b.get("action_time")) and start <= b["action_time"] < end:
                    add_command(command_periods[name], a, b)
        visit_changes.append(dict(identity={field: row.get(field) for field in (
            "vehicle_id", "line_id", "trip_id", "stop_id", "stop_occurrence")},
            matched=a is not None and b is not None,
            fresh_visit_index=a.get("visit_index") if a is not None else None,
            no_fresh_visit_index=b.get("visit_index") if b is not None else None,
            fresh_vehicle_id=a.get("vehicle_id") if a else None,
            no_fresh_vehicle_id=b.get("vehicle_id") if b else None,
            fresh_command_seconds=a.get("command_hold_seconds") if a is not None else None,
            no_fresh_command_seconds=b.get("command_hold_seconds") if b is not None else None,
            fresh_departure=a.get("departure_time") if a is not None else None,
            no_fresh_departure=b.get("departure_time") if b is not None else None,
            delta=delta))
    largest = lambda rows, phase=None: sorted(rows, key=lambda item: abs(
        item["delta"]["full_day"]["ride_phases"][phase] if phase else item["delta"]["full_day"]["ride"]), reverse=True)[:4]
    stops = [dict(stop=key, **value) for key, value in by_stop.items()]
    def selected(rows):
        result = []
        for item in largest(rows) + largest(rows, "holding"):
            if item not in result:
                result.append(item)
        return result
    return dict(matched_visits=len(common), only_fresh_visits=len(fresh.keys() - nofresh.keys()),
        only_no_fresh_visits=len(nofresh.keys() - fresh.keys()), commands=commands,
        matching_identity_consistent=all(fresh[key].get(field) == nofresh[key].get(field)
            for key in common for field in ("line_id",)),
        fleet_assignment_different_visits=sum(fresh[key].get("vehicle_id") != nofresh[key].get("vehicle_id") for key in common),
        fleet_assignment_different_trips=len({key[0] for key in common if fresh[key].get("vehicle_id") != nofresh[key].get("vehicle_id")}),
        commands_by_reference_action_time=command_periods,
        by_line=by_line, largest_stop_changes=selected(stops), largest_visit_changes=selected(visit_changes),
        interpretation="Accepted command pairs use trip/stop identity, with within-stop occurrence for repeat visits. Trip-wide visit_index is descriptive only. Time groups use the reference action time. Missing or rejected commands are not counted as zero accepted dose. Visit ride includes the following route segment; phase association is not a counterfactual effect.")


def compact_scene(scene):
    if scene is None:
        return None
    passengers = copy.deepcopy(scene["passengers"])
    for field in ("by_ride_group", "by_completion_group"):
        for group in passengers[field].values():
            group["delta"] = group["delta"]["full_day"]
    visits = copy.deepcopy(scene["visits"])
    for group in visits["by_line"].values():
        group["delta"] = group["delta"]["full_day"]
        group["commands"].pop("max_increase")
        group["commands"].pop("max_decrease")
    for group in visits["commands_by_reference_action_time"].values():
        group.pop("max_increase")
        group.pop("max_decrease")
    for field in ("largest_stop_changes", "largest_visit_changes"):
        if field == "largest_visit_changes":
            visits["largest_holding_visit_changes"] = copy.deepcopy(sorted(visits[field],
                key=lambda row: abs(row["delta"]["full_day"]["ride_phases"]["holding"]), reverse=True)[:3])
        visits[field] = sorted(visits[field],
            key=lambda row: abs(row["delta"]["full_day"]["ride"]), reverse=True)[:3]
        for row in visits[field]:
            row["delta"] = row["delta"]["full_day"]
            if "commands" in row:
                row["commands"].pop("max_increase")
                row["commands"].pop("max_decrease")
    for row in visits["largest_holding_visit_changes"]:
        row["delta"] = row["delta"]["full_day"]
    return dict(fresh_minus_matched=scene["fresh_minus_matched"],
                passengers=passengers, visits=visits)


def analyze(root):
    root = Path(root)
    experiment = root / RELATIVE
    review = experiment / "review"
    review.mkdir(exist_ok=True)
    destination = review / "summary.json"
    if destination.exists():
        return read(destination)
    p = read(experiment / "protocol.json")
    paths = {scene + "/" + arm: experiment / "runs" / scene / arm
             for scene in p["scenarios"] for arm in p["arms"]}
    missing = [key for key, folder in paths.items() if not (folder / "summary.json").exists()]
    gate_path = experiment / "preflight_gate.json"
    gate = read(gate_path) if gate_path.exists() else None
    gate_failed = gate is not None and gate.get("all_pass") is False
    # A failed stable preflight prevents shock launch. Once both already-running
    # stable days terminate, emit HOLD with two unknown cells rather than wait.
    if missing and (not gate_failed or any(key.startswith("stable/") for key in missing)):
        return dict(status="pending", missing=missing)
    runs = {key: read(folder / "summary.json") for key, folder in paths.items() if key not in missing}
    reproduction, observations, errors, people, visits = {}, {}, [], {}, {}
    for scene in p["scenarios"]:
        for arm in p["arms"]:
            key = scene + "/" + arm
            if key in missing:
                reproduction[key] = None
                observations[key] = dict(all_pass=False, error="not_started_or_missing_terminal_receipt")
                errors.append(key + ":missing_terminal_receipt")
                continue
            reproduction[key] = cell_reproduction(root, p, scene, arm)
            if not reproduction[key]["all_pass"]:
                errors.append(key + ":v111_reproduction_failed")
            try:
                observations[key], people[key], visits[key] = observed_cell(paths[key], runs[key], p)
                if not observations[key]["all_pass"]:
                    errors.append(key + ":observer_ledger_failed")
            except (OSError, ValueError, KeyError, TypeError) as error:
                observations[key] = dict(all_pass=False, error=str(error))
                errors.append(key + ":observer_missing_or_invalid")
    if gate is None or gate.get("all_pass") is not True:
        errors.append("stable_preflight_not_passed")
    scenes = {}
    for scene in p["scenarios"]:
        a, b = scene + "/fresh", scene + "/no_fresh"
        if not all(observations[key]["all_pass"] for key in (a, b)):
            scenes[scene] = None
            continue
        passenger = passenger_attribution(people[a], people[b], p["full_day_departed_passengers"])
        if not passenger["same_departed_ids"]:
            errors.append(scene + ":different_passenger_cohorts")
        scenes[scene] = dict(fresh_minus_matched=report_blocks(delta_blocks(
            observations[a]["totals"], observations[b]["totals"]), p["full_day_departed_passengers"]),
            passengers=passenger, visits=visit_attribution(visits[a], visits[b], p["full_day_departed_passengers"]))
        if not scenes[scene]["visits"]["matching_identity_consistent"]:
            errors.append(scene + ":visit_identity_mismatch")
    native = [run.get("budget", {}).get("native_seconds") for run in runs.values()]
    known = [value for value in native if finite(value)]
    previous = read(root / p["v111_root"] / "summary.json")
    result = dict(status="complete" if not errors else "technical_hold",
        decision=dict(state="DIAGNOSTIC_REPLAY_COMPLETE" if not errors else "HOLD_REPLAY_OR_LEDGER_MISMATCH",
            automatic_new_experiments=False, original_v111_decision=previous.get("decision")),
        integrity=dict(all_pass=not errors, errors=errors),
        attribution_to_original_v111_allowed=not errors,
        reproduction=reproduction, observer_checks={key: {
            field: value for field, value in row.items() if field != "totals"}
            for key, row in observations.items()},
        original_v111_cost_gates=previous.get("cost_gates"), missing=missing,
        scenes={key: compact_scene(value) for key, value in scenes.items()},
        resident_full_attribution_path=str(RELATIVE / "review/full_attribution.json"),
        budget=dict(new_native_seconds=sum(known) if len(known) == len(paths) else None,
            known_native_seconds=sum(known) if known else None,
            native_seconds_is_lower_bound=len(known) != len(paths) or any(
                run.get("budget", {}).get("native_seconds_is_lower_bound") is not False for run in runs.values()),
            actual_by_cell={key: row.get("budget", {}) for key, row in runs.items()}),
        interpretation=dict(comparator="v111 recipe fresh minus matched no_fresh, independently replayed",
            units="All wait,ride,cost and ride_phases are passenger-seconds; explicitly named per_passenger fields use15079.",
            exact_scope="Attribution to v111 requires every reproduction and observer check. The old run had no post-update action trace; full-day action equality is not claimed.",
            windows="The four disjoint windows partition the full day.12600_12900 is an additional overlapping diagnostic window.",
            phases="Observed service/holding/moving/traffic_or_other occupancy time; traffic_or_other is not certified pure congestion. Observed accounting attribution is not an action-specific counterfactual causal effect.",
            original_candidate="This diagnostic does not reopen the closed v111 performance candidate.",
            pairing_revision="Original summary retained. Match scheduled trip/physical stop/within-stop occurrence and line; FIFO fleet_id is a policy-dependent assignment, reported separately rather than required identical.",
            ignored_summary_paths=[".".join(path) for path in EXCLUDED_SUMMARY_PATHS]))
    # Full group/window and top-visit decompositions remain resident; only the
    # bounded summary is a download artifact.
    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    if len(encoded.encode("utf-8")) > p["summary_max_bytes"]:
        raise ValueError("Small summary exceeds the declared byte limit")
    with (review / "full_attribution.json").open("x") as handle:
        json.dump(dict(scenes=scenes, observer_checks=observations), handle,
                  ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    with destination.open("x") as handle:
        handle.write(encoded)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    result = preflight(args.root, args.out_dir or args.root / RELATIVE) if args.preflight else analyze(args.root)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
