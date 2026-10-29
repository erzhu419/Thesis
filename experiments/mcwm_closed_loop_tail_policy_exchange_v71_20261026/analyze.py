"""Resident readout of four restored native entries and eight frozen tails."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
SEEDS = (6201, 6203)
ARMS = ("fresh", "no_fresh")
BRANCHES = {"low": "fresh", "high": "no_fresh"}
PERIODS = {"tail": (18300, 24000), "post20100": (20100, 24000)}
DENOMINATOR = 15079
MAX_BYTES = 48000
COST_FIELDS = ("wait", "ride", "cost", "cost_per_passenger")
SERVICE_EVENT_FIELDS = ("bus_id", "line_id", "clock_seconds", "observed_at", "waiting",
    "eligible_waiting", "onboard", "capacity", "free_capacity", "native_board_count", "native_stranded_count")


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def near(left, right):
    return finite(left) and finite(right) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-6)


def integrity(value):
    audit = value.get("integrity", {})
    checks = audit.get("checks")
    return audit.get("all_pass") is True and bool(checks) and all(x is True for x in checks.values())


def overlap(children):
    if set(children) != set(BRANCHES):
        return None
    if not all(all(finite(row.get(key)) for key in ("native_started_wall", "native_finished_wall"))
               for row in children.values()):
        return None
    return min(x["native_finished_wall"] for x in children.values()) - max(
        x["native_started_wall"] for x in children.values())


def prefix_checks(row, seed, entry):
    budget, cell = row.get("budget", {}), row.get("cell", {})
    checks = dict(complete=row.get("status") == "complete", native_integrity=integrity(row),
        correct_cell=cell.get("seed") == seed and cell.get("scenario") == "stable"
            and cell.get("entry_arm") == entry,
        restored285=budget.get("restoration_feedback_slots") == 285,
        cutpoint=finite(row.get("prefix_clock")) and 18300 <= row["prefix_clock"] < 24000,
        actual_prefix_budget=budget.get("native_prefix_seconds") == row.get("prefix_clock"),
        complete_prefix_exact=type(row.get("trace_rows")) is int and row["trace_rows"] > 0
            and row.get("matched_prefix_rows") == row["trace_rows"])
    checks.update({family+"456": budget.get("restoration_"+family+"_updates") == 456
                   for family in ("critic", "value", "actor", "alpha")})
    return checks


def fork_checks(row, prefix, branches):
    routes, children = row.get("route_streams", {}), row.get("children", {})
    actual_overlap = overlap(children)
    return dict(native_fork_integrity=row.get("all_pass") is True,
        parent_clock_unchanged=row.get("native_before") == row.get("native_after") == prefix.get("prefix_clock"),
        parent_counters_unchanged=bool(row.get("counters_before")) and row.get("counters_before") == row.get("counters_after"),
        parent_rng_unchanged=row.get("parent_rng_unchanged") is True,
        parent_trace_unchanged=row.get("parent_trace_cookie_unchanged") is True,
        route_positions_unchanged=routes.get("parent_offsets_unchanged") is True
            and routes.get("parent_offsets_before") == routes.get("parent_offsets_after"),
        actual_parallel=actual_overlap is not None and actual_overlap > 0,
        recorded_overlap_matches=near(row.get("overlap_seconds"), actual_overlap),
        native_wall_matches=all(children.get(branch) == value.get("execution_wall")
            for branch, value in branches.items()) and len(branches) == 2)


def window_checks(row, period):
    start, end = PERIODS[period]
    endpoints = row.get("endpoints", {})
    low, high = endpoints.get(str(start), {}), endpoints.get(str(end), {})
    window = row.get("cost_windows", {}).get(period, {})
    ready = all(finite(window.get(k)) for k in COST_FIELDS)
    points = all(all(finite(point.get(k)) for k in ("time_seconds", "accrued_wait_seconds",
        "accrued_in_vehicle_seconds", "accrued_generalized_passenger_seconds",
        "waiting_passengers", "in_vehicle_passengers")) for point in (low, high))
    return dict(complete_finite=ready and points,
        true_clock_endpoints=low.get("time_seconds") == start and high.get("time_seconds") == end,
        denominator=window.get("denominator") == DENOMINATOR,
        passenger_cost=ready and near(window["cost"], 2*window["wait"]+window["ride"]),
        accrued_wait=ready and points and near(window["wait"], high["accrued_wait_seconds"]-low["accrued_wait_seconds"]),
        accrued_ride=ready and points and near(window["ride"], high["accrued_in_vehicle_seconds"]-low["accrued_in_vehicle_seconds"]),
        accrued_cost=ready and points and near(window["cost"], high["accrued_generalized_passenger_seconds"]-low["accrued_generalized_passenger_seconds"]),
        passenger_normalization=ready and near(window["cost_per_passenger"], window["cost"]/DENOMINATOR))


def suffix_checks(row, seed, entry, policy, prefix):
    cell, endpoints = row.get("cell", {}), row.get("endpoints", {})
    checks = dict(complete=row.get("status") == "complete", native_integrity=integrity(row),
        correct_cell=cell.get("seed") == seed and cell.get("scenario") == "stable"
            and cell.get("entry_arm") == entry and cell.get("policy_arm") == policy,
        exact_switch=row.get("switch_time") == prefix.get("prefix_clock"),
        baseline18300=bool(row.get("baseline18300")) and row.get("baseline18300") == endpoints.get("18300"),
        suffix_zero_training=row.get("zero_suffix_training") is True,
        final_waiting_matches=finite(row.get("final_waiting")) and row["final_waiting"] == endpoints.get("24000", {}).get("waiting_passengers"))
    for period in PERIODS:
        checks.update({period+"/"+key: value for key, value in window_checks(row, period).items()})
    if entry == policy:
        reference = row.get("diagonal_reference", {})
        checks.update({"diagonal/"+key: reference.get(key) is True
                       for key in ("ledger_exact", "metrics_exact", "full_remaining_trace_exact")})
    return checks


def subtraction(left, right):
    return {key: left[key]-right[key] for key in COST_FIELDS}


def decomposition(cells):
    """Preserve each entrance; P+S equals the original diagonal difference."""
    output = {}
    for period in PERIODS:
        ff, fn, nf, nn = (cells[entry][policy]["cost_windows"][period]
                         for entry, policy in (("fresh", "fresh"), ("fresh", "no_fresh"),
                                              ("no_fresh", "fresh"), ("no_fresh", "no_fresh")))
        fresh_entry, nofresh_entry = subtraction(ff, fn), subtraction(nf, nn)
        policy = {key: (fresh_entry[key]+nofresh_entry[key])/2 for key in COST_FIELDS}
        state = {key: ((ff[key]-nf[key])+(fn[key]-nn[key]))/2 for key in COST_FIELDS}
        output[period] = dict(fresh_entry_policy_fresh_minus_no_fresh=fresh_entry,
            no_fresh_entry_policy_fresh_minus_no_fresh=nofresh_entry,
            policy_component_P=policy, state_component_S=state,
            interaction={key: fresh_entry[key]-nofresh_entry[key] for key in COST_FIELDS},
            original_diagonal_fresh_minus_no_fresh=subtraction(ff, nn),
            additive_identity_exact=all(near(policy[key]+state[key], ff[key]-nn[key]) for key in COST_FIELDS))
    stocks = {}
    for clock in ("20100", "24000"):
        values = [cells[entry][policy]["endpoints"][clock]["waiting_passengers"]
                  for entry, policy in (("fresh", "fresh"), ("fresh", "no_fresh"),
                                        ("no_fresh", "fresh"), ("no_fresh", "no_fresh"))]
        ff, fn, nf, nn = values
        stocks[clock] = dict(fresh_entry_policy_fresh_minus_no_fresh=ff-fn,
            no_fresh_entry_policy_fresh_minus_no_fresh=nf-nn,
            policy_component_P=((ff-fn)+(nf-nn))/2,
            state_component_S=((ff-nf)+(fn-nn))/2,
            interaction=(ff-fn)-(nf-nn), original_diagonal_fresh_minus_no_fresh=ff-nn)
    output["waiting_stock_decomposition"] = stocks
    tail = output["tail"]
    rows = [tail[entry+"_entry_policy_fresh_minus_no_fresh"] for entry in ("fresh", "no_fresh")]
    # The fixed plan requires both state entrances to agree in both seeds.
    output["feedback_redistribution_supported"] = all(row["cost"] > 0 for row in rows) and rows[0]["wait"] > 0
    output["fresh_entry_rescued"] = rows[0]["cost"] > 0 and rows[0]["wait"] > 0
    return output


def waiting_comparison(left, right):
    """Compare actual passenger memberships, never reconstruct IDs from counts."""
    ready = all(isinstance(point, dict) and point.get("waiting_inventory_complete") is True
        for point in (left, right))
    if not ready:
        return dict(status="inventory_incomplete", membership_difference=None)
    maps = [{person["passenger_id"]: person for person in point["waiting_passengers"]}
            for point in (left, right)]
    if any(len(people) != len(point["waiting_passengers"]) for people, point in zip(maps, (left, right))):
        return dict(status="duplicate_inventory_ids", membership_difference=None)
    ids = [set(value) for value in maps]
    exclusives = [ids[0]-ids[1], ids[1]-ids[0]]
    groups = []
    for people, exclusive in zip(maps, exclusives):
        counts = {}
        for identity in exclusive:
            row = people[identity]
            key = (row["station_id"], None if row.get("passable_lines") is None
                   else tuple(row["passable_lines"]))
            counts[key] = counts.get(key, 0)+1
        groups.append([dict(station_id=key[0], passable_lines=key[1], passengers=count)
                       for key, count in sorted(counts.items(), key=lambda pair: str(pair[0]))])
    return dict(status="complete", common_waiting=len(ids[0] & ids[1]), left_only=len(exclusives[0]),
        right_only=len(exclusives[1]), left_minus_right=len(ids[0])-len(ids[1]),
        actual_unique_left_right=[len(value) for value in ids],
        ledger_waiting_left_right=[point.get("ledger_waiting") for point in (left, right)],
        actual_inventory_matches_ledger_left_right=[point.get("waiting_matches_ledger") for point in (left, right)],
        clock_note="Native membership is at the recorded tick; bridge ledger inventory can retain the preceding tick. Actual IDs are not inferred from ledger counts.",
        left_only_station_groups=groups[0], right_only_station_groups=groups[1])


def service_comparison(left, right, *, last_services):
    result = {}
    for clock in ("18300", "20100", "24000"):
        result[clock] = waiting_comparison(left.get("milestones", {}).get(clock),
                                           right.get("milestones", {}).get(clock))
    if last_services and result["24000"]["status"] == "complete":
        groups = [{(row["station_id"], None if row.get("passable_lines") is None else tuple(row["passable_lines"])): row
                   for row in value.get("final_waiting_groups", [])} for value in (left, right)]
        changed = []
        for key in sorted(groups[0].keys() | groups[1].keys(), key=str):
            values = [group.get(key, {}) for group in groups]
            if set(values[0].get("passenger_ids", [])) == set(values[1].get("passenger_ids", [])):
                continue
            services = []
            for row in values:
                kinds = {}
                for kind, event in row.get("last_compatible_service", {}).items():
                    kinds[kind] = None if event is None else [event.get(field) for field in SERVICE_EVENT_FIELDS]
                services.append(kinds)
            changed.append(dict(station_id=key[0], passable_lines=key[1],
                waiting_left_right=[row.get("waiting", 0) for row in values],
                last_compatible_service_left_right=services))
        result["final_changed_groups_last_service"] = changed
    elif last_services:
        result["final_changed_groups_last_service"] = None
    return result


def service_analysis(services):
    resident = {}
    for row in services:
        info = row.get("service") or {}
        path = info.get("resident_summary_path")
        value = read(Path(path)) if path else None
        if value:
            resident[row["seed"], row["entry_arm"], row["policy_arm"]] = value
    pairs = []
    for seed in SEEDS:
        for entry in ARMS:
            keys = [(seed, entry, policy) for policy in ARMS]
            ready = all(key in resident for key in keys)
            pairs.append(dict(seed=seed, entry_arm=entry, comparison="policy_fresh_minus_no_fresh",
                status="complete" if ready else "resident_service_missing",
                observations=service_comparison(*(resident[key] for key in keys), last_services=True) if ready else None))
        for policy in ARMS:
            keys = [(seed, entry, policy) for entry in ARMS]
            ready = all(key in resident for key in keys)
            pairs.append(dict(seed=seed, policy_arm=policy, comparison="entry_fresh_minus_no_fresh",
                status="complete" if ready else "resident_service_missing",
                observations=service_comparison(*(resident[key] for key in keys), last_services=False) if ready else None))
    return pairs


def analyze(root=HERE):
    root = Path(root)
    checks, missing, failures, rows, services, seeds = {}, [], [], [], [], []
    prefix_seconds, suffix_seconds, prefix_count, suffix_count = 0., 0., 0, 0
    for seed in SEEDS:
        cells, seed_keys = {entry: {} for entry in ARMS}, []
        for entry in ARMS:
            name = f"seed{seed}/{entry}"
            base = root/"runs"/f"seed{seed}"/entry
            prefix = read(base/"prefix_summary.json")
            if prefix is None:
                missing.append(name+"/prefix_summary.json")
                failure = read(base/"failure.json")
                if failure:
                    failures.append(dict(cell=name, failure=failure))
                continue
            entry_checks = prefix_checks(prefix, seed, entry)
            prefix_count += int(prefix.get("status") == "complete")
            if prefix.get("status") == "complete" and finite(prefix.get("budget", {}).get("native_prefix_seconds")):
                prefix_seconds += prefix["budget"]["native_prefix_seconds"]
            branches = {}
            for branch, policy in BRANCHES.items():
                value = read(base/"suffixes"/branch/"summary.json")
                if value is None:
                    missing.append(name+"/suffixes/"+branch+"/summary.json")
                    failure = read(base/"suffixes"/branch/"failure.json")
                    if failure:
                        failures.append(dict(cell=name+"/"+branch, failure=failure))
                    continue
                branches[branch] = value
                cells[entry][policy] = value
                suffix_count += int(value.get("status") == "complete")
                if value.get("status") == "complete" and finite(value.get("switch_time")):
                    suffix_seconds += 24000-value["switch_time"]
                entry_checks.update({branch+"/"+key: condition for key, condition in
                                     suffix_checks(value, seed, entry, policy, prefix).items()})
                rows.append(dict(seed=seed, entry_arm=entry, policy_arm=policy,
                    switch_time=value.get("switch_time"), cost_windows=value.get("cost_windows"),
                    endpoints=value.get("endpoints"), final_waiting=value.get("final_waiting"),
                    diagonal_reference=value.get("diagonal_reference"),
                    zero_suffix_training=value.get("zero_suffix_training")))
                services.append(dict(seed=seed, entry_arm=entry, policy_arm=policy, service=value.get("service")))
            live = read(base/"suffixes/live_fork.json")
            if live is None:
                missing.append(name+"/suffixes/live_fork.json")
            elif len(branches) == 2:
                entry_checks.update({"fork/"+key: condition for key, condition in fork_checks(live, prefix, branches).items()})
            if len(branches) == 2:
                entry_checks["same_complete_entry_baseline"] = branches["low"].get("baseline18300") == branches["high"].get("baseline18300")
            for key, condition in entry_checks.items():
                full = name+"/"+key
                checks[full] = condition is True
                seed_keys.append(full)
        seed_ready = all(len(cells[entry]) == 2 for entry in ARMS) and seed_keys and all(checks[key] for key in seed_keys) \
            and not any(path.startswith(f"seed{seed}/") for path in missing) \
            and not any(row["cell"].startswith(f"seed{seed}/") for row in failures)
        values = decomposition(cells) if seed_ready else None
        seeds.append(dict(seed=seed, status="complete" if seed_ready else "unscored", decomposition=values))
    technical = len(rows) == 8 and prefix_count == 4 and not missing and not failures and bool(checks) and all(checks.values())
    failed_checks = [key for key, value in checks.items() if not value]
    status = "complete" if technical else "integrity_failed" if failed_checks or failures else "pending"
    reason = "incomplete_or_integrity_failed"
    supported = False
    if technical:
        supported = all(seed["decomposition"]["feedback_redistribution_supported"] for seed in seeds)
        no_rescue = all(not seed["decomposition"]["fresh_entry_rescued"] for seed in seeds)
        reason = "may_consider_fixed_late_feedback_candidate" if supported else (
            "prior_state_service_diagnosis" if no_rescue else "interaction_stop_no_seed_expansion")
    budget = dict(restored_prefixes=prefix_count, completed_suffixes=suffix_count,
        restored_feedback_slots=285*prefix_count, restoration_updates_per_family=456*prefix_count,
        native_prefix_seconds=prefix_seconds, native_suffix_seconds=suffix_seconds,
        known_total_native_seconds=prefix_seconds+suffix_seconds,
        known_equivalent24000_second_days=(prefix_seconds+suffix_seconds)/24000,
        accounting="Only present complete summaries counted; incomplete/failed work remains separately recorded, never zero.",
        new_initialization_updates=0, suffix_training_updates=0 if technical else None,
        diagnostic_truth_entered_training=False)
    output = dict(version="v71", status=status, complete=bool(technical),
        scope="Two exposed stable development seeds; complete native state exchange, not independent confirmation.",
        metric="J=(2W+R)/15079; both windows use full-day departed denominator; positive differences are worse.",
        windows=PERIODS, cells=rows, seeds=seeds, budget=budget,
        integrity=dict(all_pass=bool(technical), checked_conditions=len(checks), failed_checks=failed_checks),
        missing=missing, failures=failures,
        decision=dict(reason=reason, feedback_redistribution_may_be_considered=bool(technical and supported),
            new_training_authorized_by_diagnostic=False, automatic_seed_expansion=False),
        service_summary_file="service.json", old_code_and_results_modified=False)
    service = dict(version="v71", status=status, cells=services, comparisons=service_analysis(services),
        last_service_event_fields=SERVICE_EVENT_FIELDS,
        interpretation="Passenger membership differences use native person IDs resident on server; only counts/station groups and last actual compatible service are exported. Missing inventories are not empty sets.")
    return output, service


def export(root, out):
    output, service = analyze(root)
    paths = {Path(out): output, Path(out).with_name("service.json"): service}
    payloads = {path: (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n").encode()
                for path, value in paths.items()}
    for path, data in payloads.items():
        if len(data) > MAX_BYTES:
            raise ValueError(f"{path.name} exceeds 48KB: {len(data)}")
        if path.exists():
            raise FileExistsError(path)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for path, data in payloads.items():
        with path.open("xb") as handle:
            handle.write(data)
    return dict(status=output["status"], decision=output["decision"],
                files={str(path): len(data) for path, data in payloads.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.root, args.out), ensure_ascii=False))
