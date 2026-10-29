"""Select time-eligible service records; retain complete labels only on the server."""
from bisect import bisect_right
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from experiments.mcwm_service_contract_v119_20261029.features import bind_action
from experiments.mcwm_service_contract_v119_20261029.labels import mature_labels


RECEIVER_FIELDS = ("capacity", "onboard", "free_seats", "service_ready_age_seconds")
FEEDER_FIELDS = tuple("receiver_"+key for key in RECEIVER_FIELDS) + (
    "feeder_onboard", "feeder_speed", "same_lane", "gap_m",
    "feeder_next_stop_matches", "current_leg_dest_count")


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            yield json.loads(line)


def unknown_fields(context):
    missing = Counter()
    if context.get("capture", {}).get("status") != "complete":
        missing["capture.incomplete"] += 1
    for field in context.get("capture", {}).get("missing", []):
        missing["capture."+field.rsplit("/", 1)[-1]] += 1
    for key in RECEIVER_FIELDS:
        if context.get("receiver", {}).get(key) is None:
            missing["receiver."+key] += 1
    for feeder in context.get("feeders", []):
        for key in FEEDER_FIELDS:
            if feeder.get("x", {}).get(key) is None:
                missing["feeder.x."+key] += 1
        if feeder.get("current_stage0_complete") is not True:
            missing["feeder.current_stage0_incomplete"] += 1
    return missing


def event_index(path, keys, *, exchanges=False):
    indexed = defaultdict(list)
    for row in rows(path):
        key = (row["trip_id"], row["stop_id"])
        if key in keys and (not exchanges or row.get("observed_at_stop") is True):
            indexed[key].append(row)
    result = {}
    for key, events in indexed.items():
        events.sort(key=lambda event: event["time"])
        result[key] = ([event["time"] for event in events], events)
    return result


def window(index, key, start, end):
    times, events = index.get(key, ([], []))
    return events[bisect_right(times, start):bisect_right(times, end)]


def select_and_label(run_dir, feedback_contract, horizon=24000, window_seconds=60):
    run_dir, p = Path(run_dir), feedback_contract
    first, last = p["feedback_first_deadline"], p["feedback_last_deadline"]
    step, limit = p["feedback_interval_seconds"], p["feedback_records_per_packet"]
    delay, budget = p["feedback_delay_seconds"], p["target_record_budget"]
    deadlines = list(range(first, last+1, step))
    if len(deadlines)*limit > budget:
        raise ValueError("Declared packet slots exceed the frozen root-action budget")
    buckets, candidate_counts = defaultdict(list), Counter()
    totals = Counter(dict.fromkeys(("contexts", "contexts_with_unknown_inputs",
        "contexts_with_incomplete_capture", "window_crosses_horizon", "complete_window_candidates",
        "available_after_feedback_budget", "feedback_window_candidates"), 0))
    all_missing, seen = Counter(), set()
    for context in rows(run_dir/"contexts.jsonl"):
        meta = context["meta"]
        sequence, start = meta["sequence"], meta["observed_at"]
        if sequence in seen or not isinstance(start, (int, float)) or not math.isfinite(start):
            raise ValueError("Context requires a unique sequence and finite observation clock")
        seen.add(sequence)
        totals["contexts"] += 1
        missing = unknown_fields(context)
        all_missing.update(missing)
        totals["contexts_with_unknown_inputs"] += bool(missing)
        totals["contexts_with_incomplete_capture"] += context.get("capture", {}).get("status") != "complete"
        end, available = start+window_seconds, start+window_seconds+delay
        if end > horizon:
            totals["window_crosses_horizon"] += 1
            continue
        totals["complete_window_candidates"] += 1
        deadline = first+max(0, math.ceil((available-first)/step))*step
        if deadline > last:
            totals["available_after_feedback_budget"] += 1
            continue
        # Each row belongs to only this release window; unused older rows never roll over.
        candidate_counts[deadline] += 1
        totals["feedback_window_candidates"] += 1
        buckets[deadline].append((available, sequence, context))
        buckets[deadline].sort(key=lambda row: row[:2])
        buckets[deadline] = buckets[deadline][-limit:]

    packets, selected = [], []
    for index, deadline in enumerate(deadlines):
        chosen = buckets[deadline]
        packets.append(dict(deadline=deadline,
            release_lower_exclusive=None if index == 0 else deadlines[index-1],
            release_upper_inclusive=deadline, candidate_records=candidate_counts[deadline],
            selected_sequences=[row[1] for row in chosen], selected_records=len(chosen)))
        selected.extend((deadline, *row) for row in chosen)
    selected_ids = {row[2] for row in selected}
    command_rows = defaultdict(list)
    for command in rows(run_dir/"commands.jsonl"):
        if command.get("sequence") in selected_ids:
            command_rows[command["sequence"]].append(command)
    keys = set()
    for _, _, _, context in selected:
        station = context["meta"]["station"]
        keys.add((context["meta"]["receiver"], station))
        keys.update((feeder["trip_id"], station) for feeder in context["feeders"])
    visits = event_index(run_dir/"visit_events.jsonl", keys)
    exchanges = event_index(run_dir/"exchanges.jsonl", keys, exchanges=True)
    observer = json.loads((run_dir/"observer_summary.json").read_text())
    coverage = (observer.get("status") == "complete" and observer.get("start") == 0
        and observer.get("end") == horizon and bool(observer.get("checks"))
        and all(value is True for value in observer["checks"].values()))
    statistics = Counter(dict.fromkeys(("selected_with_unknown_inputs", "selected_with_incomplete_capture",
        "observed_root_cohort_positive", "selected_with_incomplete_cohort", "actions_bound",
        "label_records_built", "receiver_departure_mature", "receiver_visit_count_mature",
        "complete_cohort_count_mature", "valid_mature_records", "observed_any_boarding_positive",
        "observed_root_cohort_boarding_positive"), 0))
    records, selected_missing, errors, holds = [], Counter(), [], []
    for deadline, available, sequence, context in selected:
        start = context["meta"]["observed_at"]
        end, station = start+window_seconds, context["meta"]["station"]
        missing = unknown_fields(context)
        selected_missing.update(missing)
        statistics["selected_with_unknown_inputs"] += bool(missing)
        cohorts = {feeder["trip_id"]: set(feeder["cohort_ids"]) for feeder in context["feeders"]}
        statistics["observed_root_cohort_positive"] += any(cohorts.values())
        capture_complete = context.get("capture", {}).get("status") == "complete"
        statistics["selected_with_incomplete_capture"] += not capture_complete
        capture_checks = context.get("capture", {}).get("checks", {})
        cohort_coverage = all(capture_checks.get(key) is True for key in
            ("receiver_native_present", "receiver_at_declared_stop", "feeder_selection_complete"))
        complete_cohort = cohort_coverage and all(
            feeder["current_stage0_complete"] is True for feeder in context["feeders"])
        statistics["selected_with_incomplete_cohort"] += not complete_cohort
        record = dict(context=context, action=None, labels=None,
            selection=dict(sequence=sequence, packet_deadline=deadline, window_start=start,
                window_end=end, feedback_available_at=available),
            missing_inputs=dict(missing), cohort_definition_complete=complete_cohort, errors=[])
        try:
            if len(command_rows[sequence]) != 1:
                raise ValueError("Selected sequence does not have exactly one native command")
            bound = bind_action(context, command_rows[sequence][0])
            record["action"] = bound["action"]
            holds.append(bound["action"]["hold_seconds"])
            statistics["actions_bound"] += 1
            if not coverage:
                raise ValueError("Full observer coverage is not valid")
            related = {context["meta"]["receiver"], *cohorts}
            evidence = dict(start=start, end=end, continuous=True,
                visit_events=[event for trip in related for event in window(visits, (trip, station), start, end)],
                exchanges=[event for trip in related for event in window(exchanges, (trip, station), start, end)])
            label = mature_labels(bound, evidence, available_at=end)
            record["labels"] = label
            statistics["label_records_built"] += 1
            mature_departure = label["receiver_departure"]["value"] is not None
            mature_count = label["cohort_boarded_count"]["value"] is not None
            statistics["receiver_departure_mature"] += mature_departure
            statistics["receiver_visit_count_mature"] += label["receiver_post_issue_boarded_count"]["value"] is not None
            statistics["complete_cohort_count_mature"] += mature_count and complete_cohort
            statistics["valid_mature_records"] += mature_departure and mature_count and complete_cohort
            statistics["observed_any_boarding_positive"] += label["receiver_post_issue_boarded_count"]["observed_count_lower_bound"] > 0
            statistics["observed_root_cohort_boarding_positive"] += label["cohort_boarded_count"]["observed_count_lower_bound"] > 0
        except (ValueError, KeyError) as error:
            record["errors"].append(str(error))
            errors.append(dict(sequence=sequence, error=str(error)))
        records.append(record)
    for name, values in (("selected_records.jsonl", records), ("packets.jsonl", packets)):
        with (run_dir/name).open("x") as handle:
            for row in values:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n")
    checks = dict(observer_coverage=coverage, selected_unique=len(selected)==len(selected_ids),
        within_budget=len(selected)<=budget, all_packet_slots_filled=all(row["selected_records"]==limit for row in packets),
        selected_commands_and_label_builds_valid=not errors)
    return dict(status="complete" if all(checks.values()) else "technical_hold", all_pass=all(checks.values()),
        checks=checks, counts=dict(totals), selected_records=len(selected), unique_selected=len(selected_ids),
        packet_slots=len(packets), complete_packet_slots=sum(row["selected_records"]==limit for row in packets),
        statistics=dict(statistics), unknown_input_counts_all=dict(all_missing),
        unknown_input_counts_selected=dict(selected_missing),
        action_seconds=dict(count=len(holds), zero=sum(value==0 for value in holds),
            minimum=min(holds) if holds else None, maximum=max(holds) if holds else None,
            mean=sum(holds)/len(holds) if holds else None),
        error_count=len(errors), first_errors=errors[:3],
        budget=dict(root_action_records=len(selected), limit=budget, new_native_seconds=0,
                    optimizer_updates=0, training_admissions=0),
        paths=dict(selected_records="selected_records.jsonl", packets="packets.jsonl"),
        semantics="Selection uses only fixed window end plus reporting delay and latest availability/sequence within each release window. Unknown inputs, zero hold, zero boarding and censored labels never change selection. Label evidence ends at issue+window, not at packet delivery. Valid mature records require closed receiver labels and complete root cohort; unknown numeric inputs remain recorded. Passengers/feeders are dependent targets, not extra budget units.")
