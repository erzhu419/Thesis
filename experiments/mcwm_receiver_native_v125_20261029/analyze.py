"""Describe sealed current-leg captures against already retained outcomes."""
import json
from pathlib import Path


def describe_captures(capture_dir, evidence_path):
    """Compare IDs in memory; return only bounded, aggregate diagnostics.

    Call after every declared pre-action capture has been sealed. Future
    outcomes never alter the captured count or determine capture eligibility.
    """
    capture_dir, evidence_path = Path(capture_dir), Path(evidence_path)
    evidence = json.loads(evidence_path.read_text())
    paths = sorted(capture_dir.glob("*.json"), key=lambda path: int(path.stem))
    if {path.stem for path in paths} != set(evidence):
        raise ValueError("All declared captures must be sealed before diagnostic comparison")
    descriptions = []
    for path in paths:
        saved = json.loads(path.read_text())
        observation, expected = saved["observation"], saved["expected_root"]
        meta, receiver = observation["meta"], observation["receiver"]
        seq, issue = meta["sequence"], meta["observed_at"]
        data = evidence[path.stem]
        record, visit = data["record"], data["visit"]
        old_meta = record["context"]["meta"]
        binding = (str(seq) == path.stem and seq == expected["sequence"] == record["selection"]["sequence"]
            and issue == expected["issue_clock"] == record["selection"]["window_start"] == old_meta["observed_at"]
            and meta["receiver"] == expected["receiver"] == visit["trip_id"] == old_meta["receiver"]
            and meta["station"] == expected["station"] == visit["stop_id"] == old_meta["station"]
            and expected["visit_index"] == visit["visit_index"])
        if not binding:
            raise ValueError("Current capture and retained diagnostic root differ: "+str(seq))
        membership_known = receiver["onboard_count"] is not None
        people = observation["people"]
        onboard = {person["person_id"] for person in people}
        complete = receiver["current_stage0_complete"]
        matched = {person["person_id"] for person in people
                   if person["current_stage0"]["known"] is True
                   and person["current_stage0"]["destStop"] == meta["station"]}
        if membership_known and (len(people) != len(onboard) or len(onboard) != receiver["onboard_count"]):
            raise ValueError("Captured onboard membership count is inconsistent: "+str(seq))
        if complete:
            if (not membership_known or any(person["current_stage0"]["known"] is not True for person in people)
                    or receiver["current_leg_dest_count"] != len(matched)):
                raise ValueError("Complete current-stage count is inconsistent: "+str(seq))
        elif receiver["current_leg_dest_count"] is not None:
            raise ValueError("Incomplete current-stage capture must retain an unknown total")
        departure = float(visit["departure_time"])
        exchanges = [event for event in data["exchanges"]
            if event["trip_id"] == meta["receiver"] and event["visit_index"] == visit["visit_index"]
            and issue < event["time"] <= departure]
        boarded = {pid for event in exchanges for pid in event["boarded_ids"]}
        alighted = {pid for event in exchanges for pid in event["alighted_ids"]}
        future_onboard_alighted = alighted & onboard if membership_known else None
        last_at_stop = max((event["time"] for event in exchanges
            if event.get("observed_at_stop") is True and event.get("stop_id") == meta["station"]
            and (event["boarded_ids"] or event["alighted_ids"])), default=None)
        passivity = saved["passivity"]
        capture_checks = observation["capture"]["checks"]
        passive = (bool(passivity) and all(value is True for value in passivity.values())
            and capture_checks["clock_unchanged"] is True and capture_checks["pending_event_unassigned"] is True
            and observation["capture"]["native_clock_before"] == observation["capture"]["native_clock_after"] == issue)
        descriptions.append(dict(sequence=seq, issue=issue, receiver=meta["receiver"], station=meta["station"],
            onboard=receiver["onboard_count"], current_dest_count=receiver["current_leg_dest_count"],
            complete=complete, estimated_ready_age=receiver["estimated_service_ready_age_seconds"],
            future_alighted_onboard_count=None if future_onboard_alighted is None else len(future_onboard_alighted),
            matched_IDs_equal_or_null=(matched == future_onboard_alighted) if complete else None,
            postissue_boarded_people=len(boarded), postissue_alighted_people=len(alighted),
            postissue_boarding_events=sum(len(event["boarded_ids"]) for event in exchanges),
            postissue_alighting_events=sum(len(event["alighted_ids"]) for event in exchanges),
            last_at_stop_exchange_time=last_at_stop, actual_departure=departure,
            capture_status=observation["capture"]["status"], read_error_count=len(observation["capture"]["errors"]),
            passivity_passed=passive))
    totals = dict(roots=len(descriptions), complete_current_stage_roots=sum(row["complete"] for row in descriptions),
        current_dest_positive_roots=sum(row["current_dest_count"] is not None and row["current_dest_count"] > 0 for row in descriptions),
        current_dest_zero_roots=sum(row["current_dest_count"] == 0 for row in descriptions),
        current_dest_unknown_roots=sum(row["current_dest_count"] is None for row in descriptions),
        current_dest_count_sum_complete_roots=sum(row["current_dest_count"] for row in descriptions if row["complete"]),
        future_alighted_onboard_count_sum_known_memberships=sum(row["future_alighted_onboard_count"] for row in descriptions
                                                               if row["future_alighted_onboard_count"] is not None),
        future_alighted_onboard_unknown_roots=sum(row["future_alighted_onboard_count"] is None for row in descriptions),
        matched_equal_roots=sum(row["matched_IDs_equal_or_null"] is True for row in descriptions),
        matched_unequal_roots=sum(row["matched_IDs_equal_or_null"] is False for row in descriptions),
        matched_unknown_roots=sum(row["matched_IDs_equal_or_null"] is None for row in descriptions),
        postissue_boarded_people_sum=sum(row["postissue_boarded_people"] for row in descriptions),
        postissue_alighted_people_sum=sum(row["postissue_alighted_people"] for row in descriptions))
    result = dict(status="complete", all_pass=all(row["passivity_passed"] for row in descriptions),
        roots=descriptions, totals=totals,
        semantics="Current observations remain unchanged. Future comparison uses issue<time<=actual departure in the same native visit, intersected with actual issue-time onboard IDs. Equality is diagnostic, never an admission gate. Unknown current stages retain null counts/equality; known membership can still support a future alighting count. People counts are unique within each root; totals sum root memberships, not independent people. Last exchange and departure are observed tick times, not a universal service formula. No features or training labels are backfilled.")
    if len(json.dumps(result, separators=(",", ":"), allow_nan=False).encode()) > 8000:
        raise ValueError("Diagnostic summary exceeds the fixed8KB export bound")
    return result
