"""Two fixed connection cases from existing v114 logs; no native/learner imports."""
import argparse
from itertools import chain
import json
from pathlib import Path


SOURCE = "experiments/mcwm_same_state_holding_v114_20261029"
STABLE_STOP = "122S17_311S23_406X15"
SHOCK_STOP = "122X22_311X22_406S29"
EVENT_COLUMNS = ("line_id", "station_index", "sim_time_seconds", "forward_headway_seconds",
    "backward_headway_seconds", "target_forward_headway_seconds", "target_backward_headway_seconds",
    "eligible_stranded_passengers", "onboard_passengers", "eligible_arrival_rate_per_second",
    "base_dwell_seconds", "forward_bus_present", "backward_bus_present", "bus_capacity",
    "system_waiting_passengers", "system_in_vehicle_passengers")


def rows(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def full_rows(parent, child, name):
    return chain(rows(parent / (name + ".jsonl")), rows(child / (name + ".jsonl")))


def action(row):
    return dict(sequence=row["sequence"], time=row["started_at"], trip_id=row["trip_id"],
        stop_id=row["station_id"], visit_index=row["visit_index"],
        proposed=row["proposed_a"], executed=row["a"], event=row["event"])


def visit_record(row, command, observed_action):
    value = {key: row[key] for key in ("trip_id", "line_id", "stop_id", "visit_index",
        "arrival_time", "service_ready_time", "action_time", "command_hold_seconds",
        "command_accepted", "departure_time", "boarded_ids", "alighted_ids")}
    value["command"] = command
    value["action_event"] = None if observed_action is None else action(observed_action)
    return value


def stable(root):
    parent = root / SOURCE / "runs/stable"
    result = dict(passenger_id="transfer_8937.0", observed_low_wait_start=18973., arms={})
    for arm in ("high", "low"):
        child = parent / "branches" / arm
        visits = [v for v in rows(child / "visits.jsonl")
                  if v["line_id"] == "122S" and v["stop_id"] == STABLE_STOP]
        targets = [v for v in visits if v["trip_id"] == "122S_33"]
        if len(targets) != 1:
            raise ValueError("Expected the fixed 122S_33 station visit")
        target = targets[0]
        later = sorted([v for v in visits if v["arrival_time"] <= 24000 and
            (v["arrival_time"] >= 18973 or
             (v["departure_time"] is not None and v["departure_time"] >= 18973))],
            key=lambda v: (v["arrival_time"], v["trip_id"])) if arm == "low" else []
        selected = later if len(later) <= 8 else later[:4] + later[-4:]
        selected = [target] + [v for v in selected if v is not target]
        keys = {(v["trip_id"], v["visit_index"]) for v in selected}
        commands = {(r["trip_id"], r["visit_index"]): r for r in
            full_rows(parent, child, "commands") if (r["trip_id"], r["visit_index"]) in keys}
        actions = {(r["trip_id"], r["visit_index"]): r for r in
            full_rows(parent, child, "actions") if (r["trip_id"], r["visit_index"]) in keys}
        points = {(v["trip_id"], v[field]) for v in selected
                  for field in ("arrival_time", "action_time", "departure_time") if v[field] is not None}
        snapshots = []
        for tick in rows(child / "ticks.jsonl"):
            if tick["start"] > max(t for _, t in points):
                break
            for vehicle in tick["vehicles"]:
                if (vehicle["trip_id"], tick["start"]) in points:
                    snapshots.append(dict(time=tick["start"], **{k: vehicle[k] for k in
                        ("trip_id", "stop_id", "visit_index", "at_stop", "speed", "onboard", "native_stop_state")}))
        records = {v["trip_id"]: visit_record(v, commands.get((v["trip_id"], v["visit_index"])),
            actions.get((v["trip_id"], v["visit_index"]))) for v in selected}
        exchanges = []
        for r in full_rows(parent, child, "exchanges"):
            if (r["trip_id"], r["visit_index"]) == (target["trip_id"], target["visit_index"]) and r["stop_id"] == STABLE_STOP:
                exchanges.append(r)
        result["arms"][arm] = dict(connection=records["122S_33"],
            observed_tick_loads=snapshots, connection_exchanges=exchanges)
        if arm == "low":
            result["low_remaining_122S_services"] = dict(window=[18973, 24000],
                inclusion="arrives during window or recorded departure overlaps its start",
                total_visits=len(later), returned_visits=[records[v["trip_id"]] for v in selected if v in later],
                omitted_middle=max(0, len(later)-8))
    return result


def shock(root):
    parent = root / SOURCE / "runs/local_speed_drop"
    trips = {"311X_19", "406S_22"}
    actions, events, last = [], [], {}
    for r in rows(parent / "actions.jsonl"):
        if r["trip_id"] in trips and r["started_at"] <= 14547:
            last[r["trip_id"]] = action(r)
            if r["started_at"] >= 14487:
                actions.append(action(r))
    for r in rows(parent / "visit_events.jsonl"):
        if r["trip_id"] in trips and 14487 <= r["time"] <= 14547:
            events.append(r)
    result = dict(clock=14547., history_window=[14487, 14547],
        preceding_actions=actions, preceding_visit_events=events,
        last_pre_root_action={trip: dict(record=last.get(trip), age_seconds=
            None if trip not in last else 14547-last[trip]["time"]) for trip in sorted(trips)}, arms={})
    for arm in ("high", "low"):
        child = parent / "branches" / arm
        tick = next(rows(child / "ticks.jsonl"))
        if tick["start"] != 14547:
            raise ValueError("The first suffix tick is not the fixed root clock")
        snapshots = [{key: v[key] for key in ("trip_id", "line_id", "stop_id", "visit_index",
            "at_stop", "speed", "native_stop_state", "onboard")} for v in tick["vehicles"] if v["trip_id"] in trips]
        root_action = next(r for r in rows(child / "actions.jsonl") if r["sequence"] == 10045)
        future = []
        for r in rows(child / "exchanges.jsonl"):
            if r["time"] > 14560:
                break
            if "transfer_15055.0" in r["boarded_ids"] or "transfer_15055.0" in r["alighted_ids"]:
                future.append({**{key: r[key] for key in ("time", "trip_id", "line_id", "stop_id", "observed_at_stop")},
                    "boarded": "transfer_15055.0" in r["boarded_ids"], "alighted": "transfer_15055.0" in r["alighted_ids"]})
        result["arms"][arm] = dict(first_tick=dict(start=tick["start"], end=tick["end"], vehicles=snapshots),
            root_action=action(root_action), subsequent_observed_exchanges=future)
    return result


def extract(root):
    return dict(new_native_seconds=0, new_optimizer_updates=0, source=SOURCE,
        event_columns=EVENT_COLUMNS, stable=stable(root), local_speed_drop=shock(root),
        semantics="Actual visit and exchange records are retrospective. Tick14547 was sampled after root commands were applied but before the first native advance; its physical fields are not a recorded actor input. Prefix actions/visits occurred before that root, with last-action age explicit. No vehicle position, road-edge distance, ETA or future passenger intention is stored in ticks. Eligible-stranded is the original event field, not an inferred current-stage transfer queue. Waiting entry is an observed membership change, not exact arrival. No capacity, timetable or last-bus cause is assigned.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    value = extract(args.root)
    encoded = json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n"
    if len(encoded.encode()) > 12000:
        raise ValueError("Fixed-case summary exceeds12000 bytes; compact before retrieval")
    print(encoded, end="")
