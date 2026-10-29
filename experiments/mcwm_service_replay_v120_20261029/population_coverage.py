"""Coverage of all observed roots versus the fixed service-record selection."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.mcwm_service_replay_v120_20261029.samples import event_index, rows, window


FIELDS = ("roots", "currentcohortpositive", "immediatecohortpositive",
          "same_lane_immediatepositive", "observedcohortboardpositive",
          "maturecohortboardpositive", "mature_departure", "incomplete_cohort",
          "window_truncated_at_horizon")


def describe(run, selected_run, protocol):
    p = protocol["feedback_contract"]
    seconds, horizon = protocol["service_window_seconds"], protocol["horizon_seconds"]
    observer = json.loads((run/"observer_summary.json").read_text())
    if (observer["status"] != "complete" or observer["start"] != 0 or observer["end"] != horizon
            or not all(value is True for value in observer["checks"].values())):
        raise ValueError("Original observer does not provide complete native evidence")
    selected = {(row["context"]["meta"]["sequence"], row["context"]["meta"]["observed_at"]): row
                for row in rows(selected_run/"selected_records.jsonl")}
    roots, keys, packet_order, identities = [], set(), defaultdict(list), set()
    for context in rows(run/"contexts.jsonl"):
        meta = context["meta"]
        sequence, clock = meta["sequence"], meta["observed_at"]
        identity = (sequence, clock)
        if identity in identities:
            raise ValueError("Duplicate root identifier in original contexts")
        identities.add(identity)
        cohort, immediate = set(), set()
        same_lane_immediate = False
        capture = context.get("capture", {}).get("checks", {})
        complete = all(capture.get(key) is True for key in
            ("receiver_native_present", "receiver_at_declared_stop", "feeder_selection_complete"))
        for feeder in context["feeders"]:
            members, x = set(feeder["cohort_ids"]), feeder["x"]
            cohort.update(members)
            complete &= feeder["current_stage0_complete"] is True
            if x.get("feeder_next_stop_matches") is True:
                immediate.update(members)
                same_lane_immediate |= bool(members) and x.get("same_lane") is True and x.get("gap_m") is not None
        end = clock+seconds
        available = end+p["feedback_delay_seconds"]
        eligible = end <= horizon and available <= p["feedback_last_deadline"]
        deadline = (p["feedback_first_deadline"] + max(0,
            math.ceil((available-p["feedback_first_deadline"])/p["feedback_interval_seconds"]))
            *p["feedback_interval_seconds"]) if eligible else None
        if eligible:
            packet_order[deadline].append((available, sequence))
        key = (meta["receiver"], meta["station"])
        keys.add(key)
        roots.append(dict(sequence=sequence, clock=clock, key=key, cohort=cohort,
            immediate=bool(immediate), same_lane_immediate=same_lane_immediate,
            cohort_complete=complete, end=end, available=available, deadline=deadline, eligible=eligible))
    if not set(selected).issubset(identities):
        raise ValueError("Selected roots are absent from the original contexts")
    for order in packet_order.values():
        order.sort()
    visits = event_index(run/"visit_events.jsonl", keys)
    exchanges = event_index(run/"exchanges.jsonl", keys, exchanges=True)
    counts = {name: Counter(dict.fromkeys(FIELDS, 0)) for name in ("all", "eligible", "outside_budget", "selected")}
    positives, selected_differences = [], []
    for root in sorted(roots, key=lambda row: (row["clock"], row["sequence"])):
        start, end, key = root["clock"], min(root["end"], horizon), root["key"]
        departures = [event["time"] for event in window(visits, key, start, end)
                      if event["event"] == "departure"]
        departure = min(departures) if departures else None
        boarded = set()
        for event in window(exchanges, key, start, departure if departure is not None else end):
            boarded.update(root["cohort"].intersection(event["boarded_ids"]))
        positive, mature = bool(boarded), departure is not None
        flags = dict(roots=1, currentcohortpositive=bool(root["cohort"]),
            immediatecohortpositive=root["immediate"], same_lane_immediatepositive=root["same_lane_immediate"],
            observedcohortboardpositive=positive, maturecohortboardpositive=positive and mature,
            mature_departure=mature, incomplete_cohort=not root["cohort_complete"],
            window_truncated_at_horizon=root["end"]>horizon)
        record = selected.get((root["sequence"], start))
        groups = ["all", "eligible" if root["eligible"] else "outside_budget"]
        if record is not None:
            groups.append("selected")
            label = record.get("labels")
            if label is None or (label["cohort_boarded_count"]["observed_count_lower_bound"] != len(boarded)
                    or (label["receiver_departure"]["value"] is not None) != mature):
                selected_differences.append(root["sequence"])
        for group in groups:
            counts[group].update({name:int(value) for name,value in flags.items()})
        if root["eligible"] and positive and len(positives)<3:
            order = packet_order[root["deadline"]]
            positives.append(dict(sequence=root["sequence"], clock=start,
                packet_deadline=root["deadline"], selected=record is not None,
                how_many_more_recent_eligible_in_same_packet=len(order)-bisect_right(order,(root["available"],root["sequence"])),
                observed_cohort_boarded=len(boarded), receiver_departure_mature=mature))
    return dict(counts={name:dict(value) for name,value in counts.items()},
        first_eligible_observed_positive=positives,
        fixed_selection_reconciliation=dict(all_pass=not selected_differences,
            rows_compared=len(selected), mismatched_rows=len(selected_differences),
            first_mismatched_sequences=selected_differences[:3]),
        no_alternative_selection_written=True)


def analyze(runs, selected_runs):
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
    scenes = {scene:describe(Path(runs)/scene,Path(selected_runs)/scene,protocol)
              for scene in protocol["scenarios"]}
    return dict(status="complete", scenes=scenes,
        definitions=dict(eligible="issue+60+30<=18000 with complete fixed window inside horizon",
            observedcohortboardpositive="A recorded root-cohort ID boards this receiver at this station after issue, within60s, and no later than its first observed departure",
            maturecohortboardpositive="Observed cohort boarding positive and receiver departure observed within the window",
            same_lane_immediatepositive="A next-stop-matching feeder has a recorded root-cohort member, same_lane=true and a nonmissing lane gap",
            incomplete_cohort="Recorded cohort may be a known subset; missing coverage is not treated as known zero",
            rank="Count eligible roots with a larger (window_end+delay,sequence) in the same original packet window; no outcome filtering"),
        interpretation="Coverage is descriptive and outcome-conditioned examples are not newly selected training samples. The two scenarios share a pre-shock prefix and must not be summed as independent roots. Count zero is no observed cohort boarding in this fixed window, not proof that every longer hold is useless.",
        budget=dict(new_native_seconds=0, optimizer_updates=0, new_selected_records=0))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--selected-runs",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.runs,args.selected_runs)
    payload=json.dumps(result,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n"
    if len(payload.encode())>6000:
        raise ValueError("Population coverage summary exceeds6000bytes")
    with args.out.open("x") as handle:
        handle.write(payload)
    print(payload,end="")
