"""Small resident-only local versus network accounting after verified v114."""
import argparse
from itertools import chain
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASES = ("service", "holding", "moving", "traffic_or_other")


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            yield json.loads(line)


def trip_records(path, trip):
    records, occurrences = [], {}
    for row in rows(path):
        if row["trip_id"] == trip:
            stop = row["stop_id"]
            occurrence = occurrences[stop] = occurrences.get(stop, 0) + 1
            records.append(((stop, occurrence), row))
    return records


def ride_total(records):
    return dict(ride=sum(row["ride"] for row in records),
        phases={phase: sum(row["ride_phases"][phase] for row in records) for phase in PHASES})


def ride_delta(low, high):
    return dict(ride=low["ride"] - high["ride"],
        phases={phase: low["phases"][phase] - high["phases"][phase] for phase in PHASES})


def root_membership_and_boardings(parent, child, trip, visit, clock):
    onboard, at_root, boarding, alighting = set(), None, [], []
    departure = visit["departure_time"]
    end = departure if departure is not None else clock
    for event in chain(rows(parent / "exchanges.jsonl"), rows(child / "exchanges.jsonl")):
        if event["time"] > end:
            break
        if at_root is None and event["time"] > clock:
            at_root = set(onboard)
        if event["trip_id"] != trip:
            continue
        onboard.difference_update(event["alighted_ids"])
        onboard.update(event["boarded_ids"])
        if (event.get("observed_at_stop") is True and event.get("stop_id") == visit["stop_id"]
                and event["visit_index"] == visit["visit_index"]):
            boarding.extend(event["boarded_ids"])
            alighting.extend(event["alighted_ids"])
    if at_root is None:
        at_root = set(onboard)
    observed, first_tick = {}, None
    for tick in rows(child / "ticks.jsonl"):
        if first_tick is None:
            first_tick = tick["start"]
        if tick["start"] > end:
            break
        if tick["start"] in (clock, departure):
            vehicle = next((row for row in tick["vehicles"] if row["trip_id"] == trip), None)
            if vehicle is not None:
                observed[tick["start"]] = vehicle["onboard"]
    return dict(root_ids=at_root, root_native_count=observed.get(clock),
        root_ids_match_native_count=first_tick == clock and observed.get(clock) == len(at_root),
        first_suffix_tick=first_tick,
        departure_native_count=observed.get(departure),
        station_boarding_events=len(boarding), station_boarding_people=len(set(boarding)),
        station_alighting_events=len(alighting), station_alighting_people=len(set(alighting)))


def root_detail(parent, child, anchor):
    records = trip_records(child / "visits.jsonl", anchor["trip_id"])
    key = (anchor["station_id"], 1)
    index = next(i for i, (identity, _) in enumerate(records) if identity == key)
    visit = records[index][1]
    later = records[index + 1:]
    membership = root_membership_and_boardings(parent, child, anchor["trip_id"], visit, anchor["clock"])
    arrival, ready, applied, departed, hold = (visit.get(field) for field in (
        "arrival_time", "service_ready_time", "action_time", "departure_time", "command_hold_seconds"))
    known = all(value is not None for value in (arrival, ready, applied, departed, hold))
    h = max(0., min(departed, applied + hold) - max(arrival, ready, applied)) if known else None
    s = max(0., min(ready, departed) - arrival) if known else None
    phases = visit["ride_phases"]
    detail = {field: visit.get(field) for field in (
        "vehicle_id", "trip_id", "stop_id", "visit_index", "arrival_time", "service_ready_time",
        "action_time", "command_hold_seconds", "command_accepted", "departure_time", "segment_end")}
    detail.update(ride_including_next_link=visit["ride"], phases=phases,
        observed_visit_boarding_events=len(visit["boarded_ids"]),
        observed_visit_alighting_events=len(visit["alighted_ids"]),
        holding_seconds=h, service_seconds=s,
        holding_mean_load=phases["holding"] / h if h is not None and h > 0 else None,
        service_mean_load=phases["service"] / s if s is not None and s > 0 else None,
        next_stop=later[0][1]["stop_id"] if later else None,
        next_stop_arrival=later[0][1]["arrival_time"] if later else None,
        road_seconds=visit["segment_end"] - departed if departed is not None else None,
        **{field: value for field, value in membership.items() if field != "root_ids"})
    return detail, membership["root_ids"], dict(later), ride_total([visit])


def passenger_changes(low, high, ids, denominator):
    changes = [(pid, low[pid]["wait"] - high[pid]["wait"], low[pid]["ride"] - high[pid]["ride"])
               for pid in ids]
    costs = [(pid, 2 * wait + ride) for pid, wait, ride in changes]
    waiting, riding = sum(row[1] for row in changes), sum(row[2] for row in changes)
    return dict(passengers=len(changes), delta_wait=waiting, delta_ride=riding,
        delta_cost=2 * waiting + riding, cost_per_total_passenger=(2 * waiting + riding) / denominator,
        changed_wait=sum(wait != 0 for _, wait, _ in changes),
        changed_ride=sum(ride != 0 for _, _, ride in changes),
        cost_better=sum(value < 0 for _, value in costs), cost_worse=sum(value > 0 for _, value in costs),
        cost_equal=sum(value == 0 for _, value in costs),
        gross_cost_increase=sum(value for _, value in costs if value > 0),
        gross_cost_decrease=sum(value for _, value in costs if value < 0))


def scene_readout(parent, anchor, denominator):
    details, ids, later, root_totals, people = {}, {}, {}, {}, {}
    for arm in ("high", "low"):
        child = parent / "branches" / arm
        details[arm], ids[arm], later[arm], root_totals[arm] = root_detail(parent, child, anchor)
        people[arm] = {row["passenger_id"]: row for row in rows(child / "passengers.jsonl")}
    if people["high"].keys() != people["low"].keys() or len(people["high"]) != denominator:
        raise ValueError("Full departed passenger identities differ; fixed-person attribution unavailable")
    fixed = ids["high"] == ids["low"] and all(
        details[arm]["root_ids_match_native_count"] for arm in ("high", "low"))
    cohort = ids["high"] if fixed else ids["high"] | ids["low"]
    low, high = people["low"], people["high"]
    all_ids = low.keys()
    totals = passenger_changes(low, high, all_ids, denominator)
    runs = {arm: read(parent / "branches" / arm / "summary.json")["periods"]["full_day"]
            for arm in ("high", "low")}
    if totals["delta_wait"] != runs["low"]["wait"] - runs["high"]["wait"] or totals["delta_ride"] != runs["low"]["ride"] - runs["high"]["ride"]:
        raise ValueError("Paired passenger changes do not equal the verified full-day ledger")
    later_totals = {arm: ride_total(later[arm].values()) for arm in ("high", "low")}
    later_delta = ride_delta(later_totals["low"], later_totals["high"])
    root_delta = ride_delta(root_totals["low"], root_totals["high"])
    ranked = sorted(((2 * (low[pid]["wait"] - high[pid]["wait"]) + low[pid]["ride"] - high[pid]["ride"], pid)
                     for pid in all_ids), reverse=True)
    def top(rows):
        return [dict(passenger_id=pid, delta_wait=low[pid]["wait"] - high[pid]["wait"],
                     delta_ride=low[pid]["ride"] - high[pid]["ride"], delta_cost=value) for value, pid in rows]
    return dict(root=details, root_visit_low_minus_high=root_delta,
        later_same_trip=dict(high=later_totals["high"], low=later_totals["low"], delta=later_delta,
            matched_visits=len(later["high"].keys() & later["low"].keys()),
            only_high_visits=len(later["high"].keys() - later["low"].keys()),
            only_low_visits=len(later["low"].keys() - later["high"].keys())),
        passengers=dict(all=totals, root_cohort=passenger_changes(low, high, cohort, denominator),
            other_passengers=passenger_changes(low, high, all_ids - cohort, denominator),
            fixed_root_clock_membership_verified=fixed,
            cohort_definition="Same root-clock IDs verified against both first-suffix-tick counts" if fixed else
                "Descriptive union of high/low root-clock IDs; fixed pre-intervention cohort not established",
            largest_cost_increases=top([row for row in ranked if row[0] > 0][:3]),
            largest_cost_decreases=top([row for row in reversed(ranked) if row[0] < 0][:3])),
        remainder_network_ride_delta=totals["delta_ride"] - root_delta["ride"] - later_delta["ride"])


def analyze(root):
    experiment = Path(root) / "experiments" / HERE.name
    main = read(experiment / "summary.json")
    if main.get("status") != "complete" or main.get("integrity", {}).get("all_pass") is not True:
        raise ValueError("Wait for complete, verified main results before posthoc readout")
    p = read(experiment / "protocol.json")
    return dict(new_native_seconds=0, new_optimizer_updates=0,
        scenes={scene: scene_readout(experiment / "runs" / scene, p["anchors"][scene],
                                    p["full_day_departed_passengers"]) for scene in p["scenarios"]},
        interpretation="All differences are low minus high; raw costs are passenger-seconds, cost=2W+R. Root visit includes its following road segment; later_same_trip excludes root. Root cohort uses identities at the shared root clock before the first suffix advance, never all riders of the later trip. Boarding counts are native membership changes observed at that station. Phase and passenger splits locate the paired total effect, not separate causal mediation effects. Remainder ride excludes root and later visits of the selected trip. No new root selection, policy or cost gate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    if len(encoded.encode()) > 12000:
        raise ValueError("Posthoc summary exceeds12000 bytes")
    with (args.out or args.root / "experiments" / HERE.name / "posthoc_summary.json").open("x") as handle:
        handle.write(encoded)
    print(encoded, end="")
