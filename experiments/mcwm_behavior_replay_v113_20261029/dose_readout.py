"""Resident-only station holding/service dose from the completed passive replay."""
import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
WINDOWS = {"full": (0, 24000), "0_9000": (0, 9000), "9000_12000": (9000, 12000),
           "12000_18300": (12000, 18300), "18300_24000": (18300, 24000)}
COUNTS = ("visits", "route_start", "rejected", "uncommanded", "missing_arrival",
          "missing_ready", "missing_departure", "missing_apply", "missing_hold", "complete")
TIMES = ("arrival_time", "service_ready_time", "departure_time", "action_time", "command_hold_seconds")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def new_group():
    return dict(n=[0] * len(COUNTS), raw=[[0., 0., 0., 0.] for _ in WINDOWS],
                conflicts=[[0, 0] for _ in WINDOWS])


def accumulate(group, visit):
    group["n"][0] += 1
    if visit["stop_id"] is None:
        group["n"][1] += 1
        return
    if visit.get("command_accepted") is not True:
        group["n"][2 if visit.get("command_accepted") is False else 3] += 1
        return
    values = [visit.get(key) for key in TIMES]
    absent = [not finite(value) for value in values]
    absent[-1] |= finite(values[-1]) and values[-1] < 0
    if any(absent):
        for index, missing in enumerate(absent):
            group["n"][4 + index] += int(missing)
        return
    arrival, ready, departure, applied, hold = values
    group["n"][9] += 1
    for index, (name, (low, high)) in enumerate(WINDOWS.items()):
        holding = max(0., min(departure, applied + hold, high) - max(arrival, ready, applied, low))
        service = max(0., min(ready, departure, high) - max(arrival, low))
        phases = (visit if name == "full" else visit["periods"][name])["ride_phases"]
        observed = [holding, phases["holding"], service, phases["service"]]
        group["raw"][index] = [a + b for a, b in zip(group["raw"][index], observed)]
        for phase_index, duration, person_seconds in ((0, holding, phases["holding"]),
                                                     (1, service, phases["service"])):
            group["conflicts"][index][phase_index] += duration == 0 and person_seconds > 1e-6


def load(row, conflicts):
    h, hp, s, sp = row
    return hp / h if h > 0 and not conflicts[0] else None, sp / s if s > 0 and not conflicts[1] else None


def display(value):
    return None if value is None else round(value, 2)


def compact_group(group):
    if not group["n"][9]:
        return dict(n=group["n"], v=[[None] * 6 for _ in WINDOWS],
                    z=group["conflicts"])
    rows = []
    for raw, conflicts in zip(group["raw"], group["conflicts"]):
        hl, sl = load(raw, conflicts)
        h, hp, s, sp = raw
        rows.append([display(value) for value in (h, hp, hl, s, sp, sl)])
    return dict(n=group["n"], v=rows, z=group["conflicts"])


def read_case(folder):
    receipt = json.loads((folder / "observer_summary.json").read_text())
    groups = {"all": new_group()}
    with (folder / "visits.jsonl").open() as handle:
        for line in handle:
            visit = json.loads(line)
            accumulate(groups["all"], visit)
            accumulate(groups.setdefault(str(visit["line_id"]), new_group()), visit)
    return groups, dict(status=receipt.get("status"), ledger_exact=receipt.get("ledger_exact"),
                        rejected_commands=receipt.get("rejected_commands"))


def matched(fresh, nofresh):
    if not fresh["n"][9] or not nofresh["n"][9]:
        return [[None] * 6 for _ in WINDOWS]
    rows = []
    for index in range(len(WINDOWS)):
        a, b = fresh["raw"][index], nofresh["raw"][index]
        la = load(a, fresh["conflicts"][index])[0]
        lb = load(b, nofresh["conflicts"][index])[0]
        duration_term = (a[0] - b[0]) * (la + lb) / 2 if la is not None and lb is not None else None
        load_term = (la - lb) * (a[0] + b[0]) / 2 if la is not None and lb is not None else None
        rows.append([display(value) for value in (
            a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3], duration_term, load_term)])
    return rows


def analyze(root):
    experiment = Path(root) / "experiments" / HERE.name
    p = json.loads((experiment / "protocol.json").read_text())
    cases, raw, receipts = {}, {}, {}
    for scene in p["scenarios"]:
        for arm in p["arms"]:
            key = scene + "/" + arm
            raw[key], receipts[key] = read_case(experiment / "runs" / scene / arm)
            cases[key] = {line: compact_group(group) for line, group in raw[key].items()}
    return dict(windows=list(WINDOWS), coverage_columns=COUNTS,
        z_columns=["zero_H_duration_positive_person_seconds", "zero_S_duration_positive_person_seconds"],
        value_columns=["H_seconds", "H_person_seconds", "H_mean_load", "S_seconds", "S_person_seconds", "S_mean_load"],
        matched_columns=["delta_H_seconds", "delta_H_person_seconds", "delta_S_seconds",
                         "delta_S_person_seconds", "holding_duration_term", "holding_load_term"],
        cases=cases, input_receipts=receipts,
        all_inputs_complete=all(row["status"] == "complete" and row["ledger_exact"] is True
                                for row in receipts.values()),
        matched={scene: matched(raw[scene + "/fresh"]["all"], raw[scene + "/no_fresh"]["all"])
                 for scene in p["scenarios"]},
        scope="Complete accepted station visits only; route-start,unaccepted and missing timing excluded and counted. Missing reasons may overlap. H=intersection(arrival,ready,apply,apply+hold,departure); S=arrival to min(ready,departure). Native interval overlap assigns each window. Person-seconds use holding/service phases, never visit ride including road travel. Load=person-seconds/vehicle-seconds; zero duration or inconsistent phase evidence yields null. Matched terms use unrounded symmetric duration/load decomposition, an aggregate accounting identity, not a causal effect. Display rounds2 decimals; no new gate,simulation,training or labels.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    if len(encoded.encode()) > 24000:
        raise ValueError("Dose summary exceeds24000 bytes")
    destination = args.out or args.root / "experiments" / HERE.name / "dose_summary.json"
    with destination.open("x") as handle:
        handle.write(encoded)
    print(encoded, end="")
