"""Read completed resident logs; observed boardings have resolution (t-1,t]."""

import json
from pathlib import Path


CATEGORIES = ("base", "holding", "afterhold", "protected", "straddle", "offstop")


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def visit_key(row):
    return str(row["trip_id"]), int(row["visit_index"])


def category(exchange, visit, command):
    """Classify only when the entire native observation interval is located."""
    high, low = float(exchange["time"]), float(exchange["time"])-1.
    arrival, ready, departure = (visit[key] for key in
        ("arrival_time", "service_ready_time", "departure_time"))
    applied = float(command["applied_at"])
    expiry = applied+float(command["requested_additional_hold_seconds"])
    boundaries = (arrival, ready, applied, expiry, departure)
    if low < 0 or any(point is not None and low < float(point) <= high for point in boundaries):
        return "straddle"
    if not exchange["observed_at_stop"]:
        return "offstop"
    if arrival is None or ready is None or low < float(arrival):
        return "straddle"
    if departure is not None and high >= float(departure):
        return "straddle"
    if high < float(ready):
        return "base"
    if high < applied:
        return "protected"
    if low >= max(float(ready), applied) and high < expiry:
        return "holding"
    if low >= max(float(ready), expiry):
        return "afterhold"
    return "straddle"


class Counts:
    def __init__(self):
        self.events, self.people, self.visits = 0, set(), set()

    def add(self, exchange):
        ids = exchange["boarded_ids"]
        self.events += len(ids)
        self.people.update(map(str, ids))
        self.visits.add(visit_key(exchange))

    def summary(self, denominator=None):
        result = dict(boarding_events=self.events, unique_people=len(self.people), visits=len(self.visits))
        if denominator is not None:
            result["visit_fraction"] = round(len(self.visits)/denominator, 6) if denominator else None
        return result


def summarize_case(case_dir):
    """No native imports/calls; only final receipts and complete JSONL records."""
    case_dir = Path(case_dir)
    receipt = json.loads((case_dir/"observer_summary.json").read_text())
    # Original v113 bridge uses step_length=1. Sparse exchanges are not a clock.
    if (receipt["status"] != "complete" or receipt["start"] != 0
            or receipt["end"] != 24000 or receipt["ticks"] != 24000
            or receipt["ledger_exact"] is not True
            or receipt["checks"]["tick_contiguous"] is not True):
        raise ValueError("Boarding readout requires the complete one-second v113 receipt")
    visits = {visit_key(row): row for row in rows(case_dir/"visits.jsonl")}
    commands = {}
    for command in rows(case_dir/"commands.jsonl"):
        key = visit_key(command)
        if key in commands:
            raise ValueError("The fixed run must have one command per stop visit")
        visit = visits[key]
        if (str(command["stop_id"]) != visit["stop_id"]
                or command["accepted"] is not visit["command_accepted"]):
            raise ValueError("Command and visit records disagree")
        commands[key] = command
    accepted = {key: command for key, command in commands.items() if command["accepted"] is True}
    counts = {name: Counts() for name in CATEGORIES}
    excluded = Counts()
    total = Counts()
    for exchange in rows(case_dir/"exchanges.jsonl"):
        if not exchange["boarded_ids"]:
            continue
        time = float(exchange["time"])
        if time != int(time) or not 0 <= time <= 24000:
            raise ValueError("Boarding observation is not on the native one-second clock")
        total.add(exchange)
        key = visit_key(exchange)
        command = accepted.get(key)
        if command is None:
            excluded.add(exchange)
            continue
        counts[category(exchange, visits[key], command)].add(exchange)
    result = dict(status="complete", accepted_command_visits=len(accepted),
        nonzero_hold_visits=sum(c["requested_additional_hold_seconds"] > 0 for c in accepted.values()),
        zero_hold_visits=sum(c["requested_additional_hold_seconds"] == 0 for c in accepted.values()),
        no_proposal_visits=sum(c["sequence"] is None for c in accepted.values()),
        rejected_commands=sum(c["accepted"] is not True for c in commands.values()),
        categories={name: value.summary(len(accepted)) for name, value in counts.items()},
        unaccepted_or_uncommanded=excluded.summary(), all_observed=total.summary())
    if sum(group["boarding_events"] for group in result["categories"].values())+excluded.events != total.events:
        raise RuntimeError("Observed boarding events were not partitioned")
    return result


def summarize_panel(root):
    root = Path(root)
    return {scene+"/"+arm: summarize_case(root/"runs"/scene/arm)
            for scene in ("stable", "local_speed_drop") for arm in ("fresh", "no_fresh")}


def self_test():
    """Synthetic fixtures only, including missing/zero/rejected commands."""
    from tempfile import TemporaryDirectory
    import unittest

    class Tests(unittest.TestCase):
        def test_intervals_and_fractional_boundaries(self):
            visit = dict(arrival_time=1., service_ready_time=4.5, departure_time=12.)
            command = dict(applied_at=6., requested_additional_hold_seconds=3.5)
            expected = {1: "straddle", 3: "base", 5: "straddle", 6: "straddle",
                        7: "holding", 9: "holding", 10: "straddle", 11: "afterhold", 12: "straddle"}
            for time, name in expected.items():
                self.assertEqual(category(dict(time=time, observed_at_stop=True), visit, command), name)
            visit["service_ready_time"] = 3.5
            self.assertEqual(category(dict(time=5, observed_at_stop=True), visit, command), "protected")
            self.assertEqual(category(dict(time=14, observed_at_stop=False), visit, command), "offstop")

        def test_complete_fixture_includes_zero_hold_and_no_proposal(self):
            with TemporaryDirectory() as folder:
                root = Path(folder)
                receipt = dict(status="complete", start=0, end=24000, ticks=24000,
                               ledger_exact=True, checks=dict(tick_contiguous=True))
                (root/"observer_summary.json").write_text(json.dumps(receipt))
                visits = [dict(trip_id="trip", visit_index=i, stop_id=str(i), arrival_time=1.,
                    service_ready_time=3., departure_time=20., command_accepted=i != 3) for i in (1, 2, 3)]
                commands = [dict(trip_id="trip", visit_index=i, stop_id=str(i), applied_at=5.,
                    requested_additional_hold_seconds=4. if i == 1 else 0., accepted=i != 3,
                    sequence=10 if i == 1 else None) for i in (1, 2, 3)]
                exchanges = [dict(trip_id="trip", visit_index=i, time=7., observed_at_stop=True,
                                 boarded_ids=["p"+str(i)], alighted_ids=[]) for i in (1, 2, 3)]
                for name, values in (("visits", visits), ("commands", commands), ("exchanges", exchanges)):
                    (root/(name+".jsonl")).write_text("".join(json.dumps(value)+"\n" for value in values))
                result = summarize_case(root)
                self.assertEqual(result["accepted_command_visits"], 2)
                self.assertEqual(result["zero_hold_visits"], 1)
                self.assertEqual(result["no_proposal_visits"], 1)
                self.assertEqual(result["categories"]["holding"]["boarding_events"], 1)
                self.assertEqual(result["categories"]["afterhold"]["boarding_events"], 1)
                self.assertEqual(result["categories"]["holding"]["visit_fraction"], .5)
                self.assertEqual(result["unaccepted_or_uncommanded"]["boarding_events"], 1)
                receipt["ticks"] = 23999
                (root/"observer_summary.json").write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, "one-second"):
                    summarize_case(root)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    elif len(sys.argv) == 3 and sys.argv[1] == "--panel":
        print(json.dumps(summarize_panel(sys.argv[2]), separators=(",", ":"), allow_nan=False))
    elif len(sys.argv) == 2:
        print(json.dumps(summarize_case(sys.argv[1]), separators=(",", ":"), allow_nan=False))
    else:
        raise SystemExit("Use CASE_DIR, --panel RUN_ROOT, or --self-test")
