"""Stream recorded proposals and reconstruct the existing native ledger at issue."""
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path


LEDGER_FIELDS = ("time_seconds", "waiting_passengers", "in_vehicle_passengers",
                 "accrued_wait_seconds", "accrued_in_vehicle_seconds")
ACTION_FIELDS = ("agent", "vehicle_id", "line_id", "station_id", "trip_id", "event",
                 "terminal", "s", "raw_s", "a", "proposed_a", "started_at", "sequence",
                 "episode_id", "domain")


class ReplayMismatch(RuntimeError):
    pass


def clean(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def records(paths):
    for path in paths:
        with Path(path).open() as handle:
            for line in handle:
                yield json.loads(line)


class ReplayReference:
    """The caller stops before apply when check_event returns target_reached.

    A native tick stores start-state counts integrated over [start, end).
    At an issue at end, the ledger still has those counts, not the subsequent
    tick's counts. Multiple issues in one atomic batch share this same ledger.
    """
    def __init__(self, action_paths, tick_paths, target_sequence, target_clock,
                 initial_ledger=None):
        self.action_paths = [str(path) for path in action_paths]
        self.tick_paths = [str(path) for path in tick_paths]
        self.actions, self.ticks = records(self.action_paths), records(self.tick_paths)
        self.target_sequence, self.target_clock = int(target_sequence), float(target_clock)
        self.ledger = dict(time_seconds=0., waiting_passengers=0, in_vehicle_passengers=0,
                          accrued_wait_seconds=0., accrued_in_vehicle_seconds=0.)
        if initial_ledger is not None:
            row = clean(initial_ledger)
            self.ledger = {key: row[key] for key in LEDGER_FIELDS}
        self.current = None
        self.actions_read = self.actions_checked = self.ticks_integrated = 0
        self.reached = False
        self.failure = None

    def fail(self, reason, **details):
        if self.failure is None:
            self.failure = dict(reason=reason, **details)
        raise ReplayMismatch(reason)

    def next_action(self, state):
        if self.reached:
            self.fail("Policy called after the frozen observation target")
        if self.current is not None:
            self.fail("Previous proposal has not been checked by on_issue")
        row = next(self.actions, None)
        if row is None:
            self.fail("Recorded actions ended before the frozen observation target",
                      expected_sequence=self.actions_checked)
        self.actions_read += 1
        missing = [key for key in ACTION_FIELDS if key not in row]
        if missing:
            self.fail("Recorded action is incomplete", fields=missing)
        if row["sequence"] != self.actions_checked:
            self.fail("Recorded action sequence is discontinuous", expected=self.actions_checked,
                      observed=row["sequence"])
        if row["sequence"] > self.target_sequence or row["started_at"] > self.target_clock:
            self.fail("Recorded action passed the frozen observation target")
        if clean(state) != row["s"]:
            self.fail("Canonical policy state differs", sequence=row["sequence"])
        self.current = row
        return list(row["proposed_a"])

    def ledger_at(self, clock):
        if clock < self.ledger["time_seconds"]:
            self.fail("Issue time moved backwards", clock=clock)
        while self.ledger["time_seconds"] < clock:
            row = next(self.ticks, None)
            if row is None:
                self.fail("Recorded ticks ended before issue", issue_clock=clock,
                          last_tick_end=self.ledger["time_seconds"])
            start, end = row["start"], row["end"]
            if start != self.ledger["time_seconds"] or end <= start:
                self.fail("Recorded native ticks are not contiguous",
                          previous_end=self.ledger["time_seconds"], start=start, end=end)
            if end > clock:
                self.fail("Issue clock is not an exact recorded tick boundary", clock=clock,
                          start=start, end=end)
            self.ledger["accrued_wait_seconds"] += (end-start)*row["waiting"]
            self.ledger["accrued_in_vehicle_seconds"] += (end-start)*row["in_vehicle"]
            self.ledger.update(time_seconds=end, waiting_passengers=row["waiting"],
                               in_vehicle_passengers=row["in_vehicle"])
            self.ticks_integrated += 1
        return dict(self.ledger)

    def check_event(self, event, snapshot):
        if self.current is None:
            self.fail("on_issue has no pending recorded proposal")
        actual, expected = clean(event), self.current
        # visit_index is added by the old passive Observer, not by EventStream.
        keys = set(expected) - {"visit_index"}
        differences = sorted(key for key in keys if key not in actual or actual[key] != expected[key])
        if differences:
            self.fail("Issued event differs from the recorded action",
                      sequence=expected["sequence"], fields=differences)
        reference = self.ledger_at(expected["started_at"])
        observed = clean(snapshot)
        differences = [key for key in LEDGER_FIELDS if key not in observed or observed[key] != reference[key]]
        if differences:
            self.fail("Native issue ledger differs from the recorded tick ledger",
                      sequence=expected["sequence"], fields=differences,
                      expected=reference, observed={key: observed.get(key) for key in LEDGER_FIELDS})
        target = expected["sequence"] == self.target_sequence
        if target and expected["started_at"] != self.target_clock:
            self.fail("Target sequence reached at a different clock",
                      expected=self.target_clock, observed=expected["started_at"])
        self.actions_checked += 1
        self.current, self.reached = None, target
        return dict(target_reached=target, reference_ledger=reference)

    def report(self):
        return dict(all_pass=self.reached and self.failure is None,
            status="failed" if self.failure else "target_reached" if self.reached else "pending",
            target_sequence=self.target_sequence, target_clock=self.target_clock,
            actions_read=self.actions_read, actions_checked=self.actions_checked,
            ticks_integrated=self.ticks_integrated, last_reference_ledger=dict(self.ledger),
            action_paths=self.action_paths, tick_paths=self.tick_paths, failure=self.failure,
            action_comparison="All recorded EventStream fields exact; observer visit_index excluded",
            ledger_semantics="Integrate tick-start counts through end=issue clock; retain last tick counts")

    def close(self):
        self.actions.close()
        self.ticks.close()
