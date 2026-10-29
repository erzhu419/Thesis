"""In-memory recorded continuation and small receipts for the existing live fork."""
from collections import Counter
import json
from pathlib import Path
import random

from experiments.mcwm_logged_action_observation_v116_20261029.replay import clean, LEDGER_FIELDS


def rows(paths):
    for path in paths:
        with Path(path).open() as handle:
            for line in handle:
                yield json.loads(line)


def close_root(reference, stream):
    """Close old read descriptors before fork; the root issue remains unobserved."""
    if reference.current is None:
        raise ValueError("Root proposal must be pending before detaching its reference")
    event = dict(reference.current)
    expected = reference.ledger_at(event["started_at"])
    actual = clean(stream._snapshot())
    if any(actual[key] != expected[key] for key in LEDGER_FIELDS):
        raise ValueError("Root ledger differs before the native fork")
    value = dict(root_action=event, reference_ledger=expected,
                 prefix=reference.report(), root_issue_not_yet_observed=True)
    reference.close()
    value["reference_readers_closed"] = True
    return value


class RecordedContinuation:
    def __init__(self, action_paths, targetseq, endclock, tick_paths):
        self.targetseq, self.endclock = int(targetseq), float(endclock)
        self.table, self.window, self.prefix_counts = {}, [], Counter()
        counts, expected_sequence = Counter(), 0
        self.root_action = None
        for row in rows(action_paths):
            if row["sequence"] != expected_sequence:
                raise ValueError("Original action sequence is discontinuous")
            expected_sequence += 1
            identity = (row["trip_id"], row["station_id"])
            counts[identity] += 1
            self.table[(*identity, counts[identity])] = list(row["proposed_a"])
            if row["sequence"] < self.targetseq:
                self.prefix_counts[identity] += 1
            elif row["sequence"] == self.targetseq:
                self.root_action = row
            if row["sequence"] >= self.targetseq and row["started_at"] < self.endclock:
                self.window.append(row)
        if self.root_action is None or not self.window:
            raise ValueError("Fixed root is absent from the original continuation")
        self.rootclock = self.root_action["started_at"]
        self.tick_rows = []
        reader = rows(tick_paths)
        try:
            for row in reader:
                if row["start"] >= self.endclock:
                    break
                if row["start"] >= self.rootclock:
                    self.tick_rows.append({key: row[key] for key in ("start", "end", "waiting", "in_vehicle")})
        finally:
            reader.close()
        self.occurrences = self.prefix_counts.copy()
        self.lookup_count = self.issue_count = self.checked = self.tick_index = 0
        self.ledger = None
        self.pending = None
        self.end_checked = False
        self.revisited_trip_stops = sum(count > 1 for count in counts.values())
        self.original_actions = expected_sequence

    def proposal(self, state, stream):
        controls = [event for event in stream._batch() if not event["terminal"]]
        index = round(float(state[2])*15)
        if not 0 <= index < len(controls) or abs(float(state[2])*15-index) > 1e-5:
            raise ValueError("Canonical batch index does not identify a current control event")
        event = controls[index]
        identity = (event["trip_id"], event["station_id"])
        key = (*identity, self.occurrences[identity]+1)
        if key not in self.table:
            raise ValueError("Current trip/stop occurrence has no recorded continuation")
        if self.pending is not None:
            raise ValueError("Previous continuation proposal has not reached on_issue")
        self.pending = key
        self.lookup_count += 1
        return list(self.table[key])

    def note_issue(self, event):
        identity = (event["trip_id"], event["station_id"])
        key = (*identity, self.occurrences[identity]+1)
        if key not in self.table or (self.pending is not None and self.pending != key):
            raise ValueError("Issued identity differs from its continuation lookup")
        self.occurrences[identity] += 1
        self.issue_count += 1
        self.pending = None
        return key

    def close_root(self, reference, stream):
        binding = close_root(reference, stream)
        if binding["root_action"] != self.root_action:
            raise ValueError("Detached root differs from the continuation root")
        self.ledger = dict(binding["reference_ledger"])
        return binding

    def _check_ledger(self, snapshot):
        if self.ledger is None:
            raise ValueError("Root ledger has not been bound before fork")
        actual = clean(snapshot)
        clock = actual["time_seconds"]
        if not self.ledger["time_seconds"] <= clock <= self.endclock:
            raise ValueError("Continuation ledger clock is outside the fixed window")
        while self.ledger["time_seconds"] < clock:
            if self.tick_index >= len(self.tick_rows):
                raise ValueError("Original tick coverage ended before the fixed endpoint")
            row = self.tick_rows[self.tick_index]
            if row["start"] != self.ledger["time_seconds"] or not row["start"] < row["end"] <= clock:
                raise ValueError("Original tick boundary does not match the continuation")
            dt = row["end"]-row["start"]
            self.ledger["accrued_wait_seconds"] += dt*row["waiting"]
            self.ledger["accrued_in_vehicle_seconds"] += dt*row["in_vehicle"]
            self.ledger.update(time_seconds=row["end"], waiting_passengers=row["waiting"],
                               in_vehicle_passengers=row["in_vehicle"])
            self.tick_index += 1
        if any(actual[key] != self.ledger[key] for key in LEDGER_FIELDS):
            raise ValueError("Unchanged branch native ledger did not reproduce")

    def check_issue(self, event, snapshot):
        if self.checked >= len(self.window):
            raise ValueError("Unchanged branch produced an extra issue in the window")
        expected, actual = self.window[self.checked], clean(event)
        if any(key not in actual or actual[key] != value for key, value in expected.items()
               if key != "visit_index"):
            raise ValueError("Unchanged branch issue did not exactly reproduce")
        self._check_ledger(snapshot)
        self.checked += 1

    def check_end(self, snapshot):
        if clean(snapshot)["time_seconds"] != self.endclock or self.checked != len(self.window):
            raise ValueError("Unchanged branch did not cover the declared issues and endpoint")
        self._check_ledger(snapshot)
        self.end_checked = True
        return self.report()

    def report(self):
        return dict(original_actions=self.original_actions, revisited_trip_stops=self.revisited_trip_stops,
            root_sequence=self.targetseq, root_clock=self.rootclock, end_clock=self.endclock,
            preloaded_window_issues=len(self.window), preloaded_ticks=len(self.tick_rows),
            continuation_lookups=self.lookup_count, continuation_issues=self.issue_count,
            exact_reference_issues=self.checked, exact_reference_ticks=self.tick_index,
            unchanged_branch_exact=self.end_checked, open_reference_readers=0)


def state_receipt(stream, counters):
    """Visible root and complete global CPU RNG; native memory is inherited by COW."""
    import numpy as np
    import torch
    return clean(dict(native_ledger=stream._snapshot(), counters=counters,
        stream=dict(episode_id=stream.episode_id, issued_count=stream.issued_count,
            completed_count=stream.completed_count, released_count=stream.released_count,
            thinned_count=stream.thinned_count,
            pending_sequence=None if stream._pending is None else stream._pending["sequence"],
            vehicle_pending={str(key): row["sequence"] for key,row in stream._vehicle_pending.items()},
            queued_sequences=[row.sequence for row in stream._queue],
            queued_interval_sequences=[row.sequence for row in stream._interval_queue],
            current_batch=stream._batch()),
        global_rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                        torch=torch.get_rng_state()),
        native_state_basis="Existing v34 live os.fork COW; native RNG is inherited, not serialized here"))
