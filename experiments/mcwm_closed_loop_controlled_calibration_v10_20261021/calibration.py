"""Fixed randomized holding calibration inside the original feedback budget.

The first eligible diagnostic issue in each window receives a jointly
executable 0/7.5-second command from that line's seed-fixed balanced block.
A completed response bundle replaces one ordinary feedback slot; its global,
same-vehicle and fixed-time endpoints never count as separate labels.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


LINES = ("7X", "102X", "705X")


def _value(record, key):
    return record[key] if isinstance(record, dict) else getattr(record, key)


class ControlledCalibration:
    """Select by issue clock/state, before any response label is observed."""

    def __init__(self, protocol, seed, transform, *, station_id, lines=LINES):
        self.p, self.transform = protocol, transform
        self.lines, self.station_id = tuple(lines), str(station_id)
        self.interval = float(protocol["feedback_interval_seconds"])
        deadlines = range(protocol["feedback_first_deadline"],
                          protocol["feedback_last_deadline"] + 1,
                          protocol["feedback_interval_seconds"])
        count = len(deadlines)
        endpoint_seconds = float(protocol.get("response_endpoint_seconds", 300.))
        self.first_issue = float(protocol.get("calibration_first_issue_seconds",
            protocol["feedback_first_deadline"] - self.interval - endpoint_seconds
            - protocol["feedback_delay_seconds"]))
        self.plan = []
        self.action_sequences = {}
        heldout_start = float(protocol.get("calibration_heldout_first_issue_seconds",
                                          protocol["feedback_last_deadline"]))
        heldout_end = float(protocol["horizon_seconds"])-endpoint_seconds-float(protocol["feedback_delay_seconds"])
        heldout_count = max(0, int(np.ceil((heldout_end-heldout_start)/self.interval)))
        for role, start, end, role_count, offset in (
            ("train", self.first_issue, self.first_issue+count*self.interval, count, 0),
            ("heldout", heldout_start, heldout_end, heldout_count, 10000),
        ):
            rng = np.random.default_rng(int(seed) + int(protocol.get("calibration_seed_offset", 410000)) + offset)
            by_line = {line: [] for line in self.lines}
            for line in self.lines:
                while len(by_line[line]) < role_count:
                    by_line[line].extend([0., 7.5] if rng.integers(2) == 0 else [7.5, 0.])
            self.action_sequences[role] = by_line
            for index in range(role_count):
                self.plan.append(dict(role=role, window_index=index,
                    issue_start=start + index*self.interval,
                    issue_end=min(end, start + (index+1)*self.interval)))
        self.issued, self.attempts = {}, []
        self.line_successes = Counter()

    def _action_for_hold(self, state, seconds):
        """Invert the existing monotone F, then verify the executed command.

        For the existing F, 7.5 seconds is reached either by the direct proposal
        or the maximum proposal after its one-eighth base rule. Completion
        floors can make the requested command unreachable. No substitute is
        then issued as calibration.
        """
        def evaluate(proposal):
            action = np.asarray(self.transform.numpy(state,
                np.asarray([proposal], np.float32)), np.float32).reshape(1)
            hold = float((action[0]+1.)*.5*self.transform.max_hold)
            return action, hold

        direct = 2.*seconds/self.transform.max_hold-1.
        for proposal in (-1., direct, 1.):
            action, hold = evaluate(proposal)
            if abs(hold-seconds) <= 1e-5:
                return np.asarray([proposal], np.float32), action, hold
        return None

    def choose(self, issued, *, role="train"):
        now = float(issued["started_at"])
        plans = [row for row in self.plan if row["role"] == role
                 and row["issue_start"] <= now < row["issue_end"]]
        if not plans:
            return None
        plan = plans[0]
        index, key = plan["window_index"], (role, plan["window_index"])
        if key in self.issued:
            return None
        line = str(issued["line_id"])
        if line not in self.lines or str(issued["station_id"]) != self.station_id:
            return None
        target = self.action_sequences[role][line][self.line_successes[(role, line)]]
        targets = {hold: self._action_for_hold(issued["s"], hold) for hold in (0., 7.5)}
        sequence = int(issued["sequence"])
        attempt = dict(role=role, window_index=index, sequence=sequence, started_at=now,
            line_id=line, target_hold_seconds=target)
        if any(value is None for value in targets.values()):
            endpoints = [self.transform.numpy(issued["s"], np.asarray([u], np.float32))
                         for u in (-1., 1.)]
            holds = [float((a[0]+1.)*.5*self.transform.max_hold) for a in endpoints]
            attempt.update(status="collapsed" if abs(holds[1]-holds[0]) <= 1e-5 else "ineligible",
                minimum_executable_hold=holds[0], maximum_executable_hold=holds[1])
            self.attempts.append(attempt)
            return None
        proposal, action, actual = targets[target]
        decision = dict(attempt, proposal=proposal, action=action,
            assignment="low" if target == 0. else "high",
            executed_hold_seconds=actual, jointly_executable=True)
        self.issued[key] = {name: value.tolist() if isinstance(value, np.ndarray) else value
                           for name, value in decision.items()}
        self.attempts.append(dict(attempt, status="issued", executed_hold_seconds=actual))
        self.line_successes[(role, line)] += 1
        return decision

    def summary(self, now):
        closed = [row for row in self.plan if row["issue_end"] <= float(now)]
        missed = [row for row in closed if (row["role"], row["window_index"]) not in self.issued]
        coverage = {}
        for role in ("train", "heldout"):
            coverage[role] = {}
            for line in self.lines:
                coverage[role][line] = {}
                for hold in (0., 7.5):
                    name = "low" if hold == 0. else "high"
                    matching = lambda row: row["role"] == role and row["line_id"] == line and row["target_hold_seconds"] == hold
                    coverage[role][line][name] = dict(
                        scheduled_quota=None,
                        predefined_assignments=self.action_sequences[role][line].count(hold),
                        eligible=sum(matching(row) and row["status"] == "issued" for row in self.attempts),
                        issued=sum(matching(row) for row in self.issued.values()))
        return dict(planned_windows=len(self.plan), issued=len(self.issued),
            statuses=dict(Counter(row["status"] for row in self.attempts)),
            issued_by_line_action={line: {name: sum(row["line_id"] == line and row["assignment"] == name
                for row in self.issued.values()) for name in ("low", "high")} for line in self.lines},
            missed_windows=missed, pending_windows=len(self.plan)-len(closed),
            assignments=list(self.issued.values()), attempts=self.attempts, coverage=coverage,
            plan=self.plan, predefined_action_order=self.action_sequences,
            balanced_blocks="Seed-fixed per-line low/high blocks consumed only by successful issues; missing windows retain the next assignment")


class CalibrationPacketBudget:
    """Five slots per original deadline, including at most one response bundle.

    ``reserve`` is called at the controlled issue, before its global row exists.
    Complete delayed bundles arrive via ``ingest_calibration``. ``take`` returns
    their original global rows alongside ordinary rows, preserving their true
    zero duration.  The runner must exclude zero/within-batch rows from learned
    global fitting, and uses ``last_calibration_records`` for endpoint fitting.
    """

    def __init__(self, protocol):
        self.p, self.rows, self.index, self.receipts = protocol, [], 0, []
        self.deadlines = list(range(protocol["feedback_first_deadline"],
            protocol["feedback_last_deadline"]+1, protocol["feedback_interval_seconds"]))
        if len(self.deadlines)*protocol["feedback_records_per_packet"] != protocol["target_record_budget"]:
            raise ValueError("Target label budget differs from the fixed packet schedule")
        self.reserved, self.reserved_global, self.calibrations = set(), {}, []
        self.last_calibration_records, self.selected_sequences = [], set()
        self.admitted, self.zero_global_admitted = [], 0

    def reserve(self, sequence):
        sequence = int(sequence)
        if sequence in self.selected_sequences or any(row.sequence == sequence for row in self.rows):
            raise ValueError("Reserve calibration at issue, before its global outcome is ingested")
        self.reserved.add(sequence)

    def ingest(self, rows):
        for row in rows:
            if row.sequence in self.reserved:
                self.reserved_global[row.sequence] = row
            else:
                self.rows.append(row)

    def ingest_calibration(self, records):
        for record in records:
            if _value(record, "role") != "train":
                raise ValueError("Held-out outcomes cannot enter training feedback")
            if _value(record, "status") != "complete":
                raise ValueError("Incomplete endpoints cannot enter feedback")
            sequence = int(_value(record, "sequence"))
            if sequence not in self.reserved:
                raise ValueError("A response bundle must own a reserved calibration slot")
            if any(int(_value(old, "sequence")) == sequence for old in self.calibrations):
                raise ValueError("The same calibration bundle was ingested twice")
            if int(_value(record, "global_next").sequence) != sequence:
                raise ValueError("Bundle and global endpoint refer to different issued actions")
            self.calibrations.append(record)

    def due(self, now):
        return self.index < len(self.deadlines) and now >= self.deadlines[self.index]

    def take(self, now):
        deadline = self.deadlines[self.index]
        if now < deadline:
            raise ValueError("Feedback deadline has not arrived")
        lower = deadline-self.p["feedback_interval_seconds"]
        ready = [row for row in self.rows if row.available_at <= deadline]
        ordinary = [row for row in ready if row.available_at > lower and row.dt > 0 and not row.terminal]
        available = sorted((record for record in self.calibrations
            if float(_value(record, "available_at")) <= deadline),
            key=lambda record: (float(_value(record, "available_at")), int(_value(record, "sequence"))))
        calibration = available[:1]
        n = self.p["feedback_records_per_packet"]
        ordinary_n = n-len(calibration)
        if len(ordinary) < ordinary_n:
            raise RuntimeError(f"Insufficient feedback at {deadline}: {len(ordinary)} ordinary records, {ordinary_n} required")
        selected = sorted(ordinary, key=lambda row: (row.available_at, row.sequence))[-ordinary_n:]
        selected += [_value(record, "global_next") for record in calibration]
        selected.sort(key=lambda row: (row.available_at, row.sequence))
        ids = {int(row.sequence) for row in selected}
        if len(ids) != n or ids & self.selected_sequences:
            raise ValueError("A feedback slot must represent one distinct issued action")
        discarded = [int(row.sequence) for row in ready if row.sequence not in ids]
        self.rows = [row for row in self.rows if row.available_at > deadline]
        self.calibrations = [record for record in self.calibrations
            if int(_value(record, "sequence")) not in ids]
        for record in calibration:
            sequence = int(_value(record, "sequence"))
            self.reserved.remove(sequence)
            self.reserved_global.pop(sequence, None)
        self.selected_sequences.update(ids)
        self.last_calibration_records = calibration
        self.admitted.extend(calibration)
        zeros = sum(row.dt == 0 for row in selected)
        self.zero_global_admitted += zeros
        self.receipts.append(dict(deadline=deadline, processed_at=float(now), records=n,
            sequence_ids=[int(row.sequence) for row in selected],
            max_available_at=max([float(row.available_at) for row in selected]
                + [float(_value(record, "available_at")) for record in calibration]),
            ordinary_records=ordinary_n, calibration_records=len(calibration),
            calibration_sequence_ids=[int(_value(record, "sequence")) for record in calibration],
            positive_global_records=sum(row.dt > 0 and not row.terminal for row in selected),
            learned_global_records=sum(row.dt > 0 and not row.terminal for row in selected),
            zero_global_records=zeros))
        self.index += 1
        return selected, discarded

    def summary(self):
        def arm(record):
            assignment = _value(record, "assignment")
            if isinstance(assignment, dict):
                assignment = assignment["assignment"]
            return assignment
        return dict(target_labels=len(self.selected_sequences), packets=len(self.receipts),
            calibration_admitted=len(self.admitted), zero_global_admitted=self.zero_global_admitted,
            calibration_pending_or_unused=len(self.reserved),
            completed_calibration_unused=len(self.calibrations),
            pending_sequence_ids=sorted(self.reserved),
            calibration_by_line={line: sum(str(_value(record, "line_id")) == line
                for record in self.admitted) for line in LINES},
            calibration_selected_by_line_action={line: {name: sum(
                str(_value(record, "line_id")) == line and arm(record) == name
                for record in self.admitted) for name in ("low", "high")} for line in LINES},
            endpoints_per_calibration_label="One composite issue slot, never separate endpoint labels")
