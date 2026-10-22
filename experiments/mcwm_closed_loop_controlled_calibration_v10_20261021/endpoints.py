"""One budgeted controlled issue with three distinct observed response endpoints.

Only the chronological ``global_next`` is a global MDP training transition.
The same-trip next stop and the exact 300-second ledger are separate responses,
released together after the last endpoint plus the original feedback delay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from experiments.mcwm_closed_loop_v1.stream import EI


ENDPOINT_NAMES = ("global_next", "same_vehicle_next", "fixed300")
LINE_NAMES = ("7X", "102X", "705X")
RESPONSE_FEATURE_NAMES = tuple(f"{line}_intercept" for line in LINE_NAMES) + tuple(
    f"{line}_holding" for line in LINE_NAMES) + (
    "forward_headway_deviation", "backward_headway_deviation", "queue_capacity",
    "onboard_capacity", "issue_clock", "pending_intervals", "remaining_hold")
RESPONSE_TARGET_NAMES = ("duration_seconds", "waiting_passenger_seconds",
                         "in_vehicle_passenger_seconds")
TARGET_SCALES = np.asarray((300., 10000., 10000.))


def _ledger(snapshot):
    return {key: float(getattr(snapshot, key)) for key in (
        "time_seconds", "waiting_passengers", "in_vehicle_passengers",
        "accrued_wait_seconds", "accrued_in_vehicle_seconds")}


def _outcome(start, end):
    dt = end["time_seconds"]-start["time_seconds"]
    wait = end["accrued_wait_seconds"]-start["accrued_wait_seconds"]
    ride = end["accrued_in_vehicle_seconds"]-start["accrued_in_vehicle_seconds"]
    if min(dt, wait, ride) < 0:
        raise RuntimeError("Observed response clocks or cumulative passenger ledger moved backwards")
    return dict(status="complete", completed_at=end["time_seconds"], dt=dt,
        wait=wait, ride=ride, cost=2*wait+ride,
        waiting_change=end["waiting_passengers"]-start["waiting_passengers"],
        onboard_change=end["in_vehicle_passengers"]-start["in_vehicle_passengers"])


@dataclass
class ResponseRecord:
    sequence: int
    line_id: str
    vehicle_id: str
    trip_id: str
    station_id: str
    station_index: int
    started_at: float
    hold_seconds: float
    role: str
    s: np.ndarray
    a: np.ndarray
    event: np.ndarray
    raw_s: np.ndarray
    context: dict
    issue_ledger: dict
    assignment: dict = field(default_factory=dict)
    calibration: bool = True
    global_next: object | None = None
    endpoints: dict = field(default_factory=lambda: {
        key: dict(status="pending") for key in ENDPOINT_NAMES})
    status: str = "pending"
    completed_at: float | None = None
    available_at: float | None = None
    released: bool = False
    selected_for_training: bool = False

    def summary(self):
        return dict(sequence=self.sequence, line_id=self.line_id,
            vehicle_id=self.vehicle_id, trip_id=self.trip_id, station_id=self.station_id,
            started_at=self.started_at, hold_seconds=self.hold_seconds, role=self.role,
            issue_event_detected_at=float(self.event[EI['sim_time_seconds']]),
            issue_delivery_lag_seconds=self.started_at-float(self.event[EI['sim_time_seconds']]),
            status=self.status, completed_at=self.completed_at, available_at=self.available_at,
            selected_for_training=self.selected_for_training,
            assignment=self.assignment, endpoints=self.endpoints)


class ResponseEndpointCollector:
    """Observe native clocks without changing controls, randomness or label count.

    Register only the controlled diagnostic issues. Call ``observe_event`` on
    every real event before issuing its control, ``observe_global`` on every
    actual chronological closure, and ``observe_tick`` after each native bridge
    tick has advanced its passenger ledger. No interpolation supplies endpoints.
    """

    def __init__(self, *, delay_seconds=30., duration_seconds=300.):
        self.delay_seconds, self.duration_seconds = float(delay_seconds), float(duration_seconds)
        self.records: dict[int, ResponseRecord] = {}
        self._seen_global = set()
        self.observed_now = 0.

    def register(self, issued, snapshot, context=None, *, role="train", hold_seconds=None,
                 assignment=None):
        if role not in ("train", "heldout"):
            raise ValueError("A controlled issue must be train or heldout")
        sequence = int(issued["sequence"])
        if sequence in self.records:
            raise ValueError("A controlled issue consumes one record, never duplicate endpoint labels")
        ledger = _ledger(snapshot)
        if float(issued["started_at"]) != ledger["time_seconds"]:
            raise ValueError("Capture the passenger ledger at the actual action issue clock")
        hold = 30.*(1.+float(np.asarray(issued["a"]).reshape(-1)[0])) if hold_seconds is None else float(hold_seconds)
        if not (np.isclose(hold, 0., atol=1e-6) or np.isclose(hold, 7.5, atol=1e-6)):
            raise ValueError("v10 calibration uses the actually executed 0 / 7.5 second actions")
        context = {} if context is None else {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in context.items()}
        row = ResponseRecord(sequence=sequence, line_id=str(issued["line_id"]),
            vehicle_id=str(issued["vehicle_id"]), trip_id=str(issued["trip_id"]),
            station_id=str(issued["station_id"]),
            station_index=int(round(float(issued["event"][EI["station_index"]]))),
            started_at=float(issued["started_at"]), hold_seconds=hold, role=role,
            s=issued["s"].copy(), a=issued["a"].copy(), event=issued["event"].copy(),
            raw_s=issued["raw_s"].copy(), context=context, issue_ledger=ledger,
            assignment={} if assignment is None else {key: assignment[key] for key in (
                "assignment", "window_index", "target_hold_seconds", "executed_hold_seconds")
                if key in assignment})
        self.records[sequence] = row
        self.observed_now = max(self.observed_now, row.started_at)
        return row

    def _settle(self, row):
        if row.status != "pending" or any(e["status"] == "pending" for e in row.endpoints.values()):
            return
        row.status = "complete" if all(e["status"] == "complete" for e in row.endpoints.values()) else "missing"
        row.completed_at = max(e["completed_at"] for e in row.endpoints.values())
        row.available_at = row.completed_at+self.delay_seconds

    def observe_global(self, transition, snapshot=None):
        """Capture the real next global record, including zero-duration rows.

        A native snapshot at closure is required to retain the actual wait/ride
        decomposition; reward scaling is never used to reconstruct that ledger.
        """
        sequence = int(transition.sequence)
        if sequence in self._seen_global:
            return
        self._seen_global.add(sequence)
        end = float(transition.completed_at)
        self.observed_now = max(self.observed_now, end)
        for row in self.records.values():
            if sequence < row.sequence:
                continue
            for name in ("same_vehicle_next", "fixed300"):
                endpoint = row.endpoints[name]
                within = endpoint["status"] == "pending" and (
                    name != "fixed300" or end <= row.started_at+self.duration_seconds)
                within |= endpoint["status"] == "complete" and end <= endpoint["completed_at"]
                if within:
                    endpoint["global_events"] = endpoint.get("global_events", 0)+1
        row = self.records.get(sequence)
        if row is None:
            return
        if snapshot is None or float(snapshot.time_seconds) != end:
            raise ValueError("The global endpoint needs its actual closure-clock ledger snapshot")
        if not (np.array_equal(row.s, transition.s) and np.array_equal(row.a, transition.a)):
            raise ValueError("The global response must match the saved executed controlled issue")
        row.global_next = transition
        row.endpoints["global_next"] = dict(_outcome(row.issue_ledger, _ledger(snapshot)),
            global_events=1, next_vehicle_id=str(transition.next_vehicle_id),
            terminal=bool(transition.terminal), zero_duration=float(transition.dt) == 0.)
        self._settle(row)

    def observe_event(self, event, snapshot):
        end = _ledger(snapshot)
        self.observed_now = max(self.observed_now, end["time_seconds"])
        for row in self.records.values():
            if row.endpoints["same_vehicle_next"]["status"] != "pending":
                continue
            if (str(event["line_id"]), str(event["vehicle_id"])) != (row.line_id, row.vehicle_id):
                continue
            station = int(round(float(event["event"][EI["station_index"]])))
            if str(event["trip_id"]) != row.trip_id:
                endpoint = dict(status="missing", reason="vehicle_started_another_trip",
                                completed_at=end["time_seconds"])
            elif station == row.station_index:
                continue
            elif station != row.station_index+1:
                endpoint = dict(status="missing", reason="next_stop_not_observed",
                                completed_at=end["time_seconds"])
            else:
                detected_at = float(event["event"][EI["sim_time_seconds"]])
                # Native detection at t precedes delivery at t+1. Both ends
                # use the actual action-issue/delivery ledger, not raw t.
                if detected_at > end["time_seconds"]:
                    raise ValueError("A delivered event cannot precede its native detection")
                endpoint = dict(_outcome(row.issue_ledger, end),
                    next_station_id=str(event["station_id"]),
                    event_detected_at=detected_at, event_delivered_at=end["time_seconds"],
                    delivery_lag_seconds=end["time_seconds"]-detected_at,
                    global_events=row.endpoints["same_vehicle_next"].get("global_events", 0),
                    semantics="same_trip_action_issue_to_next_stop_event_delivery")
            row.endpoints["same_vehicle_next"] = endpoint
            self._settle(row)

    def observe_tick(self, snapshot):
        end = _ledger(snapshot)
        now = end["time_seconds"]
        self.observed_now = max(self.observed_now, now)
        for row in self.records.values():
            if row.endpoints["fixed300"]["status"] != "pending":
                continue
            deadline = row.started_at+self.duration_seconds
            if now < deadline:
                continue
            if now == deadline:
                row.endpoints["fixed300"] = dict(_outcome(row.issue_ledger, end),
                    global_events=row.endpoints["fixed300"].get("global_events", 0),
                    semantics="fixed_native_clock_system_passenger_ledger")
            else:
                row.endpoints["fixed300"] = dict(status="missing", completed_at=now,
                    reason="exact_native_endpoint_clock_not_observed", scheduled_at=deadline)
            self._settle(row)

    def release(self, now=None):
        now = self.observed_now if now is None else float(now)
        if now > self.observed_now:
            raise ValueError("Cannot release response feedback beyond the observed native clock")
        ready = [row for row in self.records.values() if row.status == "complete"
                 and not row.released and row.available_at <= now]
        for row in ready:
            row.released = True
        return ready

    def finish(self, now, *, reason="observation_stopped"):
        """Preserve incomplete states; missing endpoints never become zero labels."""
        now = float(now)
        self.observed_now = max(self.observed_now, now)
        for row in self.records.values():
            for name, endpoint in row.endpoints.items():
                if endpoint["status"] == "pending":
                    row.endpoints[name] = dict(status="missing", reason=reason, completed_at=now)
            self._settle(row)
        return [row for row in self.records.values() if row.status == "missing"]

    def summary(self):
        return dict(issued=len(self.records), complete=sum(r.status == "complete" for r in self.records.values()),
            missing=sum(r.status == "missing" for r in self.records.values()),
            pending=sum(r.status == "pending" for r in self.records.values()),
            train_records=sum(r.role == "train" for r in self.records.values()),
            heldout_records=sum(r.role == "heldout" for r in self.records.values()),
            endpoint_statuses={name: {status: sum(r.endpoints[name]["status"] == status
                for r in self.records.values()) for status in ("complete", "pending", "missing")}
                for name in ENDPOINT_NAMES})


def response_design(record, *, hold_seconds=None):
    """Small issue-time design; future endpoint values are never covariates."""
    hold = record.hold_seconds if hold_seconds is None else float(hold_seconds)
    route = np.asarray([record.line_id == line for line in LINE_NAMES], dtype=float)
    local, idx = record.event, EI
    capacity = max(1., float(local[idx["bus_capacity"]]))
    forward = float(local[idx["forward_headway_seconds"]])/max(1., float(local[idx["target_forward_headway_seconds"]]))-1
    backward = float(local[idx["backward_headway_seconds"]])/max(1., float(local[idx["target_backward_headway_seconds"]]))-1
    memory = np.asarray(record.context.get("memory_features", np.zeros(18)), dtype=float).reshape(3, 6)
    context = [forward, backward, float(local[idx["eligible_stranded_passengers"]])/capacity,
        float(local[idx["onboard_passengers"]])/capacity, record.started_at/24000.,
        memory[:, 0].sum(), memory[:, 1].sum()]
    return np.concatenate((route, route*(2*hold/7.5-1), np.asarray(context)))


class ControlledResponseReadout:
    """Independent endpoint fits from budget-selected controlled train records.

    Counts remain issue counts: one selected record fits several response
    fields, not several additional training labels. This candidate is passive.
    """

    def __init__(self, *, ridge=1.):
        self.ridge = float(ridge)
        self.records: list[ResponseRecord] = []
        self.readouts = {}
        self.design = {}
        self.endpoint_designs = {}

    def accept(self, records, now):
        existing = {row.sequence for row in self.records}
        for row in records:
            if row.role != "train":
                raise ValueError("Held-out controlled responses cannot enter training")
            if row.status != "complete" or not row.released or row.available_at > float(now):
                raise ValueError("Response fitting needs the complete, released budget-selected issue")
            if row.sequence in existing:
                raise ValueError("A selected controlled issue cannot consume another training slot")
            row.selected_for_training = True
            self.records.append(row)
            existing.add(row.sequence)

    def refit(self):
        if not self.records:
            return
        x = np.stack([response_design(row) for row in self.records])
        singular = np.linalg.svd(x, compute_uv=False)
        maximum = float(singular[0])
        rank = int((singular > max(x.shape)*np.finfo(float).eps*maximum).sum())
        minimum = float(singular[-1]) if len(singular) == x.shape[1] else 0.
        self.design = dict(rows=len(x), columns=x.shape[1], rank=rank,
            active_columns=int((x != 0).any(0).sum()), ridge=self.ridge,
            ridge_precision_condition=(maximum**2+self.ridge)/(minimum**2+self.ridge))
        for endpoint in ENDPOINT_NAMES:
            records = [row for row in self.records if endpoint != "global_next"
                       or row.endpoints["global_next"]["dt"] > 0]
            if not records:
                self.endpoint_designs[endpoint] = dict(rows=0, columns=len(RESPONSE_FEATURE_NAMES), rank=0)
                continue
            design = np.stack([response_design(row) for row in records])
            self.endpoint_designs[endpoint] = dict(rows=len(design), columns=design.shape[1],
                rank=int(np.linalg.matrix_rank(design)))
            y = np.asarray([[row.endpoints[endpoint][key] for key in ("dt", "wait", "ride")]
                            for row in records])/TARGET_SCALES
            precision = design.T@design+self.ridge*np.eye(design.shape[1])
            self.readouts[endpoint] = np.linalg.solve(precision, design.T@y)

    def coverage(self):
        return {line: dict(n=sum(row.line_id == line for row in self.records),
            zero_holding=sum(row.line_id == line and bool(np.isclose(row.hold_seconds, 0.)) for row in self.records),
            high_holding=sum(row.line_id == line and bool(np.isclose(row.hold_seconds, 7.5)) for row in self.records))
            for line in LINE_NAMES}

    def endpoint_coverage(self):
        return {name: {line: dict(
            n=sum(row.line_id == line and (name != "global_next" or row.endpoints[name]["dt"] > 0)
                  for row in self.records),
            zero_holding=sum(row.line_id == line and bool(np.isclose(row.hold_seconds, 0.))
                and (name != "global_next" or row.endpoints[name]["dt"] > 0) for row in self.records),
            high_holding=sum(row.line_id == line and bool(np.isclose(row.hold_seconds, 7.5))
                and (name != "global_next" or row.endpoints[name]["dt"] > 0) for row in self.records))
            for line in LINE_NAMES} for name in ENDPOINT_NAMES}

    def predict(self, record, *, hold_seconds=None):
        support = self.coverage().get(record.line_id, {})
        coverage = self.endpoint_coverage()
        x = response_design(record, hold_seconds=hold_seconds)
        endpoints, statuses = {}, {}
        for endpoint in ENDPOINT_NAMES:
            local_support = coverage[endpoint].get(record.line_id, {})
            if endpoint not in self.readouts or not (local_support.get("zero_holding") and local_support.get("high_holding")):
                endpoints[endpoint] = None
                statuses[endpoint] = "unsupported"
                continue
            weight = self.readouts[endpoint]
            duration, wait, ride = (x@weight)*TARGET_SCALES
            # Retain unconstrained response levels for honest slope diagnostics.
            endpoints[endpoint] = dict(dt=float(duration), wait=float(wait), ride=float(ride),
                                      cost=float(2*wait+ride))
            statuses[endpoint] = "supported"
        raw_global = None if endpoints["global_next"] is None else dict(endpoints["global_next"])
        if int(round(float(record.s[2])*15))+1 < int(round(float(record.s[1])*16)):
            endpoints["global_next"] = dict(dt=0., wait=0., ride=0., cost=0.,
                                            zero_duration_analytic=True)
            statuses["global_next"] = "analytic_zero_duration"
        return dict(status="supported" if any(v != "unsupported" for v in statuses.values()) else "unsupported",
            support=support, endpoint_support={name: coverage[name].get(record.line_id, {}) for name in ENDPOINT_NAMES},
            endpoint_status=statuses, endpoints=endpoints, raw_global_next=raw_global)

    def action_difference(self, record):
        low = self.predict(record, hold_seconds=0.)
        high = self.predict(record, hold_seconds=7.5)
        return dict(status=low["status"], support=low["support"],
            endpoint_support=low["endpoint_support"], endpoint_status=low["endpoint_status"],
            gains={name: low["endpoints"][name]["cost"]-high["endpoints"][name]["cost"]
                   if low["endpoints"][name] is not None else None for name in ENDPOINT_NAMES},
            high_minus_low={name: {key: high["endpoints"][name][key]-low["endpoints"][name][key]
                                  for key in ("dt", "wait", "ride", "cost")}
                            if low["endpoints"][name] is not None else None for name in ENDPOINT_NAMES},
            raw_global_next_gain=low["raw_global_next"]["cost"]-high["raw_global_next"]["cost"]
                if low["raw_global_next"] is not None else None,
            raw_global_next_high_minus_low={key: high["raw_global_next"][key]-low["raw_global_next"][key]
                                           for key in ("dt", "wait", "ride", "cost")}
                if low["raw_global_next"] is not None else None,
            low=low, high=high)

    def predict_issue(self, issued, context=None):
        """Save both candidate predictions before executing/observing the issue.

        This is an action-contrast prediction, not a paired truth observation;
        only the action actually executed will acquire an outcome in this run.
        """
        view = SimpleNamespace(line_id=str(issued["line_id"]), s=issued["s"],
            event=issued["event"], started_at=float(issued["started_at"]),
            hold_seconds=30.*(1.+float(np.asarray(issued["a"]).reshape(-1)[0])),
            context={} if context is None else context)
        return self.action_difference(view)

    @staticmethod
    def chosen_outcome_errors(record, issue_prediction):
        if issue_prediction["status"] != "supported":
            return dict(status="unsupported", endpoints=None)
        candidate = "low" if np.isclose(record.hold_seconds, 0.) else "high"
        predicted = issue_prediction[candidate]["endpoints"]
        return dict(status="complete" if record.status == "complete" else record.status,
            chosen_action=candidate, endpoints={name: dict(
                signed_errors={key: predicted[name][key]-record.endpoints[name][key]
                               for key in ("dt", "wait", "ride", "cost")})
                if record.endpoints[name]["status"] == "complete" and predicted[name] is not None
                else dict(status="unsupported" if predicted[name] is None else record.endpoints[name]["status"])
                for name in ENDPOINT_NAMES},
            interpretation="Prequential chosen-action error; unexecuted action has no paired truth here")

    def summary(self):
        return dict(selected_issue_labels=len(self.records), feature_names=list(RESPONSE_FEATURE_NAMES),
            target_names=list(RESPONSE_TARGET_NAMES), design=self.design, coverage=self.coverage(),
            endpoint_designs=self.endpoint_designs, endpoint_coverage=self.endpoint_coverage(),
            endpoint_names=list(self.readouts),
            coefficients={name: value.tolist() for name, value in self.readouts.items()},
            semantics="Distinct response fields of selected controlled issues; no held-out fitting or global-target substitution")
