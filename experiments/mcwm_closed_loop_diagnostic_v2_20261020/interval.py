"""Local same-trip stop-interval prediction, separate from global control models.

The label is decision-ready at the upstream stop to decision-ready at the next
stop: executed holding + motion/signals + service at the next stop.  It is not
pure running time.  Only real source PhysicalInterval rows initialize the
predictors; historical global control transitions are not interval labels.
"""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"mcwm_closed_loop_formal_20261020"))
sys.path.insert(0, os.environ.get("THESIS_H2O_ROOT", "/home/erzhu419/mine_code/sumo-rl/H2Oplus"))
from scenarios import AFFECTED_LINES, DOWNSTREAM_STOP, is_direct_interval
from bus_h2o.mdp_contract_v7 import RAW_EVENT_STATE_INDEX as EI, policy_action_to_hold_seconds_v7


MODELS = ("frozen_source_prior", "online_linear", "local_context")
PERIODS = ("warmup", "before", "crossing", "early", "middle", "late", "drain")
SEMANTICS = "decision_ready_to_next_stop_decision_ready"
FEATURES = tuple(f"line_intercept_{line}" for line in AFFECTED_LINES) + (
    "forward_headway_ratio", "backward_headway_ratio", "current_queue_capacity_ratio",
    "current_onboard_capacity_ratio", "current_base_dwell_scaled")


def serialize(interval):
    """Return bounded JSON fields from one native PhysicalInterval, without pickle."""
    if isinstance(interval, dict):
        return dict(interval)
    result = {key: getattr(interval, key) for key in (
        "domain", "episode_id", "sequence", "vehicle_id", "trip_id", "line_id",
        "station_id", "next_station_id", "started_at", "completed_at", "available_at",
        "dt", "semantics")}
    result["a"] = np.asarray(interval.a, dtype=float).reshape(-1).tolist()
    result["event"] = np.asarray(interval.event, dtype=float).reshape(-1).tolist()
    return result


def _key(row):
    return str(row["domain"]), int(row["episode_id"]), int(row["sequence"])


def _physical(row):
    if row.get("semantics") != SEMANTICS:
        raise ValueError("Initialization and labels must be native same-trip physical intervals")
    if not np.isclose(row["completed_at"]-row["started_at"], row["dt"], atol=1e-5, rtol=1e-7):
        raise ValueError("Physical interval duration disagrees with its observed endpoint clocks")
    if row["dt"] <= 0:
        raise ValueError("Physical interval duration must be positive")


class IntervalDiagnostic:
    """Frozen source ridge, shared online ridge, and a fast per-line context.

    The online model accumulates available target sufficient statistics with
    its source fit.  ``local_context`` uses exactly those same parameters plus
    a per-line EMA residual in seconds.  This local predictor is distinct from
    the four-output global context in the previous A3 control arm.
    """

    def __init__(self, source_rows, contract, *, shock_at=9000., delay_seconds=30.,
                 learning_end_seconds=18000., min_source_per_line=5,
                 max_source_per_line=64, ridge=1., context_rate=.2):
        self.contract, self.shock_at = contract, float(shock_at)
        self.delay_seconds = float(delay_seconds)
        self.learning_end_seconds = float(learning_end_seconds)
        self.context_rate = float(context_rate)
        if ridge <= 0 or not 0 < context_rate <= 1 or delay_seconds < 0:
            raise ValueError("Ridge and context rate must be positive, and delay nonnegative")
        self.pending, self.records, self.label_records, self._seen_labels = {}, [], [], set()
        self.issued_by_line, self.labels_by_line = Counter(), Counter()
        self.matched_by_line, self.missing_by_line = Counter(), Counter()
        self.learned_by_line = Counter()
        self.source_available, selected, seen = Counter(), [], set()
        for value in source_rows:
            row = serialize(value)
            if row["domain"] != "source":
                raise ValueError("Interval initialization accepts source calibration only")
            if not is_direct_interval(row["line_id"], row["station_id"], row["next_station_id"]):
                continue
            _physical(row)
            if _key(row) in seen:
                raise ValueError("Duplicate source physical interval")
            seen.add(_key(row))
            line = str(row["line_id"])
            self.source_available[line] += 1
            if self.source_available[line] <= max_source_per_line:
                selected.append(row)
        self.source_used = Counter(str(row["line_id"]) for row in selected)
        missing = [line for line in AFFECTED_LINES if self.source_used[line] < min_source_per_line]
        if missing:
            raise ValueError(f"Insufficient real source calibration for lines {missing}: {dict(self.source_used)}")
        xs = np.stack([self._features(row["line_id"], row["event"]) for row in selected])
        ys = np.asarray([row["dt"]-self._holding(row["a"]) for row in selected], dtype=float)
        self.precision = xs.T@xs + float(ridge)*np.eye(len(FEATURES))
        self.information = xs.T@ys
        self.frozen_weights = np.linalg.solve(self.precision, self.information)
        self.online_weights = self.frozen_weights.copy()
        self.context = {line: 0. for line in AFFECTED_LINES}
        self.parameter_version = self.update_batches = 0
        self.source_label_stats = self._label_summary([
            dict(line_id=str(row["line_id"]), period="source", duration_seconds=float(row["dt"]),
                 holding_seconds=self._holding(row["a"]),
                 nonholding_seconds=float(row["dt"])-self._holding(row["a"])) for row in selected])

    def _holding(self, action):
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.size != 1:
            raise ValueError("The physical diagnostic uses the executed holding-only action")
        return float(policy_action_to_hold_seconds_v7(action[0], self.contract))

    def _features(self, line, raw):
        raw = np.asarray(raw, dtype=float).reshape(-1)
        features = [float(str(line) == name) for name in AFFECTED_LINES]
        for direction in ("forward", "backward"):
            present = raw[EI[f"{direction}_bus_present"]] > .5
            target = raw[EI[f"target_{direction}_headway_seconds"]]
            value = raw[EI[f"{direction}_headway_seconds"]]/target if present and target > 0 else 1.
            features.append(float(np.clip(value, 0., self.contract.max_headway_ratio)))
        capacity = raw[EI["bus_capacity"]]
        features.extend((
            float(np.clip(raw[EI["eligible_stranded_passengers"]]/capacity,
                          0., self.contract.max_queue_capacity_ratio)),
            float(np.clip(raw[EI["onboard_passengers"]]/capacity, 0., 1.)),
            float(np.clip(raw[EI["base_dwell_seconds"]]/self.contract.base_dwell_scale_seconds, 0., 2.))))
        return np.asarray(features)

    def _period(self, started, completed):
        if started < self.shock_at <= completed:
            return "crossing"
        if started < 6000.:
            return "warmup"
        if started < self.shock_at:
            return "before"
        if started < 12000.:
            return "early"
        if started < 15000.:
            return "middle"
        if started < 18000.:
            return "late"
        return "drain"

    def register(self, issue):
        """Save all three pre-label predictions using the action actually executed."""
        if issue["domain"] != "target":
            raise ValueError("The shared diagnostic prediction stream must be target-domain")
        line = str(issue["line_id"])
        if not is_direct_interval(line, issue["station_id"], DOWNSTREAM_STOP):
            return None
        key = _key(issue)
        if key in self.pending:
            raise ValueError("A physical issue was registered twice")
        x, holding = self._features(line, issue["event"]), self._holding(issue["a"])
        nonholding = dict(frozen_source_prior=float(x@self.frozen_weights),
                         online_linear=float(x@self.online_weights))
        nonholding["local_context"] = nonholding["online_linear"]+self.context[line]
        saved = dict(line_id=line, vehicle_id=str(issue["vehicle_id"]), trip_id=str(issue["trip_id"]),
            station_id=str(issue["station_id"]), started_at=float(issue["started_at"]),
            x=x, holding_seconds=holding, parameter_version=self.parameter_version,
            predictions={model: holding+max(0., value) for model, value in nonholding.items()})
        self.pending[key] = saved
        self.issued_by_line[line] += 1
        return dict(sequence=issue["sequence"], predicted_seconds=dict(saved["predictions"]),
                    holding_seconds=holding, parameter_version=self.parameter_version)

    def observe(self, intervals, now):
        """Score the complete arrived batch before updating any predictor parameter."""
        rows = [serialize(value) for value in intervals]
        updates, logs = [], []
        for row in rows:
            if not is_direct_interval(row["line_id"], row["station_id"], row["next_station_id"]):
                continue
            _physical(row)
            if row["domain"] != "target":
                raise ValueError("Only target physical labels enter online diagnostic evaluation")
            available = max(float(row["available_at"]), float(row["completed_at"])+self.delay_seconds)
            if available > now:
                raise ValueError("A physical label was supplied before its feedback delay elapsed")
            key, line = _key(row), str(row["line_id"])
            if key in self._seen_labels:
                raise ValueError("A physical label was observed twice")
            self._seen_labels.add(key)
            self.labels_by_line[line] += 1
            holding = self._holding(row["a"])
            actual, period = float(row["dt"]), self._period(row["started_at"], row["completed_at"])
            self.label_records.append(dict(line_id=line, period=period, duration_seconds=actual,
                holding_seconds=holding, nonholding_seconds=actual-holding))
            saved = self.pending.pop(key, None)
            if saved is None:
                self.missing_by_line[line] += 1
                continue
            for field in ("vehicle_id", "trip_id", "line_id", "station_id"):
                if str(row[field]) != saved[field]:
                    raise ValueError(f"The same-trip physical label has the wrong {field}")
            if (not np.isclose(row["started_at"], saved["started_at"], atol=1e-5)
                    or not np.isclose(holding, saved["holding_seconds"], atol=1e-5)):
                raise ValueError("Physical label and saved issue disagree on time or executed action")
            self.matched_by_line[line] += 1
            eligible = available <= self.learning_end_seconds
            record = dict(line_id=line, period=period, duration_seconds=actual,
                holding_seconds=holding, nonholding_seconds=actual-holding,
                learning_eligible=eligible,
                errors={model: prediction-actual for model, prediction in saved["predictions"].items()})
            self.records.append(record)
            for model, prediction in saved["predictions"].items():
                logs.append(dict(model=model, sequence=int(row["sequence"]), line_id=line,
                    vehicle_id=row["vehicle_id"], trip_id=row["trip_id"], period=period,
                    issued_at=float(row["started_at"]), completed_at=float(row["completed_at"]),
                    available_at=available, scored_at=float(now), parameter_version=saved["parameter_version"],
                    predicted_seconds=prediction, actual_seconds=actual, holding_seconds=holding,
                    nonholding_seconds=actual-holding, error_seconds=prediction-actual,
                    absolute_error_seconds=abs(prediction-actual), learning_eligible=eligible))
            if eligible:
                updates.append((line, saved["x"], actual-holding))
        # These exact parameters serve both adaptive variants.  No label from
        # this batch could have changed any prediction scored above.
        if updates:
            for line, x, y in updates:
                self.precision += np.outer(x, x)
                self.information += x*y
                self.learned_by_line[line] += 1
            self.online_weights = np.linalg.solve(self.precision, self.information)
            for line, x, y in updates:
                residual = float(y-x@self.online_weights)
                self.context[line] = (1-self.context_rate)*self.context[line]+self.context_rate*residual
            self.parameter_version += 1
            self.update_batches += 1
        return logs

    @staticmethod
    def _label_summary(records):
        def stats(rows):
            return dict(n=len(rows), **{f"mean_{field}": (
                float(np.mean([row[field] for row in rows])) if rows else None)
                for field in ("duration_seconds", "holding_seconds", "nonholding_seconds")})
        return dict(overall=stats(records),
            by_period={period: stats([row for row in records if row["period"] == period]) for period in PERIODS},
            by_line={line: dict(overall=stats([row for row in records if row["line_id"] == line]),
                by_period={period: stats([row for row in records if row["line_id"] == line
                                         and row["period"] == period]) for period in PERIODS})
                for line in AFFECTED_LINES})

    def summary(self):
        def errors(rows, model):
            values = [row["errors"][model] for row in rows]
            return dict(n=len(values), mae_seconds=float(np.mean(np.abs(values))) if values else None,
                        bias_seconds=float(np.mean(values)) if values else None)
        models = {model: dict(overall=errors(self.records, model),
            by_period={period: errors([row for row in self.records if row["period"] == period], model)
                       for period in PERIODS},
            by_line={line: dict(overall=errors([row for row in self.records if row["line_id"] == line], model),
                by_period={period: errors([row for row in self.records if row["line_id"] == line
                    and row["period"] == period], model) for period in PERIODS}) for line in AFFECTED_LINES})
            for model in MODELS}
        return dict(models=models, labels=self._label_summary(self.label_records),
            coverage=dict(direct_issued=sum(self.issued_by_line.values()),
                direct_completed_labels=sum(self.labels_by_line.values()),
                matched_labels=sum(self.matched_by_line.values()),
                missing_saved_prediction=sum(self.missing_by_line.values()),
                pending_predictions=len(self.pending), learned_labels=sum(self.learned_by_line.values()),
                learned_by_period={period: sum(row["period"] == period and row["learning_eligible"]
                    for row in self.records) for period in PERIODS},
                by_line={line: dict(issued=self.issued_by_line[line], completed=self.labels_by_line[line],
                    matched=self.matched_by_line[line], missing=self.missing_by_line[line],
                    learned=self.learned_by_line[line],
                    learned_by_period={period: sum(row["line_id"] == line and row["period"] == period
                        and row["learning_eligible"] for row in self.records) for period in PERIODS})
                    for line in AFFECTED_LINES}),
            initialization=dict(domain="source", available_by_line=dict(self.source_available),
                used_by_line=dict(self.source_used), labels=self.source_label_stats,
                features=list(FEATURES), frozen_weights=self.frozen_weights.tolist()),
            online=dict(parameter_version=self.parameter_version, update_batches=self.update_batches,
                weights=self.online_weights.tolist(), local_context_seconds=dict(self.context)),
            delay_seconds=self.delay_seconds, learning_end_seconds=self.learning_end_seconds,
            semantics=SEMANTICS,
            prediction_semantics="executed holding + predicted motion/signals/next-stop service; bias is predicted minus actual seconds",
            model_scope="local_context shares online_linear parameters and adds per-line EMA seconds; distinct from the old A3 global four-output context")
