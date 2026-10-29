"""Passive mu14 finite300 response fits from the nine budget-admitted pairs."""
from __future__ import annotations

import numpy as np

LINES = ("7X", "102X", "705X")
HOLDS = (0.0, 7.5)
FEATURE_NAMES = tuple(line + "_intercept" for line in LINES) + (
    "forward_minus_backward_headway_deviation",)
TARGET_SCALE = 1000.0
REWARD_SCALE = 1e-4


def issue_features(issued):
    """The unchanged v37 four issue-visible covariates; no outcome inputs."""
    from experiments.mcwm_closed_loop_v1.stream import EI

    line, event = str(issued["line_id"]), issued["event"]
    if line not in LINES:
        raise ValueError("Unsupported line")
    forward = float(event[EI["forward_headway_seconds"]]) / max(
        1.0, float(event[EI["target_forward_headway_seconds"]])) - 1.0
    backward = float(event[EI["backward_headway_seconds"]]) / max(
        1.0, float(event[EI["target_backward_headway_seconds"]])) - 1.0
    return [float(line == value) for value in LINES] + [forward - backward]


def _features(values, line=None):
    vector = np.asarray(values, dtype=float)
    if vector.shape != (4,) or not np.isfinite(vector).all():
        raise ValueError("Expected four finite issue-time covariates")
    index = int(np.argmax(vector[:3]))
    if not np.array_equal(vector[:3], np.eye(3)[index]):
        raise ValueError("The first three covariates identify one declared line")
    if line is not None and LINES[index] != str(line):
        raise ValueError("Issue features and line disagree")
    return vector, LINES[index]


def _gain(wait_ride):
    wait, ride = map(float, wait_ride)
    cost = 2.0 * wait + ride
    return dict(wait=wait, ride=ride, cost=cost, scaled_cost=REWARD_SCALE * cost)


def physical_cost(outcome):
    """Exact-clock ledger labels do not require an invented decision state."""
    value = outcome["truth"]
    if value.get("status") != "complete" or value.get("dt") != 300.0:
        raise ValueError("A complete exactly300-second physical response is required")
    wait, ride = float(value["wait"]), float(value["ride"])
    if not np.isfinite([wait, ride]).all():
        raise ValueError("Physical response costs must be finite")
    cost = 2.0 * wait + ride
    if "cost" in value and not np.isclose(float(value["cost"]), cost, rtol=1e-10, atol=1e-7):
        raise ValueError("Recorded cost does not equal2*wait+ride")
    return dict(wait=wait, ride=ride, cost=cost)


def actual_holds(record):
    holds = tuple(map(float, record["holds"]))
    if holds != HOLDS:
        raise ValueError("The readout requires0/7.5-second actions")
    executed = [float(record["outcomes"][arm]["execution"]["native_applied_hold_seconds"])
                for arm in ("low", "high")]
    if not np.allclose(executed, HOLDS, rtol=0.0, atol=1e-5):
        raise ValueError("Native executed actions differ from0/7.5seconds")
    return executed


class DirectDifferenceReadout:
    """Unchanged ridge1, target-scale1000 fit; no path to Q, actor, V or F."""

    def __init__(self, *, line_only=False):
        self.line_only = bool(line_only)
        self.ridge = 1.0
        self.records, self.weights, self.refits = [], None, 0

    @property
    def columns(self):
        return 3 if self.line_only else 4

    def accept(self, records, now):
        existing, admitted = {row["id"] for row in self.records}, []
        for row in records:
            if row["role"] != "train" or row.get("selected_for_training") is not True:
                raise ValueError("Only budget-admitted new training pairs may fit")
            available, at = float(row["available_at"]), float(row["admitted_at"])
            if at < available or max(at, available) > float(now):
                raise ValueError("Pair fit precedes its native release or packet admission")
            if row["id"] in existing:
                raise ValueError("A pair cannot supply a second contrast fit row")
            actual_holds(row)
            vector, line = _features(row["features"], row["line_id"])
            costs = [physical_cost(row["outcomes"][arm]) for arm in ("low", "high")]
            admitted.append(dict(id=row["id"], line_id=line, features=vector.tolist(),
                available_at=available, admitted_at=at,
                gain=[costs[0][key] - costs[1][key] for key in ("wait", "ride")]))
            existing.add(row["id"])
        self.records.extend(admitted)

    def _training_arrays(self):
        return (np.asarray([row["features"][:self.columns] for row in self.records], dtype=float),
                np.asarray([row["gain"] for row in self.records], dtype=float))

    def refit(self):
        if self.records:
            matrix, target = self._training_arrays()
            self.weights = np.linalg.solve(matrix.T @ matrix + self.ridge * np.eye(self.columns),
                                           matrix.T @ (target / TARGET_SCALE))
            self.refits += 1

    def predict(self, features):
        vector, line = _features(features)
        observed = [row["features"][3] for row in self.records if row["line_id"] == line]
        supported = self.weights is not None and bool(observed)
        return dict(status="supported" if supported else "unsupported", line_id=line,
            support_pairs=len(observed), features=vector[:self.columns].tolist(),
            headway_range=[min(observed), max(observed)] if observed else None,
            headway_extrapolation=bool(observed and not min(observed) <= vector[3] <= max(observed)),
            gain=_gain(vector[:self.columns] @ self.weights * TARGET_SCALE) if supported else None,
            fixed300_clock=dict(low=300.0, high=300.0), continuation="common_frozen_mu14")

    def predict_issue(self, issued):
        return self.predict(issue_features(issued))

    def summary(self):
        design = dict(rows=len(self.records), columns=self.columns, rank=0)
        error = coefficients = None
        if self.weights is not None:
            matrix, target = self._training_arrays()
            residual = matrix @ self.weights * TARGET_SCALE - target
            design["rank"] = int(np.linalg.matrix_rank(matrix))
            error = dict(wait_mae=float(abs(residual[:, 0]).mean()),
                ride_mae=float(abs(residual[:, 1]).mean()),
                cost_mae=float(abs(2.0 * residual[:, 0] + residual[:, 1]).mean()))
            coefficients = {key: _gain(value * TARGET_SCALE)
                            for key, value in zip(FEATURE_NAMES[:self.columns], self.weights)}
        return dict(model="passive_finite300_wait_ride_difference", line_only=self.line_only,
            continuation="common_frozen_mu14", holds=list(HOLDS), ridge=self.ridge,
            target_scale=TARGET_SCALE, reward_scale=REWARD_SCALE,
            feature_names=list(FEATURE_NAMES[:self.columns]), pairs=len(self.records),
            outcome_slots=2 * len(self.records), pair_ids=[row["id"] for row in self.records],
            by_line={line: sum(row["line_id"] == line for row in self.records) for line in LINES},
            design=design, coefficients=coefficients, training_error=error, refits=self.refits,
            target="J300_low-J300_high=2*(W_low-W_high)+(R_low-R_high)",
            optimizer_updates=0, controls_native_policy=False, modifies_bellman_target=False)
