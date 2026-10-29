"""Passive finite-300s contrasts, separate from the native long-horizon critic."""
from __future__ import annotations

import numpy as np

LINES = ("7X", "102X", "705X")
HOLDS = (0.0, 1.875)
FEATURE_NAMES = tuple(line + "_intercept" for line in LINES) + (
    "forward_minus_backward_headway_deviation",
)
TARGET_SCALE = 1000.0


def issue_features(issued):
    """Read only the line and the headways visible at this issue."""
    from experiments.mcwm_closed_loop_v1.stream import EI

    line, event = str(issued["line_id"]), issued["event"]
    if line not in LINES:
        raise ValueError("The finite300 readout supports the three declared lines")
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
        raise ValueError("The first three covariates must identify one declared line")
    if line is not None and LINES[index] != str(line):
        raise ValueError("Issue features and recorded line disagree")
    return vector, LINES[index]


def _gain(wait_ride):
    wait, ride = map(float, wait_ride)
    return dict(wait=wait, ride=ride, cost=2.0 * wait + ride)


class DirectDifferenceReadout:
    """Ridge-fit admitted pair contrasts; never changes Q, V, actor or replay."""

    def __init__(self, *, line_only=False, ridge=1.0):
        self.line_only, self.ridge = bool(line_only), float(ridge)
        if not np.isfinite(self.ridge) or self.ridge <= 0.0:
            raise ValueError("The fixed ridge penalty must be positive")
        self.records, self.weights, self.refits = [], None, 0

    @property
    def columns(self):
        return 3 if self.line_only else 4

    def accept(self, records, now):
        """Accept only released training pairs; each pair supplies one contrast."""
        existing = {row["id"] for row in self.records}
        admitted = []
        for row in records:
            if row["role"] != "train" or row.get("selected_for_training") is not True:
                raise ValueError("Only budget-admitted training pairs may fit the readout")
            available, at = float(row["available_at"]), float(row["admitted_at"])
            if at < available or max(at, available) > float(now):
                raise ValueError("A paired response cannot fit before native feedback release")
            if row["id"] in existing:
                raise ValueError("A pair cannot supply a second contrast fit row")
            if tuple(map(float, row["holds"])) != HOLDS:
                raise ValueError("The readout requires actual executed 0/1.875-second actions")
            vector, line = _features(row["features"], row["line_id"])
            outcomes = []
            for arm in ("low", "high"):
                truth = row["outcomes"][arm]["truth"]
                if truth.get("status") != "complete" or truth.get("dt") != 300.0:
                    raise ValueError("Both outcomes must cover exactly300 native seconds")
                outcomes.append(np.asarray([truth["wait"], truth["ride"]], dtype=float))
            if not np.isfinite(outcomes).all():
                raise ValueError("A complete finite300 outcome must contain finite physical costs")
            admitted.append(dict(id=row["id"], line_id=line, features=vector.tolist(),
                available_at=available, admitted_at=at,
                gain=(outcomes[0] - outcomes[1]).tolist()))
            existing.add(row["id"])
        self.records.extend(admitted)

    def refit(self):
        if not self.records:
            return
        matrix, target = self._training_arrays()
        self.weights = np.linalg.solve(matrix.T @ matrix + self.ridge * np.eye(self.columns),
            matrix.T @ (target / TARGET_SCALE))
        self.refits += 1

    def _training_arrays(self):
        return (np.asarray([row["features"][:self.columns] for row in self.records], dtype=float),
            np.asarray([row["gain"] for row in self.records], dtype=float))

    def predict(self, features):
        vector, line = _features(features)
        support = [row for row in self.records if row["line_id"] == line]
        supported = self.weights is not None and bool(support)
        return dict(status="supported" if supported else "unsupported", line_id=line,
            support_pairs=len(support), features=vector[:self.columns].tolist(),
            gain=_gain(vector[:self.columns] @ self.weights * TARGET_SCALE) if supported else None,
            fixed300_clock=dict(low=300.0, high=300.0), continuation="common_frozen_zero_actor")

    def predict_issue(self, issued):
        return self.predict(issue_features(issued))

    def summary(self):
        design, error, coefficients = dict(rows=len(self.records), columns=self.columns, rank=0), None, None
        if self.weights is not None:
            matrix, target = self._training_arrays()
            residual = matrix @ self.weights * TARGET_SCALE - target
            cost_error = 2.0 * residual[:, 0] + residual[:, 1]
            design["rank"] = int(np.linalg.matrix_rank(matrix))
            error = dict(wait_mae=float(abs(residual[:, 0]).mean()),
                ride_mae=float(abs(residual[:, 1]).mean()), cost_mae=float(abs(cost_error).mean()))
            coefficients = {key: _gain(value * TARGET_SCALE)
                for key, value in zip(FEATURE_NAMES[:self.columns], self.weights)}
        return dict(model="passive_finite300_wait_ride_difference", line_only=self.line_only,
            ridge=self.ridge, target_scale=TARGET_SCALE, feature_names=list(FEATURE_NAMES[:self.columns]),
            pairs=len(self.records), outcome_slots=2 * len(self.records),
            pair_ids=[row["id"] for row in self.records],
            by_line={line: sum(row["line_id"] == line for row in self.records) for line in LINES},
            design=design, coefficients=coefficients, training_error=error, refits=self.refits,
            target="C_low-C_high=2*(W_low-W_high)+(R_low-R_high), both clocks300",
            optimizer_updates=0, controls_native_policy=False, modifies_bellman_target=False)
