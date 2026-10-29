"""Direct paired300-second waiting/ride differences with four fixed covariates."""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np

LINES = ('7X', '102X', '705X')
FEATURE_NAMES = tuple(line+'_intercept' for line in LINES)+('forward_minus_backward_headway_deviation',)
TARGET_SCALE = 1000.


def issue_features(issued, context=None):
    """Use the same issue-time headway covariates as the preceding readouts."""
    from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.endpoints import response_design
    record = SimpleNamespace(line_id=str(issued['line_id']), event=np.asarray(issued['event']),
        started_at=float(issued['started_at']), hold_seconds=0., context={} if context is None else context)
    original = response_design(record)
    return [float(record.line_id == line) for line in LINES]+[float(original[6]-original[7])]


def _physical(wait_ride):
    wait, ride = (float(value) for value in wait_ride)
    return dict(dt=0., wait=wait, ride=ride, cost=2.*wait+ride)


def _design_summary(matrix, ridge):
    rows, columns = matrix.shape
    result = dict(rows=rows, columns=columns, rank=0, active_columns=0,
        ridge=float(ridge), ridge_precision_condition=1., raw_condition=None)
    if rows:
        singular = np.linalg.svd(matrix, compute_uv=False)
        full = len(singular) == columns and singular[-1] > 0.
        minimum = float(singular[-1]) if len(singular) == columns else 0.
        result.update(rank=int(np.linalg.matrix_rank(matrix)),
            active_columns=int(np.any(matrix != 0., axis=0).sum()),
            ridge_precision_condition=float((singular[0]**2+ridge)/(minimum**2+ridge)),
            raw_condition=float(singular[0]/singular[-1]) if full else None)
    return result


class DirectDifferenceReadout:
    """Fit only admitted paired labels; no action-specific background level fit."""

    def __init__(self, *, line_only=False, ridge=1.):
        self.line_only, self.ridge = bool(line_only), float(ridge)
        self.records, self.weights, self.design, self.refits = [], None, {}, 0

    @property
    def columns(self):
        return 3 if self.line_only else 4

    def accept(self, records, now):
        existing = {record['id'] for record in self.records}
        for record in records:
            if record['role'] != 'train' or record.get('selected_for_training') is not True:
                raise ValueError('Difference fitting accepts budget-admitted training pairs only')
            if record['available_at'] > float(now) or record['admitted_at'] > float(now):
                raise ValueError('A paired contrast cannot be fitted before feedback release')
            if record['id'] in existing:
                raise ValueError('An admitted pair cannot supply a second contrast fit row')
            for arm in ('low', 'high'):
                truth = record['outcomes'][arm]['truth']
                if truth.get('status') != 'complete' or truth.get('dt') != 300.:
                    raise ValueError('Paired difference targets require both exact300-second outcomes')
            self.records.append(record)
            existing.add(record['id'])

    def refit(self):
        if not self.records:
            return
        matrix = np.asarray([record['features'][:self.columns] for record in self.records], dtype=float)
        target = np.asarray([[record['outcomes']['low']['truth'][key]
            -record['outcomes']['high']['truth'][key] for key in ('wait', 'ride')]
            for record in self.records], dtype=float)/TARGET_SCALE
        self.weights = np.linalg.solve(matrix.T@matrix+self.ridge*np.eye(self.columns), matrix.T@target)
        self.design = _design_summary(matrix, self.ridge)
        self.refits += 1

    def predict(self, features):
        features = np.asarray(features, dtype=float)
        line = LINES[int(np.argmax(features[:3]))]
        count = sum(record['line_id'] == line for record in self.records)
        supported = self.weights is not None and count > 0
        values = _physical(features[:self.columns]@self.weights*TARGET_SCALE) if supported else None
        return dict(status='supported' if supported else 'unsupported', line_id=line,
            support_pairs=int(count), features=features[:self.columns].tolist(), gain=values,
            fixed300_clock=dict(low=300., high=300.))

    def predict_issue(self, issued, context=None):
        return self.predict(issue_features(issued, context))

    def summary(self):
        error = None
        coefficients = None
        if self.weights is not None:
            matrix = np.asarray([record['features'][:self.columns] for record in self.records], dtype=float)
            target = np.asarray([[record['outcomes']['low']['truth'][key]
                -record['outcomes']['high']['truth'][key] for key in ('wait', 'ride')]
                for record in self.records], dtype=float)
            physical_error = matrix@self.weights*TARGET_SCALE-target
            cost_error = 2.*physical_error[:, 0]+physical_error[:, 1]
            error = dict(wait_mae=float(np.mean(abs(physical_error[:, 0]))),
                ride_mae=float(np.mean(abs(physical_error[:, 1]))),
                cost_mae=float(np.mean(abs(cost_error))), cost_rmse=float(np.sqrt(np.mean(cost_error**2))))
            coefficients = {name: _physical(row*TARGET_SCALE)
                for name, row in zip(FEATURE_NAMES[:self.columns], self.weights)}
        return dict(model='direct300_paired_wait_ride_difference', line_only=self.line_only,
            ridge=self.ridge, target_scale=TARGET_SCALE,
            feature_names=list(FEATURE_NAMES[:self.columns]), targets=['wait', 'ride'],
            pairs=len(self.records), pair_ids=[record['id'] for record in self.records],
            outcome_labels=2*len(self.records),
            by_line={line: sum(record['line_id'] == line for record in self.records) for line in LINES},
            design=self.design, coefficients=coefficients, training_error=error, refits=self.refits,
            semantics='Predict low-minus-high waiting/ride differences directly; cost=2*wait+ride, dt difference0 and both clocks300. Same admitted pairs; no fitted background, no scene feature or heldout fit.')
