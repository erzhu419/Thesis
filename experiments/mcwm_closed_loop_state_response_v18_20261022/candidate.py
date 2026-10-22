"""One prespecified issue-load term for the paired 300-second ride response."""
from __future__ import annotations

import numpy as np

from experiments.mcwm_closed_loop_v1.stream import EI
from experiments.mcwm_closed_loop_state_response_v18_20261022.difference import (
    DirectDifferenceReadout, FEATURE_NAMES, TARGET_SCALE, _design_summary,
    _physical, issue_features,
)


RIDE_FEATURE_NAMES = FEATURE_NAMES + ('onboard_capacity_ratio',)


def issue_state_features(issued, context=None):
    """Append the decision-ready bus load without changing the original four."""
    event = np.asarray(issued['event'])
    load = float(event[EI['onboard_passengers']])/float(event[EI['bus_capacity']])
    return issue_features(issued, context) + [load]


class StateResponseReadout(DirectDifferenceReadout):
    """Keep the shared waiting fit exact and add one observed ride covariate."""

    def __init__(self, *, ridge=1.):
        super().__init__(ridge=ridge)
        self.ride_weights, self.ride_design = None, {}

    def refit(self):
        if not self.records:
            return
        # Reuse the complete four-column solve and prediction path for exact
        # waiting parity with the shared control on the same admitted pairs.
        super().refit()
        matrix = np.asarray([record['features'][:5] for record in self.records], dtype=float)
        target = np.asarray([record['outcomes']['low']['truth']['ride']
            - record['outcomes']['high']['truth']['ride']
            for record in self.records], dtype=float)/TARGET_SCALE
        self.ride_weights = np.linalg.solve(matrix.T@matrix+self.ridge*np.eye(5), matrix.T@target)
        self.ride_design = _design_summary(matrix, self.ridge)

    def predict(self, features):
        result = super().predict(features)
        features = np.asarray(features, dtype=float)
        result['features'] = features[:5].tolist()
        if result['gain'] is not None:
            ride = float(features[:5]@self.ride_weights*TARGET_SCALE)
            result['gain'] = _physical((result['gain']['wait'], ride))
        return result

    def predict_issue(self, issued, context=None):
        return self.predict(issue_state_features(issued, context))

    def summary(self):
        result = super().summary()
        coefficients, error = None, None
        if self.ride_weights is not None:
            coefficients = dict(
                wait={name: float(value*TARGET_SCALE)
                    for name, value in zip(FEATURE_NAMES, self.weights[:, 0])},
                ride={name: float(value*TARGET_SCALE)
                    for name, value in zip(RIDE_FEATURE_NAMES, self.ride_weights)})
            errors = []
            for record in self.records:
                gain = self.predict(record['features'])['gain']
                errors.append([gain[key]-(record['outcomes']['low']['truth'][key]
                    -record['outcomes']['high']['truth'][key]) for key in ('wait', 'ride')])
            errors = np.asarray(errors)
            cost_error = 2.*errors[:, 0]+errors[:, 1]
            error = dict(wait_mae=float(np.mean(abs(errors[:, 0]))),
                ride_mae=float(np.mean(abs(errors[:, 1]))),
                cost_mae=float(np.mean(abs(cost_error))),
                cost_rmse=float(np.sqrt(np.mean(cost_error**2))))
        result.update(model='direct300_paired_wait4_ride5_issue_load',
            feature_names=dict(wait=list(FEATURE_NAMES), ride=list(RIDE_FEATURE_NAMES)),
            design=dict(wait=self.design, ride=self.ride_design),
            coefficients=coefficients, training_error=error,
            semantics='Waiting exactly reuses shared four-column prediction; ride adds only issue onboard/capacity. Same admitted pairs and fixed ridge; cost=2*wait+ride, dt difference0 and clocks300. No background fit, scene feature, future service feature or heldout fit.')
        return result
