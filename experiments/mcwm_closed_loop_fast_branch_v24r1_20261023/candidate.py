"""One issue-visible running-history column for budgeted300-second response."""
import numpy as np

from experiments.mcwm_closed_loop_state_response_v18_20261022.difference import (
    DirectDifferenceReadout, FEATURE_NAMES, TARGET_SCALE, LINES, _design_summary, _physical,
)

HISTORY_FEATURE = 'mean_recent_completed_same_link_runtime_residual_over60'


class HistoryResponseReadout(DirectDifferenceReadout):
    def __init__(self, *, ridge=1.):
        super().__init__(ridge=ridge)
        self.used_records = []

    def refit(self):
        # Keep all admitted records for budget provenance; unavailable history
        # supplies neither a fitted row nor a fictitious zero covariate.
        self.used_records = [r for r in self.records if
            r['service_history'].get('status') == 'supported']
        if not self.used_records:
            return
        matrix = np.asarray([r['features'][:4]+[r['service_history']['mean_residual_scaled']]
            for r in self.used_records], dtype=float)
        target = np.asarray([[r['outcomes']['low']['truth'][key]-r['outcomes']['high']['truth'][key]
            for key in ('wait','ride')] for r in self.used_records], dtype=float)/TARGET_SCALE
        self.weights = np.linalg.solve(matrix.T@matrix+self.ridge*np.eye(5),matrix.T@target)
        self.design = _design_summary(matrix,self.ridge)
        self.refits += 1

    def predict(self, features, history):
        line = LINES[int(np.argmax(features[:3]))]
        count = sum(r['line_id']==line for r in self.used_records)
        history_ready = history.get('status')=='supported'
        supported = self.weights is not None and count>0 and history_ready
        x = list(features[:4])+[history['mean_residual_scaled']] if history_ready else None
        return dict(status='supported' if supported else 'unsupported',line_id=line,
            support_pairs=count,history_supported=history_ready,features=x,
            gain=_physical(np.asarray(x)@self.weights*TARGET_SCALE) if supported else None,
            fixed300_clock=dict(low=300.,high=300.))

    def summary(self):
        return dict(model='history_conditioned_direct300_difference',
            feature_names=list(FEATURE_NAMES)+[HISTORY_FEATURE],ridge=self.ridge,
            target_scale=TARGET_SCALE,pairs=len(self.records),outcome_labels=2*len(self.records),
            fitted_pairs=len(self.used_records),missing_history_pairs=len(self.records)-len(self.used_records),
            pair_ids=[r['id'] for r in self.records],fitted_pair_ids=[r['id'] for r in self.used_records],
            by_line={line:sum(r['line_id']==line for r in self.used_records) for line in LINES},
            design=self.design,refits=self.refits,
            coefficients=None if self.weights is None else self.weights.tolist(),
            semantics='Only own-day budget-admitted training pairs. Both wait/ride share history5; cost=2*wait+ride. Missing history unsupported. Prior stages and heldout truth never fitted.')
