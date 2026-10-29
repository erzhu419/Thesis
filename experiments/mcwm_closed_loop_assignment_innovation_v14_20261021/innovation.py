"""Passive action innovations from randomized block starts and prior feedback."""
from __future__ import annotations

import numpy as np

from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.endpoints import (
    ENDPOINT_NAMES, LINE_NAMES, TARGET_SCALES,
)
from experiments.mcwm_closed_loop_response_residual_v11_20261021.residual import (
    RESIDUAL_TARGET_NAMES, ResidualResponseReadout, _view, endpoint_baseline,
)
from experiments.mcwm_closed_loop_separated_contrast_v13_20261021.separated import (
    ACTION_NAMES, BACKGROUND_NAMES, FEATURE_NAMES, SeparatedContrastReadout,
    _error_metric, _rank_summary, _ridge, separated_design,
)
from experiments.mcwm_closed_loop_state_interaction_v12_20261021.interaction import (
    _physical_component,
)


class AssignmentInnovationReadout(SeparatedContrastReadout):
    """Keep block-second responses for levels; learn slopes only from fresh bits."""

    def __init__(self, *, ridge=1., line_only=False):
        super().__init__(ridge=ridge)
        self.line_only = bool(line_only)
        self.background_readouts = {}
        self.background_sources = {}
        self.innovation_diagnostics = {}

    def capture_context(self, issued, context):
        """Freeze the prediction available before this action's outcome exists."""
        z, _ = separated_design(_view(issued, context))
        snapshot = dict(issued_at=float(issued['started_at']), sequence=int(issued['sequence']), endpoints={})
        for endpoint, names in RESIDUAL_TARGET_NAMES.items():
            weight = self.background_readouts.get(endpoint)
            value = np.zeros(len(names)) if weight is None else z@weight
            source = self.background_sources.get(endpoint, dict(fit_rows=0, max_available_at=None, max_sequence=None))
            snapshot['endpoints'][endpoint] = dict(status='cold_start_zero' if weight is None else 'fitted',
                mean=value.tolist(), **source)
        context['innovation_background'] = snapshot
        return snapshot

    @staticmethod
    def _first(row):
        return int(row.context['assignment_position']) == 0

    def _first_records(self, endpoint):
        return [row for row in self._endpoint_records(endpoint) if self._first(row)]

    def endpoint_coverage(self):
        return {endpoint: {line: dict(
            n=sum(row.line_id == line for row in self._first_records(endpoint)),
            zero_holding=sum(row.line_id == line and bool(np.isclose(row.hold_seconds, 0.))
                             for row in self._first_records(endpoint)),
            high_holding=sum(row.line_id == line and bool(np.isclose(row.hold_seconds, 7.5))
                             for row in self._first_records(endpoint)))
            for line in LINE_NAMES} for endpoint in ENDPOINT_NAMES}

    @staticmethod
    def _scope(row, snapshot, source):
        return (snapshot['sequence'] == row.sequence and snapshot['issued_at'] == row.started_at
            and (source['max_available_at'] is None or source['max_available_at'] <= row.started_at)
            and (source['max_sequence'] is None or source['max_sequence'] < row.sequence))

    def refit(self):
        if not self.records:
            return
        self.design = _rank_summary(np.asarray([np.concatenate(separated_design(row)) for row in self.records]), self.ridge)
        for endpoint, names in RESIDUAL_TARGET_NAMES.items():
            rows = self._endpoint_records(endpoint)
            if not rows:
                continue
            z = np.asarray([separated_design(row)[0] for row in rows])
            t = np.asarray([separated_design(row)[1] for row in rows])
            scales = TARGET_SCALES if endpoint != 'fixed300' else TARGET_SCALES[1:]
            y = np.asarray([[row.endpoints[endpoint][key]-endpoint_baseline(row, endpoint)[key]
                             for key in names] for row in rows])/scales
            first = np.asarray([self._first(row) for row in rows])
            first_rows = [row for row in rows if self._first(row)]
            # These means were saved at issue. A current refit must never recompute them.
            snapshots = [row.context['innovation_background'] for row in first_rows]
            sources = [snapshot['endpoints'][endpoint] for snapshot in snapshots]
            saved_m = np.asarray([source['mean'] for source in sources]).reshape(-1, len(names))
            innovation = y[first]-saved_m
            action = t[first]
            keep = slice(0, 3 if self.line_only else 9)
            beta = np.zeros((9, len(names)))
            if len(first_rows):
                beta[keep] = _ridge(action[:, keep], innovation, self.ridge)
            # Attribution at the fitted precision: retain cold rows in the fit,
            # and measure their exact linear contribution without selecting an arm.
            cold = np.asarray([source['status'] == 'cold_start_zero' for source in sources], dtype=bool)
            cold_beta, warm_beta = np.zeros_like(beta), np.zeros_like(beta)
            if len(first_rows):
                a = action[:, keep]
                precision = a.T@a+self.ridge*np.eye(a.shape[1])
                cold_beta[keep] = np.linalg.solve(precision, a[cold].T@innovation[cold])
                warm_beta[keep] = np.linalg.solve(precision, a[~cold].T@innovation[~cold])
            reconstruction = [_physical_component(value, names, scales)
                for value in beta-cold_beta-warm_beta]
            gamma = _ridge(z, y-t@beta, self.ridge)
            self.readouts[endpoint] = np.concatenate((gamma, beta), axis=0)
            # Direct outcome-only nuisance, common to both action arms. Only future
            # issues will use these coefficients; stored issue predictions stay fixed.
            self.background_readouts[endpoint] = _ridge(z, y, self.ridge)
            self.background_sources[endpoint] = dict(fit_rows=len(rows),
                max_available_at=max(float(row.available_at) for row in rows),
                max_sequence=max(int(row.sequence) for row in rows))
            self.endpoint_designs[endpoint] = _rank_summary(np.concatenate((z, t), axis=1), self.ridge)
            self.endpoint_designs[endpoint]['learned_targets'] = list(names)
            self.innovation_diagnostics[endpoint] = dict(status='fitted' if len(first_rows) else 'no_randomized_first_rows',
                background_rows=len(rows), first_rows=len(first_rows), second_rows=int((~first).sum()),
                first_coverage=self.endpoint_coverage()[endpoint],
                cold_start_first_rows=int(cold.sum()),
                cold_start_action_coefficient_physical={name: _physical_component(value, names, scales)
                    for name, value in zip(ACTION_NAMES, cold_beta)},
                cold_warm_beta_reconstruction_error=max(abs(value) for row in reconstruction for value in row.values()),
                snapshot_scope_valid=all(self._scope(row, snapshot, source)
                    for row, snapshot, source in zip(first_rows, snapshots, sources)),
                action_design=_rank_summary(action[:, keep], self.ridge),
                innovation_error_rmse=_error_metric(innovation, names, scales) if len(first_rows) else None,
                score_error_rmse=_error_metric(innovation-action@beta, names, scales) if len(first_rows) else None,
                action_coefficient_physical={name: _physical_component(value, names, scales)
                    for name, value in zip(ACTION_NAMES, beta)},
                fit_sequence_ids=[int(row.sequence) for row in rows],
                first_sequence_ids=[int(row.sequence) for row in first_rows],
                second_sequence_ids=[int(row.sequence) for row in rows if not self._first(row)])

    def summary(self):
        result = ResidualResponseReadout.summary(self)
        result.update(model='assignment_first_prequential_background_innovation',
            line_only=self.line_only, feature_names=list(FEATURE_NAMES),
            background_names=list(BACKGROUND_NAMES), action_names=list(ACTION_NAMES),
            innovation_diagnostics=self.innovation_diagnostics,
            background_coefficients={key: value.tolist() for key, value in self.background_readouts.items()},
            semantics='Block-first fresh assignment only: known E[u|past]=0, no fitted q; saved issue-time background; cold-start m=0 included. All selected labels fit levels. Global positive-duration support remains conditional; no causal identification guarantee.')
        return result
