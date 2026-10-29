"""Bounded issue-only census and execution-map diagnostics; never learning data."""
from __future__ import annotations

from collections import Counter, defaultdict
import json

import numpy as np
import torch

from experiments.mcwm_closed_loop_stratified_pairs_v23_20261023.sampling import visible_stratum


MAPPING_COLUMNS = ('issues', 'floor_condition', 'floor_changed', 'floor_increment_sum_seconds',
    'floor_increment_max_seconds', 'native_tail_changed', 'native_tail_absolute_sum_seconds',
    'native_tail_absolute_max_seconds', 'reserve_changed', 'native_hold_sum_seconds',
    'tail_hold_sum_seconds', 'no_floor_hold_sum_seconds', 'actor_clock_mismatch')
LINES = ('7X', '102X', '705X')
DEFAULT_WINDOWS = {
    '7X': [('bootstrap', 1200., 2700.), ('response', 3000., 6600.), ('late', 15000., 17670.)],
    '102X': [('bootstrap', 3600., 6600.), ('response', 7200., 11700.), ('late', 15000., 17670.)],
    '705X': [('bootstrap', 3600., 6600.), ('response', 7200., 11700.), ('late', 15000., 17670.)],
}
EPS = 1.e-4


def _bin(value):
    return 'below_minus_hold' if value < -7.5 else 'above_hold' if value > 7.5 else 'within_hold'


def _range_update(row, key, value):
    previous = row[key]
    row[key] = [value, value] if previous is None else [min(previous[0], value), max(previous[1], value)]


def headway_feature(event):
    """Exact v18 raw-event covariate, retaining imbalances clipped in canonical s.

    v7 retains v4 raw-event indices3/4 for headways and5/6 for their targets.
    Each deviation subtracts1 before their difference, as in v18 issue_features.
    """
    raw = np.asarray(event['event'])
    forward = float(raw[3])/max(1., float(raw[5]))-1.
    backward = float(raw[4])/max(1., float(raw[6]))-1.
    return forward-backward


class IssueAudit:
    def __init__(self, protocol, transform):
        self.transform = transform
        self.horizon = float(protocol['horizon_seconds'])
        self.station = str(protocol['diagnostic_station_id'])
        windows = protocol.get('diagnostic_windows', {
            line:[dict(name=name, start=start, end=end) for name, start, end in rows]
            for line, rows in DEFAULT_WINDOWS.items()})
        self.windows = {}
        for line in LINES:
            for window in windows[line]:
                self.windows[(line, window['name'])] = dict(start=float(window['start']), end=float(window['end']),
                    issues=0, action_supported=0, action_unsupported=0, history_supported=0,
                    history_missing=0, history_noncausal=0, history_feature_missing=0,
                    joint_eligible=0, bins_agree=0, third_joint_counts=Counter(), mean_joint_counts=Counter(),
                    raw_headway_range=None, history_mean_seconds_range=None)
        self.mapping = defaultdict(lambda: np.zeros(len(MAPPING_COLUMNS), dtype=np.float64))

    @torch.no_grad()
    def _execution(self, event, proposal, need_choices):
        state = torch.as_tensor(np.asarray(event['s'], dtype=np.float32)).reshape(1, 229)
        action = torch.as_tensor(np.asarray(proposal, dtype=np.float32)).reshape(1, 1)
        # One F call for the factual proposal; only window issues also inspect F's
        # algebraic action support. These are commands, never environment queries.
        if need_choices:
            candidates = torch.tensor([[-1.], [2.*7.5/self.transform.max_hold-1.], [1.]], dtype=torch.float32)
            proposals = torch.cat((action, candidates))
            executed = self.transform(state.expand(4, -1), proposals)
            holds = self.transform.hold_seconds(executed).reshape(-1)
            choices = all(bool(torch.any(abs(holds[1:]-seconds) <= 1.e-5)) for seconds in (0., 7.5))
            native = holds[0]
        else:
            native = self.transform.hold_seconds(self.transform(state, action)).reshape(-1)[0]
            choices = False
        parent = self.transform.hold_seconds(action).reshape(-1)[0]
        tail = .125*parent
        current = torch.round(state[0, 2]*15).long().clamp(0, 15)
        local = state[:, 5:].reshape(16, 14)[current]
        forward, backward = local[11] > .5, local[12] > .5
        forward_error = (local[3]-1.)*local[5]*self.transform.headway_scale
        backward_error = (local[4]-1.)*local[6]*self.transform.headway_scale
        reserve = (.5*(backward_error*backward-forward_error*forward)).clamp(0., self.transform.max_hold)
        no_floor = torch.where((state[0, 0] < .75) & (parent > 0.),
            torch.minimum(parent, torch.maximum(tail, reserve)), tail).clamp(0., self.transform.max_hold)
        floor = ((state[0, 0] >= .15) & (state[0, 0] < .75) & forward & ~backward
            & (local[9] > 0.) & (local[8] <= .25))
        increment, difference = float(native-no_floor), abs(float(native-tail))
        expected_clock = np.float32(np.float32(event['started_at'])/self.horizon)
        return np.asarray([1., bool(floor), increment > EPS, increment, increment,
            difference > EPS, difference, difference, abs(float(no_floor-tail)) > EPS,
            float(native), float(tail), float(no_floor), float(state[0, 0]) != float(expected_clock)]), choices

    def observe(self, event, proposal, history_snapshot=None):
        """Consume issue-visible inputs, retain counters only, return no training row."""
        line, now = str(event['line_id']), float(event['started_at'])
        matching = [row for (candidate, _), row in self.windows.items()
            if candidate == line and str(event['station_id']) == self.station and row['start'] <= now < row['end']]
        value, action_supported = self._execution(event, proposal, bool(matching))
        phase = 'pre_floor' if now < .15*self.horizon else 'reserve_floor' if now < .75*self.horizon else 'tail'
        total = self.mapping[(line, phase)]
        for i in range(len(MAPPING_COLUMNS)):
            total[i] = max(total[i], value[i]) if i in (4, 7) else total[i]+value[i]
        for row in matching:
            row['issues'] += 1
            row['action_supported' if action_supported else 'action_unsupported'] += 1
            history = history_snapshot or {}
            segments = history.get('segments', [])
            if history.get('status') != 'supported' or len(segments) != 3 or any(
                    not segment.get('available') or segment.get('latest') is None for segment in segments):
                row['history_missing'] += 1
                continue
            if any(segment['latest']['completed_at'] >= now or segment['latest']['observed_at'] >= now for segment in segments):
                row['history_noncausal'] += 1
                continue
            if history.get('mean_residual_scaled') is None:
                row['history_feature_missing'] += 1
                continue
            features = [float(line == target) for target in LINES]+[headway_feature(event)]
            third = visible_stratum(issued_at=now, features=features, history=history)
            if third is None:
                row['history_missing'] += 1
                continue
            row['history_supported'] += 1
            if not action_supported:
                continue
            mean = float(history['mean_residual_scaled'])*60.
            mean_bin, third_bin = _bin(mean), third['third_residual_bin']
            row['joint_eligible'] += 1
            row['bins_agree'] += mean_bin == third_bin
            row['third_joint_counts'][f"{third['headway_sign']}|{third_bin}"] += 1
            row['mean_joint_counts'][f"{third['headway_sign']}|{mean_bin}"] += 1
            _range_update(row, 'raw_headway_range', features[3])
            _range_update(row, 'history_mean_seconds_range', mean)

    def summary(self):
        result = dict(status='complete', role='passive_issue_diagnostic', training_rows=0,
            environment_queries=0, retained_full_states=0,
            mapping_columns=list(MAPPING_COLUMNS),
            mapping=[dict(line=line, phase=phase, values=[int(value) if i in (0, 1, 2, 5, 8, 12)
                else float(value) for i, value in enumerate(values)])
                for (line, phase), values in sorted(self.mapping.items())],
            windows=[dict(line=line, name=name, **{key:dict(value) if isinstance(value, Counter) else value
                for key, value in row.items()}) for (line, name), row in self.windows.items()],
            semantics=dict(census='Every factual upstream issue throughout each half-open window; no stop after first eligible issue, no sampling or labels.',
                eligible='Both0/7.5-second commands are executable under unchanged F, plus all three history segments completed and observed strictly before issue.',
                covariates='Headway is the original unclipped raw forward/target minus backward/target deviation; third residual and actual mean_residual_scaled*60 use the same +/-7.5-second bins.',
                mapping='Native F uses the original canonical state and actor proposal. Shadow tail=proposal_hold/8; no_floor recomputes F with only completion floor removed. Condition counts and actual increments are distinct; clocks are read and never changed.',
                phase_seconds=[.15*self.horizon, .75*self.horizon],
                joint_key='headway_sign|history_bin',
                windows='Diagnostic support only; no service-reversal gate for the factual fresh-attribution experiment.'))
        if len(json.dumps(result, separators=(',', ':'), allow_nan=False).encode()) > 20000:
            raise ValueError('Passive issue diagnostic exceeds the small-summary budget')
        return result
