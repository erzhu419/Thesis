"""Issue-visible stratified pair selection; outcomes never select an issue.

Each line has three fixed cells. Bootstrap selects its first eligible issue;
later cells prefer a joint stratum not selected previously on that line. The
last third of each later window takes the first eligible issue as a declared
fallback. An expired cell stays missing rather than selecting retrospectively.
"""
from copy import deepcopy
from math import isfinite


HOLD_SECONDS = 7.5


def _sign(value):
    return 1 if value > 0 else -1 if value < 0 else 0


def visible_stratum(*, issued_at, features, history):
    """Return only strictly pre-issue history and original headway covariates."""
    segments = history.get('segments', [])
    if history.get('status') != 'supported' or len(segments) != 3:
        return None
    residuals = []
    for segment in segments:
        latest = segment.get('latest')
        if not segment.get('available') or latest is None:
            return None
        if not latest['completed_at'] < issued_at or not latest['observed_at'] < issued_at:
            return None
        residuals.append(float(latest['residual']))
    headway = float(features[3])
    if not all(isfinite(value) for value in residuals + [headway]):
        return None
    third = residuals[2]
    bucket = 'below_minus_hold' if third < -HOLD_SECONDS else (
        'above_hold' if third > HOLD_SECONDS else 'within_hold')
    return dict(headway_sign=_sign(headway), third_residual_bin=bucket,
        third_residual_seconds=third,
        history_spread_seconds=max(residuals)-min(residuals),
        history_signs=[_sign(value) for value in residuals])


def _joint(stratum):
    return stratum['headway_sign'], stratum['third_residual_bin']


class StratifiedPairSampler:
    def __init__(self, plans, lines):
        self.plans = [deepcopy(row) for row in plans if row['role'] == 'train']
        self.lines = list(lines)
        self.selected, self.chosen = [], set()
        self.eligible = {row['id']: 0 for row in self.plans}
        self.waited = {row['id']: 0 for row in self.plans}
        self.history_unsupported = {row['id']: 0 for row in self.plans}

    def consider(self, plan, *, issued_at, line_id, features, history):
        """Commit at this issue, or wait; the caller checks action/fit support.

        Non-bootstrap calls require a previous admitted own-line pair, checked
        by the runner using pair counts only. No truth or prediction is an
        argument to this sampler.
        """
        if plan['role'] != 'train' or plan['id'] in self.chosen:
            return None
        start, end = plan['window']['start'], plan['window']['end']
        if line_id != plan['line_id'] or not start <= issued_at < end:
            return None
        stratum = visible_stratum(issued_at=issued_at, features=features, history=history)
        if stratum is None:
            self.history_unsupported[plan['id']] += 1
            return None
        self.eligible[plan['id']] += 1
        previous = [row for row in self.selected if row['line_id'] == line_id]
        novel = _joint(stratum) not in {_joint(row['stratum']) for row in previous}
        bootstrap = plan['window_name'] == 'bootstrap'
        fallback = False
        if bootstrap:
            reason = 'bootstrap_first_eligible'
        elif novel:
            reason = 'first_unseen_joint_stratum'
        elif issued_at < start + (end-start)*2./3.:
            self.waited[plan['id']] += 1
            return None
        else:
            reason, fallback = 'last_third_first_eligible_fallback', True
        result = dict(id=plan['id'], line_id=line_id, window_name=plan['window_name'],
            issued_at=float(issued_at), selection_reason=reason, stratum=stratum,
            target_met=bool(bootstrap or novel), fallback=fallback)
        self.chosen.add(plan['id'])
        self.selected.append(result)
        return deepcopy(result)

    def report(self):
        expected = {row['id'] for row in self.plans}
        by_line = {}
        for line in self.lines:
            selected = [row for row in self.selected if row['line_id'] == line]
            unique = sorted({_joint(row['stratum']) for row in selected})
            by_line[line] = dict(selected_pairs=len(selected),
                unique_joint_strata=[dict(headway_sign=sign, third_residual_bin=bucket)
                    for sign, bucket in unique], unique_joint_strata_count=len(unique),
                headway_signs=sorted({row['stratum']['headway_sign'] for row in selected}),
                target_met_by_window={row['window_name']: row['target_met'] for row in selected})
        checks = dict(exactly_nine_cells=len(self.plans) == 9,
            selected_once=len(self.selected) == len(self.chosen),
            selected_only_planned=self.chosen <= expected,
            all_planned_cells_selected=self.chosen == expected,
            three_pairs_per_line=all(row['selected_pairs'] == 3 for row in by_line.values()))
        return dict(all_pass=all(checks.values()), checks=checks,
            selected=deepcopy(self.selected), by_line=by_line,
            missing_ids=sorted(expected-self.chosen), eligible_issues=dict(self.eligible),
            waited_for_novel_stratum=dict(self.waited),
            history_unsupported_issues=dict(self.history_unsupported),
            stratum_definition=dict(headway='sign of unchanged issue_features[3]',
                third_history='third inbound completed-runtime residual',
                third_history_bins=dict(below_minus_hold='residual < -7.5 seconds',
                    within_hold='-7.5 <= residual <= 7.5 seconds',
                    above_hold='residual > 7.5 seconds'),
                fallback='first eligible issue in last third, never retrospective'),
            semantics='Selection uses only strict pre-issue history and issue covariates. '
                'Truth, prediction error, reversal labels and old heldouts are not inputs. '
                'Structural all_pass is separate from two-strata-per-line scientific coverage.')
