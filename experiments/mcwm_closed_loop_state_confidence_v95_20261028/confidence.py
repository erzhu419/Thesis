"""Two-parameter residual bias and a fixed-design support scale.

Calibration groups are external seeds. The support scale is not an identified
epistemic standard deviation. Native eligibility is locked by the caller.
"""
from math import inf, isfinite, isnan, sqrt

import numpy as np

from experiments.mcwm_closed_loop_calibration_design_v94_20261028.calibration import calibrated_margin


def _psi(q, q_range):
    lo, hi = map(float, q_range)
    if not isfinite(float(q)) or not isfinite(lo) or not isfinite(hi) or hi <= lo:
        raise ValueError('Finite q and a nonconstant frozen q range are required')
    return np.asarray([1., 2.*(float(q)-lo)/(hi-lo)-1.], dtype=np.float64)


def fit_bias(rows, q_range):
    """One ridge-1 solve; independent future data alone define deployable fits."""
    rows = list(rows)
    design = np.asarray([_psi(row['q'], q_range) for row in rows], dtype=np.float64).reshape(-1, 2)
    rank = int(np.linalg.matrix_rank(design)) if len(rows) else 0
    if rank != 2:
        raise ValueError('The fixed two-column residual design must have rank2')
    residual = np.asarray([float(row['delta'])-float(row['prediction']) for row in rows])
    if not np.isfinite(residual).all():
        raise ValueError('Every fitting row requires a finite paired residual')
    gram = np.eye(2)+design.T@design
    solved = np.linalg.solve(gram, np.column_stack((design.T@residual, np.eye(2))))
    return dict(q_range=list(map(float, q_range)), coef=solved[:, 0].tolist(),
                gram_inverse=solved[:, 1:].tolist(), n=len(rows), rank=rank,
                ridge=1., linear_solves=1)


def predict(model, q, m):
    """Return bias, support scale and corrected center without reading a label."""
    features = _psi(q, model['q_range'])
    bias = float(features@np.asarray(model['coef']))
    support_scale = sqrt(1.+float(features@np.asarray(model['gram_inverse'])@features))
    return dict(bias=bias, support_scale=support_scale, center=float(m)+bias)


def seed_score(rows, bias_model):
    """Maximum standardized positive error over all locked negative proposals."""
    score = 0.
    for row in rows:
        if not row['active']:
            continue
        m = float(row['m'])
        if not isfinite(m):
            raise ValueError('An active frozen prediction must be finite')
        if m >= 0.:
            continue
        delta = row.get('delta')
        if delta is None or not isfinite(float(delta)):
            return inf
        prediction = predict(bias_model, row['q'], m)
        score = max(score, (float(delta)-prediction['center'])/prediction['support_scale'])
    return score


def upper_bound(m, q, bias_model, margin):
    margin = float(margin)
    if isnan(margin) or margin < 0.:
        raise ValueError('The calibrated margin must be nonnegative')
    result = predict(bias_model, q, m)
    return result['center']+margin*result['support_scale']


def select_first(rows, bias_model, margin):
    """Call separately per scene; select at most once using issue-visible fields."""
    ordered = sorted((row for row in rows if row['active']),
                     key=lambda row:(row['issue_clock'], row['sequence']))
    for row in ordered:
        m = float(row['m'])
        if not isfinite(m):
            raise ValueError('An active frozen prediction must be finite')
        if m < 0. and upper_bound(m, row['q'], bias_model, margin) < 0.:
            return row
    return None
