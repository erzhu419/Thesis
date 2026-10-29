"""Seed-block one-sided calibration for frozen, pre-action locked proposals.

Rows contain active, predictions[model], delta (candidate-minus-reference cost),
issue_clock and sequence. Call select_first separately for each scene.
"""
from math import ceil, inf, isfinite, isnan


def seed_score(locked_rows, model):
    """Cover every potentially accepted negative point in one whole seed block.

    Eligibility is locked before outcomes. Physical separation and outcome sign
    never filter a point; any required missing outcome keeps the block at inf.
    """
    score = 0.
    for row in locked_rows:
        if not row['active']:
            continue
        prediction = float(row['predictions'][model])
        if not isfinite(prediction):
            raise ValueError('An active frozen prediction must be finite')
        if prediction >= 0.:
            continue
        delta = row.get('delta')
        if delta is None or not isfinite(float(delta)):
            return inf
        score = max(score, float(delta)-prediction)
    return score


def calibrated_margin(scores, alpha):
    """Use the finite-sample upper order statistic, conservatively clipped at0."""
    if not 0. < alpha < 1.:
        raise ValueError('alpha must lie strictly between0 and1')
    values = sorted(float(value) for value in scores)
    if any(isnan(value) for value in values):
        raise ValueError('A seed score cannot be NaN; missing outcomes score inf')
    index = ceil((len(values)+1)*(1.-alpha))
    return inf if index > len(values) else max(0., values[index-1])


def select_first(rows, model, margin):
    """Choose at most one pre-action proposal; no outcome field is read."""
    margin = float(margin)
    if isnan(margin) or margin < 0.:
        raise ValueError('The calibrated upper margin must be nonnegative')
    for row in sorted((value for value in rows if value['active']),
                      key=lambda value:(value['issue_clock'], value['sequence'])):
        prediction = float(row['predictions'][model])
        if not isfinite(prediction):
            raise ValueError('An active frozen prediction must be finite')
        if prediction+margin < 0.:
            return row
    return None
