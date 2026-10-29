"""Issue-only support and frozen forward predictions; no label access."""
from math import isfinite

LINES = ('7X', '102X', '705X')
MODELS = ('line3', 'service5', 'service_history6')


def lock_first(selection, line, identity):
    """Lock the physical opportunity before context/history support is known."""
    slot = selection[line]
    if slot['selected'] is not None:
        return False
    slot.update(selected=dict(identity), status='locked')
    return True


def context_reason(service, condition, line, model, issue_clock):
    if not service['observed']:
        return service['reason'] or 'native_context_unsupported'
    if condition['status'] != 'supported' or not condition['historical_only']:
        return condition['reason'] or 'history_not_strictly_past'
    if not (condition['latest_completed_at'] < issue_clock
            and condition['latest_observed_at'] < issue_clock
            and condition['latest_age_seconds'] <= 900.):
        return 'history_availability_or_age'
    q = condition['value']
    if q is None or not isfinite(q):
        return 'history_value_missing_or_nonfinite'
    return None


def model_support(model, line, q):
    """Describe the old empirical q range, without granting collection or deployment."""
    low, high = model['common_support']['q_training_ranges'][line]
    available = q is not None and isfinite(q)
    inside = low <= q <= high if available else None
    return dict(q=float(q) if available else None, training_range=[low, high],
                in_training_range=inside,
                extrapolation=not inside if available else None,
                status=('within_training_range' if inside else 'outside_training_range')
                       if available else 'unavailable')


def frozen_predictions(model, line, service, q, ref_hold, cand_hold):
    a = (cand_hold-ref_hold)/model['hold_scale']
    center = model['design']['line_centers'][line]
    x = [a*float(line == other) for other in LINES]
    x += [a*(value-mu) for value, mu in zip(service['features']+[q], center)]
    return {name:sum(value*coef for value, coef in zip(x, model['models'][name]['coef']))
            for name in MODELS}
