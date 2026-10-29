"""Unweighted factual Brier readout and the frozen two-scene admission gate."""
import math


MODELS = ('fresh', 'frozen', 'hold_only', 'past_prevalence', 'clock_zero_risk')
BASELINES = MODELS[1:]
SCENES = ('stable', 'local_speed_drop')


def _summary(rows):
    count = len(rows)
    brier = {model: math.fsum((row['probabilities'][model] - row['target']) ** 2
                             for row in rows) / count if count else None
             for model in MODELS}
    return dict(count=count, brier=brier,
                fresh_minus={model: brier['fresh'] - brier[model] if count else None
                             for model in BASELINES})


def score(predictions):
    """Score every supplied slot without filtering, weighting or changing rows."""
    rows = list(predictions)
    for row in rows:
        if row['target'] not in (0, 1):
            raise ValueError('Brier requires a known binary target')
        for model in MODELS:
            probability = row['probabilities'][model]
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError('Invalid probability for ' + model)
        n = row['inputs'][0]
        if not math.isfinite(n) or n < 0:
            raise ValueError('The descriptive n strata require a known nonnegative count')
    result = _summary(rows)
    result['by_label'] = {str(label): _summary([row for row in rows if row['target'] == label])
                          for label in (0, 1)}
    result['by_n'] = {
        'n=0': _summary([row for row in rows if row['inputs'][0] == 0]),
        'n>0': _summary([row for row in rows if row['inputs'][0] > 0]),
    }
    return result


def gate(cases, technical_pass):
    """Apply direct arithmetic signs; descriptive strata never change the gate."""
    failed = []
    if technical_pass is not True:
        failed.append('technical_pass')
    if set(cases) != set(SCENES):
        failed.append('complete_two_scenes')
    noninferiority = {}
    strict = {'frozen': False, 'hold_only': False}
    for scene in SCENES:
        result = cases.get(scene)
        differences = result['fresh_minus'] if result is not None else {}
        checks = {}
        for model in BASELINES:
            difference = differences.get(model)
            known = (result is not None and result['count'] > 0 and difference is not None
                     and math.isfinite(difference))
            checks[model] = bool(known and difference <= 0)
            if not checks[model]:
                failed.append(scene + ':fresh_not_noninferior_to_' + model)
            if model in strict and known and difference < 0:
                strict[model] = True
        noninferiority[scene] = checks
    for model, passed in strict.items():
        if not passed:
            failed.append('no_strict_improvement_vs_' + model)
    return dict(passed=not failed, per_scene_noninferiority=noninferiority,
                strict_vs_frozen=strict['frozen'], strict_vs_hold_only=strict['hold_only'],
                failed_conditions=failed)
