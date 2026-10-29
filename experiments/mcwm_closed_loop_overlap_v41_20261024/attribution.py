"""Three-factor attribution of an executed-action log trace coefficient.

Every one of the eight corners uses the same logged executed action and F
geometry.  Only the Hurdle gate probabilities, Gaussian mean and Gaussian std
are exchanged.  Thus this is a distribution-parameter diagnostic conditional
on an existing state/action, not an action-selection or path-reweighting claim.
"""
from __future__ import annotations

import math

from experiments.mcwm_closed_loop_trace_design_v40_20261024.behavior import (
    execution_measure, ratio_from_measures,
)


FACTORS = ('gate', 'mean', 'std')
FACTOR_BITS = dict(gate=1, mean=2, std=4)


def _corner_parameters(behavior, target, mask):
    parameters = dict(behavior)
    if mask & FACTOR_BITS['gate']:
        # Exchange the probabilities already read from the actor.  Recomputing
        # sigmoid from a logged gate would introduce a different rounding path.
        for name in ('positive_probability', 'zero_probability'):
            parameters[name] = target[name]
        if 'gate' in target:
            parameters['gate'] = target['gate']
    if mask & FACTOR_BITS['mean']:
        parameters['mean'] = target['mean']
    if mask & FACTOR_BITS['std']:
        parameters['std'] = target['std']
        if 'log_std' in target:
            parameters['log_std'] = target['log_std']
    return parameters


def _shapley(corners, field):
    values = {row['mask']: row[field] for row in corners}
    contributions = {}
    for factor in FACTORS:
        bit = FACTOR_BITS[factor]
        marginals = []
        for mask in range(8):
            if mask & bit:
                continue
            size = mask.bit_count()
            weight = 1. / 6. if size == 1 else 1. / 3.
            marginals.append(weight * (values[mask | bit] - values[mask]))
        contributions[factor] = math.fsum(marginals)
    closure = math.fsum(contributions.values()) - (values[7] - values[0])
    return contributions, closure


def _continuous_fixed_path(behavior_measure, behavior, target, actual_log_ratio):
    """Analytic log-ratio decomposition with declared gate -> mean -> std order.

    This additional decomposition is *unclipped*.  The Shapley result, rather
    than this fixed path, attributes the nonlinear min(0, log ratio).
    """
    if behavior_measure['measure_kind'] != 'continuous':
        return None
    z = behavior_measure['continuous_preimage']['gaussian_value']
    mean_b, mean_t = behavior['mean'], target['mean']
    std_b, std_t = behavior['std'], target['std']
    components = dict(
        gate=math.log(target['positive_probability']) - math.log(behavior['positive_probability']),
        mean=.5 * (((z - mean_b) / std_b) ** 2 - ((z - mean_t) / std_b) ** 2),
        std=math.log(std_b) - math.log(std_t)
            + .5 * (((z - mean_t) / std_b) ** 2 - ((z - mean_t) / std_t) ** 2),
    )
    return dict(order=['gate', 'mean', 'std'], quantity='unclipped_log_ratio',
                gaussian_value=z, components=components,
                closure_error=math.fsum(components.values()) - actual_log_ratio)


def decompose_log_c(transform, state, logged_action, behavior_params, target_params, *,
                    geometry=None, denominator_params=None):
    """Score eight hybrid distributions and return order-independent Shapley terms.

    Bit assignments are gate=1, mean=2, std=4.  Mask 0 is the behavior
    distribution scored against the denominator; mask 7 is the supplied target.  Gate
    means the *pair* of supplied positive/zero probabilities, never a sigmoid
    recomputed from the optional diagnostic gate logit.

    The denominator normally equals behavior_params, giving an exact zero
    baseline.  With denominator_params, behavior_params can instead be the
    initializer at the target's visible input and denominator_params the
    initializer at the old actor's actual atomic-batch input.  Then mask 0
    isolates the input effect, and the three Shapley terms attribute parameter
    drift beyond that baseline.  All corners still use the supplied same F.

    Unsupported or numerically unresolved measure parts keep their status and
    null attribution.  They are never interpreted as zero log coefficient.
    """
    denominator = behavior_params if denominator_params is None else denominator_params
    behavior = execution_measure(transform, state, logged_action, denominator, geometry=geometry)
    geometry = behavior['geometry']
    corners = []
    for mask in range(8):
        parameters = _corner_parameters(behavior_params, target_params, mask)
        target = behavior if mask == 0 and denominator == behavior_params else execution_measure(
            transform, state, logged_action, parameters, geometry=geometry)
        ratio = ratio_from_measures(behavior, target)
        corners.append(dict(mask=mask,
            target_factors=[factor for factor in FACTORS if mask & FACTOR_BITS[factor]],
            status=ratio['status'], reason=ratio['reason'], log_ratio=ratio['log_ratio'],
            log_c=ratio['log_c'], c=ratio['c'], target_log_measure=ratio['target_log_measure'],
            target_atom_mass=ratio['target_atom_mass'], target_measure_kind=ratio['target_measure_kind']))
    actual = corners[7]
    result = dict(status=actual['status'], actual_log_ratio=actual['log_ratio'],
        actual_log_c=actual['log_c'], actual_c=actual['c'], corners=corners,
        baseline_log_ratio=corners[0]['log_ratio'], baseline_log_c=corners[0]['log_c'],
        behavior_measure_kind=behavior['measure_kind'], behavior_log_measure=behavior['log_measure'],
        behavior_atom_mass=behavior['atom_mass'], geometry=geometry,
        factor_bits=FACTOR_BITS, factor_order_independent=True,
        old_log_authorized_for_training=False)
    unresolved = [row['mask'] for row in corners if row['status'] != 'supported'
                  or row['log_c'] is None or row['log_ratio'] is None
                  or not math.isfinite(row['log_c']) or not math.isfinite(row['log_ratio'])]
    if unresolved:
        return dict(**result, attribution_status='unresolved_measure', unresolved_corners=unresolved,
                    shapley_log_c=None, shapley_log_ratio=None,
                    closure_error=None, continuous_fixed_path=None)
    clipped, clipped_error = _shapley(corners, 'log_c')
    raw, raw_error = _shapley(corners, 'log_ratio')
    return dict(**result, attribution_status='complete', unresolved_corners=[],
        shapley_log_c=clipped, shapley_log_ratio=raw,
        closure_error=dict(log_c=clipped_error, log_ratio=raw_error),
        continuous_fixed_path=_continuous_fixed_path(
            behavior, behavior_params, target_params,
            actual['log_ratio'] - corners[0]['log_ratio']))
