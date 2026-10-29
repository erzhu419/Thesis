"""A fixed-capacity bound on future physical-time Bellman residuals.

The native one-step mixed backup is unchanged.  The future residual scale is
eta=9/269, fixed from the 270-row response capacity, not the observed chain
length or its future weights.  It is applied once to each suffix's complete
physical trace; it is never applied recursively at every event link.
"""
from __future__ import annotations

import math

import numpy as np

from experiments.mcwm_closed_loop_physical_credit_v50_20261025.credit import (
    physical_trace, probe_bootstrap_shift, probe_reward_band, probe_uniform_q_shift,
)


CAPACITY = 270
RESIDUAL_MASS_BOUND = 9.
ETA = RESIDUAL_MASS_BOUND / (CAPACITY - 1)


def bounded_trace(rows, c, lambda_trace):
    """Return native-compatible decomposition of one bounded physical trace.

    G_i = Y_i + lambda*eta*(Gphysical_i-Y_i), evaluated in float32.
    The physical trace retains all actual endpoints, execution ratios and
    matching subset-Q references.  Returned reach weights describe the root's
    additive expansion.  The links retain the *underlying physical* recurrence;
    recursively multiplying the outer scale along these links would be wrong.
    """
    if len(rows) > CAPACITY:
        raise ValueError('The response capacity is 270 real transitions')
    lam = float(lambda_trace)
    if not math.isfinite(lam) or not 0. <= lam <= 1.:
        raise ValueError('lambda_trace must be in [0, 1]')
    physical = physical_trace(rows, c, 1.)
    native = np.asarray(physical['native_targets'], dtype=np.float32)
    original = np.asarray(physical['targets'], dtype=np.float32)
    scale = np.float32(np.float32(lam) * np.float32(ETA))
    targets = native.copy()
    blend_parts = []
    for index, (one_step, long_target) in enumerate(zip(native, original)):
        if lam == 0.:
            # Preserve every native bit, including the sign of zero.
            parts = dict(subtraction=0., multiplication=0., addition=0., total=0.)
        else:
            delta = np.float32(long_target - one_step)
            correction = np.float32(scale * delta)
            targets[index] = np.float32(one_step + correction)
            subtraction = float(scale) * (float(delta) - (float(long_target) - float(one_step)))
            multiplication = float(correction) - float(scale) * float(delta)
            addition = float(targets[index]) - (float(one_step) + float(correction))
            parts = dict(subtraction=subtraction, multiplication=multiplication,
                         addition=addition, total=math.fsum((subtraction, multiplication, addition)))
        blend_parts.append(parts)

    reach = [1.] + [float(scale) * value for value in physical['reach_weights'][1:]]
    per_event = []
    for index, (source, weight) in enumerate(zip(physical['per_event'], reach)):
        entry = dict(source)
        entry.update(reach_weight=weight, target=float(targets[index]),
                     physical_target=float(original[index]))
        for field, value_field in (
            ('reward_root_contribution', 'reward_scaled'),
            ('bootstrap_root_contribution', 'bootstrap_contribution'),
        ):
            entry[field] = weight * source[value_field]
        baseline = source['previous_link_matching_target_q']
        entry['minus_matching_q_root_contribution'] = (
            0. if baseline is None else -weight * baseline)
        native_rounding = (source['native_target'] - source['reward_scaled'] -
                           source['bootstrap_contribution'])
        entry['native_rounding_root_contribution'] = weight * native_rounding
        scaled_rounding = float(scale) * source['recurrence_rounding_root_contribution']
        root_blend = blend_parts[0]['total'] if index == 0 else 0.
        entry.update(physical_recurrence_rounding_root_contribution=scaled_rounding,
                     root_blend_rounding_root_contribution=root_blend,
                     recurrence_rounding_root_contribution=scaled_rounding + root_blend)
        if 'value_contribution' in rows[index] and 'sac_contribution' in rows[index]:
            value, sac = float(rows[index]['value_contribution']), float(rows[index]['sac_contribution'])
            entry.update(value_root_contribution=weight * value,
                         sac_root_contribution=weight * sac,
                         bootstrap_split_rounding_root_contribution=(
                             weight * (source['bootstrap_contribution'] - value - sac)))
        per_event.append(entry)

    def total(field, entries=None):
        return math.fsum(entry[field] for entry in (per_event if entries is None else entries))

    components = dict(reward=total('reward_root_contribution'),
        bootstrap=total('bootstrap_root_contribution'),
        minus_matching_target_q=total('minus_matching_q_root_contribution'),
        native_rounding=total('native_rounding_root_contribution'),
        recurrence_rounding=total('recurrence_rounding_root_contribution'))
    reconstructed = math.fsum(components.values())
    root = dict(native_target=float(native[0]), trace_target=float(targets[0]),
        correction=float(targets[0]) - float(native[0]), components=components,
        reconstructed_target=reconstructed,
        reconstruction_residual=float(targets[0]) - reconstructed,
        root_reward=per_event[0]['reward_scaled'], root_bootstrap=per_event[0]['bootstrap_contribution'],
        future_reward=total('reward_root_contribution', per_event[1:]),
        future_bootstrap=total('bootstrap_root_contribution', per_event[1:]))
    if 'bootstrap_split' in physical['root']:
        root['bootstrap_split'] = dict(value=total('value_root_contribution'),
            sac=total('sac_root_contribution'),
            assembly_rounding=total('bootstrap_split_rounding_root_contribution'))

    tail = per_event[-1]
    terminal = dict(physical['terminal_dependency'])
    terminal.update(coefficient=reach[-1], weighted_native_target=reach[-1] * float(native[-1]),
        weighted_reward=tail['reward_root_contribution'],
        weighted_bootstrap=tail['bootstrap_root_contribution'],
        weighted_minus_incoming_matching_q=tail['minus_matching_q_root_contribution'],
        weighted_native_minus_incoming_matching_q=math.fsum((
            reach[-1] * float(native[-1]), tail['minus_matching_q_root_contribution'])))
    links = [dict(link, coefficient_scope='underlying_physical_recurrence_before_outer_scale',
                  outer_residual_scale=float(scale), outer_scale_applied_per_link=False)
             for link in physical['links']]
    if lam == 0.:
        for link in links:
            link.update(active=False, reason='lambda_zero', coefficient=0.)
    first_stop = next((dict(event=link['to_event'], reason=link['reason'])
                       for link in links if not link['active']), None)
    future_mass = math.fsum(reach[1:])
    ideal_mass = lam * ETA * math.fsum(physical['reach_weights'][1:])
    result = dict(physical)
    result.update(lambda_trace=lam, targets=targets.tolist(), native_targets=native.tolist(),
        reach_weights=reach, links=links, per_event=per_event, root=root,
        terminal_dependency=terminal, first_stop=first_stop,
        full_chain_supported=all(link['active'] for link in links),
        reached_event_count=sum(weight != 0. for weight in reach),
        lambda_zero_native_exact=(np.array_equal(targets.view(np.uint32), native.view(np.uint32))
                                  if lam == 0. else None),
        trace_kernel='native_Y_plus_once_scaled_physical_future_residuals',
        fixed_capacity=CAPACITY, eta=ETA, eta_float32=float(np.float32(ETA)),
        future_residual_scale=float(scale), physical_trace_gain=physical['gain'],
        suffix_blend_rounding=blend_parts,
        rounding_breakdown=dict(
            physical_recurrence=physical['root']['components']['recurrence_rounding'],
            scaled_physical_recurrence=total('physical_recurrence_rounding_root_contribution'),
            root_blend=blend_parts[0],
            combined_recurrence=components['recurrence_rounding']),
        gain=dict(sum_reach=1. + future_mass, future_sum_reach=future_mass,
            uniform_q_offset_gain=.9 - .1 * future_mass,
            ideal_future_sum_reach=ideal_mass,
            ideal_uniform_q_offset_gain=.9 - .1 * ideal_mass,
            ideal_future_mass_bound=lam * RESIDUAL_MASS_BOUND,
            float32_mass_rounding=future_mass - ideal_mass,
            all_nonterminal_reference=not any(row['terminal'] for row in rows),
            scope='Fixed V and weights; synchronous scalar Q offset only; not a contraction or training guarantee'))
    return result


__all__ = ['bounded_trace', 'probe_bootstrap_shift', 'probe_reward_band',
           'probe_uniform_q_shift', 'CAPACITY', 'RESIDUAL_MASS_BOUND', 'ETA']
