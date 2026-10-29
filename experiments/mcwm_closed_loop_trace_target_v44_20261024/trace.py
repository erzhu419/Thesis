"""Read-only expansion of the original mixed backup on one fixed real chain.

Y_i is the *native float32* one-step target, not a reconstructed approximation.
The candidate is G_i = Y_i + gamma_i*.9*lambda*c_(i+1)*(G_(i+1)-Qbar_(i+1)).
The .9 is the original SAC-Q mixing weight, not the temporal discount.  This is
a diagnostic candidate for that mixed backup, not a Retrace convergence claim.
The final real transition retains its native bootstrap; a collector cutoff is
not a terminal.  No learner, optimizer, environment, or endpoint selection is
available in this module.
"""
from __future__ import annotations

import math

import numpy as np


def _f32(value):
    return np.float32(value)


def _bits(values):
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def decompose_trace(rows, c, lambda_trace, *, mixed_path_weight=.9):
    """Expand a supplied complete chain without choosing a shorter endpoint.

    Each row supplies native_target, reward_scaled, bootstrap_contribution,
    terminal, dt and recorded_next_target_q.  The last Qbar is unused.  Optional
    gamma defaults to the project's 1; value_contribution and sac_contribution
    retain the native bootstrap split when both are supplied.  c has n-1 actual
    F-pushforward ratios, with the root intervention excluded.  Missing support
    or a matching target-subset Qbar stops the link and is reported explicitly.

    Float32 recurrence and double-precision additive accounting are separate.
    Native reward-plus-bootstrap rounding and recurrence rounding are explicit,
    so a large cancellation is not silently called a reward contribution.
    The final coefficient is an algebraic unrolled coefficient, not a claim
    about an SGD gradient or a finite perturbation through float32 rounding.
    """
    if not rows:
        raise ValueError('A trace requires real closed transitions')
    count = len(rows)
    if len(c) != count-1:
        raise ValueError('Execution ratios must match the real successor links')
    lam, mixing = float(lambda_trace), float(mixed_path_weight)
    if not math.isfinite(lam) or not 0. <= lam <= 1.:
        raise ValueError('lambda_trace must be in [0, 1]')
    if not math.isfinite(mixing) or not 0. <= mixing <= 1.:
        raise ValueError('Mixed Q path weight must be in [0, 1]')
    native = np.asarray([row['native_target'] for row in rows], dtype=np.float32)
    reward = np.asarray([row['reward_scaled'] for row in rows], dtype=np.float32)
    bootstrap = np.asarray([row['bootstrap_contribution'] for row in rows], dtype=np.float32)
    if not all(np.isfinite(value).all() for value in (native, reward, bootstrap)):
        raise ValueError('Native reward, bootstrap and targets must be finite')
    assembly_exact = np.array_equal(_bits(reward+bootstrap), _bits(native))
    if not assembly_exact:
        raise ValueError('Supplied reward and bootstrap do not assemble to the native target')
    for row, boot in zip(rows, bootstrap):
        if float(row['dt']) < 0. or not math.isfinite(float(row['dt'])):
            raise ValueError('A real event must have a nonnegative duration')
        if row['terminal'] and boot != 0.:
            raise ValueError('A true terminal cannot retain native bootstrap')

    links = []
    for index in range(count-1):
        row, ratio, reason = rows[index], c[index], None
        gamma = float(row.get('gamma', 1.))
        if not math.isfinite(gamma) or not 0. <= gamma <= 1.:
            raise ValueError('The actual temporal discount must be in [0, 1]')
        if row['terminal']:
            reason = 'true_terminal_transition'
        elif lam == 0.:
            reason = 'lambda_zero'
        elif gamma == 0. or mixing == 0.:
            reason = 'zero_native_continuation'
        elif ratio is None:
            reason = 'executed_action_support_unavailable'
        elif not math.isfinite(float(ratio)) or not 0. <= float(ratio) <= 1.:
            raise ValueError('Execution c must be the clipped actual ratio in [0, 1]')
        elif float(ratio) == 0.:
            reason = 'zero_execution_support'
        elif (row.get('recorded_next_target_q') is None or
              not math.isfinite(float(row['recorded_next_target_q']))):
            reason = 'matching_subset_target_q_unavailable'
        alpha = (_f32(_f32(gamma)*_f32(mixing)*_f32(lam)*_f32(ratio))
                 if reason is None else _f32(0.))
        links.append(dict(from_event=index, to_event=index+1, active=reason is None,
                          reason=reason, coefficient=float(alpha), gamma=gamma,
                          mixed_path_weight=mixing, c=ratio))

    # Copying is essential: even an algebraically equivalent reconstruction can
    # alter the native target's last bit, or the sign of zero at lambda=0.
    targets = native.copy()
    recurrence_rounding = [0.]*count
    for index in range(count-2, -1, -1):
        if links[index]['active']:
            alpha = _f32(links[index]['coefficient'])
            baseline = _f32(rows[index]['recorded_next_target_q'])
            delta = _f32(targets[index+1]-baseline)
            correction = _f32(alpha*delta)
            targets[index] = _f32(native[index]+correction)
            exact_expression = float(native[index])+float(alpha)*(float(targets[index+1])-float(baseline))
            recurrence_rounding[index] = float(targets[index])-exact_expression

    reach = [1.]
    for link in links:
        reach.append(reach[-1]*link['coefficient'] if link['active'] else 0.)
    per_event = []
    split_available = all('value_contribution' in row and 'sac_contribution' in row for row in rows)
    for index, (row, weight) in enumerate(zip(rows, reach)):
        baseline = (_f32(rows[index-1]['recorded_next_target_q'])
                    if index and weight != 0. else None)
        native_rounding = float(native[index])-(float(reward[index])+float(bootstrap[index]))
        entry = dict(event=index, dt=float(row['dt']), terminal=bool(row['terminal']),
            reach_weight=weight, native_target=float(native[index]), target=float(targets[index]),
            reward_scaled=float(reward[index]), bootstrap_contribution=float(bootstrap[index]),
            reward_root_contribution=weight*float(reward[index]),
            bootstrap_root_contribution=weight*float(bootstrap[index]),
            previous_link_matching_target_q=None if baseline is None else float(baseline),
            minus_matching_q_root_contribution=0. if baseline is None else -weight*float(baseline),
            native_rounding_root_contribution=weight*native_rounding,
            recurrence_rounding_root_contribution=weight*recurrence_rounding[index])
        if split_available:
            value, sac = float(row['value_contribution']), float(row['sac_contribution'])
            entry.update(value_root_contribution=weight*value, sac_root_contribution=weight*sac,
                bootstrap_split_rounding_root_contribution=weight*(float(bootstrap[index])-value-sac))
        per_event.append(entry)

    def total(field, subset=None):
        return math.fsum(row[field] for row in (per_event if subset is None else subset))

    components = dict(reward=total('reward_root_contribution'),
        bootstrap=total('bootstrap_root_contribution'),
        minus_matching_target_q=total('minus_matching_q_root_contribution'),
        native_rounding=total('native_rounding_root_contribution'),
        recurrence_rounding=total('recurrence_rounding_root_contribution'))
    reconstructed = math.fsum(components.values())
    root = dict(native_target=float(native[0]), trace_target=float(targets[0]),
        correction=float(targets[0])-float(native[0]), components=components,
        reconstructed_target=reconstructed,
        reconstruction_residual=float(targets[0])-reconstructed,
        root_reward=float(reward[0]), root_bootstrap=float(bootstrap[0]),
        future_reward=total('reward_root_contribution', per_event[1:]),
        future_bootstrap=total('bootstrap_root_contribution', per_event[1:]))
    if split_available:
        root['bootstrap_split'] = dict(value=total('value_root_contribution'),
            sac=total('sac_root_contribution'),
            assembly_rounding=total('bootstrap_split_rounding_root_contribution'))
    tail = per_event[-1]
    first_stop = next((dict(event=link['to_event'], reason=link['reason'])
                       for link in links if not link['active']), None)
    return dict(lambda_trace=lam, mixed_path_weight=mixing, event_count=count,
        targets=targets.tolist(), native_targets=native.tolist(), reach_weights=reach,
        links=links, per_event=per_event, root=root,
        terminal_dependency=dict(event=count-1, true_terminal=bool(rows[-1]['terminal']),
            coefficient=reach[-1], native_target=float(native[-1]),
            raw_reward=float(reward[-1]), raw_bootstrap=float(bootstrap[-1]),
            weighted_native_target=reach[-1]*float(native[-1]),
            weighted_reward=tail['reward_root_contribution'],
            weighted_bootstrap=tail['bootstrap_root_contribution'],
            weighted_minus_incoming_matching_q=tail['minus_matching_q_root_contribution'],
            weighted_native_minus_incoming_matching_q=math.fsum((
                reach[-1]*float(native[-1]), tail['minus_matching_q_root_contribution'])),
            coefficient_scope='algebraic_unrolled_coefficient_not_SGD_gradient'),
        native_assembly_exact=assembly_exact,
        lambda_zero_native_exact=(np.array_equal(_bits(targets), _bits(native)) if lam == 0. else None),
        full_chain_supported=all(link['active'] for link in links), first_stop=first_stop,
        reached_event_count=sum(value != 0. for value in reach),
        zero_duration_events=sum(float(row['dt']) == 0. for row in rows),
        terminal_rule='stop_at_true_terminal_otherwise_keep_last_real_native_bootstrap',
        synthetic300_state=False, new_training_updates=0)
