"""Frozen one-row target factors: positive-action draw versus native head draw.

The head seed denotes the subset drawn *after* the native actor draw for that
seed.  It is not a fresh subset sampled at the beginning of that RNG stream.
Each diagonal therefore preserves v44; off-diagonals cross those cached factors.
No training, replay insertion or synthetic transition is available here.
"""
from __future__ import annotations

import numpy as np
import torch

from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.learning import critic_states
from experiments.mcwm_closed_loop_multievent_credit_v38_20261024.frozen import (
    frozen_state, row_from_record, unchanged,
)
from experiments.mcwm_closed_loop_trace_target_v44_20261024.probe import optimizer_versions
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import (
    _batch, _equal, _restore, _rng,
)


SEEDS = (336202, 336203, 336204)
PART_FIELDS = ('reward_scaled', 'bootstrap_contribution', 'value_contribution',
               'sac_contribution', 'next_value', 'sac_q_without_entropy', 'sac_entropy_bonus')


def _scalar(tensor):
    return float(tensor.reshape(-1)[0])


def _legal_successor(current, following):
    return (current.completed_at == following.started_at
            and np.array_equal(current.ns, following.s)
            and current.memory_ns['observed_at'] == following.memory_s['observed_at']
            and current.memory_ns['vector'] == following.memory_s['vector'])


@torch.no_grad()
def factorize_row(learner, supplied, following, *, seeds=SEEDS, cached_diagonal=None):
    """Return 3x3 trace-compatible rows without changing model, Adam or RNG.

    ``cached_diagonal`` optionally maps str(seed) to the existing resident v44
    row, avoiding another native target call.  Equality checks include every
    saved target component, sampled head subset and matching executed-action Q.
    Cells are indexed as ``actor_seed/head_seed``.  ``factor_detail`` retains
    selected zero/positive Q heads, entropy and both weighted and raw terms.
    The matching executed-action Q uses native duplicated two-action inputs,
    selected-head forward and slot 0, exactly as v44.
    """
    seeds = tuple(seeds)
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError('The declared factor diagnostic requires three distinct seeds')
    current = row_from_record(supplied) if isinstance(supplied, dict) else supplied
    following = (row_from_record(following) if isinstance(following, dict) else following)
    if following is not None and not current.terminal and not _legal_successor(current, following):
        raise RuntimeError('Factor diagnostic requires the real contiguous state and causal memory')
    before, rng = frozen_state(learner), _rng()
    adam = optimizer_versions(learner)
    try:
        batch = _batch([current], learner.device)
        states, next_states = batch['observations'], batch['next_observations']
        memory = torch.as_tensor(current.memory_s['vector'], dtype=states.dtype,
                                 device=learner.device).reshape(1, -1)
        next_memory = torch.as_tensor(current.memory_ns['vector'], dtype=states.dtype,
                                      device=learner.device).reshape(1, -1)
        explicit_s = critic_states(learner, states, memory)
        explicit_ns = critic_states(learner, next_states, next_memory)
        features = learner.target_critic.encode_states(explicit_ns)
        repeated_features = torch.cat((features, features))
        value = learner.value(next_states)
        discount = batch['discounts']*(1.-batch['terminals'])
        mixing = learner.config.mixed_backup_lambda
        if mixing != .1:
            raise RuntimeError('The original mixed backup must remain .1 V plus .9 SAC')
        reward = learner.config.reward_scale*batch['rewards']
        value_part = discount*mixing*value
        draws, heads = {}, {}
        for seed in seeds:
            torch.manual_seed(seed)
            components = learner.actor.soft_action_components(next_states)
            # Capture the native draw order, including the actor rsample before
            # torch.randperm/randint selects the target-head subset.
            chosen = learner._sample_target_subset()
            draws[seed] = components
            heads[seed] = chosen
        missing = ('terminal_transition' if current.terminal else
                   'no_next_real_recorded_action' if following is None else None)
        recorded = {}
        if missing is None:
            actual = torch.as_tensor(following.a, dtype=states.dtype,
                                     device=learner.device).reshape(1, 1)
            actual_actions = torch.cat((actual, actual))
            for seed, chosen in heads.items():
                values = learner.target_critic.forward_features_subset(
                    repeated_features, actual_actions, chosen)
                recorded[seed] = values
        cells = {}
        for actor_seed, parts in draws.items():
            actions = torch.cat((learner.action_transform(next_states, parts['zero_action']),
                                 learner.action_transform(next_states, parts['positive_action'])))
            for head_seed, chosen in heads.items():
                selected = learner.target_critic.forward_features_subset(
                    repeated_features, actions, chosen)
                zero_heads, positive_heads = selected.split(1, dim=1)
                zero, positive = zero_heads.min(0).values, positive_heads.min(0).values
                sac_q = parts['zero_probability']*zero+parts['hold_probability']*positive
                if learner.config.backup_entropy:
                    zero -= learner.alpha.detach()*parts['zero_log_prob']
                    positive -= learner.alpha.detach()*parts['positive_log_prob']
                sac = parts['zero_probability']*zero+parts['hold_probability']*positive
                entropy = sac-sac_q
                sac_part = discount*(1.-mixing)*sac
                bootstrap = discount*(mixing*value+(1.-mixing)*sac)
                target = reward+bootstrap
                q_part = discount*(1.-mixing)*sac_q
                entropy_part = discount*(1.-mixing)*entropy
                qbar_values = recorded.get(head_seed)
                qbar = None if qbar_values is None else _scalar(qbar_values[:, 0].min())
                detail = dict(actor_seed=actor_seed, head_seed=head_seed,
                    actor_components={key: tensor.tolist() for key, tensor in parts.items()},
                    executed_target_actions=actions.reshape(-1).tolist(),
                    target_holding_seconds=learner.action_transform.hold_seconds(actions).reshape(-1).tolist(),
                    target_subset=chosen.tolist(), zero_q_heads=zero_heads[:, 0].tolist(),
                    positive_q_heads=positive_heads[:, 0].tolist(),
                    zero_min_head=int(chosen[zero_heads[:, 0].argmin()]),
                    positive_min_head=int(chosen[positive_heads[:, 0].argmin()]),
                    sac_q_contribution=_scalar(q_part), entropy_contribution=_scalar(entropy_part),
                    bootstrap_component_rounding=(_scalar(bootstrap)-_scalar(value_part)
                                                  -_scalar(q_part)-_scalar(entropy_part)),
                    matching_q_heads=None if qbar_values is None else qbar_values[:, 0].tolist(),
                    matching_q_min_head=None if qbar_values is None else int(chosen[qbar_values[:, 0].argmin()]),
                    matching_q_slot=0, native_two_action_batch=True,
                    subset_draw_after_native_actor=True)
                row = dict(sequence=current.sequence, line_id=current.line_id,
                    started_at=current.started_at, completed_at=current.completed_at,
                    available_at=current.available_at, dt=current.dt, terminal=bool(current.terminal),
                    gamma=1., executed_action=float(current.a[0]),
                    holding_seconds=float(learner.action_transform.hold_seconds(torch.as_tensor(current.a))[0]),
                    native_target=_scalar(target), reward_scaled=_scalar(reward),
                    bootstrap_contribution=_scalar(bootstrap), value_contribution=_scalar(value_part),
                    sac_contribution=_scalar(sac_part), next_value=_scalar(value),
                    sac_q_without_entropy=_scalar(sac_q), sac_entropy_bonus=_scalar(entropy),
                    target_subset=chosen.tolist(), recorded_next_target_q=qbar,
                    baseline_missing_reason=missing, factor_detail=detail,
                    causal_memory_s_observed_at=current.memory_s['observed_at'],
                    causal_memory_ns_observed_at=current.memory_ns['observed_at'],
                    native_target_exact=actor_seed == head_seed)
                if actor_seed == head_seed:
                    if cached_diagonal is None:
                        torch.manual_seed(actor_seed)
                        actual = learner._target(dict(batch, observations=explicit_s,
                            next_observations=explicit_ns), backup=learner.config.online_backup)
                        if not torch.equal(target, actual):
                            raise RuntimeError('Factor diagonal differs from the native one-row target')
                    else:
                        cached = cached_diagonal[str(actor_seed)]
                        fields = ('native_target', *PART_FIELDS, 'target_subset',
                                  'recorded_next_target_q', 'baseline_missing_reason')
                        changed = [field for field in fields if row[field] != cached[field]]
                        if changed:
                            raise RuntimeError('Factor diagonal differs from resident v44: '+str(changed))
                cells[f'{actor_seed}/{head_seed}'] = row
    finally:
        _restore(rng)
        proof = unchanged(learner, before, rng)
        if not _equal(adam, optimizer_versions(learner)):
            raise RuntimeError('Frozen factor scoring changed an Adam state tensor')
    memberships = {seed: sorted(set(chosen.tolist())) for seed, chosen in heads.items()}
    return dict(cells=cells, diagonal_exact=True, integrity=proof,
                actor_seeds=list(seeds), head_seeds=list(seeds), new_training_updates=0,
                new_native_branches=0, matching_q_independent_of_actor_seed=True,
                target_config=dict(ensemble_size=learner.config.ensemble_size,
                    subset_size=learner.config.target_subset_size,
                    replacement=learner.config.target_subset_replacement,
                    sampled_subsets={str(seed): chosen.tolist() for seed, chosen in heads.items()},
                    sampled_memberships={str(seed): value for seed, value in memberships.items()},
                    distinct_memberships=len({tuple(value) for value in memberships.values()}),
                    sampled_membership_varies=len({tuple(value) for value in memberships.values()}) > 1,
                    all_sampled_memberships_cover_full_ensemble=all(
                        value == list(range(learner.config.ensemble_size)) for value in memberships.values())))
