"""Frozen positive-action joint means on the existing v43 real event rows.

Only E_positive[min Q(F(a)) - alpha log pi(a)] changes from one draw to
eight. The analytic gate, first-draw zero branch, native mixed backup and
matching recorded-action Q are fixed. No learner target is replaced here.
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


PART_FIELDS = ('native_target', 'reward_scaled', 'bootstrap_contribution',
               'value_contribution', 'sac_contribution', 'next_value',
               'sac_q_without_entropy', 'sac_entropy_bonus', 'target_subset',
               'recorded_next_target_q', 'baseline_missing_reason')


def _scalar(value):
    return float(value.reshape(-1)[0])


def _legal_successor(current, following):
    return (current.completed_at == following.started_at
            and np.array_equal(current.ns, following.s)
            and current.memory_ns['observed_at'] == following.memory_s['observed_at']
            and current.memory_ns['vector'] == following.memory_s['vector'])


@torch.no_grad()
def score_row_block(learner, supplied, following, *, draw_seeds, cached_m1=None,
                    head_seed=None):
    """Return trace-compatible M1/M8 and all eight constituent M1 rows.

    Each seed resets the native one-row actor RNG, draws its soft components,
    then draws the native target subset. The first subset and its slot-zero
    recorded-action Q are held fixed throughout the block. An optional
    ``head_seed`` selects the native-after-actor subset from that seed instead.
    In this experiment both target heads are always selected; the explicit
    assertion prevents accidental introduction of a second estimator change.

    ``cached_m1`` is the existing v44 row for draw_seeds[0]. With no cache, the
    first target is compared with learner._target. It never runs an update.
    All eight forward calls retain the native two-action critic batch shape.
    Mean reduction happens on the paired positive joint values before mixing
    with the analytic gate; Q and log-probability are never drawn separately.
    """
    seeds = tuple(int(seed) for seed in draw_seeds)
    if len(seeds) != 8 or len(set(seeds)) != 8:
        raise ValueError('The declared positive joint estimator requires eight distinct seeds')
    config = learner.config
    if (config.ensemble_size != 2 or config.target_subset_size != 2
            or config.target_subset_replacement or config.mixed_backup_lambda != .1):
        raise RuntimeError('v46 preserves the original two-of-two .1 V plus .9 SAC backup')
    current = row_from_record(supplied) if isinstance(supplied, dict) else supplied
    following = row_from_record(following) if isinstance(following, dict) else following
    if following is not None and not current.terminal and not _legal_successor(current, following):
        raise RuntimeError('Frozen scoring requires the real contiguous state and causal memory')
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
        reward = config.reward_scale*batch['rewards']
        discount = batch['discounts']*(1.-batch['terminals'])
        mixing = config.mixed_backup_lambda
        value_part = discount*mixing*value
        draws, native_subsets = [], []
        for seed in seeds:
            torch.manual_seed(seed)
            draws.append(learner.actor.soft_action_components(next_states))
            native_subsets.append(learner._sample_target_subset())
        chosen = native_subsets[0]
        if head_seed is not None and int(head_seed) != seeds[0]:
            torch.manual_seed(int(head_seed))
            learner.actor.soft_action_components(next_states)
            chosen = learner._sample_target_subset()
        if sorted(chosen.tolist()) != [0, 1]:
            raise RuntimeError('The fixed target subset must contain both original heads')
        missing = ('terminal_transition' if current.terminal else
                   'no_next_real_recorded_action' if following is None else None)
        qbar = None
        qbar_heads = None
        if missing is None:
            actual = torch.as_tensor(following.a, dtype=states.dtype,
                                     device=learner.device).reshape(1, 1)
            actual_q = learner.target_critic.forward_features_subset(
                repeated_features, torch.cat((actual, actual)), chosen)
            qbar_heads = actual_q[:, 0].tolist()
            qbar = _scalar(actual_q[:, 0].min())
        first = draws[0]
        positives_q, positives_joint, details = [], [], []
        zero_q = zero_joint = None
        for seed, parts, native_subset in zip(seeds, draws, native_subsets):
            for field in ('zero_probability', 'hold_probability', 'zero_action', 'zero_log_prob'):
                if not torch.equal(parts[field], first[field]):
                    raise RuntimeError('Only the positive draw may vary within a frozen state')
            actions = torch.cat((learner.action_transform(next_states, parts['zero_action']),
                                 learner.action_transform(next_states, parts['positive_action'])))
            selected = learner.target_critic.forward_features_subset(repeated_features, actions, chosen)
            zero_heads, positive_heads = selected.split(1, dim=1)
            zq, pq = zero_heads.min(0).values, positive_heads.min(0).values
            zj, pj = zq.clone(), pq.clone()
            if config.backup_entropy:
                zj -= learner.alpha.detach()*parts['zero_log_prob']
                pj -= learner.alpha.detach()*parts['positive_log_prob']
            if zero_q is None:
                zero_q, zero_joint = zq, zj
            elif not torch.equal(zero_q, zq) or not torch.equal(zero_joint, zj):
                raise RuntimeError('The native zero branch changed across positive draws')
            positives_q.append(pq)
            positives_joint.append(pj)
            details.append(dict(seed=seed, native_target_subset=native_subset.tolist(),
                fixed_target_subset=chosen.tolist(), positive_q=_scalar(pq),
                positive_log_prob=_scalar(parts['positive_log_prob']),
                positive_joint=_scalar(pj), zero_q=_scalar(zq), zero_joint=_scalar(zj),
                zero_probability=_scalar(parts['zero_probability']),
                hold_probability=_scalar(parts['hold_probability']),
                proposed_positive_action=_scalar(parts['positive_action']),
                executed_positive_action=_scalar(actions[1:]),
                executed_positive_hold_seconds=_scalar(learner.action_transform.hold_seconds(actions[1:])),
                positive_q_heads=positive_heads[:, 0].tolist()))

        def assemble(pq, pj, count, draw_seed):
            sac_q = first['zero_probability']*zero_q+first['hold_probability']*pq
            sac = first['zero_probability']*zero_joint+first['hold_probability']*pj
            entropy = sac-sac_q
            sac_part = discount*(1.-mixing)*sac
            bootstrap = discount*(mixing*value+(1.-mixing)*sac)
            target = reward+bootstrap
            return dict(sequence=current.sequence, line_id=current.line_id,
                started_at=current.started_at, completed_at=current.completed_at,
                available_at=current.available_at, dt=current.dt, terminal=bool(current.terminal),
                gamma=1., executed_action=float(current.a[0]),
                holding_seconds=float(learner.action_transform.hold_seconds(torch.as_tensor(current.a))[0]),
                native_target=_scalar(target), reward_scaled=_scalar(reward),
                bootstrap_contribution=_scalar(bootstrap), value_contribution=_scalar(value_part),
                sac_contribution=_scalar(sac_part), next_value=_scalar(value),
                sac_q_without_entropy=_scalar(sac_q), sac_entropy_bonus=_scalar(entropy),
                target_subset=chosen.tolist(), recorded_next_target_q=qbar,
                baseline_missing_reason=missing,
                causal_memory_s_observed_at=current.memory_s['observed_at'],
                causal_memory_ns_observed_at=current.memory_ns['observed_at'],
                native_target_exact=count == 1, estimator_M=count, draw_seed=draw_seed,
                positive_joint=_scalar(pj), positive_q=_scalar(pq),
                zero_joint=_scalar(zero_joint), zero_q=_scalar(zero_q),
                zero_probability=_scalar(first['zero_probability']),
                hold_probability=_scalar(first['hold_probability']))

        singles = [assemble(pq, pj, 1, seed)
                   for pq, pj, seed in zip(positives_q, positives_joint, seeds)]
        m1 = singles[0]
        if cached_m1 is None:
            torch.manual_seed(seeds[0])
            native = learner._target(dict(batch, observations=explicit_s,
                next_observations=explicit_ns), backup=config.online_backup)
            if np.float32(m1['native_target']).view(np.uint32) != np.float32(_scalar(native)).view(np.uint32):
                raise RuntimeError('M=1 differs from the native one-row target')
        else:
            changed = [field for field in PART_FIELDS if m1[field] != cached_m1[field]]
            if changed:
                raise RuntimeError('M=1 differs from resident v44: '+str(changed))
        m8 = assemble(torch.stack(positives_q).mean(0),
                      torch.stack(positives_joint).mean(0), 8, None)
    finally:
        _restore(rng)
        proof = unchanged(learner, before, rng)
        if not _equal(adam, optimizer_versions(learner)):
            raise RuntimeError('Frozen target scoring changed an Adam state tensor')
    return dict(m1=m1, m8=m8, singles=singles, draw_details=details, integrity=proof,
        first_m1_native_exact=True, new_training_updates=0, new_native_branches=0,
        target_config=dict(ensemble_size=2, subset_size=2, replacement=False,
            first_draw_seed=seeds[0], head_seed=seeds[0] if head_seed is None else int(head_seed),
            fixed_target_subset=chosen.tolist(), matching_q_heads=qbar_heads,
            matching_q_slot=0, native_two_action_batch=True,
            native_subsets=[value.tolist() for value in native_subsets],
            sampled_membership_varies=False, analytic_gate_unchanged=True,
            first_zero_branch_unchanged=True, F_unchanged=True,
            mixed_backup_lambda=mixing, joint_positive_mean=True))
