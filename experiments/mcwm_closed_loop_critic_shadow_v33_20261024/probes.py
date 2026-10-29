"""Common-action, read-only probes on saved contexts that have been admitted."""
from copy import deepcopy
import random

import numpy as np
import torch

from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.learning import critic_states
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _batch, _rng, _restore, _equal


def window(clock):
    return 'before9000' if clock < 9000 else '9000-12000' if clock < 12000 else \
        '12000-18000' if clock < 18000 else 'tail'


@torch.no_grad()
def probe(learner, rows, *, seed):
    """Actual observations/actions are common; neither probe outcomes nor RNG train."""
    saved = _rng()
    private = {name: deepcopy(getattr(learner, name).bit_generator.state)
               for name in ('rng', 'diagnostic_rng')}
    try:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        batch = _batch(rows, learner.device)
        states, next_states = batch['observations'], batch['next_observations']
        memory = torch.as_tensor(np.asarray([r.memory_s['vector'] for r in rows]),
                                 dtype=states.dtype, device=learner.device)
        next_memory = torch.as_tensor(np.asarray([r.memory_ns['vector'] for r in rows]),
                                      dtype=states.dtype, device=learner.device)
        explicit_s = critic_states(learner, states, memory)
        explicit_ns = critic_states(learner, next_states, next_memory)
        n = len(rows)
        proposal = (2.*torch.tensor([0., 7.5, 15.], device=learner.device)
                    /learner.action_transform.max_hold-1.).repeat(n, 1)
        repeated_states = states[:, None, :].expand(-1, 3, -1).reshape(-1, states.shape[1])
        executed = learner.action_transform(repeated_states, proposal.reshape(-1, 1)).reshape(n, 3)
        actions = torch.cat((executed, batch['actions']), dim=1)
        repeated = explicit_s[:, None, :].expand(-1, 4, -1).reshape(-1, explicit_s.shape[1])
        q = learner.critic(repeated, actions.reshape(-1, 1)).reshape(-1, n, 4)
        # Same learned weights, zero context: separates direct input effects
        # from base-critic and actor/value changes induced by prior learning.
        zero_s = critic_states(learner, states, torch.zeros_like(memory))
        zero_repeated = zero_s[:, None, :].expand(-1, 4, -1).reshape(-1, zero_s.shape[1])
        q_no_context = learner.critic(zero_repeated, actions.reshape(-1, 1)).reshape(-1, n, 4)
        q_min, q_mean = q.min(0).values, q.mean(0)
        q_masked = q_no_context.min(0).values
        current_proposal = learner.actor(states, deterministic=True)[0]
        current_action = learner.action_transform(states, current_proposal)
        parts = learner.actor.soft_action_components(next_states)
        next_actions = torch.cat((learner.action_transform(next_states, parts['zero_action']),
            learner.action_transform(next_states, parts['positive_action'])))
        chosen = learner._sample_target_subset()
        features = learner.target_critic.encode_states(explicit_ns)
        values = learner.target_critic.forward_features_subset(
            torch.cat((features, features)), next_actions, chosen)
        low, high = values.split(n, dim=1)
        low, high = low.min(0).values, high.min(0).values
        sac_q = parts['zero_probability']*low+parts['hold_probability']*high
        if learner.config.backup_entropy:
            low -= learner.alpha.detach()*parts['zero_log_prob']
            high -= learner.alpha.detach()*parts['positive_log_prob']
        sac = parts['zero_probability']*low+parts['hold_probability']*high
        value = learner.value(next_states)
        mixing = learner.config.mixed_backup_lambda
        discount = batch['discounts']*(1.-batch['terminals'])
        reward = learner.config.reward_scale*batch['rewards']
        value_part = discount*mixing*value
        sac_part = discount*(1.-mixing)*sac
        bootstrap = discount*(mixing*value+(1.-mixing)*sac)
        target = reward+bootstrap
        # Restore the same draw before exercising the learner's actual target.
        torch.manual_seed(seed)
        native_batch = dict(batch, observations=explicit_s, next_observations=explicit_ns)
        native_target = learner._target(native_batch, backup=learner.config.online_backup)
        if not torch.equal(target, native_target):
            raise RuntimeError('Diagnostic Bellman decomposition differs from the learner target')
        records = []
        hold = learner.action_transform.hold_seconds(actions)
        for i, row in enumerate(rows):
            records.append(dict(sequence=row.sequence, line_id=row.line_id, issue=row.started_at,
                window=window(row.started_at), context_observed_at=row.memory_s['observed_at'],
                holding_seconds=hold[i].tolist(), executed_actions=actions[i].tolist(),
                unique_fixed_actions=len(set(actions[i, :3].tolist())),
                fixed_holding_span=float(hold[i, :3].max()-hold[i, :3].min()),
                q_min=q_min[i].tolist(), q_mean=q_mean[i].tolist(),
                advantage_min=(q_min[i]-q_min[i, 0]).tolist(),
                advantage_mean=(q_mean[i]-q_mean[i, 0]).tolist(),
                zero_context_advantage_min=(q_masked[i]-q_masked[i, 0]).tolist(),
                proposal_seconds=float(learner.action_transform.hold_seconds(current_proposal[i])),
                actor_holding_seconds=float(learner.action_transform.hold_seconds(current_action[i])),
                best_fixed_action=int(q_min[i, :3].argmax()),
                reward_scaled=float(reward[i]), next_value=float(value[i]),
                value_contribution=float(value_part[i]), sac_contribution=float(sac_part[i]),
                sac_q_without_entropy=float(sac_q[i]), sac_entropy_bonus=float(sac[i]-sac_q[i]),
                bootstrap_contribution=float(bootstrap[i]), target=float(target[i]),
                observed_td_mean=float(q_mean[i, 3]-target[i]),
                target_subset=chosen.tolist()))
    finally:
        for name, state in private.items():
            getattr(learner, name).bit_generator.state = state
        _restore(saved)
    if not _equal(saved, _rng()):
        raise RuntimeError('Diagnostic probes changed training RNG')
    return records
