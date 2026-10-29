"""Frozen final-model TD pressure on the already selected factual outcomes.

This scores recorded outcomes at the final parameters.  It does not reconstruct
the gradients at their historical training times or execute an optimizer step.
"""
import numpy as np
import torch

from experiments.mcwm_closed_loop_capacity_decomposition_v82_20261027.data import context_vector
from experiments.mcwm_closed_loop_capacity_decomposition_v82_20261027.decompose import explicit_components
from experiments.mcwm_closed_loop_context_wiring_v80_20261027.learning import pack_states


def td_theta_terms(q, target, feature, *, delta=1., quota=.25, weights=None):
    """Per-record current-theta partial derivative of the actual Huber loss.

    The target is detached, as in the original critic update. ``gradient`` has
    shape[heads,records] and includes quota/heads, but not a record average.
    Summing its columns with normalized exposure weights gives the weighted
    mean contribution. Group contributions must retain these global weights.
    Error terms are additive; clipped Huber gradients cannot be decomposed by
    applying clipping independently to reward/continuation components.
    """
    error = q-target.detach()[None]
    derivative = error.clamp(-delta, delta)
    gradient = (quota/q.shape[0])*derivative*feature[None]
    normalized = torch.ones_like(target) if weights is None else torch.as_tensor(
        weights, dtype=target.dtype, device=target.device)
    if normalized.shape != target.shape or not bool(torch.isfinite(normalized).all()) or bool((normalized < 0).any()) or not bool(normalized.sum() > 0):
        raise ValueError('Outcome exposure weights must be aligned, finite and nonnegative')
    normalized = normalized/normalized.sum()
    return dict(error=error, huber_derivative=derivative,
        huber_clipped=error.abs() > delta, gradient=gradient,
        uniform_gradient=gradient.mean(1),
        exposure_weighted_gradient=(gradient*normalized[None]).sum(1),
        normalized_exposure_weights=normalized,
        raw_mse_gradient=(quota/q.shape[0])*2.*error*feature[None])


@torch.no_grad()
def pressure(learner, records, noise, weights=None):
    """Score real endpoints with shared explicit M8 positive-action noise.

    The frozen production target subset is both heads without replacement, so
    its minimum is deterministic. Only the positive-action expectation uses the
    caller's predeclared eight draws; no main or private RNG is consumed.
    """
    config = learner.config
    if not (config.critic_loss == 'huber' and config.huber_delta == 1.
            and config.bootstrap_probability == 1.
            and config.reward_scale == 1e-4 and config.mixed_backup_lambda == .1
            and config.target_subset_size == learner.target_critic.ensemble_size == 2
            and config.target_subset_replacement is False and config.backup_entropy):
        raise ValueError('Frozen pressure requires the actual v81 Huber/mixed-backup contract')
    if not records:
        raise ValueError('Frozen TD pressure requires actual recorded outcomes')
    device = learner.device
    tensor = lambda values: torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)
    states = tensor([row['s'] for row in records])
    successors = tensor([row['ns'] for row in records])
    actions = tensor([row['a'] for row in records]).reshape(-1,1)
    context = tensor([context_vector(row['ctx_s']) for row in records])
    next_context = tensor([context_vector(row['ctx_ns']) for row in records])
    rewards = tensor([row['r'] for row in records]).reshape(-1)
    terminal = tensor([row['terminal'] for row in records]).reshape(-1)
    noise = torch.as_tensor(noise, dtype=states.dtype, device=device)
    if noise.shape != (8,):
        raise ValueError('The frozen target uses the predeclared eight common normal draws')

    components = explicit_components(learner.actor,
        *learner.actor.distribution_parameters(successors), noise=noise)
    next_zero = learner.action_transform(successors,components['zero_action'])
    repeated_states = successors[:,None].expand(-1,8,-1).reshape(-1,229)
    next_positive = learner.action_transform(repeated_states,
        components['positive_action'].reshape(-1,1))
    features = learner.target_critic.encode_states(pack_states(successors,next_context))
    repeated_features = features[:,None].expand(-1,8,-1).reshape(-1,features.shape[1])
    zero_q = learner.target_critic.forward_features(features,next_zero).min(0).values
    positive_q = learner.target_critic.forward_features(repeated_features,next_positive).reshape(
        2,len(records),8).min(0).values.mean(1)
    expected_q = components['zero_probability']*zero_q+components['hold_probability']*positive_q
    expected_log_pi = components['zero_probability']*components['zero_log_prob']+components[
        'hold_probability']*components['positive_log_prob'].mean(1)
    continuation = 1.-terminal  # Original v25 replay stores discount=1 on each factual row.
    reward_component = config.reward_scale*rewards
    value_component = continuation*.1*learner.value(successors)
    sac_q_component = continuation*.9*expected_q
    entropy_component = continuation*.9*(-learner.alpha.detach()*expected_log_pi)
    target = reward_component+value_component+sac_q_component+entropy_component
    q = learner.critic(pack_states(states,context),actions)
    z = torch.where(context[:,1].bool(),context[:,0],torch.zeros_like(context[:,0]))
    feature = z*((actions[:,0]+1.)*.5)
    terms = td_theta_terms(q,target,feature,weights=weights)
    lower = learner.action_transform(states,torch.full_like(actions,-1.))
    upper = learner.action_transform(states,torch.ones_like(actions))
    single = (lower == upper).reshape(-1)
    holding = learner.action_transform.hold_seconds(actions).reshape(-1)
    forced_60 = single & (learner.action_transform.hold_seconds(lower).reshape(-1) == 60.)
    errors = dict(current_Q=q,
        negative_reward=-reward_component[None].expand_as(q),
        negative_V=-value_component[None].expand_as(q),
        negative_SAC_Q=-sac_q_component[None].expand_as(q),
        negative_entropy=-entropy_component[None].expand_as(q))
    reconstructed = sum(errors.values())
    return dict(**terms, q=q, target=target, feature=feature,
        holding_seconds=holding, single_support=single, forced_60=forced_60,
        reward_component=reward_component,value_component=value_component,
        sac_q_component=sac_q_component,entropy_component=entropy_component,
        error_components=errors,
        error_reconstruction_error=(reconstructed-terms['error']).abs().amax(),
        scope='Frozen final-model re-score; historical theta update causality is not reconstructed',
        samples=8, fresh_TD_quota=.25, gradient_includes_head_mean=True,
        gradient_component_decomposition_valid=False,
        hypothetical_raw_MSE_not_actual_Huber=True)
