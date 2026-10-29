"""Frozen action and distribution-gradient accounting under the original F.

No policy update, random sampling, optimizer call or imagined successor occurs.
The caller supplies common standard-normal draws for the sampled-positive mode.
"""
import torch
import torch.nn.functional as functional
from torch.distributions import Normal

from experiments.mcwm_closed_loop_context_wiring_v80_20261027.learning import pack_states


def executed_grid(transform, states, holds=(0., 7.5, 15., 30., 60.)):
    """Map each latent proposal once; exact aliases count once per state."""
    proposed_holds = states.new_tensor(holds)
    proposals = (2.*proposed_holds/transform.max_hold-1.)[None].expand(len(states), -1)
    repeated = states[:, None].expand(-1, len(holds), -1).reshape(-1, states.shape[1])
    actions = transform(repeated, proposals.reshape(-1, 1)).reshape(len(states), -1)
    unique = torch.ones_like(actions, dtype=torch.bool)
    for column in range(1, actions.shape[1]):
        unique[:, column] = ~(actions[:, column, None] == actions[:, :column]).any(1)
    return dict(proposed_hold_seconds=proposed_holds, executed_actions=actions,
        executed_hold_seconds=transform.hold_seconds(actions), unique=unique,
        unique_count=unique.sum(1), forced_single_action=unique.sum(1) == 1)


def _gather_head(values, indices):
    return values.gather(0, indices[None]).squeeze(0)


def explicit_components(actor, gate, mean, log_std, noise=None):
    """Native hurdle probabilities/densities with explicit positive noise.

    Positive arrays have shape[N,M], zero arrays/probabilities have shape[N]
    except zero_action[N,1]. No call to a stochastic distribution sampler.
    """
    if noise is None:
        pre_sigmoid = mean[:, None]
    else:
        noise = torch.as_tensor(noise, device=mean.device, dtype=mean.dtype)
        if noise.ndim == 1:
            noise = noise[None].expand(len(mean), -1)
        if noise.ndim != 2 or len(noise) != len(mean) or noise.shape[1] == 0:
            raise ValueError('Common normal draws must have shape[M] or[N,M]')
        pre_sigmoid = mean[:, None]+log_std.exp()[:, None]*noise
    log_hold, log_zero = functional.logsigmoid(gate), functional.logsigmoid(-gate)
    positive = actor._positive_transform(pre_sigmoid)
    conditional = (Normal(mean[:, None], log_std.exp()[:, None]).log_prob(pre_sigmoid)
                   -actor._positive_log_det(pre_sigmoid))
    return dict(zero_probability=log_zero.exp(), hold_probability=log_hold.exp(),
        zero_action=torch.full_like(mean[:, None],-1.), positive_action=positive,
        zero_log_prob=log_zero, positive_conditional_log_prob=conditional,
        positive_log_prob=log_hold[:, None]+conditional)


def q_decomposition(critic, states, context, executed_actions):
    """Score already executed actions without mapping them through F again.

    For every action, min(full Q) = min(base Q) + direct + head_switch.
    Direct is the residual on the full-Q selected head; head_switch is that
    head's base level minus the independent base minimum. Their action
    differences, rather than their levels, determine action advantages.
    """
    if executed_actions.ndim != 2 or len(executed_actions) != len(states):
        raise ValueError('Expected one executed action row per canonical state')
    count, width = executed_actions.shape
    features = critic.encode_states(pack_states(states, context))
    repeated = features[:, None].expand(-1, width, -1).reshape(-1, features.shape[1])
    actions = executed_actions.reshape(-1, 1)
    base = critic.base.forward_features(repeated[:, :-2], actions).reshape(-1, count, width)
    full = critic.forward_features(repeated, actions).reshape(-1, count, width)
    # The explicit analytical term avoids subtracting two large Q levels.
    z = torch.where(context[:, 1].bool(), context[:, 0], torch.zeros_like(context[:, 0]))
    residual = critic.action_weight[:, None, None]*z[None, :, None]*(executed_actions[None]+1.)*.5
    base_head, full_head = base.argmin(0), full.argmin(0)
    base_min, full_min = _gather_head(base, base_head), _gather_head(full, full_head)
    base_at_full = _gather_head(base, full_head)
    direct = _gather_head(residual, full_head)
    switch = base_at_full-base_min
    advantage = lambda values: values-values[:, :1]
    unique = torch.ones_like(executed_actions, dtype=torch.bool)
    for column in range(1, width):
        unique[:, column] = ~(executed_actions[:, column, None] == executed_actions[:, :column]).any(1)
    full_preferred = full_min.masked_fill(~unique, -torch.inf).argmax(1)
    base_preferred = base_min.masked_fill(~unique, -torch.inf).argmax(1)
    full_action = executed_actions.gather(1, full_preferred[:, None]).squeeze(1)
    base_action = executed_actions.gather(1, base_preferred[:, None]).squeeze(1)
    return dict(base_heads=base, full_heads=full, residual_heads=residual,
        base_head=base_head, full_head=full_head, head_changed=full_head != base_head,
        base_min=base_min, full_min=full_min, base_at_full_head=base_at_full,
        direct=direct, head_switch=switch,
        base_advantage=advantage(base_min), full_advantage=advantage(full_min),
        direct_advantage=advantage(direct), head_switch_advantage=advantage(switch),
        unique=unique, unique_support_count=unique.sum(1),
        preferred_action_full=full_action, preferred_action_base=base_action,
        preference_changed=full_action != base_action,
        roundoff_bound=8*torch.finfo(full.dtype).eps*
            torch.maximum(base.abs().amax((0,2)), full.abs().amax((0,2))).clamp_min(1.),
        head_reconstruction_error=(full-base-residual).abs().amax((0, 2)),
        min_reconstruction_error=(full_min-base_min-direct-switch).abs().amax(1),
        advantage_reconstruction_error=(advantage(full_min)-advantage(base_min)
            -advantage(direct)-advantage(switch)).abs().amax(1))


def actor_gradient_decomposition(learner, states, context, noise=None):
    """Per-state gradients of the frozen SAC objective w.r.t. actor outputs.

    ``noise=None`` uses deterministic positive actions; an explicit tensor
    ``noise[M]`` or ``noise[N,M]`` uses reparameterized common normal draws.
    These are local gate/mean/effective-logstd gradients, not a simulated actor
    update or reconstructed historical gradient. Averaging rows recovers the
    corresponding mean actor-output gradient. No global RNG is touched.
    """
    gate, mean, log_std = [value.detach().clone().requires_grad_(True)
                           for value in learner.actor.distribution_parameters(states)]
    components = explicit_components(learner.actor,gate,mean,log_std,noise)
    mode = 'deterministic_positive' if noise is None else 'common_reparameterized_positive'
    positive = components['positive_action']
    width = positive.shape[1]
    repeated = states[:, None].expand(-1, width, -1).reshape(-1, states.shape[1])
    executed_positive = learner.action_transform(repeated, positive.reshape(-1, 1)).reshape(len(states), width)
    zero = learner.action_transform(states, torch.full_like(mean[:, None], -1.))
    actions = torch.cat((zero, executed_positive), 1)
    values = q_decomposition(learner.critic, states, context, actions)
    hold, nonhold = components['hold_probability'],components['zero_probability']
    mixture = lambda q: nonhold*q[:, 0]+hold*q[:, 1:].mean(1)
    entropy = learner.alpha.detach()*(nonhold*components['zero_log_prob']
        +hold*components['positive_log_prob'].mean(1))
    losses = dict(base_Q=-mixture(values['base_min']),
        direct_capacity=-mixture(values['direct']),
        head_switch=-mixture(values['head_switch']),
        full_Q=-mixture(values['full_min']), entropy=entropy)
    losses['full_SAC'] = losses['full_Q']+entropy
    gradients = {}
    for name, loss in losses.items():
        terms = torch.autograd.grad(loss.sum(), (gate,mean,log_std),
            retain_graph=True, allow_unused=True)
        gradients[name] = torch.stack([torch.zeros_like(gate) if term is None else term
                                      for term in terms], 1)
    gradient_error = (gradients['full_Q']-gradients['base_Q']
        -gradients['direct_capacity']-gradients['head_switch']).abs().amax(1)
    sac_error = (gradients['full_SAC']-gradients['full_Q']-gradients['entropy']).abs().amax(1)
    # Deleting a critic term changes scores/gradients, not this frozen actor.
    deterministic_positive = learner.actor._positive_transform(mean[:, None])
    deterministic_proposal = torch.where((hold >= .5)[:, None],
        deterministic_positive, torch.full_like(deterministic_positive,-1.))
    deterministic_executed = learner.action_transform(states, deterministic_proposal)
    return dict(mode=mode, samples=width, actions=actions, q=values,
        gate_probability=hold.detach(), positive_proposals=positive.detach(),
        losses={key:value.detach() for key,value in losses.items()},
        gradients={key:value.detach() for key,value in gradients.items()},
        deployed_proposal=deterministic_proposal.detach(),
        deployed_executed_action=deterministic_executed.detach(),
        deployed_holding_seconds=learner.action_transform.hold_seconds(deterministic_executed).detach().reshape(-1),
        gradient_reconstruction_error=gradient_error.detach(),
        SAC_gradient_reconstruction_error=sac_error.detach(),
        actor_gradient_coordinates=('gate_logit','positive_mean','effective_positive_log_std'))


def audit_saved_gradients(q, gate_probability, gradients):
    """Audit saved gradients without evaluating a model or changing old arrays.

    Gate backprop subtracts p'(g)*Q0 and p'(g)*mean(Q+) before a small
    action advantage remains. Its floating-point error therefore scales with
    those pre-cancellation terms, not with the final small gradient. The
    independent FP64 gate expression below exposes that cancellation explicitly.

    Mean/logstd coordinates retain the original strict output-gradient bound;
    they are not excused by the Q-level gate bound. This helper is intended for
    the existing FP32 readouts, not a blanket relaxation of every audit.
    """
    names = dict(base_Q='base_min', full_Q='full_min',
                 direct_capacity='direct', head_switch='head_switch')
    dtype = gradients['full_Q'].dtype
    eps = torch.finfo(dtype).eps
    unit_roundoff = eps*.5
    samples = q['full_min'].shape[1]-1
    # A reduction of M terms and two probability/product/backward paths. The
    # additive operations cover the native logsigmoid/exp gate derivatives.
    operations = 2*samples+8
    gamma = operations*unit_roundoff/(1.-operations*unit_roundoff)
    probability = gate_probability.detach().double()
    factor = probability*(1.-probability)
    references, errors, bounds, scales = {}, {}, {}, {}
    for name,key in names.items():
        values = q[key].detach().double()
        references[name] = -factor*(values[:,1:].mean(1)-values[:,0])
        scales[name] = factor*(values[:,0].abs()+values[:,1:].abs().mean(1))
        bounds[name] = gamma*scales[name].clamp_min(1.)
        errors[name] = (gradients[name][:,0].detach().double()-references[name]).abs()
    floating_gradients = {name:values.detach().double() for name,values in gradients.items()}
    closure = floating_gradients['full_Q']-floating_gradients['base_Q']-floating_gradients['direct_capacity']-floating_gradients['head_switch']
    sac_closure = floating_gradients['full_SAC']-floating_gradients['full_Q']-floating_gradients['entropy']
    scale = torch.ones_like(gradients['full_Q'][:,0])
    for values in gradients.values():
        scale = torch.maximum(scale,values.abs().amax(1))
    nongate_bound = (16*eps*scale).double()
    level_error = q['full_min'].detach().double()-q['base_min'].detach().double()-q['direct'].detach().double()-q['head_switch'].detach().double()
    # The saved forward Q addition also rounds. Propagate that measured level
    # discrepancy through the independent gate expression, rather than hiding it.
    forward_gate_error = factor*(level_error[:,1:].abs().mean(1)+level_error[:,0].abs())
    gate_closure_bound = sum(bounds.values())+forward_gate_error
    sac_operations = 2*samples+16
    sac_gamma = sac_operations*unit_roundoff/(1.-sac_operations*unit_roundoff)
    sac_gate_bound = sac_gamma*(2*scales['full_Q']+
        floating_gradients['entropy'][:,0].abs()).clamp_min(1.)
    checks = dict(independent_gate_reference=all(bool((errors[name] <= bounds[name]).all()) for name in names),
        Q_gate_reconstruction=bool((closure[:,0].abs() <= gate_closure_bound).all()),
        SAC_gate_reconstruction=bool((sac_closure[:,0].abs() <= sac_gate_bound).all()),
        Q_mean_logstd_reconstruction=bool((closure[:,1:].abs() <= nongate_bound[:,None]).all()),
        SAC_mean_logstd_reconstruction=bool((sac_closure[:,1:].abs() <= nongate_bound[:,None]).all()))
    return dict(all_pass=all(checks.values()), checks=checks,
        analytic_gate_gradients=references, gate_reference_errors=errors,
        gate_reference_bounds=bounds, gate_uncancelled_scales=scales,
        Q_closure_error=closure.abs(), SAC_closure_error=sac_closure.abs(),
        Q_gate_closure_bound=gate_closure_bound, SAC_gate_closure_bound=sac_gate_bound,
        mean_logstd_bound=nongate_bound,
        scope='Independent FP64 gate expressions on saved values; original FP32 arrays unchanged')
