"""Frozen beta=0/1 future-entropy intervention on the actual v53 chain.

The root M8 mixed backup is retained. Beta=1 removes only each future row's
explicit SAC entropy contribution before it travels through the legal trace.
No action-conditioned successors are fabricated and no learner state is set.
"""
from __future__ import annotations

import torch

CAPACITY = 271
ETA = 9. / (CAPACITY - 1)


def physical_trace(native, matching_q, coefficients, entropy, *, beta=0.):
    """Native float32 recurrence, excluding entropy of the receiving root.

    matching_q[i] uses the recorded action at actual successor issue i+1.
    The final real transition retains its original mixed bootstrap.
    """
    if float(beta) not in (0., 1.):
        raise ValueError('The predeclared frozen intervention is beta=0 or beta=1')
    physical = native.clone()
    for index in range(len(native)-2, -1, -1):
        coefficient = coefficients[index]
        if coefficient != 0:
            residual = physical[index+1]-matching_q[index]
            # Branching preserves the original beta=0 arithmetic exactly.
            if beta == 1.:
                residual = residual-entropy[index+1]
            physical[index] = native[index]+coefficient*residual
    return physical


def trace_target(root_y, future=None, *, lambda_trace=1., beta=0., eta=ETA):
    """Return the original root tensor lazily at lambda=0.

    For lambda>0 the root and future arrays describe the same real rows. A
    callable future is useful to verify that the zero case performs no future
    forwards, random draws or array construction.
    """
    if float(lambda_trace) == 0.:
        return root_y
    if not 0. <= float(lambda_trace) <= 1.:
        raise ValueError('lambda_trace must lie in [0,1]')
    future = future() if callable(future) else future
    physical = physical_trace(future['native'], future['matching_q'],
        future['coefficients'], future['entropy'], beta=beta)
    scale = root_y.new_tensor(float(lambda_trace))*root_y.new_tensor(float(eta))
    return root_y+scale*(physical-future['native'])


def _future_sum64(atoms, coefficients):
    """Root i receives only k>i; float64 is accounting, not target execution."""
    atoms, coefficients = atoms.double(), coefficients.double()
    result = torch.zeros_like(atoms)
    for index in range(len(atoms)-2, -1, -1):
        result[index] = coefficients[index]*(atoms[index+1]+result[index+1])
    return result


@torch.no_grad()
def frozen_block(learner, seeds):
    """One predeclared eight-seed block, with global RNG restored on exit.

    Only the positive Q-alpha*logpi joint is averaged. Gate probabilities and
    zero-action terms come from the common first draw, as in native v54 M8.
    The complete original two-of-two head subset needs no random subset draw.
    """
    if len(seeds) != 8:
        raise ValueError('Each block must contain exactly eight predeclared seeds')
    batch, config = learner.physical_batch, learner.config
    if len(batch['rewards']) != CAPACITY or len(learner.physical_coefficients) != CAPACITY-1:
        raise ValueError('The original real 271-row chain is required')
    if (config.ensemble_size != 2 or config.target_subset_size != 2
            or config.target_subset_replacement or config.mixed_backup_lambda != .1):
        raise ValueError('The original two-of-two .1V/.9SAC target is required')
    states = batch['next_observations']
    size, critic = len(states), learner.target_critic
    features = critic.encode_states(states)
    repeated_features = torch.cat((features, features))
    subset = torch.tensor([0, 1], dtype=torch.long, device=states.device)
    positives, positive_entropies = [], []
    positive_heads, first, zero, zero_entropy, zero_heads = [], None, None, None, None
    rng = torch.get_rng_state().clone()
    try:
        for seed in seeds:
            torch.manual_seed(int(seed))
            parts = learner.sync_actor.soft_action_components(states[:, :229])
            actions = torch.cat((learner.sync_transform(states[:, :229], parts['zero_action']),
                learner.sync_transform(states[:, :229], parts['positive_action'])))
            q = critic.forward_features_subset(repeated_features, actions, subset)
            q_zero, q_positive = q.split(size, dim=1)
            zero_q, current_zero_heads = q_zero.min(0)
            positive_q, current_positive_heads = q_positive.min(0)
            if config.backup_entropy:
                entropy_zero = -learner.alpha.detach()*parts['zero_log_prob']
                entropy_positive = -learner.alpha.detach()*parts['positive_log_prob']
                # Use native subtraction order, rather than adding -alpha*logpi.
                joint_zero = zero_q-learner.alpha.detach()*parts['zero_log_prob']
                joint_positive = positive_q-learner.alpha.detach()*parts['positive_log_prob']
            else:
                entropy_zero, entropy_positive = torch.zeros_like(zero_q), torch.zeros_like(positive_q)
                joint_zero, joint_positive = zero_q, positive_q
            if first is None:
                first, zero, zero_entropy = parts, joint_zero, entropy_zero
                zero_heads = current_zero_heads
            positives.append(joint_positive)
            positive_entropies.append(entropy_positive)
            positive_heads.append(current_positive_heads)
    finally:
        torch.set_rng_state(rng)
    sac = (first['zero_probability']*zero
        + first['hold_probability']*torch.stack(positives).mean(0))
    expected_entropy = (first['zero_probability']*zero_entropy
        + first['hold_probability']*torch.stack(positive_entropies).mean(0))
    reward = config.reward_scale*batch['rewards']
    discount = batch['discounts']*(1.-batch['terminals'])
    value = learner.value(states[:, :229])
    # Exactly the native mixed assembly, not a reordered component sum.
    bootstrap = discount*(.1*value+.9*sac)
    native = reward+bootstrap
    entropy = discount*(.9*expected_entropy)
    next_actions = batch['actions'][1:]
    next_features = critic.encode_states(states[:-1])
    matching_heads = critic.forward_features_subset(torch.cat((next_features, next_features)),
        torch.cat((next_actions, next_actions)), subset)[:, :CAPACITY-1]
    matching_q, matching_head = matching_heads.min(0)
    coefficients = learner.physical_coefficients
    future = dict(native=native, matching_q=matching_q, coefficients=coefficients, entropy=entropy)
    physical0 = physical_trace(**future, beta=0.)
    physical1 = physical_trace(**future, beta=1.)
    beta0 = trace_target(native, future, beta=0.)
    beta1 = trace_target(native, future, beta=1.)
    scale = native.new_tensor(ETA)
    # In exact arithmetic beta1-beta0 is minus accumulated future entropy.
    entropy_correction64 = -ETA*_future_sum64(entropy, coefficients)
    reward_correction64 = ETA*_future_sum64(reward, coefficients)
    bootstrap_correction64 = ETA*_future_sum64(bootstrap, coefficients)
    # Qbar lives on links: put Qbar_i in the successor atom i+1.
    aligned_minus_q = torch.cat((matching_q.new_zeros(1), -matching_q))
    minus_q_correction64 = ETA*_future_sum64(aligned_minus_q, coefficients)
    total0_64 = reward_correction64+bootstrap_correction64+minus_q_correction64
    total1_64 = total0_64+entropy_correction64
    future_mass = ETA*_future_sum64(torch.ones_like(native), coefficients)
    tail_weight = torch.ones_like(native, dtype=torch.float64)
    for index in range(CAPACITY-2, -1, -1):
        tail_weight[index] = coefficients[index].double()*tail_weight[index+1]
    tail_weight = ETA*tail_weight
    tail_weight[-1] = 0.  # There is no future bootstrap for the last root.
    return dict(native=native, beta0=beta0, beta1=beta1,
        physical_beta0=physical0, physical_beta1=physical1,
        reward=reward, bootstrap=bootstrap, entropy=entropy, sac=sac,
        expected_entropy=expected_entropy, value=value, discount=discount,
        matching_q=matching_q, matching_heads=matching_heads, matching_head=matching_head,
        coefficients=coefficients, target_subset=subset, zero_heads=zero_heads,
        positive_heads=torch.stack(positive_heads),
        future_mass=future_mass, mass=future_mass, terminal_weight=tail_weight,
        entropy_correction=entropy_correction64,
        accounting64=dict(reward=reward_correction64, bootstrap=bootstrap_correction64,
            minus_q=minus_q_correction64, entropy_removal=entropy_correction64,
            beta0=total0_64, beta1=total1_64),
        rounding64=dict(beta0=(beta0-native).double()-total0_64,
            beta1=(beta1-native).double()-total1_64,
            beta1_minus_beta0=(beta1-beta0).double()-entropy_correction64),
        future_residual_beta0=scale*(physical0-native),
        future_residual_beta1=scale*(physical1-native))
