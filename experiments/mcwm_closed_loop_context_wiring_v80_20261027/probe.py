"""Zero-update integration parity using resident archived test copies only."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch

from experiments.mcwm_closed_loop_context_wiring_v80_20261027 import learning
from experiments.mcwm_closed_loop_capacity_context_v78_20261027.probe import (
    _random_state, _restore_random, _note)
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import (
    OPTIMIZERS, _equal, _rng, _restore, _state)


def archived_copies(replay):
    """Separate covariates and outcomes, with explicitly unobserved context."""
    def row(table, index, sequence):
        values = {field: np.asarray(table[field][index]).copy()
                  for field in learning.native.FIELDS.values()}
        values.update(sequence=sequence,
            ctx_s=dict(observed=False, z=None, reasons=['archived_context_not_captured']),
            ctx_ns=dict(observed=False, z=None, reasons=['archived_context_not_captured']))
        return SimpleNamespace(**values)
    return ([row(replay.history, index, index) for index in range(5)],
            [row(replay.source, index+5, index+5) for index in range(5)])


def _frozen_state(learner):
    state = _state(learner)
    state.pop('sampler')  # Temporary diagnostic sampling is restored separately.
    return state


def _gradients(loss, parameters, *, retain_graph=False):
    return torch.autograd.grad(loss, tuple(parameters), retain_graph=retain_graph)


def _original_actor_terms(learner, states):
    components = learner.actor.soft_action_components(states)
    actions = torch.cat((learner.action_transform(states, components['zero_action']),
                         learner.action_transform(states, components['positive_action'])))
    features = learner.critic.encode_states(states)
    q = learner._reduce_actor_q(learner.critic.forward_features(torch.cat((features, features)), actions))
    zero_q, positive_q = q.split(len(states))
    log_prob = components['zero_probability']*components['zero_log_prob']+components['hold_probability']*components['positive_log_prob']
    loss = (components['zero_probability']*(learner.alpha.detach()*components['zero_log_prob']-zero_q)
            + components['hold_probability']*(learner.alpha.detach()*components['positive_log_prob']-positive_q)).mean()
    return loss, log_prob


def _nonzero_wiring(candidate, batches):
    """Hand-set theta/context stress, not native observations or supervision."""
    offline, source = batches
    batch = {key: torch.cat((offline[key], source[key])) for key in offline}
    current = candidate.replay_buffer.last_covariates
    original_context = current['context']
    rng = _random_state(candidate)
    native_state = current['observations'].clone()
    feature_masks = []
    original_forward = candidate.critic.forward_features
    def observe_features(features, actions):
        feature_masks.append(bool(features[:, -1].bool().any()))
        return original_forward(features, actions)
    candidate.critic.forward_features = observe_features
    try:
        with torch.no_grad():
            candidate.critic.action_weight.fill_(8.)
            candidate.target_critic.action_weight.fill_(8.)
        results = []
        for sign in (1., -1.):
            _restore_random(candidate, rng)
            context = torch.ones((len(native_state), 2), device=candidate.device)
            context[:, 0] = sign
            current['context'] = context
            actor = candidate.context_actor_terms(candidate._actor_states(*batches))[0]
            value = candidate._iql_value_terms(offline)[1]
            start = len(feature_masks)
            penalty = candidate._cql_penalty(offline, None)
            cql_context_seen = len(feature_masks) > start and all(feature_masks[start:])
            packed_next = batch['next_observations'].clone()
            packed_next[:, -2], packed_next[:, -1] = sign, 1.
            target = candidate._target(dict(batch, next_observations=packed_next), backup='h2oplus')
            results.append((actor.detach(), value.detach(), penalty.detach(), target, cql_context_seen))
        states = native_state[:2]
        context = torch.tensor([[1., 1.], [-1., 1.]], device=states.device)
        packed = learning.pack_states(states, context)
        low, high = torch.full((2, 1), -1., device=states.device), torch.full((2, 1), -.9375, device=states.device)
        native_delta = candidate.critic.base(states, high)-candidate.critic.base(states, low)
        delta = candidate.critic(packed, high)-candidate.critic(packed, low)
        maximum_q = max(1., float(candidate.critic(packed, high).detach().abs().max()))
        direct = torch.allclose(delta-native_delta,
            candidate.critic.action_weight[:, None]*context[:, 0][None, :]/32.,
            rtol=0., atol=4*torch.finfo(delta.dtype).eps*maximum_q)
        return dict(theta_hand_set_no_step=True, synthetic_context_only=True,
            direct_action_interaction=bool(direct and torch.any(delta != native_delta)),
            target_context_reaches_Q=not torch.equal(results[0][3], results[1][3]),
            actor_context_reaches_Q=not torch.equal(results[0][0], results[1][0]),
            value_context_reaches_target_Q=not torch.equal(results[0][1], results[1][1]),
            CQL_feature_context_retained=results[0][4] and results[1][4],
            CQL_penalty_changes=not torch.equal(results[0][2], results[1][2]),
            CQL_penalty_change_required=False,
            canonical_covariates_unchanged=torch.equal(native_state, current['observations']))
    finally:
        del candidate.critic.forward_features
        current['context'] = original_context
        with torch.no_grad():
            candidate.critic.action_weight.zero_()
            candidate.target_critic.action_weight.zero_()
        _restore_random(candidate, rng)


def _run_parity(seed, arm):
    baseline, old_replay, _, _, initializer = learning.native.load_learner(seed, arm=arm)
    candidate, replay, _, _, other_initializer = learning.native.load_learner(seed, arm=arm)
    old_parameters = tuple(candidate.critic.parameters())
    old_adam = {name: (getattr(candidate, name),
        {id(parameter): id(state) for parameter, state in getattr(candidate, name).state.items()})
        for name in OPTIMIZERS}
    constructor_rng = _random_state(candidate)
    learning.attach_context(candidate)
    checks = dict(same_initializer=str(initializer) == str(other_initializer),
        constructor_all_rng_unchanged=_equal(constructor_rng, _random_state(candidate)),
        old_parameter_objects_retained=all(left is right for left, right in zip(old_parameters, candidate.critic.base.parameters())),
        initializer_50000=baseline.total_updates == candidate.total_updates == 50000,
        no_online_updates=baseline.online_critic_updates == candidate.online_critic_updates == 0)
    errors = {}
    saved_state = {name: deepcopy(_frozen_state(model)) for name, model in (('baseline', baseline), ('context', candidate))}
    saved_rng = {name: _random_state(model) for name, model in (('baseline', baseline), ('context', candidate))}
    try:
        covariates, outcomes = archived_copies(old_replay)
        old_replay.add_target(covariates, outcomes)
        replay.add_target(covariates, outcomes)
        old = old_replay.sample_mixed(baseline.config.batch_size, .5, rng=baseline.rng)
        new = replay.sample_mixed(candidate.config.batch_size, .5, rng=candidate.rng)
        checks['five_sampling_draws_exact'] = (_equal(old_replay.last_sampling, replay.last_sampling)
            and _equal(old_replay.last_outcome_sampling, replay.last_outcome_sampling))
        checks['learner_and_private_outcome_rng_exact'] = (_equal(baseline.rng.bit_generator.state, candidate.rng.bit_generator.state)
            and _equal(old_replay.outcome_rng.bit_generator.state, replay.outcome_rng.bit_generator.state))
        for label, before, after in zip(('factual', 'source'), old, new):
            for key in before:
                actual = after[key][:, :229] if key in ('observations', 'next_observations') else after[key]
                _note(checks, errors, label+'_'+key, before[key], actual)
            for key in ('observations', 'next_observations'):
                checks[label+'_'+key+'_explicit_missing'] = bool(torch.isnan(after[key][:, 229]).all() and not after[key][:, 230].bool().any())
        checks['latest5_covariates_native229'] = torch.equal(old_replay.last_covariates['observations'], replay.last_covariates['observations'])
        checks['no_fresh_outcome_context_leak'] = arm == 'fresh' or (not replay.fresh_outcomes and not replay.context_outcomes)
        old_batch = {key: torch.cat((old[0][key], old[1][key])) for key in old[0]}
        new_batch = {key: torch.cat((new[0][key], new[1][key])) for key in new[0]}
        rng = _rng()
        old_target = baseline._target(old_batch, backup='h2oplus')
        target_end = _rng()
        _restore(rng)
        new_target = candidate._target(new_batch, backup='h2oplus')
        _note(checks, errors, 'mixed_target', old_target, new_target)
        checks['target_sampling_rng_exact'] = _equal(target_end, _rng())
        old_q = baseline.critic(old_batch['observations'], old_batch['actions'])
        new_q = candidate.critic(new_batch['observations'], new_batch['actions'])
        _note(checks, errors, 'all_head_Q', old_q, new_q)
        size = len(old[0]['observations'])
        def td(model, values, target):
            return .5*model._ensemble_td_loss(values[:, :size], target[:size])+.5*model._ensemble_td_loss(values[:, size:], target[size:])
        before_td, after_td = td(baseline, old_q, old_target), td(candidate, new_q, new_target)
        _note(checks, errors, 'TD_loss', before_td, after_td)
        new_grad = _gradients(after_td, (*candidate.critic.base.parameters(), candidate.critic.action_weight))
        _note(checks, errors, 'TD_base_parameter_gradients', _gradients(before_td, baseline.critic.parameters()), new_grad[:-1])
        checks['missing_context_theta_TD_gradient_zero'] = not bool(new_grad[-1].any())
        _note(checks, errors, 'V_terms', baseline._iql_value_terms(old[0]), candidate._iql_value_terms(new[0]))
        old_states, new_states = baseline._actor_states(*old), candidate._actor_states(*new)
        _note(checks, errors, 'actor_canonical_states', old_states, new_states[:, :229])
        _note(checks, errors, 'temperature_native229', baseline._temperature_states(*old), candidate._temperature_states(*new))
        rng = _rng()
        old_loss, old_log_prob = _original_actor_terms(baseline, old_states)
        actor_end = _rng()
        _restore(rng)
        new_loss, new_log_prob, _ = candidate.context_actor_terms(new_states)
        _note(checks, errors, 'actor_loss', old_loss, new_loss)
        _note(checks, errors, 'actor_log_prob', old_log_prob, new_log_prob)
        _note(checks, errors, 'actor_parameter_gradients', _gradients(old_loss, baseline.actor.parameters()), _gradients(new_loss, candidate.actor.parameters()))
        checks['actor_sampling_rng_exact'] = _equal(actor_end, _rng())
        rng = _rng()
        old_penalty = baseline._cql_penalty(old[0], old_q[:, :size])
        cql_end = _rng()
        _restore(rng)
        new_penalty = candidate._cql_penalty(new[0], new_q[:, :size])
        _note(checks, errors, 'CQL_penalty', old_penalty, new_penalty)
        new_grad = _gradients(new_penalty, (*candidate.critic.base.parameters(), candidate.critic.action_weight))
        _note(checks, errors, 'CQL_base_parameter_gradients', _gradients(old_penalty, baseline.critic.parameters()), new_grad[:-1])
        checks['missing_context_theta_CQL_gradient_zero'] = not bool(new_grad[-1].any())
        checks['CQL_main_rng_exact'] = _equal(cql_end, _rng())
        old_audit, new_audit = baseline.cql_audit(), candidate.cql_audit()
        old_audit.pop('regularizer_wall_seconds'); new_audit.pop('regularizer_wall_seconds')
        checks['CQL_dedup_and_F_audit_exact'] = _equal(old_audit, new_audit)
        wiring = _nonzero_wiring(candidate, new)
        checks['nonzero_synthetic_wiring'] = all(wiring[key] for key in (
            'direct_action_interaction', 'target_context_reaches_Q', 'actor_context_reaches_Q',
            'value_context_reaches_target_Q', 'CQL_feature_context_retained', 'canonical_covariates_unchanged'))
    finally:
        for name, model in (('baseline', baseline), ('context', candidate)):
            _restore_random(model, saved_rng[name])
    for name, model in (('baseline', baseline), ('context', candidate)):
        checks[name+'_modules_Adam_and_update_counts_unchanged'] = _equal(saved_state[name], _frozen_state(model))
        checks[name+'_all_rng_restored'] = _equal(saved_rng[name], _random_state(model))
    checks['old_Adam_objects_retained'] = all(getattr(candidate, name) is optimizer and states ==
        {id(parameter): id(state) for parameter, state in optimizer.state.items()}
        for name, (optimizer, states) in old_adam.items())
    checks['theta_has_no_Adam_state'] = candidate.critic.action_weight not in candidate.critic_optimizer.state
    return dict(seed=seed, arm=arm, all_pass=all(checks.values()), checks=checks,
        maximum_absolute_errors=errors, synthetic_nonzero_wiring=wiring,
        fixture=dict(kind='archived test copies in isolated replay', covariates='history[0:5]',
            complete_outcomes='source[5:10]' if arm == 'fresh' else 'complete historical replacement',
            temporary_covariate_copies=5, temporary_complete_copies=5 if arm == 'fresh' else 0,
            new_target_feedback=0, v79_diagnostic_data_used=False),
        native_simulated_seconds=0, paid_training_updates=0, initializer_retraining_updates=0,
        source_initializer=str(initializer), scope='Zero-update integration; no8-step optimizer or performance claim.')


def run_parity(seed, arm):
    if seed not in (6208, 6207) or arm not in ('fresh', 'no_fresh'):
        raise ValueError('v80 fixed parity cohort is6208/6207 x fresh/no_fresh')
    outer_rng = _rng()
    try:
        return _run_parity(seed, arm)
    finally:
        _restore(outer_rng)
