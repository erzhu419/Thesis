"""Explicit capacity context on the frozen v75 replay and critic paths.

Actor, V and F receive the native229 fields. Only Q receives two appended
columns: signed downstream demand/seat margin z and an observation mask.
Archived missing context is NaN/mask0, never an observed zero margin.
"""
from collections import deque
import math

import numpy as np
import torch

from experiments.mcwm_closed_loop_frozen_validation_v75_20261027 import learning as native
from experiments.mcwm_closed_loop_capacity_context_v78_20261027.critic import CapacityContextQEnsemble
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng
from SimpleSAC.transit_v7.networks import FEATURE_DIM


STATE_DIM, CONTEXT_DIM = 229, 2


def row_context(row, field):
    observation = getattr(row, field, None)
    if observation is None or not observation['observed']:
        return np.array([np.nan, 0.], dtype=np.float32)
    z = observation['z']
    if z is None or not math.isfinite(float(z)):
        raise ValueError('Observed capacity context requires a finite issue-visible z')
    return np.array([z, 1.], dtype=np.float32)


def missing_context(count, device='cpu'):
    context = torch.zeros((count, CONTEXT_DIM), device=device)
    context[:, 0] = torch.nan
    return context


def pack_states(states, context):
    if states.shape != (len(context), STATE_DIM) or context.shape != (len(states), CONTEXT_DIM):
        raise ValueError('Critic input requires aligned native229 states plus[z,mask]')
    return torch.cat((states, context.to(dtype=states.dtype, device=states.device)), dim=1)


def split_states(states):
    if states.ndim != 2 or states.shape[1] != STATE_DIM+CONTEXT_DIM:
        raise ValueError('Context Q requires explicit231-column critic input')
    return states[:, :STATE_DIM], states[:, STATE_DIM:]


class ContextQEnsemble(CapacityContextQEnsemble):
    """Native ensemble interface, with context carried through feature batching."""
    def forward(self, states, actions):
        canonical, context = split_states(states)
        return super().forward(canonical, actions, context[:, 0], context[:, 1].bool())

    def forward_subset(self, states, actions, indices):
        canonical, context = split_states(states)
        return super().forward_subset(canonical, actions, context[:, 0], context[:, 1].bool(), indices)

    def encode_states(self, states):
        canonical, context = split_states(states)
        return torch.cat((self.base.encode_states(canonical), context), dim=1)

    def forward_features(self, features, actions):
        return (self.base.forward_features(features[:, :FEATURE_DIM], actions)
                + self._residual(actions, features[:, FEATURE_DIM], features[:, FEATURE_DIM+1].bool()))

    def forward_features_subset(self, features, actions, indices):
        return (self.base.forward_features_subset(features[:, :FEATURE_DIM], actions, indices)
                + self._residual(actions, features[:, FEATURE_DIM], features[:, FEATURE_DIM+1].bool(), indices))


class ContextReplay(native.DecoupledReplay):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize_context()

    def initialize_context(self):
        self.context_covariates = deque(maxlen=285)
        self.context_outcomes = deque(maxlen=285)

    def add_target(self, covariates, outcomes):
        if any(not hasattr(row, 'ctx_s') for row in covariates):
            raise ValueError('Every newly admitted covariate requires explicit ctx_s')
        if self.arm == 'fresh' and any(not hasattr(row, field)
                for row in outcomes for field in ('ctx_s', 'ctx_ns')):
            raise ValueError('Every newly admitted complete outcome requires explicit ctx_s/ctx_ns')
        current = [row_context(row, 'ctx_s') for row in covariates]
        complete = [(row_context(row, 'ctx_s'), row_context(row, 'ctx_ns')) for row in outcomes] if self.arm == 'fresh' else []
        super().add_target(covariates, outcomes)
        self.context_covariates.extend(current)
        self.context_outcomes.extend(complete)

    def sample_mixed(self, size, online_fraction, *, rng):
        factual, source = super().sample_mixed(size, online_fraction, rng=rng)
        count = size//4
        current = torch.as_tensor(np.asarray(self.context_covariates)[self.last_sampling['recent']],
                                  dtype=torch.float32, device=self.device)
        self.last_covariates['context'] = torch.cat((missing_context(count, self.device), current))
        selected = missing_context(count, self.device)
        next_selected = missing_context(count, self.device)
        if self.arm == 'fresh':
            window = self.last_outcome_sampling['window']
            selected_context = np.asarray(list(self.context_outcomes)[-window:])[
                self.last_outcome_sampling['local_indices']]
            selected, next_selected = (torch.as_tensor(selected_context[:, index],
                dtype=torch.float32, device=self.device) for index in (0, 1))
        for key, context in (('observations', selected), ('next_observations', next_selected)):
            factual[key] = pack_states(factual[key], torch.cat((missing_context(count, self.device), context)))
            source[key] = pack_states(source[key], missing_context(len(source[key]), self.device))
        return factual, source

    def context_summary(self):
        return dict(covariate_contexts=len(self.context_covariates),
            complete_context_pairs=len(self.context_outcomes),
            observed_covariates=sum(int(row[1]) for row in self.context_covariates),
            observed_outcome_s=sum(int(pair[0][1]) for pair in self.context_outcomes),
            observed_outcome_ns=sum(int(pair[1][1]) for pair in self.context_outcomes),
            archive_context='missing NaN/mask0', row_alignment='record order, never state-value lookup')


class ContextTrainer(native.ExecutedCQLTrainer):
    @torch.no_grad()
    def _target(self, batch, *, backup):
        states, context = split_states(batch['next_observations'])
        return super()._target(dict(batch, next_observations=states, next_context=context), backup=backup)

    @torch.no_grad()
    def _sac_bootstrap(self, batch):
        states, context = batch['next_observations'], batch['next_context']
        components = self.actor.soft_action_components(states)
        size = len(states)
        actions = torch.cat((self.action_transform(states, components['zero_action']),
                             self.action_transform(states, components['positive_action'])))
        chosen = self._sample_target_subset()
        features = self.target_critic.encode_states(pack_states(states, context))
        values = self.target_critic.forward_features_subset(torch.cat((features, features)), actions, chosen)
        zero_q, positive_q = values.split(size, dim=1)
        zero_value, positive_value = zero_q.min(0).values, positive_q.min(0).values
        if self.config.backup_entropy:
            zero_value -= self.alpha.detach()*components['zero_log_prob']
            positive_value -= self.alpha.detach()*components['positive_log_prob']
        return components['zero_probability']*zero_value+components['hold_probability']*positive_value

    def _iql_value_terms(self, batch):
        current = self.replay_buffer.last_covariates
        states = current['observations']
        with torch.no_grad():
            target_q = self.target_critic(pack_states(states, current['context']), current['actions']).min(0).values
        advantage = target_q-self.value(states)
        weights = torch.where(advantage >= 0., torch.full_like(advantage, self.config.iql_expectile),
                              torch.full_like(advantage, 1.-self.config.iql_expectile))
        value_loss = (weights*advantage.square()).mean()
        actor_weights = torch.exp(advantage.detach()*self.config.iql_advantage_temperature).clamp_max(self.config.iql_max_weight)
        return value_loss, advantage.detach(), actor_weights

    def _actor_states(self, offline_batch, online_batch):
        current = self.replay_buffer.last_covariates
        return torch.cat((pack_states(current['observations'], current['context']), online_batch['observations']))

    def _temperature_states(self, offline_batch, online_batch):
        return self._actor_states(offline_batch, online_batch)[:, :STATE_DIM]

    @torch.no_grad()
    def _executed_candidates(self, states):
        canonical, _ = split_states(states)
        return super()._executed_candidates(canonical)

    def _cql_penalty(self, batch, data_q):
        current = self.replay_buffer.last_covariates
        self.replay_buffer.last_covariates = dict(current,
            observations=pack_states(current['observations'], current['context']))
        try:
            return super()._cql_penalty(batch, data_q)
        finally:
            self.replay_buffer.last_covariates = current

    def context_actor_terms(self, actor_states):
        """Original SAC terms without backward, zero_grad or optimizer.step."""
        states, _ = split_states(actor_states)
        components = self.actor.soft_action_components(states)
        size = len(states)
        zero_action = self.action_transform(states, components['zero_action'])
        positive_action = self.action_transform(states, components['positive_action'])
        features = self.critic.encode_states(actor_states)
        combined_q = self.critic.forward_features(torch.cat((features, features)), torch.cat((zero_action, positive_action)))
        zero_q, positive_q = self._reduce_actor_q(combined_q).split(size)
        zero_loss = self.alpha.detach()*components['zero_log_prob']-zero_q
        positive_loss = self.alpha.detach()*components['positive_log_prob']-positive_q
        loss = (components['zero_probability']*zero_loss+components['hold_probability']*positive_loss).mean()
        log_prob = components['zero_probability']*components['zero_log_prob']+components['hold_probability']*components['positive_log_prob']
        hold_log_prob = components['positive_log_prob']-components['positive_conditional_log_prob']
        diagnostics = {'train/hybrid_entropy': -log_prob.mean().detach(),
            'train/hybrid_gate_entropy': -(components['zero_probability']*components['zero_log_prob']+components['hold_probability']*hold_log_prob).mean().detach(),
            'train/hybrid_weighted_positive_entropy': (-components['hold_probability']*components['positive_conditional_log_prob']).mean().detach(),
            'train/hold_probability': components['hold_probability'].mean().detach(),
            'train/positive_action_mean': components['positive_action'].mean().detach(),
            'train/executed_positive_action_mean': positive_action.mean().detach(),
            'train/executed_zero_action_mean': zero_action.mean().detach()}
        return loss, log_prob, diagnostics

    def _update_sac_actor(self, actor_states, *, temperature_states=None,
                          temperature_action_samples=1, collect_metrics=True):
        # This is the unchanged456-update execution path; v80 never calls it.
        states, _ = split_states(actor_states)
        if temperature_states is None:
            temperature_states = states
        self._set_critic_grad(False)
        try:
            actor_loss, log_prob, self._last_actor_diagnostics = self.context_actor_terms(actor_states)
            if self.config.temperature_state_source == 'actor' and temperature_action_samples == 1:
                temperature_log_prob = log_prob.detach()
            else:
                with torch.no_grad():
                    temperature_log_prob = self._temperature_log_probability(temperature_states, temperature_action_samples)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad_norm = self._gradient_norm(self.actor.parameters(), collect_metrics=collect_metrics)
            self.actor_optimizer.step()
        finally:
            self._set_critic_grad(True)
        alpha_loss = -(self.log_alpha*(temperature_log_prob+self.config.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        return actor_loss, alpha_loss, actor_grad_norm


def attach_context(learner):
    """Attach before first feedback; retain all old parameter and Adam objects."""
    replay = learner.replay_buffer
    if type(learner) is not native.ExecutedCQLTrainer or type(replay) is not native.DecoupledReplay:
        raise ValueError('v80 attaches only to the actual isolated v75 learner/replay')
    if replay.target_received or replay.recent or replay.fresh_outcomes or learner.world is not None:
        raise ValueError('Attach before any target feedback to the original real-only learner')
    before = _rng()
    learner.__class__ = ContextTrainer
    replay.__class__ = ContextReplay
    replay.initialize_context()
    learner.critic = ContextQEnsemble(learner.critic)
    learner.target_critic = ContextQEnsemble(learner.target_critic)
    learner.target_critic.eval()
    learner.target_critic.action_weight.requires_grad_(False)
    learner.critic_optimizer.param_groups[0]['params'].append(learner.critic.action_weight)
    if not _equal(before, _rng()):
        raise RuntimeError('Context attachment consumed a random draw')
    return learner


def make_learner(seed, *, arm):
    learner, replay, contract, stations, initializer = native.load_learner(seed, arm=arm)
    attach_context(learner)
    return learner, replay, contract, stations, initializer
