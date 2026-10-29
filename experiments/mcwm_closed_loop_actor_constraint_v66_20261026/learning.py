"""v64 targets with one fixed-reference, full hurdle-policy KL per packet."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.mcwm_closed_loop_fresh_value_v64_20261025.learning import (
    FIELDS, ExecutionAction, FreshOutcomeReplay, FreshValueTrainer,
    TransitTrainerConfig, load_table, resident)


def hurdle_kl(current, reference):
    """Exact KL(new || reference), before their common positive transform."""
    gate, mean, log_std = current
    old_gate, old_mean, old_log_std = (value.detach() for value in reference)
    probability = torch.sigmoid(gate)
    gate_kl = (probability*(F.logsigmoid(gate)-F.logsigmoid(old_gate))
        + (1.-probability)*(F.logsigmoid(-gate)-F.logsigmoid(-old_gate)))
    conditional = (old_log_std-log_std + .5*(
        torch.exp(2.*(log_std-old_log_std))
        + (mean-old_mean).square()*torch.exp(-2.*old_log_std)-1.))
    weighted_positive = probability*conditional
    return dict(gate_kl=gate_kl, weighted_positive_kl=weighted_positive,
                total_kl=gate_kl+weighted_positive)


class ActorConstrainedTrainer(FreshValueTrainer):
    """Original eight actor/temperature updates plus beta*KL to packet entry."""

    def __init__(self, *args, actor_kl_coefficient=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.actor_kl_coefficient = float(actor_kl_coefficient)
        self._packet_actor = None
        self._actor_packet_steps = []
        self._actor_packet_kl = []
        self._constraint_actor_updates = self._constraint_alpha_updates = 0

    def begin_actor_packet(self):
        if self._actor_packet_steps and self._actor_packet_steps[-1] != 8:
            raise RuntimeError('The preceding paid packet must complete its eight updates')
        self._packet_actor = deepcopy(self.actor).requires_grad_(False)
        self._actor_packet_steps.append(0)
        self._actor_packet_kl = []
        self._reference_update = self.total_updates

    @torch.no_grad()
    def actor_comparison(self, states):
        """Same states and F; deterministic gate crossings are visible in seconds."""
        if self._packet_actor is None:
            raise RuntimeError('Begin the paid packet before comparing its actor')
        current = self.actor.distribution_parameters(states)
        reference = self._packet_actor.distribution_parameters(states)
        result = hurdle_kl(current, reference)
        for name, parameters in (('reference', reference), ('current', current)):
            gate, mean, log_std = parameters
            components = self.actor.soft_action_components_from_parameters(
                *parameters, deterministic_positive=True)
            choose_hold = components['hold_probability'] >= .5
            proposal = torch.where(choose_hold[:, None], components['positive_action'],
                                   components['zero_action'])
            result.update({f'hold_probability_{name}': torch.sigmoid(gate),
                f'positive_mean_{name}': mean, f'positive_log_std_{name}': log_std,
                f'{name}_hold_seconds': self.action_transform.hold_seconds(
                    self.action_transform(states, proposal)).reshape(-1)})
        result['holding_delta_seconds'] = result['current_hold_seconds']-result['reference_hold_seconds']
        result['gate_flip'] = ((result['hold_probability_current'] >= .5)
                              != (result['hold_probability_reference'] >= .5))
        return result

    def _update_sac_actor(self, actor_states, *, temperature_states=None,
                          temperature_action_samples=1, collect_metrics=True):
        if self._packet_actor is None:
            raise RuntimeError('Begin the paid packet before its actor updates')
        if self._actor_packet_steps[-1] >= 8:
            raise RuntimeError('A paid packet contains exactly eight actor updates')
        # Zero is used only by the exact v64 equivalence test, never production.
        if self.actor_kl_coefficient == 0.:
            result = super()._update_sac_actor(actor_states,
                temperature_states=temperature_states,
                temperature_action_samples=temperature_action_samples,
                collect_metrics=collect_metrics)
            self._record_actor_update(None)
            return result
        if temperature_states is None:
            temperature_states = actor_states
        self._last_actor_diagnostics = {}
        self._set_critic_grad(False)
        try:
            # These are the same parameter forward and sole positive draw as v3.
            parameters = self.actor.distribution_parameters(actor_states)
            components = self.actor.soft_action_components_from_parameters(*parameters)
            with torch.no_grad():
                reference = self._packet_actor.distribution_parameters(actor_states)
            kl = hurdle_kl(parameters, reference)
            size = len(actor_states)
            zero_action = self.action_transform(actor_states, components['zero_action'])
            positive_action = self.action_transform(actor_states, components['positive_action'])
            actions = torch.cat((zero_action, positive_action))
            features = self.critic.encode_states(actor_states)
            combined_q = self.critic.forward_features(torch.cat((features, features)), actions)
            zero_q, positive_q = self._reduce_actor_q(combined_q).split(size)
            zero_loss = self.alpha.detach()*components['zero_log_prob']-zero_q
            positive_loss = self.alpha.detach()*components['positive_log_prob']-positive_q
            original_actor_loss = (components['zero_probability']*zero_loss
                + components['hold_probability']*positive_loss).mean()
            actor_loss = original_actor_loss+self.actor_kl_coefficient*kl['total_kl'].mean()
            actor_log_prob = (components['zero_probability']*components['zero_log_prob']
                + components['hold_probability']*components['positive_log_prob'])
            hold_log_prob = components['positive_log_prob']-components['positive_conditional_log_prob']
            gate_entropy = -(components['zero_probability']*components['zero_log_prob']
                + components['hold_probability']*hold_log_prob)
            self._last_actor_diagnostics = {
                'train/hybrid_entropy': (-actor_log_prob).mean().detach(),
                'train/hybrid_gate_entropy': gate_entropy.mean().detach(),
                'train/hybrid_weighted_positive_entropy': (
                    -components['hold_probability']*components['positive_conditional_log_prob']).mean().detach(),
                'train/hold_probability': components['hold_probability'].mean().detach(),
                'train/positive_action_mean': components['positive_action'].mean().detach(),
                'train/executed_positive_action_mean': positive_action.mean().detach(),
                'train/executed_zero_action_mean': zero_action.mean().detach(),
                'train/actor_unconstrained_loss': original_actor_loss.detach(),
                **{f'train/actor_packet_{key}': value.mean().detach() for key, value in kl.items()},
            }
            if self.config.temperature_state_source == 'actor' and temperature_action_samples == 1:
                temperature_log_prob = actor_log_prob.detach()
            else:
                with torch.no_grad():
                    temperature_log_prob = self._temperature_log_probability(
                        temperature_states, temperature_action_samples)
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
        self._record_actor_update(kl)
        return actor_loss, alpha_loss, actor_grad_norm

    def _record_actor_update(self, kl):
        self._actor_packet_steps[-1] += 1
        self._constraint_actor_updates += 1
        self._constraint_alpha_updates += 1
        if kl is not None:
            self._actor_packet_kl.append({key: float(value.detach().mean()) for key, value in kl.items()})

    def packet_actor_audit(self):
        return dict(packet=len(self._actor_packet_steps), coefficient=self.actor_kl_coefficient,
            reference_update=self._reference_update,
            actor_updates=self._actor_packet_steps[-1], alpha_updates=self._actor_packet_steps[-1],
            step_kl=list(self._actor_packet_kl))

    def actor_constraint_audit(self):
        return dict(coefficient=self.actor_kl_coefficient, packets=len(self._actor_packet_steps),
            reference_refreshes=len(self._actor_packet_steps), updates=self._constraint_actor_updates,
            actor_updates=self._constraint_actor_updates, alpha_updates=self._constraint_alpha_updates,
            packet_step_counts=list(self._actor_packet_steps), packet_reference_fixed=True,
            full_hurdle_kl_scope=True, kl_orientation='new||packet_reference', additional_random_draws=0)


def load_learner(seed, *, arm):
    """Exactly v64's resident initializer/replay construction; new actor loss only."""
    torch.set_num_threads(1)
    coefficient = json.loads(Path(__file__).with_name('protocol.json').read_text())['actor_kl_coefficient']
    if coefficient != 1.0:
        raise ValueError('v66 production KL coefficient is fixed at 1.0')
    inputs = resident.INPUTS
    metadata = json.loads((inputs/'history_metadata.json').read_text())
    with np.load(inputs/'history.npz') as archive:
        history = {key: archive[key] for key in (*FIELDS.values(), 'sample_weight')}
    source = load_table(inputs/'source_9101.npz', 4096)
    contract = resident.formal.TransitMDPConfigV7(**metadata['contract'])
    stations = np.asarray(metadata['station_max_by_line'], np.float32)
    initializer = inputs/f'h2oplus_seed{seed}.pt'
    payload = torch.load(initializer, map_location='cpu', weights_only=False)
    if (payload['total_updates'] != 50000 or payload['online_critic_updates'] != 0
            or payload['extra']['source_training_seed'] != seed):
        raise ValueError('Wrong read-only initializer')
    config = replace(TransitTrainerConfig(**payload['source_config']), use_ratio_correction=False)
    if config.batch_size != 256:
        raise ValueError('Actual control requires the original128/64/64 quotas')
    del payload
    replay = FreshOutcomeReplay(history, source, contract, stations, arm=arm, window=285)
    learner = ActorConstrainedTrainer(replay, config=config, device='cpu', seed=seed,
        action_transform=ExecutionAction(contract), actor_kl_coefficient=coefficient)
    learner.load_offline_initializer(initializer)
    resident.formal.set_cpu_optimizer_runtime(learner)
    return learner, replay, contract, stations, initializer
