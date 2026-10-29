"""Synchronized M8 backup and once-bounded physical credit on v53 real rows.

The frozen issue actor supplies the SAC evaluator only. V, alpha, target Q,
actor updates and all optimizer objects remain live. The trace adds future
residuals only to the 271 newly collected, consecutive response transitions.
"""
from __future__ import annotations

from copy import deepcopy
import math
import time

import numpy as np
import torch

from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.learning import MemoryOutcomeTrainer
from experiments.mcwm_closed_loop_joint_shadow_v47_20261024.mean_target import (
    Mean8OutcomeTrainer, install_mean8, mean8_state, restore_mean8_state, mean8_stats,
)

CAPACITY = 271
ETA = 9. / (CAPACITY - 1)
TIME_CONSTANT = 300.


def _get(row, field):
    return row[field] if isinstance(row, dict) else getattr(row, field)


def _rng(seed):
    return torch.Generator(device='cpu').manual_seed(int(seed)).get_state().clone()


class ReferenceSyncMean8Trainer(Mean8OutcomeTrainer):
    """Native target dispatcher with the frozen actor and v47 M8 estimator."""

    @torch.no_grad()
    def _sac_bootstrap(self, batch):
        actor, transform = self.actor, self.action_transform
        self.actor, self.action_transform = self.sync_actor, self.sync_transform
        try:
            return Mean8OutcomeTrainer._sac_bootstrap(self, batch)
        finally:
            self.actor, self.action_transform = actor, transform


class SyncPhysicalOutcomeTrainer(ReferenceSyncMean8Trainer):
    def _sample_target_subset(self):
        if self._future_subset is not None:
            return self._future_subset
        return super()._sample_target_subset()

    def set_lambda(self, value):
        value = float(value)
        if not math.isfinite(value) or not 0. <= value <= 1.:
            raise ValueError('The declared trace lambda must lie in [0, 1]')
        self.lambda_trace = value

    def _continuous_ids(self, batch):
        """Recover ids from the unchanged actual 64/64/128 sampler ordering."""
        count = len(batch['rewards'])
        ids = torch.full((count,), -1, dtype=torch.long, device=self.device)
        if 'continuous_ids' in batch:  # Direct frozen target probes, not replay.
            ids = batch['continuous_ids'].to(device=self.device, dtype=torch.long)
        else:
            sampling = self.replay_buffer.last_sampling
            if sampling is None or count != 256:
                raise RuntimeError('Trace training requires the actual native mixed batch')
            recent = torch.as_tensor(sampling['recent'], dtype=torch.long, device=self.device)
            history = len(sampling['history'])
            if (history, len(recent), len(sampling['source'])) != (64, 64, 128):
                raise RuntimeError('The 2:1:1 factual sampler changed')
            ids[history:history+len(recent)] = torch.where(recent >= 14, recent-14, -1)
        selected = ids >= 0
        if bool(selected.any()):
            actual = ids[selected]
            if int(actual.max()) >= CAPACITY:
                raise RuntimeError('Only the admitted 271 continuous rows support credit')
            chain = self.physical_batch
            for field in ('observations', 'next_observations', 'actions', 'rewards',
                          'delta_t_seconds', 'terminals'):
                if not torch.equal(batch[field][selected], chain[field][actual]):
                    raise RuntimeError('Trace row identity or causal memory mismatch: '+field)
        return ids

    @torch.no_grad()
    def _future_targets(self, chosen):
        """Use private persistent streams; do not consume native or M8 RNG."""
        started = time.perf_counter()
        native_rng, native_mean = torch.get_rng_state(), mean8_state(self)
        try:
            torch.set_rng_state(self.future_rng)
            self.mean8_extra_rng_state = self.future_extra_rng
            self._future_subset = chosen
            self._collect_td, collect = False, self._collect_td
            try:
                native = ReferenceSyncMean8Trainer._target(self, self.physical_batch,
                    backup='h2oplus')
            finally:
                self._collect_td = collect
            future_last = deepcopy(self.mean8_last)
            self.future_rng = torch.get_rng_state().clone()
            self.future_extra_rng = self.mean8_extra_rng_state.clone()
            self.target_compute['future_extra_positive_samples'] += 7 * CAPACITY
            self.target_compute['future_actor_forwards'] += 8
            self.target_compute['future_q_forwards'] += 9  # Eight SAC plus matching Q.
            state = self.physical_batch['next_observations']
            reward = self.config.reward_scale * self.physical_batch['rewards']
            discount = self.physical_batch['discounts'] * (1.-self.physical_batch['terminals'])
            bootstrap = discount * (.1*self.value(state[:, :229])+.9*future_last['mean8_sac'])
            if not torch.equal(native, reward+bootstrap):
                raise RuntimeError('Future mixed backup no longer matches native assembly')
            actions = self.physical_batch['actions']
            # Exactly 270 actual successor actions exist inside the admitted
            # chain. The last real transition retains bootstrap and needs no Qbar.
            next_actions = actions[1:]
            features = self.target_critic.encode_states(state[:-1])
            qs = self.target_critic.forward_features_subset(torch.cat((features, features)),
                torch.cat((next_actions, next_actions)), chosen)
            matching_q = qs[:, :CAPACITY-1].min(0).values
            physical = native.clone()
            future_reward = torch.zeros_like(native)
            future_bootstrap = torch.zeros_like(native)
            future_minus_q = torch.zeros_like(native)
            future_mass = torch.zeros_like(native)
            terminal_weight = torch.ones_like(native)
            for index in range(CAPACITY-2, -1, -1):
                coefficient = self.physical_coefficients[index]
                terminal_weight[index] = coefficient*terminal_weight[index+1]
                if coefficient != 0:
                    physical[index] = native[index] + coefficient*(
                        physical[index+1]-matching_q[index])
                    future_reward[index] = coefficient*(reward[index+1]+future_reward[index+1])
                    future_bootstrap[index] = coefficient*(bootstrap[index+1]+future_bootstrap[index+1])
                    future_minus_q[index] = coefficient*(-matching_q[index]+future_minus_q[index+1])
                    future_mass[index] = coefficient*(1.+future_mass[index+1])
            self.target_compute['future_calls'] += 1
            return dict(native=native, physical=physical, future_residual=physical-native,
                future_reward=future_reward, future_bootstrap=future_bootstrap,
                future_minus_matching_q=future_minus_q,
                future_reach_mass=future_mass, terminal_weight=terminal_weight,
                reward=reward, bootstrap=bootstrap, matching_q=matching_q,
                target_subset=chosen.clone())
        finally:
            self._future_subset = None
            torch.set_rng_state(native_rng)
            restore_mean8_state(self, native_mean)
            self.target_compute['future_seconds'] += time.perf_counter()-started

    @torch.no_grad()
    def _target(self, batch, *, backup):
        # This is the identical native dispatcher used by the reference arm.
        native = ReferenceSyncMean8Trainer._target(self, batch, backup=backup)
        if backup != 'h2oplus':
            raise RuntimeError('v54 preserves the declared mixed online backup')
        ids = self._continuous_ids(batch)
        selected = ids >= 0
        self.target_compute['target_calls'] += 1
        self.target_compute['supported_sampled_rows'] += int(selected.sum())
        record = dict(lambda_trace=self.lambda_trace, native=native.clone(),
            target=native.clone(), continuous_ids=ids.clone(), future=None)
        # Return the native tensor itself: no extra draw, reconstruction or blend.
        if self.lambda_trace == 0. or not bool(selected.any()):
            self.target_last = record
            return native
        future = self._future_targets(self.mean8_last['target_subset'])
        scale = native.new_tensor(self.lambda_trace) * native.new_tensor(ETA)
        result = native.clone()
        result[selected] = native[selected] + scale*future['future_residual'][ids[selected]]
        record.update(target=result.clone(), future={key: value.clone() for key, value in future.items()})
        self.target_last = record
        self.target_compute['trace_corrected_rows'] += int(selected.sum())
        correction = result[selected]-native[selected]
        self.target_compute['correction_sum'] += float(correction.double().sum())
        self.target_compute['correction_abs_sum'] += float(correction.double().abs().sum())
        self.target_compute['correction_max_abs'] = max(self.target_compute['correction_max_abs'],
            float(correction.abs().max()))
        parts = {}
        for label, field in (('future_reward', 'future_reward'),
                             ('future_bootstrap', 'future_bootstrap'),
                             ('future_minus_matching_q', 'future_minus_matching_q')):
            contribution = scale*future[field][ids[selected]]
            parts[label] = contribution
            self.target_compute[label+'_sum'] += float(contribution.double().sum())
            self.target_compute[label+'_abs_sum'] += float(contribution.double().abs().sum())
        # Includes recursive, subtraction, scaling and final-addition float32
        # rounding; do not attribute their combined residual to physical reward.
        rounding = correction.double()-sum(value.double() for value in parts.values())
        parts['rounding'] = rounding
        record['correction_components'] = parts
        self.target_compute['decomposition_rounding_sum'] += float(rounding.sum())
        self.target_compute['decomposition_rounding_abs_sum'] += float(rounding.abs().sum())
        self.target_compute['decomposition_rounding_max_abs'] = max(
            self.target_compute['decomposition_rounding_max_abs'], float(rounding.abs().max()))
        mass = scale*future['future_reach_mass'][ids[selected]]
        tail = scale*future['terminal_weight'][ids[selected]]
        tail = torch.where(ids[selected] == CAPACITY-1, 0., tail)
        self.target_compute['bounded_future_mass_max'] = max(
            self.target_compute['bounded_future_mass_max'], float(mass.max()))
        self.target_compute['tail_bootstrap_weight_sum'] += float(tail.double().sum())
        if self._collect_td:
            from experiments.mcwm_closed_loop_diagnostic_v2_20261020.td_diagnostics import distribution
            partition = self._model_batches if self._making_model else self._real_batches
            partition[-1]['native_one_step_target_q'] = partition[-1]['target_q']
            partition[-1]['target_q'] = distribution(result)
        return result


def install_reference_sync_m8(learner, evaluator, *, seed=546202):
    """Retain model/optimizer objects; use native _target without trace dispatch."""
    install_mean8(learner, seed=seed)
    learner.sync_actor = deepcopy(evaluator.actor).eval()
    learner.sync_transform = deepcopy(evaluator.action_transform)
    for parameter in learner.sync_actor.parameters():
        parameter.requires_grad_(False)
    learner.__class__ = ReferenceSyncMean8Trainer
    return learner


def install_sync_target(learner, evaluator, continuous_rows, execution_c, *,
                        lambda_trace=0., seed=546202):
    """execution_c[i] is the measured c at successor issue i+1; root c excluded."""
    if len(continuous_rows) != CAPACITY or len(execution_c) != CAPACITY-1:
        raise ValueError('v54 requires 271 continuous TDs and 270 genuine successor ratios')
    for left, right in zip(continuous_rows, continuous_rows[1:]):
        if (_get(left, 'completed_at') != _get(right, 'started_at')
                or not np.array_equal(_get(left, 'ns'), _get(right, 's'))
                or _get(left, 'memory_ns') != _get(right, 'memory_s')):
            raise ValueError('Physical credit requires the real adjacent state and memory chain')
    install_reference_sync_m8(learner, evaluator, seed=seed)
    learner.__class__ = SyncPhysicalOutcomeTrainer
    tensor = lambda values: torch.as_tensor(np.asarray(values), dtype=torch.float32, device=learner.device)
    chain = {}
    for key, field in (('observations', 's'), ('next_observations', 'ns')):
        states = tensor([_get(row, field) for row in continuous_rows])
        memory = tensor([_get(row, 'memory_'+field)['vector'] for row in continuous_rows])
        chain[key] = torch.cat((states, memory), dim=1)
    chain.update(actions=tensor([_get(row, 'a') for row in continuous_rows]).reshape(CAPACITY, 1),
        rewards=tensor([_get(row, 'r') for row in continuous_rows]).reshape(CAPACITY),
        delta_t_seconds=tensor([_get(row, 'dt') for row in continuous_rows]).reshape(CAPACITY),
        terminals=tensor([_get(row, 'terminal') for row in continuous_rows]).reshape(CAPACITY),
        discounts=torch.ones(CAPACITY, device=learner.device))
    coefficients = []
    for index, c in enumerate(execution_c):
        dt = float(_get(continuous_rows[index], 'dt'))
        if not math.isfinite(dt) or dt < 0:
            raise ValueError('The real physical duration must be nonnegative')
        if c is not None and (not math.isfinite(float(c)) or not 0 <= float(c) <= 1):
            raise ValueError('Execution c must be the measured clipped ratio')
        coefficient = (0. if c is None or _get(continuous_rows[index], 'terminal') else
            float(np.float32(np.float32(math.exp(-dt/TIME_CONSTANT))*np.float32(c))))
        coefficients.append(coefficient)
    learner.physical_batch = chain
    learner.physical_coefficients = tensor(coefficients)
    learner.future_rng, learner.future_extra_rng = _rng(seed+1), _rng(seed+2)
    learner._future_subset = None
    learner.target_last = None
    learner.target_compute = dict(target_calls=0, supported_sampled_rows=0,
        trace_corrected_rows=0, future_calls=0, future_actor_forwards=0,
        future_q_forwards=0, future_extra_positive_samples=0, future_seconds=0.,
        correction_sum=0., correction_abs_sum=0., correction_max_abs=0.,
        future_reward_sum=0., future_reward_abs_sum=0.,
        future_bootstrap_sum=0., future_bootstrap_abs_sum=0.,
        future_minus_matching_q_sum=0., future_minus_matching_q_abs_sum=0.,
        decomposition_rounding_sum=0., decomposition_rounding_abs_sum=0.,
        decomposition_rounding_max_abs=0., bounded_future_mass_max=0.,
        tail_bootstrap_weight_sum=0.)
    learner.set_lambda(lambda_trace)
    return learner


def set_lambda(learner, value):
    learner.set_lambda(value)


def target_state(learner):
    state = dict(mean8=mean8_state(learner))
    if isinstance(learner, SyncPhysicalOutcomeTrainer):
        state.update(future_rng=learner.future_rng.clone(),
            future_extra_rng=learner.future_extra_rng.clone(), lambda_trace=learner.lambda_trace,
            compute=deepcopy(learner.target_compute), last=deepcopy(learner.target_last))
    return state


def restore_target_state(learner, state):
    restore_mean8_state(learner, state['mean8'])
    if isinstance(learner, SyncPhysicalOutcomeTrainer):
        learner.future_rng = state['future_rng'].clone()
        learner.future_extra_rng = state['future_extra_rng'].clone()
        learner.lambda_trace = state['lambda_trace']
        learner.target_compute, learner.target_last = deepcopy(state['compute']), deepcopy(state['last'])


def target_stats(learner):
    result = dict(mean8=mean8_stats(learner), frozen_evaluator=True,
        current_V_alpha_Q=True, root_native_first_draw=True)
    if isinstance(learner, SyncPhysicalOutcomeTrainer):
        result.update(lambda_trace=learner.lambda_trace, capacity=CAPACITY, eta=ETA,
            physical_time_constant=TIME_CONSTANT, **deepcopy(learner.target_compute))
    return result
