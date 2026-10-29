"""One target change: mean eight paired positive Q-minus-entropy samples.

The native target, diagnostics and mixed-backup assembly remain inherited.
Only the memory trainer's SAC bootstrap is overridden. Its first draw retains
the native actor/subset/2B-critic order; seven extra draws use a persistent,
separately seeded CPU torch stream that never advances training randomness.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import time

import torch

from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.learning import (
    MemoryOutcomeTrainer, critic_states,
)
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _restore, _rng


class Mean8OutcomeTrainer(MemoryOutcomeTrainer):
    """Top-level pickleable class; model/Adam objects are retained on install."""

    @torch.no_grad()
    def _sac_bootstrap(self, batch):
        started = time.perf_counter()
        states, memory = batch['next_observations'], batch['next_memory']
        # Keep these first-draw operations in the original native order and
        # original batch shapes. The extra stream starts only after this path.
        components = self.actor.soft_action_components(states)
        size = len(states)
        actions = torch.cat((self.action_transform(states, components['zero_action']),
                             self.action_transform(states, components['positive_action'])))
        chosen = self._sample_target_subset()
        features = self.target_critic.encode_states(critic_states(self, states, memory))
        repeated = torch.cat((features, features))
        values = self.target_critic.forward_features_subset(repeated, actions, chosen)
        zero_q, positive_q = values.split(size, dim=1)
        zero_value, positive_value = zero_q.min(0).values, positive_q.min(0).values
        if self.config.backup_entropy:
            zero_value -= self.alpha.detach()*components['zero_log_prob']
            positive_value -= self.alpha.detach()*components['positive_log_prob']
        native_sac = (components['zero_probability']*zero_value
                      + components['hold_probability']*positive_value)

        positive_joint = [positive_value]
        live_torch = torch.get_rng_state()
        extra_started = time.perf_counter()
        try:
            torch.set_rng_state(self.mean8_extra_rng_state)
            for _ in range(7):
                extra = self.actor.soft_action_components(states)
                extra_actions = torch.cat((self.action_transform(states, extra['zero_action']),
                    self.action_transform(states, extra['positive_action'])))
                extra_values = self.target_critic.forward_features_subset(repeated, extra_actions, chosen)
                # Retain the native two-action batch even though its zero slot
                # is not used. The original first-draw zero value is fixed.
                joint = extra_values[:, size:].min(0).values
                if self.config.backup_entropy:
                    joint -= self.alpha.detach()*extra['positive_log_prob']
                positive_joint.append(joint)
            self.mean8_extra_rng_state = torch.get_rng_state().clone()
        finally:
            torch.set_rng_state(live_torch)
        extra_seconds = time.perf_counter()-extra_started
        joint_mean = torch.stack(positive_joint).mean(0)
        result = (components['zero_probability']*zero_value
                  + components['hold_probability']*joint_mean)
        stats = self.mean8_compute
        stats['calls'] += 1
        stats['batch_rows'] += size
        stats['native_actor_forwards'] += 1
        stats['native_q_forwards'] += 1
        stats['native_q_rows'] += 2*size
        stats['extra_positive_samples'] += 7*size
        stats['extra_actor_forwards'] += 7
        stats['extra_q_forwards'] += 7
        stats['extra_q_rows'] += 14*size
        stats['extra_seconds'] += extra_seconds
        stats['total_sac_seconds'] += time.perf_counter()-started
        self.mean8_last = dict(native_sac=native_sac.detach().clone(),
            mean8_sac=result.detach().clone(), zero_value=zero_value.detach().clone(),
            zero_probability=components['zero_probability'].detach().clone(),
            hold_probability=components['hold_probability'].detach().clone(),
            positive_joint=torch.stack(positive_joint).detach().clone(),
            positive_joint_mean=joint_mean.detach().clone(), target_subset=chosen.detach().clone(),
            batch_rows=size, native_q_batch_rows=2*size)
        return result


def install_mean8(learner, *, seed=476202):
    """Install before replay; M1 must simply keep its original learner class."""
    if learner.__class__ is not MemoryOutcomeTrainer or learner.device.type != 'cpu':
        raise RuntimeError('The v47 candidate attaches to the existing CPU memory trainer')
    config = learner.config
    if (learner.world is not None or config.mixed_backup_lambda != .1
            or config.ensemble_size != 2 or config.target_subset_size != 2
            or config.target_subset_replacement):
        raise RuntimeError('The unchanged native no-world two-of-two mixed backup is required')
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    learner.mean8_extra_seed = int(seed)
    learner.mean8_extra_rng_state = generator.get_state().clone()
    learner.mean8_compute = dict(calls=0, batch_rows=0, native_actor_forwards=0,
        native_q_forwards=0, native_q_rows=0, extra_positive_samples=0,
        extra_actor_forwards=0, extra_q_forwards=0, extra_q_rows=0,
        extra_seconds=0., total_sac_seconds=0.)
    learner.mean8_last = None
    learner.__class__ = Mean8OutcomeTrainer
    return learner


def mean8_stats(learner):
    return dict(samples=8, extra_seed=learner.mean8_extra_seed,
        persistent_extra_stream=True, native_first_draw_order_preserved=True,
        extra_target_head_draws=0, extra_V_forwards=0,
        **deepcopy(learner.mean8_compute))


def mean8_state(learner):
    """Copy the candidate-only stream and counters for frozen diagnostic reads."""
    return dict(extra_rng=learner.mean8_extra_rng_state.clone(),
        compute=deepcopy(learner.mean8_compute), last=deepcopy(learner.mean8_last))


def restore_mean8_state(learner, state):
    learner.mean8_extra_rng_state = state['extra_rng'].clone()
    learner.mean8_compute = deepcopy(state['compute'])
    learner.mean8_last = deepcopy(state['last'])


@contextmanager
def native_scoring(learner):
    """Use the old native M1 evaluator on either trained model, without draws.

    This context is for frozen scoring only. Model parameters stay as trained;
    its target-estimator class is temporarily native so the existing v46 M1
    audit remains well-defined. Global RNG is restored; the extra stream and
    counters are not touched because its override is never entered.
    """
    original, rng = learner.__class__, _rng()
    if original is Mean8OutcomeTrainer:
        learner.__class__ = MemoryOutcomeTrainer
    try:
        yield learner
    finally:
        learner.__class__ = original
        _restore(rng)
