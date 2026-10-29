"""Explicit execution-memory critic on the unchanged factual replay schedule.

Only the memory learner receives 229+84 critic tensors. Actor, value, entropy
and F always receive the original 229 fields. The zero-memory comparator is the
native learner, exactly equivalent to a permanently zero residual, including its
original diagnostics and optimizer state.
"""
import numpy as np
import torch

from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.learning import (
    FreshOutcomeReplay, FreshOutcomeTrainer, load_learner)
from experiments.mcwm_closed_loop_critic_memory_v32_20261023.critic import ExecutedMemoryCritic
from bus_h2o.mdp_contract_v7 import CANONICAL_STATE_DIM
from SimpleSAC.transit_v7.networks import FEATURE_DIM


MEMORY_DIM = 84
STATE_DIM = CANONICAL_STATE_DIM
ARMS = ("memory", "zero_memory")


def split_states(states):
    if states.ndim != 2 or states.shape[1] != STATE_DIM+MEMORY_DIM:
        raise ValueError("Memory critic states must explicitly contain 229+84 fields")
    return states[:, :STATE_DIM], states[:, STATE_DIM:]


def critic_states(learner, states, memory):
    """Build an explicit critic input; never lookup context by state values."""
    if not isinstance(learner.critic, MemoryQEnsemble):
        return states
    context = torch.as_tensor(memory, dtype=states.dtype, device=states.device)
    if states.shape != (len(context), STATE_DIM) or context.shape[1:] != (MEMORY_DIM,):
        raise ValueError("Each native state needs its own 84-field context")
    return torch.cat((states, context), dim=1)


class MemoryQEnsemble(ExecutedMemoryCritic):
    """The explicit context travels through native critic feature batching."""

    def __init__(self, native):
        super().__init__(native, MEMORY_DIM)
        # The caller already owns an isolated learner. Retain its exact native
        # parameter objects so the existing Adam state needs no remapping.
        self.base = native

    def forward(self, states, actions):
        canonical, memory = split_states(states)
        return super().forward(canonical, actions, memory)

    def forward_subset(self, states, actions, indices):
        canonical, memory = split_states(states)
        return super().forward_subset(canonical, actions, memory, indices)

    def encode_states(self, states):
        canonical, memory = split_states(states)
        return torch.cat((self.base.encode_states(canonical), memory), dim=1)

    def forward_features(self, features, actions):
        return (self.base.forward_features(features[:, :FEATURE_DIM], actions)
                + self._residual(actions, features[:, FEATURE_DIM:]))

    def forward_features_subset(self, features, actions, indices):
        return (self.base.forward_features_subset(features[:, :FEATURE_DIM], actions, indices)
                + self._residual(actions, features[:, FEATURE_DIM:], indices))


class MemoryOutcomeReplay(FreshOutcomeReplay):
    """The original four sampler draws, with aligned context only on fresh rows."""

    def add_target(self, rows):
        for row in rows:
            super().add_target([row])
            for field in ("memory_s", "memory_ns"):
                value = np.asarray(getattr(row, field)["vector"], dtype=np.float32)
                if value.shape != (MEMORY_DIM,):
                    raise ValueError("Expected the unchanged v32 execution context")
                self.recent[-1][field] = value.copy()

    def sample_mixed(self, size, online_fraction, *, rng):
        factual, source = super().sample_mixed(size, online_fraction, rng=rng)
        ids = self.last_sampling["recent"]
        count = len(ids)
        zero = torch.zeros((count, MEMORY_DIM), device=self.device)
        contexts = {field: torch.as_tensor(np.asarray([self.recent[index][field] for index in ids]),
                                          dtype=torch.float32, device=self.device)
                    for field in ("memory_s", "memory_ns")}
        for key, field in (("observations", "memory_s"), ("next_observations", "memory_ns")):
            aligned = torch.cat((zero, contexts[field]))
            factual[key] = torch.cat((factual[key], aligned), dim=1)
            source[key] = torch.cat((source[key], torch.zeros(
                (len(source[key]), MEMORY_DIM), dtype=source[key].dtype, device=self.device)), dim=1)
        # These original actor/value current observations remain 229 fields.
        self.last_covariates["memory_s"] = torch.cat((zero, contexts["memory_s"]))
        return factual, source


class MemoryOutcomeTrainer(FreshOutcomeTrainer):
    @torch.no_grad()
    def _target(self, batch, *, backup):
        canonical, memory = split_states(batch["next_observations"])
        explicit = dict(batch, next_observations=canonical, next_memory=memory)
        # Retain DiagnosticTrainer's actual-batch target statistics. Native V
        # sees canonical ns; only the SAC target-Q branch reads next_memory.
        return super()._target(explicit, backup=backup)

    @torch.no_grad()
    def _sac_bootstrap(self, batch):
        states, memory = batch["next_observations"], batch["next_memory"]
        components = self.actor.soft_action_components(states)
        size = len(states)
        actions = torch.cat((self.action_transform(states, components["zero_action"]),
                             self.action_transform(states, components["positive_action"])))
        chosen = self._sample_target_subset()
        features = self.target_critic.encode_states(critic_states(self, states, memory))
        repeated = torch.cat((features, features))
        values = self.target_critic.forward_features_subset(repeated, actions, chosen)
        zero_q, positive_q = values.split(size, dim=1)
        zero_value, positive_value = zero_q.min(0).values, positive_q.min(0).values
        if self.config.backup_entropy:
            zero_value -= self.alpha.detach()*components["zero_log_prob"]
            positive_value -= self.alpha.detach()*components["positive_log_prob"]
        return components["zero_probability"]*zero_value + components["hold_probability"]*positive_value

    def _iql_value_terms(self, batch):
        current = self.replay_buffer.last_covariates
        states = current["observations"]
        with torch.no_grad():
            target_q = self.target_critic(critic_states(self, states, current["memory_s"]),
                                          current["actions"]).min(dim=0).values
        value = self.value(states)
        advantage = target_q-value
        expectile = self.config.iql_expectile
        weights = torch.where(advantage >= 0., torch.full_like(advantage, expectile),
                              torch.full_like(advantage, 1.-expectile))
        value_loss = (weights*advantage.square()).mean()
        actor_weights = torch.exp(advantage.detach()*self.config.iql_advantage_temperature).clamp_max(
            self.config.iql_max_weight)
        return value_loss, advantage.detach(), actor_weights

    def _actor_states(self, offline_batch, online_batch):
        current = self.replay_buffer.last_covariates
        # The native scheduler carries context with its matching state across
        # delayed/aggregated actor updates. The actor itself receives only s.
        return torch.cat((critic_states(self, current["observations"], current["memory_s"]),
                          online_batch["observations"]))

    def _temperature_states(self, offline_batch, online_batch):
        return self._actor_states(offline_batch, online_batch)[:, :STATE_DIM]

    def _update_sac_actor(self, actor_states, *, temperature_states=None,
                          temperature_action_samples=1, collect_metrics=True):
        states, memory = split_states(actor_states)
        if temperature_states is None:
            temperature_states = states
        self._last_actor_diagnostics = {}
        self._set_critic_grad(False)
        try:
            components = self.actor.soft_action_components(states)
            size = len(states)
            zero_action = self.action_transform(states, components["zero_action"])
            positive_action = self.action_transform(states, components["positive_action"])
            actions = torch.cat((zero_action, positive_action))
            features = self.critic.encode_states(critic_states(self, states, memory))
            combined_q = self.critic.forward_features(torch.cat((features, features)), actions)
            zero_q, positive_q = self._reduce_actor_q(combined_q).split(size)
            zero_loss = self.alpha.detach()*components["zero_log_prob"]-zero_q
            positive_loss = self.alpha.detach()*components["positive_log_prob"]-positive_q
            actor_loss = (components["zero_probability"]*zero_loss
                          + components["hold_probability"]*positive_loss).mean()
            actor_log_prob = (components["zero_probability"]*components["zero_log_prob"]
                             + components["hold_probability"]*components["positive_log_prob"])
            hold_log_prob = components["positive_log_prob"]-components["positive_conditional_log_prob"]
            gate_entropy = -(components["zero_probability"]*components["zero_log_prob"]
                             + components["hold_probability"]*hold_log_prob)
            self._last_actor_diagnostics = {
                "train/hybrid_entropy": (-actor_log_prob).mean().detach(),
                "train/hybrid_gate_entropy": gate_entropy.mean().detach(),
                "train/hybrid_weighted_positive_entropy": (
                    -components["hold_probability"]*components["positive_conditional_log_prob"]).mean().detach(),
                "train/hold_probability": components["hold_probability"].mean().detach(),
                "train/positive_action_mean": components["positive_action"].mean().detach(),
                "train/executed_positive_action_mean": positive_action.mean().detach(),
                "train/executed_zero_action_mean": zero_action.mean().detach(),
            }
            if self.config.temperature_state_source == "actor" and temperature_action_samples == 1:
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
        return actor_loss, alpha_loss, actor_grad_norm


def attach_memory(learner):
    """Attach to an isolated fresh learner without replacing any old Adam state."""
    if learner.replay_buffer.arm != "fresh" or learner.replay_buffer.recent:
        raise ValueError("Attach before admitting factual feedback")
    if learner.config.online_cql_alpha != 0 or learner.world is not None:
        raise ValueError("v33 retains the resident real-only, non-CQL update")
    learner.__class__ = MemoryOutcomeTrainer
    learner.replay_buffer.__class__ = MemoryOutcomeReplay
    learner.critic = MemoryQEnsemble(learner.critic)
    learner.target_critic = MemoryQEnsemble(learner.target_critic)
    learner.target_critic.eval()
    for parameter in learner.target_critic.parameters():
        parameter.requires_grad_(False)
    learner.critic_optimizer.param_groups[0]["params"].extend(
        (learner.critic.level_weight, learner.critic.action_weight))
    return learner


def make_learner(seed, *, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    learner, replay, contract, stations, initializer = load_learner(seed, arm="fresh")
    if arm == "memory":
        attach_memory(learner)
    learner.memory_arm = arm
    return learner, replay, contract, stations, initializer
