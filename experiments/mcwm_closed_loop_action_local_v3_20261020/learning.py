"""Executed-action Q objectives, imagination and Bellman trust for isolated v3."""

# Import the existing bootstrap first: it loads the old top-level learning and
# H2O modules without letting this package module shadow either one.
from experiments.mcwm_closed_loop_diagnostic_v2_20261020 import bootstrap as _bootstrap
from experiments.mcwm_closed_loop_diagnostic_v2_20261020.td_diagnostics import DiagnosticTrainer
from experiments.mcwm_closed_loop_v1.learning import ModelAugmentation, learned_rows, row_table, targets

import numpy as np
import torch


class ActionConsistentWorldMixin:
    """Compose ahead of DiagnosticWorld; set action_transform in its constructor."""

    def update_trust(self, actor, size=64):
        rows = learned_rows(row_table(self.replay.recent))
        if not len(rows["s"]):
            return
        y = targets({key: rows[key] for key in ("s", "a", "ns", "r", "dt")},
                    self.stations, self.device)
        from experiments.mcwm_closed_loop_v1.learning import losses
        with torch.no_grad():
            error = losses(self.predict(y["s"], y["a"]), y, self.norm).mean(1)
        ids = self.rng.integers(len(error), size=size)
        ns = torch.as_tensor(rows["ns"][ids], device=self.device)
        return self.trust.update(y["s"][ids], y["a"][ids], ns,
            torch.zeros(size, device=self.device), error[ids],
            lambda states: self.action_transform(states, actor(states, deterministic=True)[0]), n_iters=1)

    @torch.no_grad()
    def generate(self, actor, size):
        rows = learned_rows(row_table(self.replay.recent))
        eligible = np.flatnonzero(rows["s"][:, 0] < .8)
        if not len(eligible):
            return None
        ids = self.rng.choice(eligible, size=size)
        s = torch.as_tensor(rows["s"][ids], device=self.device)
        a = self.action_transform(s, actor(s, deterministic=True)[0])
        ns, reward, dt = ModelAugmentation.decode(self, s, self.predict(s, a))
        batch = dict(observations=s, actions=a, next_observations=ns, rewards=reward,
            delta_t_seconds=dt, discounts=torch.ones(size, device=self.device),
            terminals=torch.zeros(size, device=self.device))
        return batch, self.trust.weight(s, a)


class ActionConsistentTrainer(DiagnosticTrainer):
    """SAC in latent proposal space with Q(s,F(s,u)) and executed-action TD.

    Historical/recent observed actions already are physical actions.  They are
    never transformed again.  The IQL value branch remains the shared frozen
    prefix continuation; SAC's policy-dependent branch is composed with F.
    """

    def __init__(self, *args, action_transform, **kwargs):
        self.action_transform = action_transform
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def _sac_bootstrap(self, batch):
        states = batch["next_observations"]
        components = self.actor.soft_action_components(states)
        size = len(states)
        actions = torch.cat((self.action_transform(states, components["zero_action"]),
                             self.action_transform(states, components["positive_action"])))
        chosen = self._sample_target_subset()
        features = self.target_critic.encode_states(states)
        repeated = torch.cat((features, features))
        values = self.target_critic.forward_features_subset(repeated, actions, chosen)
        zero_q, positive_q = values.split(size, dim=1)
        zero_value, positive_value = zero_q.min(0).values, positive_q.min(0).values
        if self.config.backup_entropy:
            zero_value -= self.alpha.detach() * components["zero_log_prob"]
            positive_value -= self.alpha.detach() * components["positive_log_prob"]
        return components["zero_probability"]*zero_value + components["hold_probability"]*positive_value

    def _update_sac_actor(self, actor_states, *, temperature_states=None,
                          temperature_action_samples=1, collect_metrics=True):
        if temperature_states is None:
            temperature_states = actor_states
        self._last_actor_diagnostics = {}
        self._set_critic_grad(False)
        try:
            components = self.actor.soft_action_components(actor_states)
            size = len(actor_states)
            zero_action = self.action_transform(actor_states, components["zero_action"])
            positive_action = self.action_transform(actor_states, components["positive_action"])
            actions = torch.cat((zero_action, positive_action))
            features = self.critic.encode_states(actor_states)
            combined_q = self.critic.forward_features(torch.cat((features, features)), actions)
            zero_q, positive_q = self._reduce_actor_q(combined_q).split(size)
            zero_loss = self.alpha.detach()*components["zero_log_prob"] - zero_q
            positive_loss = self.alpha.detach()*components["positive_log_prob"] - positive_q
            actor_loss = (components["zero_probability"]*zero_loss
                          + components["hold_probability"]*positive_loss).mean()
            actor_log_prob = (components["zero_probability"]*components["zero_log_prob"]
                             + components["hold_probability"]*components["positive_log_prob"])
            hold_log_prob = components["positive_log_prob"] - components["positive_conditional_log_prob"]
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
                    temperature_log_prob = self._temperature_log_probability(temperature_states,
                                                                            temperature_action_samples)
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
