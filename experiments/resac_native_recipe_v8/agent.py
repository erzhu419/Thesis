"""Restore the original bus learner's optimization and uncertainty channels."""

from dataclasses import replace
from pathlib import Path
import random
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bus_algorithm_matrix_v6"))
from resac_adapter import Agent as AdaptedAgent, Replay as AdaptedReplay


class Replay(AdaptedReplay):
    def sample(self, batch_size):
        indices = random.sample(range(self.size), batch_size)
        return (self.states[indices], self.actions[indices], self.rewards[indices],
                self.next_states[indices], self.dones[indices])


class Agent(AdaptedAgent):
    def __init__(self, env, arm, mode, seed, protocol, device):
        settings = dict(protocol, ensemble_size=2 if arm == "sac" else 10)
        parent_arm = "ensemble_sac_control" if arm in ("sac", "aleatoric_only") else "resac"
        super().__init__(env, parent_arm, mode, seed, settings)
        self.config = replace(self.config, training_freq=protocol["training_freq"],
            weight_reg=0. if arm in ("sac", "epistemic_only") else .01,
            beta_bc=.001 if arm == "resac" else 0.,
            beta=-2. if arm in ("resac", "epistemic_only") else 0.,
            beta_ood=.01 if arm in ("resac", "epistemic_only") else 0.)
        learner = self.learner
        learner.config, learner.device = self.config, torch.device(device)
        # The original PyTorch actor and online critic share one embedding object.
        learner.policy.embedding = learner.critic.embedding
        learner.critic.to(learner.device)
        learner.target_critic.to(learner.device)
        learner.policy.to(learner.device)
        learner.log_alpha = torch.zeros(1, device=learner.device, requires_grad=True)
        learner.critic_optimizer = torch.optim.Adam(learner.critic.parameters(), lr=self.config.learning_rate)
        learner.policy_optimizer = torch.optim.Adam(learner.policy.parameters(), lr=self.config.learning_rate)
        learner.alpha_optimizer = torch.optim.Adam([learner.log_alpha], lr=self.config.learning_rate)
        self.replay_buffer = Replay(self.config.replay_capacity)


def check(agent):
    assert agent.actor.embedding is agent.learner.critic.embedding
    assert agent.actor.embedding is not agent.learner.target_critic.embedding
    for optimizer in (agent.learner.critic_optimizer, agent.learner.policy_optimizer,
                      agent.learner.alpha_optimizer):
        assert all(group["lr"] == 1e-5 for group in optimizer.param_groups)
    assert agent.config.batch_size == 2048 and agent.config.training_freq == 5
    assert np.isfinite(agent.learner.alpha)
