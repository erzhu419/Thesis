"""Terminal predecessor supervision using only examples still in FIFO replay."""

from collections import deque
import numpy as np
import torch
from csbapr.agent import CSBAPRAgent


class TerminalBoundaryAgent(CSBAPRAgent):
    def __init__(self, state_dim, action_dim, config, boundary_weight):
        super().__init__(state_dim, action_dim, config)
        self.boundary_weight = boundary_weight
        self.terminal_examples = deque()
        self.stored_count = 0
        self.last_boundary_loss = self.last_base_q_loss = 0.
        self.boundary_updates = 0

    def store_transition(self, state, action, reward, next_state, done):
        super().store_transition(state, action, reward, next_state, done)
        self.stored_count += 1
        if done:
            self.terminal_examples.append((self.stored_count,
                np.asarray(state, dtype=np.float32).copy(),
                np.asarray(action, dtype=np.float32).copy(), float(reward)))
        oldest = self.stored_count - self.replay_buffer.capacity
        while self.terminal_examples and self.terminal_examples[0][0] <= oldest:
            self.terminal_examples.popleft()

    def _compute_q_loss_bapr_fallback(self, *args):
        result = super()._compute_q_loss_bapr_fallback(*args)
        self.last_base_q_loss = float(result[0].detach())
        self.last_boundary_loss = 0.
        if not self.terminal_examples:
            return result
        states = torch.as_tensor(np.stack([r[1] for r in self.terminal_examples]), device=self.device)
        actions = torch.as_tensor(np.stack([r[2] for r in self.terminal_examples]), device=self.device)
        rewards = torch.tensor([r[3] for r in self.terminal_examples], dtype=torch.float32, device=self.device)
        predicted = self.critic(states, actions)
        # The last executed decision has target r_T, not zero and not a bootstrap.
        boundary_loss = (predicted - rewards.unsqueeze(0)).square().mean()
        self.last_boundary_loss = float(boundary_loss.detach())
        self.boundary_updates += 1
        return (result[0] + self.boundary_weight * boundary_loss, *result[1:])

    def update(self, batch_size=None, recent_rollout=None):
        result = super().update(batch_size=batch_size, recent_rollout=recent_rollout)
        if result is not None:
            result.update(q_loss_base=self.last_base_q_loss,
                          terminal_boundary_loss=self.last_boundary_loss,
                          terminal_boundary_examples=len(self.terminal_examples))
        return result
