"""Explicit critic candidate; preflight probes do not train or control with it.

The collector supplies causal execution memory, with all-zero memory for rows
without observations. The candidate adds a per-head level and an executed-action
interaction to a private copy of the native critic. It leaves actor/value inputs
unchanged. Both residual weights start at zero without drawing random numbers.
"""
from copy import deepcopy

import torch
from torch import nn


class ExecutedMemoryCritic(nn.Module):
    """Q_k(s,a,m) = Q_native,k(s,a) + w0_k m + ((a+1)/2) w1_k m.

    ``a`` is the already executed normalized action. No F is applied here.
    Context is always an explicit argument; diagnostic selected-vehicle state
    and future service outcomes have no implicit route into the critic.
    """

    def __init__(self, native_critic, memory_dim):
        super().__init__()
        if memory_dim < 1:
            raise ValueError("memory_dim must be positive")
        self.base = deepcopy(native_critic)
        self.memory_dim = int(memory_dim)
        parameter = next(self.base.parameters())
        shape = (self.base.ensemble_size, self.memory_dim)
        self.level_weight = nn.Parameter(torch.zeros(
            shape, dtype=parameter.dtype, device=parameter.device))
        self.action_weight = nn.Parameter(torch.zeros_like(self.level_weight))

    @property
    def ensemble_size(self):
        return self.base.ensemble_size

    def _residual(self, actions, memory, indices=None):
        if memory.ndim != 2 or memory.shape != (len(actions), self.memory_dim):
            raise ValueError("memory must have one fixed-width row per action")
        if actions.ndim != 2 or actions.shape[1] != 1:
            raise ValueError("executed actions must have shape (N, 1)")
        level, interaction = self.level_weight, self.action_weight
        if indices is not None:
            level = level.index_select(0, indices)
            interaction = interaction.index_select(0, indices)
        return (memory @ level.T).T + ((actions[:, 0]+1.)*.5)[None, :] * (
            memory @ interaction.T).T

    def forward(self, states, actions, memory):
        return self.base(states, actions) + self._residual(actions, memory)

    def forward_subset(self, states, actions, memory, indices):
        return (self.base.forward_subset(states, actions, indices)
                + self._residual(actions, memory, indices))
