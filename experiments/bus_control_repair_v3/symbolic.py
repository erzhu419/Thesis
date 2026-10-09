"""Smooth one-station feedback and normalized action-space Jacobian errors."""

import sys

import numpy as np
import torch
from torch import nn

from repair_common import EXPERIMENTS, P

sys.path.insert(0, str(EXPERIMENTS / "csbapr_symbolic_bus_v2"))
from adaptation import SparseReference, equilibrium_states
from settings import P as ORIGINAL, model_folder
from csbapr.networks.policy import GaussianPolicy
from csbapr.losses.jacobian import action_jacobian, deterministic_action


class SmoothReference(SparseReference):
    def forward(self, states):
        n, width = len(states), len(self.grid)
        tiled = states[:, None].expand(n, width, 15).reshape(-1, 15)
        actions = self.grid.repeat(n)[:, None]
        index = len(ORIGINAL["sindy_features"]) + (self.objective == "passenger")
        # Direct station-cost prediction avoids rolling the two failed dynamics lines.
        costs = self.prediction(tiled, actions)[:, index].reshape(n, width)
        weights = torch.softmax(-costs / P["reference_temperature"], -1)
        return (weights @ self.grid[:, None]).clamp(-ORIGINAL["reference_action_clip"], ORIGINAL["reference_action_clip"])


def reference(seed, objective):
    with np.load(model_folder(seed) / "reference.npz") as data:
        return SmoothReference({name: data[name] for name in data.files}, objective)


class SoftResidualPolicy(GaussianPolicy):
    def __init__(self):
        super().__init__(15, 1, 64, activation="tanh")
        self.equilibrium_mean = nn.Parameter(torch.zeros(1))

    def forward(self, states):
        mean, log_std = super().forward(states)
        projected_mean, _ = super().forward(equilibrium_states(states))
        # Zero residual at projected headways does not freeze the complete hold.
        return self.equilibrium_mean + mean - projected_mean, log_std

    def compute_L_eff(self):
        return float("inf")


def normalized_jacobian_loss(policy, target, batch):
    with torch.enable_grad():
        states = batch.detach().requires_grad_(True)
        jac_policy = action_jacobian(deterministic_action(policy, states), states, True)
        jac_target = action_jacobian(target(states), states, False).detach()
        numerator = (jac_policy - jac_target).square().sum((-2, -1))
        denominator = 1 + jac_target.square().sum((-2, -1))
        return (numerator / denominator).mean()
