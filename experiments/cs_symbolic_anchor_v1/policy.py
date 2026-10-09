"""Centered holding policy and frozen action-valued SINDy feedback."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from csbapr.networks.policy import GaussianPolicy
from csbapr.losses.jacobian import action_jacobian, deterministic_action

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())


def equilibrium_states(states):
    projected = states.clone()
    projected[:, 5:7] = states[:, 8:9]
    return projected


class CenteredPolicy(GaussianPolicy):
    def __init__(self, state_dim=92, action_dim=1, hidden_dim=64):
        super().__init__(state_dim, action_dim, hidden_dim, activation="tanh")
        self.equilibrium_mean = nn.Parameter(torch.zeros(action_dim))

    def forward(self, states):
        mean, log_std = super().forward(states)
        projected_mean, _ = super().forward(equilibrium_states(states))
        return self.equilibrium_mean + mean - projected_mean, log_std


class StageReference(nn.Module):
    def __init__(self, seed):
        super().__init__()
        with np.load(HERE / "inputs" / f"seed{seed}" / "reference.npz") as data:
            powers = data["powers"].astype(int)
            # Only the passenger station-cost row is used, not fitted dynamics.
            coefficients = data["coefficients"][:, len(P["sindy_features"]) + 1]
        assert np.isfinite(coefficients).all() and powers.shape[1] == len(P["sindy_features"]) + 1
        factors = np.full((len(powers), int(powers.sum(1).max())), powers.shape[1], dtype=int)
        for i, exponents in enumerate(powers):
            ids = np.repeat(np.arange(len(exponents)), exponents)
            factors[i, :len(ids)] = ids
        self.register_buffer("coefficients", torch.as_tensor(coefficients, dtype=torch.float32))
        self.register_buffer("factors", torch.as_tensor(factors, dtype=torch.long))
        self.register_buffer("indices", torch.tensor(P["sindy_features"], dtype=torch.long))
        self.register_buffer("grid", torch.tensor(P["reference_hold_grid"], dtype=torch.float32) / 30 - 1)

    def forward(self, states):
        n, width = len(states), len(self.grid)
        local = states[:, :15]
        tiled = local[:, None].expand(n, width, 15).reshape(-1, 15)
        line = (tiled[:, 0] * 12).round().long().clamp(0, 11)
        x = torch.cat((tiled[:, self.indices], self.grid.repeat(n)[:, None]), -1)
        x = torch.cat((x, torch.ones(len(x), 1, device=x.device, dtype=x.dtype)), -1)
        features = torch.ones(len(x), len(self.factors), device=x.device, dtype=x.dtype)
        for index in self.factors.T:
            features = features * x[:, index]
        costs = (self.coefficients[line] * features).sum(-1).reshape(n, width)
        weights = torch.softmax(-costs / P["reference_temperature"], -1)
        return (weights @ self.grid[:, None]).clamp(-P["reference_action_clip"], P["reference_action_clip"])


def normalized_jacobian_loss(policy, reference, batch):
    with torch.enable_grad():
        states = batch.detach().requires_grad_(True)
        jac_pi = action_jacobian(deterministic_action(policy, states), states, True)[..., :15]
        jac_ref = action_jacobian(reference(states), states, False)[..., :15].detach()
        return ((jac_pi - jac_ref).square().sum((-2, -1)) / (1 + jac_ref.square().sum((-2, -1)))).mean()
