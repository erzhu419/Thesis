"""Action-valued sparse feedback and context-preserving residual centering."""

import numpy as np
import math
import torch
from torch import nn

from settings import P, use_snapshot

use_snapshot()
from csbapr.networks.policy import GaussianPolicy


def equilibrium_states(states):
    reference = states.clone()
    reference[:, 5:7] = states[:, 8:9]
    return reference


class RuleReference(nn.Module):
    def __init__(self, objective):
        super().__init__()
        self.objective = objective

    def forward(self, states):
        hold = ((states[:, 6:7]-states[:, 5:6])*300).clamp(0, 60)
        if self.objective == "passenger":
            hold = hold*0
        return (hold/30-1).clamp(-P["reference_action_clip"], P["reference_action_clip"])


class SparseReference(nn.Module):
    def __init__(self, payload, objective):
        super().__init__()
        self.objective = objective
        for name in ("powers", "coefficients", "cost_mean", "cost_std"):
            dtype = torch.long if name == "powers" else torch.float32
            self.register_buffer(name, torch.as_tensor(payload[name], dtype=dtype))
        self.register_buffer("grid", torch.tensor(P["reference_hold_grid"], dtype=torch.float32)/30-1)
        self.register_buffer("indices", torch.tensor(P["sindy_features"], dtype=torch.long))
        powers = np.asarray(payload["powers"], dtype=int)
        factors = np.full((len(powers), int(powers.sum(1).max())), powers.shape[1], dtype=int)
        for row, exponents in enumerate(powers):
            indices = np.repeat(np.arange(len(exponents)), exponents)
            factors[row, :len(indices)] = indices
        self.register_buffer("factors", torch.as_tensor(factors, dtype=torch.long))

    def prediction(self, states, actions):
        line = (states[:, 0]*12).round().long().clamp(0, 11)
        x = torch.cat((states[:, self.indices], actions), -1)
        x = torch.cat((x, torch.ones(len(x), 1, dtype=x.dtype, device=x.device)), -1)
        features = torch.ones(len(x), len(self.powers), dtype=x.dtype, device=x.device)
        for index in self.factors.T:
            features = features*x[:, index]
        return torch.bmm(self.coefficients[line], features.unsqueeze(-1)).squeeze(-1)

    def forward(self, states):
        n, width = len(states), len(self.grid)
        tiled = states[:, None].expand(n, width, 15).reshape(-1, 15)
        actions = self.grid.repeat(n)[:, None]
        cost_index = len(P["sindy_features"]) + (0 if self.objective == "headway" else 1)
        costs = torch.zeros(len(tiled), dtype=states.dtype, device=states.device)
        for step in range(P["reference_horizon"]):
            predicted = self.prediction(tiled, actions)
            costs = costs+P["reference_discount"]**step*predicted[:, cost_index]
            following = tiled.clone()
            following[:, self.indices] = tiled[:, self.indices]+predicted[:, :len(self.indices)]
            positive = [5, 6, 7, 8, 9, 12, 13, 14]
            following[:, positive] = following[:, positive].clamp_min(0)
            tiled = following
        costs = costs.reshape(n, width)
        weights = torch.softmax(-costs/P["reference_temperature"], -1)
        return (weights@self.grid[:, None]).clamp(-P["reference_action_clip"], P["reference_action_clip"])


class ResidualPolicy(GaussianPolicy):
    def __init__(self, reference, centered, hidden=64):
        super().__init__(15, 1, hidden, activation="tanh")
        self.reference = reference.eval().requires_grad_(False)
        self.centered = centered

    def forward(self, states):
        residual, log_std = super().forward(states)
        if self.centered:
            residual = residual-super().forward(equilibrium_states(states))[0]
        baseline = torch.atanh(self.reference(states))
        return baseline+residual, log_std

    def compute_L_eff(self):
        # The inherited plain-MLP bound does not cover the fitted feedback.
        return math.inf


def load_reference(folder, objective):
    with np.load(folder / "reference.npz") as data:
        return SparseReference({name:data[name] for name in data.files}, objective)
