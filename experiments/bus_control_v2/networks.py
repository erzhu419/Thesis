"""Bus-line categorical features with one- or two-dimensional tanh actions."""

from functools import partial

import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F

from station_common import action_dim
from SimpleSAC.transit_v3.networks import MLP, EnsembleMLP
from SimpleSAC.transit_v3.domain_ratio import FactoredDynamicsRatio


def features(states):
    line = F.one_hot(states[:, 0].round().long().clamp(0, 11), 12).to(states.dtype)
    return torch.cat((line, states[:, 1:]), -1)


class Actor(nn.Module):
    def __init__(self, hidden_dims, *, dim, layer_norm=True, activation="relu"):
        super().__init__()
        self.dim = dim
        self.network = MLP(26+dim, 2*dim, hidden_dims, layer_norm=layer_norm, activation=activation)
        with torch.no_grad():
            self.network.network[-1].bias[dim:] = -.5

    def distribution_parameters(self, states):
        mean, std = self.network(features(states)).chunk(2, -1)
        return mean, std.clamp(-5, 2)

    def forward(self, states, deterministic=False):
        mean, std = self.distribution_parameters(states)
        distribution = Normal(mean, std.exp())
        z = mean if deterministic else distribution.rsample()
        actions = z.tanh()
        logp = distribution.log_prob(z)-torch.log(1-actions.square()+1e-6)
        return actions, logp.sum(-1)

    def action_log_prob(self, states, actions):
        mean, std = self.distribution_parameters(states)
        actions = actions.clamp(-.99, .99)
        return (Normal(mean, std.exp()).log_prob(actions.atanh())-torch.log(1-actions.square()+1e-6)).sum(-1)


class Critic(nn.Module):
    def __init__(self, ensemble_size, hidden_dims, *, dim, layer_norm=True, activation="relu"):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.network = EnsembleMLP(ensemble_size, 26+2*dim, 1, hidden_dims,
                                  layer_norm=layer_norm, activation=activation)

    def forward(self, states, actions):
        return self.network(torch.cat((features(states), actions), -1)).squeeze(-1)

    def forward_subset(self, states, actions, indices):
        return self.network(torch.cat((features(states), actions), -1), indices).squeeze(-1)


class Value(nn.Module):
    def __init__(self, hidden_dims, *, dim, layer_norm=True, activation="relu"):
        super().__init__()
        self.network = MLP(26+dim, 1, hidden_dims, layer_norm=layer_norm, activation=activation)

    def forward(self, states):
        return self.network(features(states)).squeeze(-1)


def bindings(mode):
    dim = action_dim(mode)
    return dict(actor_class=partial(Actor, dim=dim), critic_class=partial(Critic, dim=dim),
                value_class=partial(Value, dim=dim), ratio_class=FactoredDynamicsRatio,
                ratio_kwargs=dict(feature_dim=26+dim, action_dim=dim, feature_fn=features,
                                  transition_feature_key="delta_t_seconds", transition_feature_scale=300.))
