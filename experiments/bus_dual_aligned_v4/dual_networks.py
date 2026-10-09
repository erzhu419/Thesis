"""The successful global encoder and exact-zero holding, extended by speed."""

import torch
from torch import nn
from torch.distributions import Normal

from aligned_common import STATE_DIM
from SimpleSAC.transit_v3.networks import MLP, EnsembleMLP
from SimpleSAC.transit_v3.domain_ratio import FactoredDynamicsRatio
from SimpleSAC.transit_v4.networks import HurdleTransitActor
from SimpleSAC.transit_v7.networks import TransitStateEncoderV7


class Encoder(nn.Module):
    def __init__(self, layer_norm=False, activation="relu"):
        super().__init__()
        self.base = TransitStateEncoderV7(layer_norm=layer_norm, activation=activation)
        self.speed = nn.Linear(16, 256, bias=False)
        nn.init.zeros_(self.speed.weight)

    def forward(self, states):
        return self.base(states[:, :229]) + self.speed(states[:, 229:STATE_DIM])


class Actor(HurdleTransitActor):
    def __init__(self, hidden_dims=(256,256), *, layer_norm=False, activation="relu"):
        nn.Module.__init__(self)
        self.encoder = Encoder(layer_norm, activation)
        self.network = MLP(256, 5, hidden_dims, layer_norm=layer_norm, activation=activation)
        with torch.no_grad():
            self.network.network[-1].bias.copy_(torch.tensor([-.62,-1.4,-.5,0.,-1.5]))
            self.network.network[-1].weight[3].zero_()

    def parameters_for_state(self, states):
        return self.network(self.encoder(states))

    def distribution_parameters(self, states):
        value = self.parameters_for_state(states)
        return value[:,0], value[:,1], value[:,2].clamp(-5,2)

    def soft_action_components(self, states, *, deterministic_positive=False):
        value = self.parameters_for_state(states)
        components = self.soft_action_components_from_parameters(value[:,0], value[:,1], value[:,2].clamp(-5,2),
            deterministic_positive=deterministic_positive)
        distribution = Normal(value[:,3], value[:,4].clamp(-5,2).exp())
        z = value[:,3] if deterministic_positive else distribution.rsample()
        speed = z.tanh()
        logp = distribution.log_prob(z)-torch.log(1-speed.square()+1e-6)
        for key in ("zero_action", "positive_action"):
            components[key] = torch.cat((components[key], speed[:,None]), -1)
        for key in ("zero_log_prob", "positive_log_prob", "positive_conditional_log_prob"):
            components[key] = components[key]+logp
        return components

    def repeated_soft_action_components(self, states, repeats, *, deterministic_positive=False):
        return self.soft_action_components(states.repeat_interleave(repeats, dim=0), deterministic_positive=deterministic_positive)

    def repeated_actions_and_log_prob(self, states, repeats, *, deterministic=False):
        return self(states.repeat_interleave(repeats, dim=0), deterministic=deterministic)

    def action_log_prob(self, states, actions):
        # Reuse the proven mixed counting/continuous holding likelihood.
        holding = super().action_log_prob(states, actions[:, :1])
        value = self.parameters_for_state(states)
        speed = actions[:,1].clamp(-.999999, .999999)
        logp = Normal(value[:,3], value[:,4].clamp(-5,2).exp()).log_prob(speed.atanh())
        return holding+logp-torch.log(1-speed.square()+1e-6)


class Critic(nn.Module):
    def __init__(self, ensemble_size, hidden_dims, *, layer_norm=False, activation="relu"):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.encoder = Encoder(layer_norm, activation)
        self.network = EnsembleMLP(ensemble_size, 258, 1, hidden_dims, layer_norm=layer_norm, activation=activation)

    def encode_states(self, states):
        return self.encoder(states)

    def forward_features(self, features, actions):
        return self.network(torch.cat((features, actions), -1)).squeeze(-1)

    def forward(self, states, actions):
        return self.forward_features(self.encode_states(states), actions)

    def forward_features_subset(self, features, actions, indices):
        return self.network(torch.cat((features, actions), -1), indices).squeeze(-1)

    def forward_subset(self, states, actions, indices):
        return self.forward_features_subset(self.encode_states(states), actions, indices)


class Value(nn.Module):
    def __init__(self, hidden_dims, *, layer_norm=False, activation="relu"):
        super().__init__()
        self.encoder = Encoder(layer_norm, activation)
        self.network = MLP(256, 1, hidden_dims, layer_norm=layer_norm, activation=activation)

    def forward(self, states):
        return self.network(self.encoder(states)).squeeze(-1)


def bindings():
    return dict(actor_class=Actor, critic_class=Critic, value_class=Value, ratio_class=FactoredDynamicsRatio,
        ratio_kwargs=dict(feature_dim=256, action_dim=2, feature_fn=Encoder(),
                          transition_feature_key="delta_t_seconds", transition_feature_scale=300.))
