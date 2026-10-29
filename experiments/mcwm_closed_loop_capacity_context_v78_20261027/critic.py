"""Frozen preflight of one downstream-capacity action interaction.

The caller supplies issue-visible z and an explicit observation mask.  Native
states/features, executed actions and the native critic remain unchanged.
There is no learner attachment, optimizer registration or update in this module.
"""
import torch
from torch import nn


class CapacityContextQEnsemble(nn.Module):
    """Q_k(s,a,z,m) = Q_native,k(s,a) + theta_k * m*z*(a+1)/2.

    ``base`` is the resident TransitQEnsembleV7 object, retained by reference.
    ``actions`` are already executed normalized actions; no F is applied here.
    ``z`` and boolean ``observed`` each have shape (N,). Missing z may be NaN:
    it is removed before arithmetic rather than treated as an observed zero.
    Every added theta starts at zero without drawing random numbers.
    """

    def __init__(self, base):
        super().__init__()
        self.base = base
        parameter = next(base.parameters())
        self.action_weight = nn.Parameter(torch.zeros(
            base.ensemble_size, dtype=parameter.dtype, device=parameter.device))

    @property
    def ensemble_size(self):
        return self.base.ensemble_size

    def _residual(self, actions, z, observed, indices=None):
        if actions.ndim != 2 or actions.shape[1] != 1:
            raise ValueError("Executed actions must have shape (N, 1)")
        if z.shape != (len(actions),) or observed.shape != z.shape:
            raise ValueError("z and observed must each have one scalar per action")
        if observed.dtype != torch.bool:
            raise ValueError("observed must be an explicit boolean mask")
        context = torch.where(observed, z, torch.zeros_like(z))
        weight = self.action_weight
        if indices is not None:
            weight = weight.index_select(0, indices)
        return weight[:, None] * (context * ((actions[:, 0]+1.)*.5))[None, :]

    def encode_states(self, states):
        return self.base.encode_states(states)

    def forward(self, states, actions, z, observed):
        return self.base(states, actions) + self._residual(actions, z, observed)

    def forward_subset(self, states, actions, z, observed, indices):
        return (self.base.forward_subset(states, actions, indices)
                + self._residual(actions, z, observed, indices))

    def forward_features(self, features, actions, z, observed):
        return (self.base.forward_features(features, actions)
                + self._residual(actions, z, observed))

    def forward_features_subset(self, features, actions, z, observed, indices):
        return (self.base.forward_features_subset(features, actions, indices)
                + self._residual(actions, z, observed, indices))
