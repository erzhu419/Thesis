"""Small neural remainder head for the additive residual world model."""

from __future__ import annotations

import torch
import torch.nn as nn


class SparseNeuralRemainderHead(nn.Module):
    """Learn only the part of a residual not explained by sparse regression.

    The caller supplies a fixed feature mask learned by SINDy. Inactive inputs
    are zeroed before the network, so the deployed correction cannot depend on
    features absent from the reported sparse support. The output layer starts
    at zero, making the initial additive model exactly the SINDy model.
    """

    def __init__(self, feature_dim: int, output_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.output_dim = int(output_dim)
        self.register_buffer("feature_mask", torch.ones(self.feature_dim))
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def set_feature_mask(self, mask: torch.Tensor) -> None:
        mask = mask.detach().to(device=self.feature_mask.device,
                                dtype=self.feature_mask.dtype).reshape(-1)
        if mask.numel() != self.feature_dim:
            raise ValueError(
                f"feature mask has {mask.numel()} entries; expected {self.feature_dim}"
            )
        self.feature_mask.copy_((mask > 0).to(self.feature_mask.dtype))

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return self.net(theta * self.feature_mask)

    def regularization_loss(self) -> torch.Tensor:
        """Mild L1 penalty keeps the remainder small relative to SINDy."""
        return sum(p.abs().mean() for p in self.net.parameters())

    @property
    def lipschitz_upper_bound(self) -> float:
        """Global network Lipschitz upper bound from layer spectral norms.

        This is a property of the neural function, not an OOD prediction-error
        guarantee. The distinction is intentional.
        """
        bound = 1.0
        with torch.no_grad():
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    bound *= float(torch.linalg.matrix_norm(layer.weight, ord=2))
                elif isinstance(layer, nn.SiLU):
                    # sup_x |d SiLU(x) / dx| < 1.1
                    bound *= 1.1
        return bound

    @property
    def L_eff(self) -> float:
        """Compatibility alias; use ``lipschitz_upper_bound`` in new code."""
        return self.lipschitz_upper_bound
