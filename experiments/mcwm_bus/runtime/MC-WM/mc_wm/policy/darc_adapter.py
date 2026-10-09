"""Domain-classifier reward correction for the DARC adaptation baseline.

The classifier follows DARC's decomposition into a domain classifier for
``(s, a)`` and an additional next-state-dependent logit.  A source transition
receives the log-density-ratio correction

    log p(target | s,a,s') - log p(source | s,a,s')
    - log p(target | s,a) + log p(source | s,a).

The corrected reward is exposed through a replay-buffer view so the shared
RE-SAC implementation remains unchanged for every other experiment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DARCAdapter(nn.Module):
    """Balanced source/target domain classifiers used by DARC."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        hidden_dim: int = 32,
        lr: float = 3e-4,
        input_noise_std: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.input_noise_std = float(input_noise_std)
        self.sa_classifier = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.sas_advantage = nn.Sequential(
            nn.Linear(2 * obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self._updates = 0

    def _domain_logits(self, s, a, s2, *, add_noise: bool):
        sa = torch.cat([s, a], dim=-1)
        sas = torch.cat([s, a, s2], dim=-1)
        if add_noise and self.input_noise_std > 0:
            sa = sa + self.input_noise_std * torch.randn_like(sa)
            sas = sas + self.input_noise_std * torch.randn_like(sas)
        sa_logits = self.sa_classifier(sa)
        sas_logits = sa_logits + self.sas_advantage(sas)
        return sa_logits, sas_logits

    def update(self, source_buffer, target_buffer, batch_size: int = 256):
        """Train once on equally sized source and target minibatches."""
        source = source_buffer.sample(batch_size)
        target = target_buffer.sample(batch_size)
        s_src, a_src, _, s2_src = source[:4]
        s_tgt, a_tgt, _, s2_tgt = target[:4]

        sa_src, sas_src = self._domain_logits(
            s_src, a_src, s2_src, add_noise=True)
        sa_tgt, sas_tgt = self._domain_logits(
            s_tgt, a_tgt, s2_tgt, add_noise=True)
        source_labels = torch.zeros(
            len(s_src), dtype=torch.long, device=self.device)
        target_labels = torch.ones(
            len(s_tgt), dtype=torch.long, device=self.device)

        sa_loss = (
            F.cross_entropy(sa_src, source_labels)
            + F.cross_entropy(sa_tgt, target_labels)
        )
        sas_loss = (
            F.cross_entropy(sas_src, source_labels)
            + F.cross_entropy(sas_tgt, target_labels)
        )
        loss = sa_loss + sas_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._updates += 1

        with torch.no_grad():
            sa_correct = (
                (sa_src.argmax(dim=-1) == source_labels).sum()
                + (sa_tgt.argmax(dim=-1) == target_labels).sum()
            )
            sas_correct = (
                (sas_src.argmax(dim=-1) == source_labels).sum()
                + (sas_tgt.argmax(dim=-1) == target_labels).sum()
            )
            denominator = float(2 * batch_size)
        return {
            "loss": float(loss.detach()),
            "sa_accuracy": float(sa_correct) / denominator,
            "sas_accuracy": float(sas_correct) / denominator,
            "updates": self._updates,
        }

    @torch.no_grad()
    def reward_delta(self, s, a, s2):
        """Return DARC's source-to-target log-density-ratio correction."""
        sa_logits, sas_logits = self._domain_logits(
            s, a, s2, add_noise=False)
        sa_logp = F.log_softmax(sa_logits, dim=-1)
        sas_logp = F.log_softmax(sas_logits, dim=-1)
        sas_log_ratio = sas_logp[:, 1] - sas_logp[:, 0]
        sa_log_ratio = sa_logp[:, 1] - sa_logp[:, 0]
        return (sas_log_ratio - sa_log_ratio).unsqueeze(-1)


class RewardAdjustedBuffer:
    """Read-only buffer view that adds DARC's correction at sample time."""

    def __init__(self, source_buffer, adapter: DARCAdapter):
        self.source_buffer = source_buffer
        self.adapter = adapter

    def sample(self, batch_size: int):
        sample = self.source_buffer.sample(batch_size)
        s, a, r, s2, d = sample[:5]
        adjusted_r = r + self.adapter.reward_delta(s, a, s2)
        if len(sample) == 6:
            return s, a, adjusted_r, s2, d, sample[5]
        return s, a, adjusted_r, s2, d
