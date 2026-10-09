"""Confidence scale and covered scalar backups; event states are never averaged."""

import torch
import torch.nn.functional as F


def trust_backup(log_confidence, following, done, gamma, normalized):
    return ((1-gamma) if normalized else 1.)*log_confidence + gamma*(1-done)*following


def blend_targets(source, target, confidence):
    return source + confidence*(target-source)


def current_features(states, actions):
    slots = states[:, 5:].reshape(-1, 16, 14)
    current = (states[:, 2]*15).round().long()
    slot = slots[torch.arange(len(states), device=states.device), current]
    line = slot[:, 1].round().long()
    # Time, network occupancy, current station/local quantities, action.
    features = torch.cat((states[:, [0, 3, 4]], slot[:, 2:13], actions.reshape(-1, 1)), 1)
    return line, features


class TargetSupport:
    def __init__(self, fit, stop, max_rows, quantile):
        index = torch.linspace(0, len(fit["s"])-1, min(max_rows, len(fit["s"])),
                               device=fit["s"].device).long()
        self.lines, raw = current_features(fit["s"][index], fit["a"][index])
        self.mean, self.scale = raw.mean(0), raw.std(0).clamp_min(.05)
        self.reference = (raw-self.mean)/self.scale
        distances = self.distance(stop["s"], stop["a"])
        finite = distances[torch.isfinite(distances)]
        if not len(finite):
            raise ValueError("No target stopping rows share a fitted route")
        self.radius = torch.quantile(finite, quantile)

    @torch.no_grad()
    def distance(self, states, actions):
        lines, raw = current_features(states, actions)
        query = (raw-self.mean)/self.scale
        distances = torch.cdist(query, self.reference)/query.shape[1]**.5
        distances.masked_fill_(lines[:, None] != self.lines[None], torch.inf)
        return distances.min(1).values

    def accepted(self, states, actions):
        return self.distance(states, actions) <= self.radius


def update_normalized_trust(trust, s, a, ns, done, errors, policy, n_iters):
    trust._update_tau(errors)
    log_c = trust._c_step(errors).clamp_min(1e-8).log().unsqueeze(-1)
    with torch.no_grad():
        following = trust.target(ns, policy(ns))
        target = trust_backup(log_c, following, done.unsqueeze(-1), trust.gamma, True)
    for _ in range(n_iters):
        loss = F.mse_loss(trust.net(s, a), target)
        trust.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        trust.optimizer.step()
    with torch.no_grad():
        for param, target_param in zip(trust.net.parameters(), trust.target.parameters()):
            target_param.lerp_(param, trust.tau_polyak)
    return dict(td_loss=float(loss.detach()), tau=trust._tau)
