"""Source-normal action restraint, added to the native actor gradient."""

import random

import numpy as np
import torch

WAITING = [17] + list(range(19, 91, 6))


class SourceAnchor:
    def __init__(self, actor, teacher, cap, weight, seed, capacity=4096, batch_size=256):
        self.actor, self.teacher = actor, teacher
        self.cap, self.weight = cap, weight
        self.capacity, self.batch_size = capacity, batch_size
        self.rng = random.Random(seed + 411000)
        self.states, self.seen, self.steps = [], 0, 0
        self.loss_sum = self.gradient_sum = 0.
        self.cache = None
        self.hits = {}
        self.parameters = tuple(actor.parameters())
        for p in teacher.parameters():
            p.requires_grad_(False)
        if weight > 0:
            for index, p in enumerate(self.parameters):
                p.register_hook(self.hook(index))

    def hook(self, index):
        def add(gradient):
            if self.cache is None or self.cache[index] is None:
                return gradient
            self.hits[index] = self.hits.get(index, 0) + 1
            return gradient + self.cache[index]
        return add

    def observe(self, state):
        if self.weight == 0:
            return
        assert state[17] < 1., "Source-normal reference must be invertible"
        self.seen += 1
        if len(self.states) < self.capacity:
            self.states.append(state.copy())
        else:
            index = self.rng.randrange(self.seen)
            if index < self.capacity:
                self.states[index] = state.copy()

    def loss(self, state):
        raw = state.clone()
        raw[:, WAITING] = raw[:, WAITING] * self.cap
        with torch.no_grad():
            mean, _ = self.teacher(raw)
            reference = self.teacher.action_range * torch.tanh(mean)
        mean, _ = self.actor(state)
        action = self.actor.action_range * torch.tanh(mean)
        return torch.relu(action - reference).square().mean()

    def prepare(self):
        self.cache = None
        self.hits = {}
        if self.weight == 0 or not self.states:
            return 0., 0
        indices = self.rng.sample(range(len(self.states)), min(self.batch_size, len(self.states)))
        state = torch.as_tensor(np.stack([self.states[i] for i in indices]), device=next(self.actor.parameters()).device)
        loss = self.loss(state)
        gradients = torch.autograd.grad(self.weight * loss, self.parameters, allow_unused=True)
        self.cache = tuple(None if g is None else g.detach() for g in gradients)
        assert all(g is None or bool(torch.isfinite(g).all()) for g in self.cache)
        self.steps += 1
        self.loss_sum += float(loss.detach())
        self.gradient_sum += sum(float(g.square().sum()) for g in self.cache if g is not None) ** .5
        return float(loss.detach()), len(indices)

    def clear(self):
        if self.cache is not None:
            assert all(self.hits.get(i, 0) == 1 for i, g in enumerate(self.cache) if g is not None), self.hits
        self.cache = None

    def summary(self):
        return dict(weight=self.weight, source_normal_observations=self.seen, retained_states=len(self.states),
                    anchor_updates=self.steps, mean_anchor_loss=self.loss_sum / max(1, self.steps),
                    mean_anchor_gradient_norm=self.gradient_sum / max(1, self.steps))
