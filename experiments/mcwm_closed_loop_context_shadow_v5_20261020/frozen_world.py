"""Freeze the initially fitted neural world while retaining label-driven trust."""

import torch

from experiments.mcwm_closed_loop_action_local_v3_20261020.learning import ActionConsistentWorldMixin
from experiments.mcwm_closed_loop_diagnostic_v2_20261020.world_diagnostics import DiagnosticWorld
from experiments.mcwm_closed_loop_v1.learning import learned_rows, row_table


class FrozenWorld(ActionConsistentWorldMixin, DiagnosticWorld):
    """The action_model world without online neural fitting or local readouts.

    The same initialization fits the source, direct and residual models. Fresh
    labels still select a family and train QDelta; policy and replay updates are
    unchanged. Nominal refit minibatch draws keep the world sampling stream
    aligned with the updating arm until their realized trajectories differ.
    """

    counterfactual_batch = None
    head_counterfactual_batches = None

    def __init__(self, replay, *, action_transform, seed=6201, updates=400):
        self.action_transform = action_transform
        super().__init__(replay, seed=seed, updates=updates, context=False)
        self.device = torch.device(self.device)
        self.initial_fit_updates = int(updates)
        self.skipped_refit_packets = self.skipped_neural_gradient_updates = 0
        self.trust_update_calls = 0
        self.initial_family = self.family
        self.initial_world_state = {}
        for name, model in dict(source=self.source, **self.models).items():
            model.requires_grad_(False).eval()
            self.initial_world_state[name] = {
                key: value.detach().clone() for key, value in model.state_dict().items()}

    def refit(self, updates):
        recent = learned_rows(row_table(self.replay.recent))
        count = len(recent['s'])
        if not count:
            return False
        history = learned_rows(self.replay.history)
        self.rng.integers(len(history['s']), size=count)
        for _ in self.models:
            for _ in range(updates):
                self.rng.integers(2*count, size=min(64, 2*count))
        self.skipped_refit_packets += 1
        self.skipped_neural_gradient_updates += len(self.models)*updates
        return False

    def update_trust(self, actor, size=64):
        result = super().update_trust(actor, size)
        self.trust_update_calls += 1
        return result

    def summary(self):
        result = super().summary()
        changes = {}
        for name, model in dict(source=self.source, **self.models).items():
            before = self.initial_world_state[name]
            current = model.state_dict()
            changes[name] = dict(tensor_count=len(current),
                changed_tensor_count=sum(not torch.equal(value, before[key])
                                         for key, value in current.items()),
                max_absolute_change=max(float((value-before[key]).abs().max())
                                        for key, value in current.items()))
        result['frozen_parameters'] = dict(enabled=True,
            initial_fit_updates=self.initial_fit_updates, models=changes,
            online_neural_gradient_updates=self.model_updates,
            skipped_refit_packets=self.skipped_refit_packets,
            skipped_neural_gradient_updates=self.skipped_neural_gradient_updates,
            nominal_refit_rng_draws_preserved=True, local_readout_enabled=False,
            family_selection_adapts=not self.fixed_reference,
            initial_selected_family=self.initial_family,
            trust_update_calls=self.trust_update_calls)
        return result
