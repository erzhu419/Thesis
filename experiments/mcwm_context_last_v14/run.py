"""Two bounded routing interventions using the main campaign's identical learner."""

from last_common import (HERE, P, FIXED, CONTEXTS, module, write, smoke_protocol,
    initializer_folder, initializer_path, bundle_path, policy_folder)
from last_routing import HysteresisFilter

main_run = module("last_confirmation_training", HERE.parent / "mcwm_seed_confirmation_v13/run.py")
for implementation in (main_run, main_run.previous, main_run.previous.previous):
    implementation.HERE, implementation.P, implementation.FIXED = HERE, P, FIXED
main_run.CONTEXTS, main_run.smoke_protocol = CONTEXTS, smoke_protocol
main_run.initializer_folder, main_run.initializer_path = initializer_folder, initializer_path
main_run.bundle_path, main_run.policy_folder = bundle_path, policy_folder


class LastTrainer(main_run.MappingTrainer):
    def configure_context(self, p, scenario, bundle):
        super().configure_context(p, scenario, bundle)
        self.context_filter = HysteresisFilter(bundle["physical_recognizer"], p)
        self.coefficient = self.context_filter.coefficient
        self.gradient_allowed, self.probe, self.probe_losses = False, False, []
        self.pending_relation, self.gradient_trace = None, []

    def _ensemble_td_loss(self, predictions, targets, weights=None):
        result = super()._ensemble_td_loss(predictions, targets, weights)
        if self.probe:
            real = main_run.base.TransitH2OTrainerV7._ensemble_td_loss(self, predictions, targets, weights)
            self.probe_losses.append(real)
            if len(self.probe_losses) == 2 and self.model_loss is not None:
                fraction = self.config.online_fraction
                mixed = (1-fraction)*self.probe_losses[0]+fraction*self.probe_losses[1]
                with self.costs.phase("context_gradient_probe"):
                    self.pending_relation = main_run.previous.previous.previous.usage.diagnostic.gradient_relation(
                        mixed, self.model_loss, tuple(p for p in self.critic.parameters() if p.requires_grad),
                        self.protocol["model_fraction"])
        return result

    def update(self, *, online, online_only=False, collect_metrics=True):
        if self.arm == "gradient_gated" and online:
            requested = self.context_filter.coefficient
            self.coefficient = requested if self.gradient_allowed else self.protocol["global_coefficient"]
            self.probe = self.total_updates % self.protocol["gradient_probe_interval"] == 0
            self.probe_losses, self.pending_relation = [], None
        else:
            self.probe = False
        effective = self.coefficient
        result = super().update(online=online, online_only=online_only, collect_metrics=collect_metrics)
        if self.pending_relation is not None:
            cosine = self.pending_relation["gradient_cosine"]
            self.gradient_allowed = cosine is not None and cosine > self.protocol["gradient_min_cosine"]
            self.gradient_trace.append(dict(total_updates=self.total_updates,
                used_coefficient=effective, requested_coefficient=self.context_filter.coefficient,
                next_update_allowed=self.gradient_allowed, **self.pending_relation))
        self.probe_losses, self.pending_relation = [], None
        return result

    def save_context(self):
        super().save_context()
        write(self.output / "gradient_gate_trace.json", dict(trace=self.gradient_trace,
            interval=self.protocol["gradient_probe_interval"], threshold=self.protocol["gradient_min_cosine"],
            scope="Observed/model critic gradients on existing training batches; decision applies to subsequent updates"))


main_run.MappingTrainer = LastTrainer


if __name__ == "__main__":
    main_run.main()
