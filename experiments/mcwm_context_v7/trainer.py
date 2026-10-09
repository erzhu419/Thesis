"""Hold observed TD scaling fixed while adapting synthetic-model TD usage."""

import torch

from context_common import HERE, FIXED, module, write
from context import CompletedBlocks, ContextFilter

usage = module("context_usage", HERE.parent / "mcwm_usage_v3/run.py")
base = usage.base
raw_trainer = base.TransitH2OTrainerV7


class ContextModel(usage.UsageModel):
    def update_trust(self, actor):
        from mechanisms import update_normalized_trust
        ix = torch.as_tensor(self.rng.integers(len(self.fit_ns), size=128), device=self.device)
        return update_normalized_trust(self.trust, self.fit["s"][ix], self.fit["a"][ix],
            self.fit_ns[ix], torch.zeros(len(ix), device=self.device), self.errors[ix],
            lambda states: actor(states, deterministic=True)[0], self.protocol["qdelta_updates_per_extra_step"])


class ContextTrainer(usage.UsageTrainer):
    def configure_context(self, p, scenario, bundle):
        self.protocol, self.arm = p, p["arm"]
        self.output = HERE / p["relative_out"] / "results"
        self.blocks = CompletedBlocks(p["context_window_physical_rows"])
        self.trace, self.context_filter = [], None
        self.coefficient = FIXED.get(self.arm, p["model_fraction"])
        self.bundle = bundle
        if self.arm == "oracle_context":
            self.coefficient = bundle["coefficients"][scenario]
        elif self.arm == "recognized_context":
            classes = bundle["recognizer"]["classes"]
            self.context_filter = ContextFilter(bundle["recognizer"],
                [bundle["coefficients"][c] for c in classes], p)
            self.coefficient = self.context_filter.coefficient

    def observe_completed(self):
        def observe(block):
            if self.context_filter is not None:
                self.coefficient = self.context_filter.observe(block["features"])
                self.trace.append(dict(end_source_row=block["end_source_row"],
                    belief=self.context_filter.belief.tolist(), coefficient=self.coefficient))

        def reset():
            if self.context_filter is not None:
                self.context_filter.reset()
                self.coefficient = self.context_filter.coefficient

        self.blocks.consume(self.replay_buffer, observe, reset)

    def _ensemble_td_loss(self, predictions, targets, weights=None):
        real = raw_trainer._ensemble_td_loss(self, predictions, targets, weights)
        if self.collect_diagnostics:
            self.real_losses.append(real)
            if len(self.real_losses) == 2:
                fraction = self.config.online_fraction
                total = (1-fraction)*self.real_losses[0] + fraction*self.real_losses[1]
                self.loss_diagnostics["real_td_loss"] = float(total.detach())
                if self.model_loss is not None:
                    relation = usage.diagnostic.gradient_relation(total, self.model_loss,
                        tuple(p for p in self.critic.parameters() if p.requires_grad), self.protocol["model_fraction"])
                    relation["scaled_real_gradient_norm"] = self.protocol["real_td_coefficient"]*relation["real_gradient_norm"]
                    relation["scaled_model_gradient_norm"] = self.coefficient*relation["model_gradient_norm"]
                    self.loss_diagnostics.update(relation)
        return self.protocol["real_td_coefficient"]*real + (
            self.coefficient*self.model_loss if self.model_loss is not None else 0)

    def update(self, *, online, online_only=False, collect_metrics=True):
        collect_metrics = collect_metrics or (self.protocol["source_event_budget"] < 1000
                                               and (self.total_updates+1) % 16 == 0)
        self.collect_diagnostics, self.loss_diagnostics, self.real_losses = collect_metrics, {}, []
        self.model_loss = None
        metrics = {}
        devices = [torch.cuda.current_device()] if self.device.type == "cuda" else []
        if online and self.augmentation is not None:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.model_seed+self.total_updates)
                if self.total_updates % self.protocol["trust_update_every"] == 0:
                    with self.costs.phase("bellman_trust_update"):
                        self.augmentation.update_trust(self.actor)
                with self.costs.phase("model_training_batch"):
                    batch, weights = self.augmentation.generate(self.actor, self.protocol["model_batch_size"])
                    q = self.critic(batch["observations"], batch["actions"])
                    target = self._target(batch, backup="h2oplus")
                    self.model_loss = raw_trainer._ensemble_td_loss(self, q, target, weights)
                    metrics = dict(model_td_loss=float(self.model_loss.detach()), model_weight_mean=float(weights.mean())) if collect_metrics else {}
        try:
            result = raw_trainer.update(self, online=online, online_only=online_only, collect_metrics=collect_metrics)
            result.update(metrics)
            if collect_metrics:
                row = dict(total_updates=self.total_updates, arm=self.arm,
                    real_td_coefficient=self.protocol["real_td_coefficient"], model_td_coefficient=self.coefficient,
                    metrics=result, **self.loss_diagnostics)
                if self.augmentation is not None:
                    with self.costs.phase("mechanism_diagnostics"), torch.no_grad():
                        row["model"] = self.model_diagnostics()
                if self.context_filter is not None:
                    row["context_belief"] = self.context_filter.belief.tolist()
                write(self.output / "diagnostics" / f"update{self.total_updates:06d}.json", row)
            return result
        finally:
            self.model_loss, self.real_losses = None, []

    def save_context(self):
        write(self.output / "context_blocks.json", dict(blocks=self.blocks.blocks))
        write(self.output / "context_trace.json", dict(trace=self.trace,
            final_coefficient=self.coefficient, consumed_source_rows=self.blocks.cursor,
            selection=self.bundle))
