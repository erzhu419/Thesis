"""Observe the actual real/model TD update without drawing another minibatch."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcwm_closed_loop_v1"))
from learning import ClosedLoopTrainer, TransitH2OTrainerV7


def distribution(value):
    value = value.detach().float().reshape(-1)
    quantiles = torch.quantile(value, value.new_tensor([.05, .5, .95]))
    return dict(n=value.numel(), mean=float(value.mean()), std=float(value.std(unbiased=False)),
                min=float(value.min()), p05=float(quantiles[0]), median=float(quantiles[1]),
                p95=float(quantiles[2]), max=float(value.max()))


def cosine(left, right):
    left, right = left.detach().double(), right.detach().double()
    denominator = left.norm() * right.norm()
    return float((torch.dot(left, right) / denominator).clamp(-1., 1.)) if denominator > 0 else None


def gradient_report(real, model, *, real_scale, model_scale):
    """Model gradient already contains the absolute per-row confidence weights."""
    real_part, model_part = real_scale * real, model_scale * model
    combined = real_part + model_part
    return dict(real_raw_norm=float(real.norm()), real_scaled_norm=float(real_part.norm()),
        model_trust_weighted_norm=float(model.norm()), model_scaled_norm=float(model_part.norm()),
        combined_norm=float(combined.norm()), cosine_real_model=cosine(real_part, model_part),
        model_to_real_scaled_norm_ratio=(float(model_part.norm() / real_part.norm())
                                        if real_part.norm() > 0 else None),
        cosine_combined_real=cosine(combined, real),
        same_minibatch_gradient_directions=dict(
            real1_norm=float(real.norm()), real075_norm=float((.75 * real).norm()),
            model_or_context_current_norm=float((.75 * real + .25 * model).norm()),
            real075_cosine_real1=cosine(.75 * real, real),
            model_or_context_current_cosine_real1=cosine(.75 * real + .25 * model, real),
            interpretation="Algebraic gradient directions on this update's actual minibatch; no shadow optimizer steps or independent context counterfactual"))


def parameter_vector(module):
    return torch.cat([parameter.detach().reshape(-1) for parameter in module.parameters()])


def parameter_change(module, before):
    delta = parameter_vector(module) - before
    return dict(l2=float(delta.norm()), max_absolute=float(delta.abs().max()),
                relative_l2=float(delta.norm() / before.norm()) if before.norm() > 0 else None), delta


class DiagnosticTrainer(ClosedLoopTrainer):
    """Four explicit objectives: real1, real075, model, and context.

    ``continue_step(collect_diagnostics=True)`` additionally returns a JSON-ready
    ``td_diagnostic`` record. Use it on the first update of each feedback packet.
    Ordinary updates take the same sampling, trust, actor and critic paths as v1.
    Initialization and the CPU optimizer conversion remain the runner's job.
    """

    def __init__(self, *args, arm, **kwargs):
        if arm not in ("real1", "real075", "model", "context"):
            raise ValueError(f"Unknown diagnostic arm: {arm}")
        self.arm = arm
        self._collect_td = False
        super().__init__(*args, **kwargs)

    def _gradient(self, loss):
        parameters = tuple(self.critic.parameters())
        values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        return torch.cat([(torch.zeros_like(parameter) if value is None else value.detach()).reshape(-1)
                          for parameter, value in zip(parameters, values)])

    def _target(self, batch, *, backup):
        target = super()._target(batch, backup=backup)
        if self._collect_td:
            record = dict(rows=len(target), reward=distribution(batch["rewards"]),
                reward_scaled=distribution(batch["rewards"] * self.config.reward_scale),
                dt_seconds=distribution(batch["delta_t_seconds"]), target_q=distribution(target),
                terminal_rows=int(batch["terminals"].sum()))
            (self._model_batches if self._making_model else self._real_batches).append(record)
        return target

    def _ensemble_td_loss(self, predictions, target, weights=None):
        # Bypass ClosedLoopTrainer's mixture: capture the actual raw partition
        # loss once, then apply this arm's explicit real/model coefficients.
        real_loss = TransitH2OTrainerV7._ensemble_td_loss(self, predictions, target, weights)
        if self._collect_td:
            self._real_parts.append(dict(loss=float(real_loss.detach()), gradient=self._gradient(real_loss),
                q=distribution(predictions), target_q=distribution(target),
                absolute_td_error=distribution((predictions.detach() - target.detach()[None]).abs())))
        real_scale, model_scale = self._scales()
        return real_scale * real_loss + (model_scale * self.model_loss if self.model_loss is not None else 0.)

    def _scales(self):
        if self.arm == "real075":
            return .75, 0.
        if self.arm in ("model", "context") and self.model_loss is not None:
            return .75, .25
        return 1., 0.

    def _finish_diagnostic(self, before):
        count = len(self._real_parts)
        fractions = ([1. - self.config.online_fraction, self.config.online_fraction]
                     if count == 2 else [1.])
        real = sum(fraction * part["gradient"] for fraction, part in zip(fractions, self._real_parts))
        raw_real_loss = sum(fraction * part["loss"] for fraction, part in zip(fractions, self._real_parts))
        model = self._model_gradient if self._model_gradient is not None else torch.zeros_like(real)
        real_scale, model_scale = self._scales()
        actor_change, _ = parameter_change(self.actor, before["actor"])
        critic_change, critic_delta = parameter_change(self.critic, before["critic"])
        model_loss = float(self.model_loss.detach()) if self.model_loss is not None else 0.
        return dict(arm=self.arm, update_before=before["updates"], update_after=self.total_updates,
            real_td_loss=raw_real_loss, real_td_contribution=real_scale * raw_real_loss,
            model_td_loss_trust_weighted=model_loss, model_td_contribution=model_scale * model_loss,
            real_scale=real_scale, model_scale=model_scale,
            effective_model_mean_weight_scale=(model_scale * self._trust["mean"] if self._trust else 0.),
            absolute_trust=self._trust, model_unweighted_td_loss=self._model_unweighted_loss,
            real_partitions=[dict(fraction=fraction, **{key: value for key, value in part.items() if key != "gradient"})
                             for fraction, part in zip(fractions, self._real_parts)],
            real_batches=self._real_batches, model_batches=self._model_batches,
            model_predictions=self._model_predictions,
            gradients=gradient_report(real, model, real_scale=real_scale, model_scale=model_scale),
            actor_parameter_change=actor_change, critic_parameter_change=critic_change,
            critic_change_cosine_negative_total_gradient=cosine(
                critic_delta, -(real_scale * real + model_scale * model)),
            model_and_real_means_are_unpaired="Generated actions and states differ from real replay; batch mean differences are not prediction errors")

    def continue_step(self, collect_diagnostics=False):
        self._collect_td = collect_diagnostics
        self._making_model = False
        self.model_loss = None
        self._real_parts, self._real_batches, self._model_batches = [], [], []
        self._model_gradient = self._trust = self._model_predictions = None
        self._model_unweighted_loss = None
        before = (dict(actor=parameter_vector(self.actor).clone(), critic=parameter_vector(self.critic).clone(),
                       updates=self.total_updates) if collect_diagnostics else None)
        metrics = {}
        try:
            if self.arm in ("model", "context") and self.world is not None and self.replay_buffer.recent:
                devices = [self.device.index or 0] if self.device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(self.world.seed + 30_000 + self.total_updates)
                    self.world.update_trust(self.actor, self.config.batch_size)
                    generated = self.world.generate(self.actor, self.config.batch_size)
                    if generated is not None:
                        batch, weights = generated
                        prediction = self.critic(batch["observations"], batch["actions"])
                        self._making_model = True
                        target = self._target(batch, backup="h2oplus")
                        self._making_model = False
                        error = self._elementwise_td_loss(prediction, target)
                        self.model_loss = (error * weights[None]).mean()
                        if collect_diagnostics:
                            self._model_gradient = self._gradient(self.model_loss)
                            self._trust = distribution(weights)
                            self._model_unweighted_loss = float(error.detach().mean())
                            self._model_predictions = dict(q=distribution(prediction),
                                absolute_td_error=distribution((prediction.detach() - target[None]).abs()))
                        metrics = dict(model_td_loss=float(self.model_loss.detach()),
                                       model_trust_mean=float(weights.mean()))
            # Exactly one update; autograd.grad above never writes parameter.grad.
            result = super().update(online=True)
            if collect_diagnostics:
                result["td_diagnostic"] = self._finish_diagnostic(before)
            return dict(result, **metrics)
        finally:
            self.model_loss = None
            self._collect_td = self._making_model = False
            self._real_parts = []
            self._model_gradient = None
