"""Normalize decision errors using independent stopping-error units."""

import torch


def rms_scale(errors, floor):
    return torch.stack(errors).square().mean().sqrt().clamp_min(floor)


def balanced_model_class(usage, parent):
    from model import losses
    from decision_model import combined_error

    class BalancedModel(parent):
        @torch.no_grad()
        def select(self, event, weights):
            self._freeze_reference()
            self.decision_event = event
            stop = self.target["stop"]
            proper_errors, decision_errors = [], []
            for family in ("direct", "residual"):
                output = torch.stack([m(stop["s"], stop["a"]) for m in self.banks[family]]).mean(0)
                proper_errors.append(losses(output, stop, self.norm).mean(1).mean())
                decision_errors.append(self._decision_error(output, stop,
                    self.protocol["training_seed"]+event+71000))
            self.prediction_scale = torch.stack(proper_errors).mean().clamp_min(.1)
            self.decision_scale = rms_scale(decision_errors, self.protocol["decision_scale_floor"])
            batch = self.target["calibrate"]
            scores, components = {}, {}
            for family in ("direct", "residual"):
                output = torch.stack([m(batch["s"], batch["a"]) for m in self.banks[family]]).mean(0)
                proper = losses(output, batch, self.norm).mean(1)
                decision = self._decision_error(output, batch,
                    self.protocol["training_seed"]+event+81000)
                error = combined_error(proper, decision, self.prediction_scale,
                    self.decision_scale, self.protocol["decision_loss_coefficient"])
                scores[family] = float(error.mean() if weights is None else (error*weights).sum()/weights.sum())
                components[family] = dict(proper_error=float(proper.mean()),
                    backup_absolute_error=float(decision.mean()),
                    normalized_proper_error=float(proper.mean()/self.prediction_scale),
                    normalized_decision_error=float(decision.mean()/self.decision_scale))
            self.family = min(scores, key=scores.get)
            self.models = self.banks[self.family]
            self.history.append(dict(source_events=event, selected_family=self.family,
                calibration_loss=scores, decision_components=components,
                prediction_scale=float(self.prediction_scale), decision_scale=float(self.decision_scale),
                stopping_backup_error_rms_by_family={name:float(err.square().mean().sqrt())
                    for name, err in zip(("direct", "residual"), decision_errors)},
                scale_mode=self.protocol["decision_scale_mode"]))
            usage.write(self.out/"calibration_decision.json", dict(selected_family=self.family, history=self.history))

    return BalancedModel
