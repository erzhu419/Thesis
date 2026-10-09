"""Proper predictive fitting with decision-aware calibration and trust errors."""

import copy
from pathlib import Path

import torch


def combined_error(prediction, decision, prediction_scale, decision_scale, coefficient):
    return prediction/prediction_scale + coefficient*decision/decision_scale


def decision_model_class(usage):
    from model import losses
    from run_pilot import take

    class DecisionModel(usage.UsageModel):
        learner = None

        def __init__(self, replay, protocol, out, device, costs):
            self.device, self.protocol = device, protocol
            self.stations = torch.as_tensor(replay.station_max_by_line, dtype=torch.float32, device=device)
            super().__init__(replay, protocol, out, device, costs)
            self._refresh_decision_errors()

        def _freeze_reference(self):
            learner = self.learner
            self.reference_actor = copy.deepcopy(learner.actor).eval().requires_grad_(False)
            self.reference_critic = copy.deepcopy(learner.target_critic).eval().requires_grad_(False)
            self.reference_value = copy.deepcopy(learner.value).eval().requires_grad_(False)
            self.reference_mix = learner.config.mixed_backup_lambda
            self.reference_reward_scale = learner.config.reward_scale

        @torch.no_grad()
        def _reference_backup(self, ns, reward):
            action = self.reference_actor(ns, deterministic=True)[0]
            q = self.reference_critic(ns, action).min(0).values
            value = self.reference_value(ns).flatten()
            return self.reference_reward_scale*reward.flatten() + (
                self.reference_mix*value+(1-self.reference_mix)*q)

        @torch.no_grad()
        def _decision_error(self, output, batch, seed):
            previous = getattr(self, "generator", None)
            self.generator = torch.Generator(device=self.device).manual_seed(seed)
            actual = self._reference_backup(batch["logged_ns"], batch["logged_reward"])
            errors = []
            try:
                for _ in range(self.protocol["decision_draws"]):
                    ns, reward, _ = self.decode(batch["s"], output)
                    errors.append((self._reference_backup(ns, reward)-actual).abs())
            finally:
                if previous is None:
                    del self.generator
                else:
                    self.generator = previous
            return torch.stack(errors).mean(0)

        @torch.no_grad()
        def select(self, event, weights):
            self._freeze_reference()
            self.decision_event = event
            stop = self.target["stop"]
            stop_backup = self._reference_backup(stop["logged_ns"], stop["logged_reward"])
            self.decision_scale = stop_backup.std().clamp_min(self.protocol["decision_scale_floor"])
            stop_errors = []
            for family in ("direct", "residual"):
                output = torch.stack([m(stop["s"], stop["a"]) for m in self.banks[family]]).mean(0)
                stop_errors.append(losses(output, stop, self.norm).mean(1).mean())
            self.prediction_scale = torch.stack(stop_errors).mean().clamp_min(.1)
            batch = self.target["calibrate"]
            scores, components = {}, {}
            for family in ("direct", "residual"):
                output = torch.stack([m(batch["s"], batch["a"]) for m in self.banks[family]]).mean(0)
                proper = losses(output, batch, self.norm).mean(1)
                decision = self._decision_error(output, batch, self.protocol["training_seed"]+event+81000)
                error = combined_error(proper, decision, self.prediction_scale,
                    self.decision_scale, self.protocol["decision_loss_coefficient"])
                scores[family] = float(error.mean() if weights is None else (error*weights).sum()/weights.sum())
                components[family] = dict(proper_error=float(proper.mean()),
                    backup_absolute_error=float(decision.mean()))
            self.family = min(scores, key=scores.get)
            self.models = self.banks[self.family]
            self.history.append(dict(source_events=event, selected_family=self.family,
                calibration_loss=scores, decision_components=components,
                prediction_scale=float(self.prediction_scale), decision_scale=float(self.decision_scale)))
            usage.write(self.out/"calibration_decision.json", dict(selected_family=self.family, history=self.history))

        @torch.no_grad()
        def _refresh_decision_errors(self):
            errors, decisions = [], []
            with self.costs.phase("decision_trust_errors"):
                for start in range(0, len(self.fit["s"]), 256):
                    ix = torch.arange(start, min(start+256, len(self.fit["s"])), device=self.device)
                    batch = take(self.fit, ix)
                    output = self.predict(batch["s"], batch["a"])
                    proper = losses(output, batch, self.norm).mean(1)
                    decision = self._decision_error(output, batch,
                        self.protocol["training_seed"]+self.decision_event+start+91000)
                    decisions.append(decision)
                    errors.append(combined_error(proper, decision, self.prediction_scale,
                        self.decision_scale, self.protocol["decision_loss_coefficient"]))
                self.errors = torch.cat(errors)
            usage.write(Path(self.out)/"decision_trust.json", dict(
                source_events=self.decision_event, rows=len(self.errors),
                mean_backup_error=float(torch.cat(decisions).mean()),
                mean_combined_error=float(self.errors.mean()), reference=self.protocol["decision_reference"]))

        def refit(self, replay, event):
            super().refit(replay, event)
            self._refresh_decision_errors()

        def update_trust(self, actor):
            from mechanisms import update_normalized_trust
            ix = torch.as_tensor(self.rng.integers(len(self.fit_ns), size=128), device=self.device)
            return update_normalized_trust(self.trust, self.fit["s"][ix], self.fit["a"][ix],
                self.fit_ns[ix], torch.zeros(len(ix), device=self.device), self.errors[ix],
                lambda states: actor(states, deterministic=True)[0],
                self.protocol["qdelta_updates_per_extra_step"])

    return DecisionModel


def logged_targets_factory(original):
    def targets(data, stations, device):
        result = original(data, stations, device)
        result["logged_ns"] = torch.as_tensor(data["ns"], dtype=torch.float32, device=device)
        result["logged_reward"] = torch.as_tensor(data["r"], dtype=torch.float32, device=device).flatten()
        return result
    return targets
