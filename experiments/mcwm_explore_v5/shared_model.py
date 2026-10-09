"""A shared representation and calibration-only output-mechanism selection."""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from model import WIDTH, TransitStateEncoderV7, losses, unpack


OUTPUTS = {
    "event": list(range(4, 20))+[20+69*j+k for j in range(16) for k in range(59)],
    "duration": [0],
    "service": [2, 3]+[20+69*j+k for j in range(16) for k in range(65, 69)],
    "headway": [20+69*j+k for j in range(16) for k in range(59, 65)],
    "cost": [1],
}


class SharedBusModel(nn.Module):
    def __init__(self, base=None):
        super().__init__()
        self.encoder = TransitStateEncoderV7()
        self.head = nn.ModuleDict({name: nn.Sequential(nn.Linear(257, 128), nn.SiLU(),
            nn.Linear(128, len(indices))) for name, indices in OUTPUTS.items()})
        for name, indices in OUTPUTS.items():
            self.register_buffer("outputs_"+name, torch.tensor(indices))
        self.base = base
        if base is not None:
            base.requires_grad_(False)
            for head in self.head.values():
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

    def forward(self, states, actions):
        features = torch.cat((self.encoder(states), actions), -1)
        output = states.new_zeros((len(states), WIDTH))
        for name, head in self.head.items():
            output[:, getattr(self, "outputs_"+name)] = head(features)
        if self.base is not None:
            with torch.no_grad():
                baseline = self.base(states, actions)
            output = output+baseline
        return output


def mechanism_losses(output, batch, norm):
    pred = unpack(output)
    mask = batch["mask"].float()
    average = lambda rows: (rows*mask).sum(1)/mask.sum(1).clamp_min(1)
    global_error = (pred["g"]-(batch["g"]-norm.g_mean)/norm.g_std).square()
    local_error = (pred["local"]-(batch["local"]-norm.local_mean)/norm.local_std).square()
    ce = lambda logits, labels: average(F.cross_entropy(logits.flatten(0, 1), labels.flatten(),
        reduction="none").reshape_as(mask))
    event = (F.cross_entropy(pred["count"], batch["count"], reduction="none")+
        ce(pred["line"], batch["line"])+ce(pred["station"], batch["station"]))/3
    flags = average(F.binary_cross_entropy_with_logits(pred["flags"], batch["flags"],
        reduction="none").mean(2))
    return dict(event=event, duration=global_error[:, 0],
        service=(global_error[:, 2:4].mean(1)+average(local_error[:, :, 4:].mean(2)))/2,
        headway=(flags+average(local_error[:, :, :4].mean(2)))/2, cost=global_error[:, 1])


def select_outputs(outputs, gates):
    result = torch.empty_like(outputs["direct"])
    for name, indices in OUTPUTS.items():
        result[:, indices] = outputs[gates[name]][:, indices]
    return result


def shared_model_class(usage):
    class SharedModel(usage.UsageModel):
        def __init__(self, replay, protocol, out, device, costs):
            self.gated = protocol["arm"] == "shared_gated"
            super().__init__(replay, protocol, out, device, costs)
            self.mode = "normalized"
            model = self.banks["direct"][0]
            usage.write(Path(out)/"architecture.json", dict(
                direct_parameters=sum(p.numel() for p in model.parameters()),
                encoders_per_member=1, selection="per mechanism" if self.gated else "whole model",
                head_output_sizes={k: len(v) for k, v in OUTPUTS.items()}))
            self._refresh_gated_errors()

        @torch.no_grad()
        def select(self, event, weights):
            batch = self.target["calibrate"]
            whole_scores, group_scores = {}, {}
            for family in ("direct", "residual"):
                output = torch.stack([m(batch["s"], batch["a"]) for m in self.banks[family]]).mean(0)
                mean = lambda error: float(error.mean() if weights is None else (error*weights).sum()/weights.sum())
                whole_scores[family] = mean(losses(output, batch, self.norm).mean(1))
                group_scores[family] = {name: mean(error) for name, error in mechanism_losses(output, batch, self.norm).items()}
            self.family = min(whole_scores, key=whole_scores.get)
            self.models = self.banks[self.family]
            self.gates = {name: min(group_scores, key=lambda family: group_scores[family][name])
                if self.gated else self.family for name in OUTPUTS}
            self.history.append(dict(source_events=event, selected_family=self.family,
                calibration_loss=whole_scores, mechanism_loss=group_scores, selected_mechanisms=self.gates.copy(),
                heterogeneous_selection=len(set(self.gates.values())) > 1))
            usage.write(self.out/"calibration_decision.json", dict(selected_family=self.family, history=self.history))

        @torch.no_grad()
        def predict(self, s, a):
            if not self.gated:
                return super().predict(s, a)
            outputs = {family: torch.stack([m(s, a) for m in self.banks[family]]).mean(0)
                for family in set(self.gates.values())}
            if "direct" not in outputs:
                outputs["direct"] = next(iter(outputs.values()))
            return select_outputs(outputs, self.gates)

        @torch.no_grad()
        def _refresh_gated_errors(self):
            if not self.gated:
                return
            with self.costs.phase("gated_trust_errors"):
                self.errors = torch.cat([losses(self.predict(self.fit["s"][i:i+256], self.fit["a"][i:i+256]),
                    {k: v[i:i+256] for k, v in self.fit.items()}, self.norm).mean(1)
                    for i in range(0, len(self.fit["s"]), 256)])

        def refit(self, replay, event):
            super().refit(replay, event)
            self._refresh_gated_errors()

    return SharedModel
