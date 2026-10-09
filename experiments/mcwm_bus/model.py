"""Hybrid nonterminal event predictor; discrete topology is never teacher-forced at inference."""

import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from data import load_protocol

sys.path.insert(0, load_protocol()["h2o_root"])
from SimpleSAC.transit_v7.networks import TransitStateEncoderV7

WIDTH = 4 + 16 + 16 * 69


def targets(data, stations, device):
    s = torch.as_tensor(data["s"], dtype=torch.float32, device=device)
    ns = torch.as_tensor(data["ns"], dtype=torch.float32, device=device)
    slots = ns[:, 5:].reshape(-1, 16, 14)
    line = slots[:, :, 1].round().long()
    station = (slots[:, :, 2] * stations[line]).round().long()
    return {
        "s": s, "a": torch.as_tensor(data["a"], dtype=torch.float32, device=device).reshape(-1, 1),
        "g": torch.stack((
            torch.log1p(torch.as_tensor(data["dt"], device=device).reshape(-1)),
            torch.log1p(-torch.as_tensor(data["r"], device=device).reshape(-1)),
            (ns[:, 3] - s[:, 3]) * 20000, (ns[:, 4] - s[:, 4]) * 20000), dim=1).float(),
        "count": (ns[:, 1] * 16).round().long() - 1,
        "mask": slots[:, :, 0] > .5, "line": line, "station": station,
        "flags": slots[:, :, 11:13], "local": slots[:, :, 3:11],
    }


class Normalizer(nn.Module):
    def __init__(self, fit):
        super().__init__()
        for name, values in (("g", fit["g"]), ("local", fit["local"][fit["mask"]])):
            self.register_buffer(name + "_mean", values.mean(0))
            self.register_buffer(name + "_std", values.std(0).clamp_min(.05))


class BusModel(nn.Module):
    def __init__(self, base=None):
        super().__init__()
        self.encoder = TransitStateEncoderV7()
        self.head = nn.Sequential(nn.Linear(257, 256), nn.SiLU(), nn.Linear(256, WIDTH))
        self.base = base
        if base is not None:
            base.requires_grad_(False)
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def forward(self, s, a):
        output = self.head(torch.cat((self.encoder(s), a), dim=-1))
        if self.base is not None:
            with torch.no_grad():
                baseline = self.base(s, a)
            output = output + baseline
        return output


def unpack(output):
    slots = output[:, 20:].reshape(-1, 16, 69)
    return dict(g=output[:, :4], count=output[:, 4:20], line=slots[:, :, :12],
                station=slots[:, :, 12:59], flags=slots[:, :, 59:61], local=slots[:, :, 61:])


class MarginalModel(nn.Module):
    """Target-fit frequency/mean baseline, with no state/action conditioning."""

    def __init__(self, fit, norm):
        super().__init__()
        output = torch.zeros((1, WIDTH), device=fit["s"].device)
        p = unpack(output)
        p["g"][0] = (fit["g"].mean(0) - norm.g_mean) / norm.g_std
        p["count"][0] = torch.log(torch.bincount(fit["count"], minlength=16).float() + 1.)
        for slot in range(16):
            active = fit["mask"][:, slot]
            for key, width in (("line", 12), ("station", 47)):
                p[key][0, slot] = torch.log(torch.bincount(fit[key][active, slot], minlength=width).float() + 1.)
            if active.any():
                flags = fit["flags"][active, slot].mean(0).clamp(1e-4, 1-1e-4)
                p["flags"][0, slot] = torch.logit(flags)
                p["local"][0, slot] = (fit["local"][active, slot].mean(0) - norm.local_mean) / norm.local_std
        self.register_buffer("constant", output)

    def forward(self, s, a):
        return self.constant.expand(len(s), -1)


def losses(output, y, norm):
    pred = unpack(output)
    mask = y["mask"].float()
    size = mask.sum(1).clamp_min(1)
    per_slot = lambda value: (value * mask).sum(1) / size
    ce = lambda logits, label: F.cross_entropy(logits.flatten(0, 1), label.flatten(),
                                               reduction="none").reshape_as(mask)
    return torch.stack((
        ((pred["g"] - (y["g"] - norm.g_mean) / norm.g_std) ** 2).mean(1),
        per_slot(((pred["local"] - (y["local"] - norm.local_mean) / norm.local_std) ** 2).mean(2)),
        F.cross_entropy(pred["count"], y["count"], reduction="none"),
        per_slot(ce(pred["line"], y["line"])),
        per_slot(ce(pred["station"], y["station"])),
        per_slot(F.binary_cross_entropy_with_logits(pred["flags"], y["flags"], reduction="none").mean(2)),
    ), dim=1)


@torch.no_grad()
def metrics(output, y, norm):
    p = unpack(output)
    mask = y["mask"]
    g = p["g"] * norm.g_std + norm.g_mean
    predicted_physical = torch.expm1(g[:, :2].clamp(0, 20))
    actual_physical = torch.expm1(y["g"][:, :2])
    heads = losses(output, y, norm).mean(0)
    local = p["local"] * norm.local_std + norm.local_mean
    predicted_count = p["count"].argmax(-1)
    predicted_line = p["line"].argmax(-1)
    predicted_station = p["station"].argmax(-1)
    return {
        "rows": len(output), "selection_loss": float(heads.mean()),
        "head_losses": dict(zip(("global", "local", "count", "line", "station", "neighbor"), heads.tolist())),
        "duration_mae_s": float((predicted_physical[:, 0] - actual_physical[:, 0]).abs().mean()),
        "cost_mae_passenger_s": float((predicted_physical[:, 1] - actual_physical[:, 1]).abs().mean()),
        "duration_log_rmse": float(((g[:, 0] - y["g"][:, 0]) ** 2).mean().sqrt()),
        "cost_log_rmse": float(((g[:, 1] - y["g"][:, 1]) ** 2).mean().sqrt()),
        "waiting_increment_mae_passengers": float((g[:, 2] - y["g"][:, 2]).abs().mean()),
        "onboard_increment_mae_passengers": float((g[:, 3] - y["g"][:, 3]).abs().mean()),
        "count_accuracy": float((predicted_count == y["count"]).float().mean()),
        "line_accuracy_on_true_active_slots": float((predicted_line[mask] == y["line"][mask]).float().mean()),
        "station_accuracy_on_true_active_slots": float((predicted_station[mask] == y["station"][mask]).float().mean()),
        "slot_identity_accuracy": float(((predicted_line == y["line"]) &
                                          (predicted_station == y["station"]))[mask].float().mean()),
        "local_rmse_on_true_active_slots": float(((local[mask] - y["local"][mask]) ** 2).mean().sqrt()),
        "neighbor_accuracy_on_true_active_slots": float(((p["flags"][mask] > 0) == (y["flags"][mask] > .5)).float().mean()),
        "label_conditioning": "True active slots only index scoring; all predictions use current state and action only",
    }
