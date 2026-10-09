"""One-step bus model augmentation with the original MC-WM Bellman trust critic."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from data import HERE, load_protocol
from model import BusModel, Normalizer, losses, targets, unpack

sys.path.insert(0, str(load_protocol().get("mcwm_root", Path(load_protocol()["h2o_root"]).parents[1] / "MC-WM")))
from mc_wm.policy.qdelta_bellman import QDeltaBellman


class ModelAugmentation:
    def __init__(self, device, protocol, stations, model_run=None):
        self.device = device
        self.protocol = protocol
        self.stations = torch.as_tensor(stations, dtype=torch.float32, device=device)
        run = Path(model_run) if model_run is not None else HERE / "runs/pilot_v1"
        payload = torch.load(run / "pilot_models.pt", map_location=device, weights_only=False)
        self.family = json.loads((run / "calibration_decision.json").read_text())["selected_family"]
        self.models = []
        for state in payload["models"][self.family]:
            model = BusModel(BusModel() if self.family == "residual" else None).to(device)
            model.load_state_dict(state)
            model.eval().requires_grad_(False)
            self.models.append(model)
        self.norm = Normalizer({"g": torch.zeros(2, 4, device=device),
                                "local": torch.zeros(2, 16, 8, device=device),
                                "mask": torch.ones(2, 16, dtype=torch.bool, device=device)}).to(device)
        self.norm.load_state_dict(payload["normalizer"])
        with np.load(run / "target_fit.npz") as fit:
            self.fit = targets(dict(fit), self.stations, device)
            self.fit_ns = torch.as_tensor(fit["ns"], dtype=torch.float32, device=device)
        with torch.no_grad():
            self.errors = torch.cat([losses(self.predict(self.fit["s"][i:i+256], self.fit["a"][i:i+256]),
                                            {k: v[i:i+256] for k, v in self.fit.items()}, self.norm).mean(1)
                                     for i in range(0, len(self.fit_ns), 256)])
        self.trust = QDeltaBellman(229, 1, gamma=protocol["qdelta_gamma"], device=device)
        self.rng = np.random.default_rng(protocol["training_seed"] + 30000)
        self.generator = torch.Generator(device=device).manual_seed(protocol["training_seed"] + 40000)

    @torch.no_grad()
    def predict(self, s, a):
        return torch.stack([model(s, a) for model in self.models]).mean(0)

    def update_trust(self, actor):
        ix = torch.as_tensor(self.rng.integers(len(self.fit_ns), size=128), device=self.device)
        return self.trust.update(self.fit["s"][ix], self.fit["a"][ix], self.fit_ns[ix],
                                 torch.zeros(len(ix), device=self.device), self.errors[ix],
                                 lambda states: actor(states, deterministic=True)[0],
                                 n_iters=self.protocol["qdelta_updates_per_extra_step"])

    @torch.no_grad()
    def generate(self, actor, size):
        eligible = torch.nonzero(self.fit["s"][:, 0] < self.protocol["model_start_max_horizon_fraction"]).flatten()
        ix = eligible[torch.randint(len(eligible), (size,), generator=self.generator, device=self.device)]
        s = self.fit["s"][ix]
        a = actor(s, deterministic=True)[0]
        ns, reward, dt = self.decode(s, self.predict(s, a))
        return {"observations": s, "actions": a, "next_observations": ns,
                "rewards": reward, "delta_t_seconds": dt,
                "discounts": torch.ones(size, device=self.device),
                "terminals": torch.zeros(size, device=self.device)}, self.trust.weight(s, a)

    @torch.no_grad()
    def decode(self, s, output):
        p = unpack(output)
        global_values = p["g"] * self.norm.g_std + self.norm.g_mean
        dt = torch.expm1(global_values[:, 0].clamp(0, 10)).clamp_min(1.)
        dt = torch.minimum(dt, (1 - s[:, 0]) * 24000 - 1)
        reward = -torch.expm1(global_values[:, 1].clamp(0, 20))
        draw = lambda logits: torch.multinomial(logits.softmax(-1).flatten(0, -2), 1,
                                                 generator=self.generator).reshape(logits.shape[:-1])
        count = draw(p["count"]) + 1
        line = draw(p["line"])
        station_logits = p["station"].clone()
        legal = torch.arange(47, device=self.device)[None, None] <= self.stations[line, None]
        station_logits.masked_fill_(~legal, -torch.inf)
        station = draw(station_logits)
        mask = torch.arange(16, device=self.device)[None] < count[:, None]
        ns = torch.zeros_like(s)
        ns[:, 0] = s[:, 0] + dt / 24000
        ns[:, 1] = count / 16
        ns[:, 3:5] = (s[:, 3:5] + global_values[:, 2:4] / 20000).clamp(0, 1.25)
        slots = ns[:, 5:].reshape(-1, 16, 14)
        slots[:, :, 0] = mask
        slots[:, :, 1] = line
        slots[:, :, 2] = station / self.stations[line]
        continuous = p["local"] * self.norm.local_std + self.norm.local_mean
        lower = torch.tensor([0, 0, .001, .001, 0, 0, 0, 0], device=self.device)
        upper = torch.tensor([5, 5, 10, 10, 2, 1, 2, 2], device=self.device)
        slots[:, :, 3:11] = continuous.clamp(lower, upper)
        flags = torch.bernoulli(p["flags"].sigmoid(), generator=self.generator)
        slots[:, :, 11:13] = flags
        slots[:, :, 3:5] = torch.where(flags > .5, slots[:, :, 3:5], 1.)
        slots[:, :, :13] *= mask[:, :, None]
        slots[:, :, 13] = -1.
        return ns, reward, dt
