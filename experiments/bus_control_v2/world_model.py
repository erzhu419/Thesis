"""Direct/residual station models and the existing normalized Bellman trust."""

import copy
import importlib.util

import numpy as np
import torch
from torch import nn

from station_common import P, MCROOT, state_headway_reward, write
from networks import features
from mc_wm.policy.qdelta_bellman import QDeltaBellman
from SimpleSAC.transit_v3.networks import MLP

# The station experiment uses the neural model, not the SINDy package imports.
spec = importlib.util.spec_from_file_location("station_neural_model", MCROOT / "mc_wm/residual/world_model.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ProbabilisticMLP = module.ProbabilisticMLP


class StationModel(nn.Module):
    def __init__(self, dim, base=None):
        super().__init__()
        self.base = copy.deepcopy(base).requires_grad_(False) if base is not None else None
        self.network = (ProbabilisticMLP(26+dim, dim, hidden=128) if base is None else
                        MLP(26+2*dim, 27+dim, (128, 128), activation="relu"))
        if base is not None:
            nn.init.zeros_(self.network.network[-1].weight)
            nn.init.zeros_(self.network.network[-1].bias)

    def forward(self, states, actions):
        if self.base is None:
            return self.network(features(states), actions)
        mean, logvar = self.base(states, actions)
        delta = self.network(torch.cat((features(states), actions), -1))
        return mean + delta, logvar


class StationModels:
    def __init__(self, replay, seed, folder, smoke=False):
        self.replay, self.seed, self.folder, self.smoke = replay, seed, folder, smoke
        self.rng = np.random.default_rng(seed+50000)
        self.dim = replay.offline["actions"].shape[1]
        self.banks, self.history = {k: [] for k in ("source", "direct", "residual")}, []
        target = replay.offline
        role = target["role"].astype(int)
        partitions = {"fit": np.flatnonzero(role < 8), "stop": np.flatnonzero(role == 8), "calibrate": np.flatnonzero(role == 9)}
        if smoke:
            boundary = np.array_split(np.arange(replay.offline_size), [int(.7*replay.offline_size), int(.85*replay.offline_size)])
            partitions = dict(zip(("fit", "stop", "calibrate"), boundary))
        if any(len(v) == 0 for v in partitions.values()):
            raise ValueError("world-model fit, stopping and calibration partitions must all be nonempty")
        self.target = {name: self.take(target, ids, 8192 if name == "fit" else 2048) for name, ids in partitions.items()}
        n = replay.online_size
        source = replay.online
        self.source = {"fit": self.take(source, np.arange(int(.8*n)), 8192),
                       "stop": self.take(source, np.arange(int(.8*n), n), 2048)}
        y = self.targets(self.source["fit"])
        self.mean, self.std = y.mean(0), y.std(0).clamp_min(.02)
        for member in range(P["model_members"]):
            torch.manual_seed(seed+50000+member)
            model = StationModel(self.dim)
            self.fit(model, self.source, P["model_updates"], None)
            model.eval().requires_grad_(False)
            self.banks["source"].append(model)
            for family in ("direct", "residual"):
                torch.manual_seed(seed+51000+member)
                candidate = StationModel(self.dim, model if family == "residual" else None)
                self.fit(candidate, self.target, P["model_updates"], None)
                candidate.eval().requires_grad_(False)
                self.banks[family].append(candidate)
        self.trust = QDeltaBellman(15+self.dim, self.dim, gamma=P["trust_gamma"], device="cpu")
        spec = importlib.util.spec_from_file_location("station_policy_density", MCROOT / "mc_wm/residual/policy_density.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.Density = module.PolicyDensity
        self.select(P["warmup_events"])

    def take(self, data, ids, limit):
        ids = np.sort(self.rng.choice(ids, min(limit, len(ids)), replace=False))
        return {k: torch.as_tensor(data[k][ids], dtype=torch.float32) for k in ("observations", "actions", "next_observations", "rewards")}

    @staticmethod
    def targets(batch):
        delta = features(batch["next_observations"])-features(batch["observations"])
        return torch.cat((delta, batch["rewards"][:, None]), -1)

    def fit(self, model, data, steps, weights):
        if self.smoke:
            steps = 8
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=3e-4)
        fit_y = (self.targets(data["fit"])-self.mean)/self.std
        stop_y = (self.targets(data["stop"])-self.mean)/self.std
        best, saved = float("inf"), None
        for step in range(steps):
            ids = torch.as_tensor(self.rng.integers(len(fit_y), size=128))
            mean, logvar = model(data["fit"]["observations"][ids], data["fit"]["actions"][ids])
            error = ((mean-fit_y[ids]).square()*(-logvar).exp()+logvar).mean(-1)
            loss = error.mean() if weights is None else (error*weights[ids]).sum()/weights[ids].sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.)
            optimizer.step()
            if (step+1) % 100 == 0 or step+1 == steps:
                with torch.no_grad():
                    mean = model(data["stop"]["observations"], data["stop"]["actions"])[0]
                    score = float((mean-stop_y).square().mean())
                if score < best:
                    best, saved = score, copy.deepcopy(model.state_dict())
        model.load_state_dict(saved)

    @torch.no_grad()
    def predict(self, states, actions, family=None):
        bank = self.banks[family or self.family]
        mean = torch.stack([m(states, actions)[0] for m in bank]).mean(0)
        return mean*self.std + self.mean

    def select(self, event):
        batch = self.target["calibrate"]
        with torch.no_grad():
            y = self.targets(batch)
            scores = {family: float(((self.predict(batch["observations"], batch["actions"], family)-y)/self.std).square().mean())
                      for family in ("direct", "residual")}
        self.family = min(scores, key=scores.get)
        self.history.append(dict(source_events=event, selected_family=self.family, calibration_mse=scores))
        write(self.folder / "history.json", self.history)

    def refit(self, event):
        replay = self.replay
        start = max(0, replay.online_size-2048)
        reference = np.concatenate((replay.online["observations"][start:replay.online_size],
                                    replay.online["actions"][start:replay.online_size]), -1)
        density = self.Density(strategy="buffer", w_min=.1, random_state=self.seed)
        density.fit_reference(reference)
        batch = self.target["fit"]
        weights = torch.as_tensor(density.weights(torch.cat((batch["observations"], batch["actions"]), -1).numpy()), dtype=torch.float32)
        for family in ("direct", "residual"):
            for model in self.banks[family]:
                model.network.requires_grad_(True)
                self.fit(model, self.target, P["model_refit_updates"], weights)
                model.eval().requires_grad_(False)
        self.select(event)

    def update_trust(self, actor):
        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcwm_usage_v3"))
        from mechanisms import update_normalized_trust
        batch = self.target["calibrate"]
        ids = torch.as_tensor(self.rng.integers(len(batch["rewards"]), size=128))
        s, a, ns = (batch[k][ids] for k in ("observations", "actions", "next_observations"))
        with torch.no_grad():
            errors = ((self.predict(s, a)-self.targets({k:v[ids] for k,v in batch.items()}))/self.std).square().mean(-1)
        update_normalized_trust(self.trust, s, a, ns, torch.zeros(len(ids)), errors,
                               lambda states: actor(states, deterministic=True)[0], 1)

    @torch.no_grad()
    def generate(self, actor, size):
        batch = self.target["fit"]
        eligible = torch.nonzero(batch["observations"][:, 2] < .75).flatten()
        ids = eligible[torch.as_tensor(self.rng.integers(len(eligible), size=size))]
        s = batch["observations"][ids]
        a = actor(s, deterministic=True)[0]
        output = self.predict(s, a)
        ns = s.clone()
        ns[:, 1:] += output[:, 12:-1]
        ns[:, 1] = ns[:, 1].clamp(0, 1)
        ns[:, 2] = torch.maximum(ns[:, 2], s[:, 2]+1/24000).clamp_max(.9999)
        ns[:, 3:5] = ns[:, 3:5].clamp(0, 5)
        ns[:, 5:7] = ns[:, 5:7].clamp(.001, 10)
        ns[:, 7:11] = ns[:, 7:11].clamp(0, 2)
        ns[:, 11:13] = (ns[:, 11:13] > .5).float()
        ns[:, 13:15] = ns[:, 13:15].clamp(0, 1.25)
        ns[:, 15:] = a
        if self.replay.objective == "headway":
            rewards = torch.as_tensor(self.replay.normalize(state_headway_reward(ns.numpy())), dtype=torch.float32)
        else:
            physical = self.replay.mean + output[:, -1] * self.replay.std / 10
            rewards = (physical.clamp_max(0)-self.replay.mean) * (10/self.replay.std)
        result = dict(observations=s, actions=a, next_observations=ns, rewards=rewards,
            delta_t_seconds=(ns[:, 2]-s[:, 2])*24000,
            discounts=torch.full((size,), P["same_vehicle_discount"]), terminals=torch.zeros(size))
        return result, self.trust.weight(s, a)

    def save(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        torch.save(dict(banks={k:[m.state_dict() for m in bank] for k,bank in self.banks.items()},
                        mean=self.mean, std=self.std, family=self.family,
                        trust=self.trust.net.state_dict()), self.folder / "models.pt")
