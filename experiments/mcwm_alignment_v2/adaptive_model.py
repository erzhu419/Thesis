"""Policy-weighted refits on fixed target logs and a frozen source predictor."""

from alignment_common import write
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

from control_model import ModelAugmentation
from model import BusModel, Normalizer, losses, targets
from run_budget import target_splits
from run_pilot import predict, take
import mc_wm

# This NumPy-only module is also usable in bus runtimes without the SINDy package.
_density_spec = importlib.util.spec_from_file_location(
    "bus_policy_density", Path(mc_wm.__file__).parent / "residual/policy_density.py")
_density_module = importlib.util.module_from_spec(_density_spec)
_density_spec.loader.exec_module(_density_module)
PolicyDensity = _density_module.PolicyDensity


def fit_member(model, fit, stop, norm, updates, seed, weights=None, stop_weights=None):
    parameters = [v for v in model.parameters() if v.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=3e-4)
    rng = np.random.default_rng(seed)
    best, selected, saved = float("inf"), 0, None
    for step in range(1, updates+1):
        ix = torch.as_tensor(rng.integers(len(fit["s"]), size=128), device=fit["s"].device)
        batch = take(fit, ix)
        error = losses(model(batch["s"], batch["a"]), batch, norm).mean(1)
        loss = error.mean() if weights is None else (error*weights[ix]).sum()/weights[ix].sum()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite bus model fit")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 10.)
        optimizer.step()
        if step % min(100, updates) == 0 or step == updates:
            with torch.no_grad():
                error = losses(predict([model], stop), stop, norm).mean(1)
                score = float(error.mean() if stop_weights is None else
                              (error*stop_weights).sum()/stop_weights.sum())
            if score < best:
                best, selected = score, step
                saved = copy.deepcopy(model.state_dict())
    model.load_state_dict(saved)
    model.eval()
    return dict(stop_loss=best, selected_step=selected, updates=updates)


class AdaptiveModel(ModelAugmentation):
    def __init__(self, replay, p, out, device, costs):
        self.costs, self.p, self.out = costs, p, Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        blocks = sorted(int(x) for x in np.unique(replay.demand_realization_seeds))
        if len(blocks) < 3:
            raise ValueError("Separate target fit, stopping and calibration blocks are required")
        raw = target_splits(replay, blocks, p)
        stations = torch.as_tensor(replay.station_max_by_line, dtype=torch.float32, device=device)
        self.target = {k: targets(v, stations, device) for k,v in raw.items()}
        n = replay.online_size
        ids = np.flatnonzero((replay._online_delta_t_seconds[:n] > 0) &
                             (replay._online_terminals[:n] < .5))
        # Warmup may cover only part of a day. Keep a chronological suffix for stopping.
        boundary = int(.8*len(ids))
        rng = np.random.default_rng(p["training_seed"] + 50000)
        source = {}
        for split, candidates in (("fit", ids[:boundary]), ("stop", ids[boundary:])):
            limit = p["model_max_fit_rows"] if split == "fit" else p["model_max_validation_rows"]
            ix = np.sort(rng.choice(candidates, min(limit, len(candidates)), replace=False))
            data = dict(s=replay._online_observations[ix], a=replay._online_actions[ix],
                        ns=replay._online_next_observations[ix], r=replay._online_rewards[ix],
                        dt=replay._online_delta_t_seconds[ix])
            source[split] = targets(data, stations, device)
        self.norm = Normalizer(source["fit"]).to(device)
        self.banks = {k: [] for k in ("source", "direct", "residual")}
        for member in range(p["model_ensemble_members"]):
            seed = p["training_seed"] + 50000 + member
            torch.manual_seed(seed)
            base = BusModel().to(device)
            with costs.phase("source_model_fit"):
                fit_member(base, source["fit"], source["stop"], self.norm,
                           p["model_updates_per_member"], seed)
            self.banks["source"].append(base)
            for family in ("direct", "residual"):
                torch.manual_seed(seed+100)
                model = BusModel(base if family == "residual" else None).to(device)
                with costs.phase("target_model_fit"):
                    fit_member(model, self.target["fit"], self.target["stop"], self.norm,
                               p["model_updates_per_member"], seed+100)
                self.banks[family].append(model)
        self.history = []
        self.select(p["warmup_events"], None)
        torch.save(dict(models={k: [m.state_dict() for m in ms] for k,ms in self.banks.items()},
                        normalizer=self.norm.state_dict()), self.out / "pilot_models.pt")
        np.savez_compressed(self.out / "target_fit.npz", **raw["fit"])
        super().__init__(device, p, replay.station_max_by_line, self.out)
        self.models = self.banks[self.family]
        write(self.out / "splits.json", dict(target_fit_blocks=blocks[:-2],
              target_stop_blocks=blocks[-2:-1], target_calibrate_blocks=blocks[-1:],
              source_split="first 80% versus last 20% of warmup physical transitions",
              target_rows={k:len(v["s"]) for k,v in raw.items()},
              source_rows={k:len(v["s"]) for k,v in source.items()}))

    def select(self, event, weights):
        with torch.no_grad():
            scores = {}
            for family in ("direct", "residual"):
                error = losses(predict(self.banks[family], self.target["calibrate"]),
                               self.target["calibrate"], self.norm).mean(1)
                scores[family] = float(error.mean() if weights is None else
                                       (error*weights).sum()/weights.sum())
        self.family = min(scores, key=scores.get)
        self.models = self.banks[self.family]
        self.history.append(dict(source_events=event, selected_family=self.family,
                                 calibration_loss=scores))
        write(self.out / "calibration_decision.json", dict(selected_family=self.family,
                                                           history=self.history))

    def refit(self, replay, event):
        n = replay.online_size
        candidates = np.flatnonzero((replay._online_delta_t_seconds[:n] > 0) &
                                     (replay._online_terminals[:n] < .5))[-2048:]
        rng = np.random.default_rng(self.p["training_seed"]+event)
        ix = rng.choice(candidates, min(len(candidates), self.p["policy_reference_rows"]), replace=False)
        reference = np.concatenate((replay._online_observations[ix], replay._online_actions[ix]), axis=1)
        density = PolicyDensity(strategy="buffer", w_min=.1, random_state=self.p["training_seed"])
        density.fit_reference(reference)
        with self.costs.phase("policy_fit_weights"):
            weights = {}
            for split, batch in self.target.items():
                sa = torch.cat((batch["s"], batch["a"]), dim=1).cpu().numpy()
                values = density.weights(sa)
                weights[split] = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        with self.costs.phase("target_model_refit"):
            for family in ("direct", "residual"):
                for member, model in enumerate(self.banks[family]):
                    model.encoder.requires_grad_(True)
                    model.head.requires_grad_(True)
                    fit_member(model, self.target["fit"], self.target["stop"], self.norm,
                               self.p["model_refit_updates"], self.p["training_seed"]+event+member,
                               weights["fit"], weights["stop"])
        self.select(event, weights["calibrate"])
        with torch.no_grad(), self.costs.phase("refresh_trust_errors"):
            self.errors = losses(predict(self.models, self.fit), self.fit, self.norm).mean(1)
        self.history[-1]["fit_weight_range"] = [float(weights["fit"].min()), float(weights["fit"].max())]
        write(self.out / "calibration_decision.json", dict(selected_family=self.family, history=self.history))
