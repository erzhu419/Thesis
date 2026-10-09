"""Independent TD-scale, additional-real-data and model-family interventions."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mcwm_seed_confirmation_v13"))
import run as original
from confirmation_common import P as PREVIOUS, write

P = dict(PREVIOUS, **json.loads((HERE / "protocol.json").read_text()))
FIXED = dict(h2o_matched=0., scaled_zero=0., target_replay=.25, direct_only=.25, residual_only=.25)
for implementation in (original.previous, original.previous.previous):
    implementation.HERE, implementation.FIXED = HERE, FIXED
original.HERE, original.FIXED = HERE, FIXED
original.initializer_path = lambda seed,smoke=False: HERE.parent / "mcwm_seed_confirmation_v13/inputs" / f"seed{seed}" / f"h2oplus_seed{seed}.pt"

OriginalModel = original.previous.previous.ContextModel


class ForcedModel(OriginalModel):
    def select(self, event, weights):
        super().select(event, weights)
        selected = self.family
        self.family = self.p["forced_family"]
        self.models = self.banks[self.family]
        self.history[-1].update(calibration_winner=selected, selected_family=self.family,
                                selection="prespecified forced-family intervention")
        write(self.out / "calibration_decision.json", dict(selected_family=self.family, history=self.history))


class TargetReplay:
    """Extra TD updates on actual target-fit tuples, with no fitted model."""
    def __init__(self, replay, protocol, out, device, costs):
        from run_budget import target_splits
        blocks = sorted(int(x) for x in np.unique(replay.demand_realization_seeds))
        raw = target_splits(replay, blocks, protocol)["fit"]
        self.data = {k:torch.as_tensor(v, dtype=torch.float32, device=device)
                     for k,v in raw.items() if k != "block"}
        self.protocol, self.device = protocol, device
        self.generator = torch.Generator(device=device).manual_seed(protocol["training_seed"]+40000)
        self.history = []

    def refit(self, replay, event):
        return None

    def update_trust(self, actor):
        return None

    def generate(self, actor, size):
        eligible = torch.nonzero(self.data["s"][:,0] < self.protocol["model_start_max_horizon_fraction"]).flatten()
        ids = eligible[torch.randint(len(eligible), (size,), generator=self.generator, device=self.device)]
        d = {k:v[ids] for k,v in self.data.items()}
        batch = dict(observations=d["s"], actions=d["a"], next_observations=d["ns"],
                     rewards=d["r"], delta_t_seconds=d["dt"],
                     terminals=torch.zeros(size, device=self.device), discounts=torch.ones(size, device=self.device))
        return batch, torch.full((size,), self.protocol["replay_weight"], device=self.device)


class Trainer(original.MappingTrainer):
    def model_diagnostics(self):
        if isinstance(self.augmentation, TargetReplay):
            return dict(kind="observed_target_replay", weight=self.protocol["replay_weight"],
                        rows=len(self.augmentation.data["s"]), fitted_models=0, actions="recorded behavior actions")
        return super().model_diagnostics()


class ScaledTrainer(Trainer):
    pass


original.MappingTrainer, original.NoModelTrainer = Trainer, ScaledTrainer
original.previous.previous.ContextModel = lambda replay,p,out,device,costs: (
    TargetReplay(replay,p,out,device,costs) if p["arm"] == "target_replay" else ForcedModel(replay,p,out,device,costs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=P["arms"], required=True)
    parser.add_argument("--scenario", choices=P["scenes"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", type=int, default=10)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    p = dict(P, arm=args.arm, training_seed=args.seed, relative_out=str(args.out.relative_to(HERE)),
             forced_family="direct" if args.arm == "direct_only" else "residual")
    if args.smoke:
        p.update(warmup_events=128, source_event_budget=192, actor_taper_event=160,
                 model_refresh_events=[128,160], model_updates_per_member=2, model_refit_updates=2,
                 model_ensemble_members=1, model_max_fit_rows=256, model_max_validation_rows=64,
                 evaluation_horizon_seconds=1200)
    if args.evaluate:
        original.evaluate_existing(args, p)
    elif not (args.out / "results/ready.json").exists():
        write(args.out / "results/runtime.json", original.runtime())
        arm = args.arm
        # v13's no-model path is reused; the protocol still names the new arm.
        if arm == "scaled_zero":
            args.arm = "h2o_matched"
        original.train(args, p)
        for name in ("train_summary.json", "ready.json"):
            path = args.out / "results" / name
            row = json.loads(path.read_text())
            row.update(arm=arm, experiment=P["name"], forced_family=p["forced_family"] if arm.endswith("_only") else None)
            write(path, row)
    print("DONE MCWM attribution", flush=True)


if __name__ == "__main__":
    main()
