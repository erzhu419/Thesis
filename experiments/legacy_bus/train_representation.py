"""Categorical-input ablation in the same existing RE-SAC bus learner."""

import argparse
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("OMP_NUM_THREADS", "1")
from evaluate_pair import ROOT, evaluate, SEEDS
import torch
from torch import nn
from torch.nn import functional as F
from bus_experiments import frozen_policy_bank_core as core


class MaskedEmbedding(core.EmbeddingLayer):
    def forward(self, values):
        return super().forward(values)*0.


class OneHot(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes
        self.output_dim = sum(sizes)

    def forward(self, values):
        return torch.cat([F.one_hot(values[:, i].long(), size).float()
                          for i, size in enumerate(self.sizes)], dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", choices=["learned", "masked", "onehot"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.representation == "masked":
        core.EmbeddingLayer = MaskedEmbedding
    elif args.representation == "onehot":
        core.EmbeddingLayer = OneHot
    out = args.out.resolve()
    result = out / "results"
    result.mkdir(parents=True, exist_ok=True)
    config = core.TrainConfig(seed=args.seed, max_episodes=args.episodes,
                              role="specialist", fixed_mode="normal", checkpoint_interval=5)
    core.atomic_json(result / "protocol.json", dict(representation=args.representation,
                     training_seed=args.seed, episodes=args.episodes,
                     learner="existing_RE-SAC_bus", evaluation_seeds=list(SEEDS),
                     evaluation_scenes=["normal", "fast_switch"],
                     parameter_matching="same hidden layers; one-hot changes input width; masked keeps embedding dimensions"))
    if not (out / "bundle/bundle_manifest.json").exists():
        core.train(config, out, out / "bundle", resume=args.resume)
    environment = core.make_environment(config)
    environment.reset()
    trainer = core.BusSAC(environment, config, torch.device("cpu"))
    payload = torch.load(out / "bundle/controller.pt", map_location="cpu", weights_only=False)
    trainer.load_controller_state(payload)
    core.atomic_json(result / "parameters.json", dict(
        policy=sum(p.numel() for p in trainer.policy.parameters()),
        critic=sum(p.numel() for p in trainer.critic.parameters())))
    shutil.copy2(out / "bundle/train_summary.json", result / "train_summary.json")
    for scene in ["normal", "fast_switch"]:
        for seed in SEEDS:
            path = result / f"{scene}_seed{seed}.json"
            if path.exists():
                continue
            row = evaluate(trainer.policy, args.representation, scene, seed)
            row.update(training_seed=args.seed, version="categorical_RE-SAC_ablation")
            core.atomic_json(path, row)
            print(json.dumps({k:v for k,v in row.items() if k != "windows"}), flush=True)
    core.atomic_json(result / "completed.json", dict(training_complete=True, evaluation_episodes=6))


if __name__ == "__main__":
    main()
