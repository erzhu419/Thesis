"""Matched reward ablation; original bus environment and learner stay unchanged."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch
from evaluate_pair import evaluate, SEEDS
from bus_experiments import frozen_policy_bank_core as core

ARMS = {"ridge": None, "exp60": (60., 0.),
        "exp60_hold": (60., .5), "exp1_hold": (1., .5)}
TRAINING_SEEDS = [17001, 17002, 17003]


def training_reward(arm, original, action, next_state):
    if arm == "ridge":
        return float(original)
    scale, weight = ARMS[arm]
    gap = abs(float(next_state[4]) - float(next_state[5]))
    hold = float(np.asarray(action).reshape(-1)[0]) / 60.
    return float(np.exp(-gap / scale) + weight * np.exp(-hold))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = args.out.resolve()
    result = out / "results"
    result.mkdir(parents=True, exist_ok=True)
    protocol = dict(arm=args.arm, training_seed=args.seed, episodes=args.episodes,
                    learner="existing_RE-SAC_bus", changed_component="replay reward only",
                    arms=ARMS, evaluation_seeds=list(SEEDS),
                    evaluation_scenes=["normal", "fast_switch"],
                    common_evaluation_reward="unchanged ridge environment reward",
                    primary_metric="generalized_seconds_per_passenger",
                    source="Wang and Sun 2020, Eq.3, doi:10.1016/j.trc.2020.102661",
                    adaptation="Holding scaled by 60s. Time scales 1s and 60s and w=0.5 are fixed thesis settings, not claimed original-paper settings.")
    core.atomic_json(result / "protocol.json", protocol)

    class RewardReplay(core.ReplayBuffer):
        def push(self, state, action, reward, next_state, done):
            super().push(state, action, training_reward(args.arm, reward, action, next_state),
                         next_state, done)

    core.ReplayBuffer = RewardReplay
    config = core.TrainConfig(seed=args.seed, max_episodes=args.episodes,
                              role="specialist", fixed_mode="normal", checkpoint_interval=5)
    if not (out / "bundle/bundle_manifest.json").exists():
        core.train(config, out, out / "bundle", resume=args.resume)
    environment = core.make_environment(config)
    environment.reset()
    trainer = core.BusSAC(environment, config, torch.device("cpu"))
    trainer.load_controller_state(torch.load(out / "bundle/controller.pt",
                                             map_location="cpu", weights_only=False))
    for scene in protocol["evaluation_scenes"]:
        for seed in SEEDS:
            path = result / f"{scene}_seed{seed}.json"
            if path.exists():
                continue
            row = evaluate(trainer.policy, args.arm, scene, seed)
            row.update(training_seed=args.seed, version="reward_ablation_v1")
            core.atomic_json(path, row)
            print(json.dumps({k: v for k, v in row.items() if k != "windows"}), flush=True)
    core.atomic_json(result / "completed.json", dict(training_complete=True, evaluation_episodes=6))


if __name__ == "__main__":
    main()
