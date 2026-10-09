"""Repeated categorical controls and actor/critic disagreement factorial."""

from study_common import EXPERIMENTS, P, arguments, write
from dataclasses import asdict
import json
import sys

sys.path.insert(0, str(EXPERIMENTS / "legacy_bus"))
from train_representation import core, MaskedEmbedding, OneHot
from evaluate_pair import evaluate
import torch


def configuration(arm, seed, episodes):
    return core.TrainConfig(
        seed=seed, max_episodes=episodes, role="specialist", fixed_mode="normal",
        beta_ood=0. if arm in ("critic_off", "both_off") else .01,
        beta=0. if arm in ("actor_off", "both_off") else -2.,
        checkpoint_interval=10)


def main():
    args = arguments("legacy")
    if args.arm == "masked_full":
        core.EmbeddingLayer = MaskedEmbedding
    elif args.arm == "onehot_full":
        core.EmbeddingLayer = OneHot
    config = configuration(args.arm, args.seed, 1 if args.smoke else P["episodes"])
    if args.evaluate:
        scenes = P["legacy_scenes"]
        scene = scenes[args.case // len(P["evaluation_seeds"]) ]
        seed = P["evaluation_seeds"][args.case % len(P["evaluation_seeds"])]
        torch.set_num_threads(1)
        env = core.make_environment(config)
        env.reset()
        trainer = core.load_controller(args.out / "bundle/controller.pt", env)
        row = evaluate(trainer.policy, args.arm, scene, seed)
        row.update(training_seed=args.seed, arm=args.arm, case=args.case)
        write(args.out / "results" / f"eval_{args.case:02d}.json", row)
        print(json.dumps({k: v for k, v in row.items() if k != "windows"}), flush=True)
        return
    if (args.out / "results/ready.json").exists() and (args.out / "bundle/controller.pt").exists():
        return
    write(args.out / "results/protocol.json", dict(arm=args.arm, config=asdict(config)))
    core.train(config, args.out, args.out / "bundle", resume=args.resume and not args.smoke)
    summary = json.loads((args.out / "bundle/train_summary.json").read_text())
    write(args.out / "results/train_summary.json", summary)
    write(args.out / "results/ready.json", dict(arm=args.arm, training_seed=args.seed,
                                               episodes=config.max_episodes))


if __name__ == "__main__":
    main()
