"""Diagnose the actual ratio estimator on its trajectory-disjoint holdouts."""

from review_common import HERE, EXPERIMENTS, P, module, write, distribution
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

base = module("round3_enhancement_mc", EXPERIMENTS / "enhancement_v1/mc.py")


@torch.no_grad()
def diagnose(trainer, fresh_source=False):
    replay, ratio = trainer.replay_buffer, trainer.ratio
    counts = {scope: {part: replay.partition_size(scope, part)
                      for part in ("ratio_train", "ratio_validation")}
              for scope in ("offline", "online")}
    result = dict(total_updates=trainer.total_updates, partitions=counts,
                  use_ratio_correction=trainer.config.use_ratio_correction,
                  ratio_update_interval=trainer.config.ratio_update_interval,
                  min_weight=ratio.min_weight, max_weight=ratio.max_weight)
    if not counts["offline"]["ratio_validation"] or (not fresh_source and not counts["online"]["ratio_validation"]):
        return dict(result, validation_available=False)
    rng = np.random.default_rng(39701)
    batches = {s: replay.sample(min(P["ratio"]["max_diagnostic_rows"],
                                    replay.online_size if s == "online" and fresh_source else counts[s]["ratio_validation"]),
                                scope=s, partition="all" if s == "online" and fresh_source else "ratio_validation", rng=rng)
               for s in counts}
    result.update(validation_available=True,
                  metrics={k: float(v) for k, v in ratio.evaluate(batches["offline"], batches["online"]).items()})
    weights, _ = ratio.weights(batches["online"])
    result["weight_distribution"] = distribution(weights.cpu().numpy())
    result["ess"] = float(weights.sum().square()/weights.square().sum())
    result["offline_holdout_blocks"] = sorted(set(replay.demand_realization_seeds[replay._offline_ratio_validation].tolist()))
    train_blocks = set(replay.demand_realization_seeds[~replay._offline_ratio_validation].tolist())
    if train_blocks.intersection(result["offline_holdout_blocks"]):
        raise RuntimeError("Target demand realizations leak across classifier split")
    result["scope"] = "Held out from classifier fitting, not from actor/value training. Does not establish the correctness of a conditional density ratio."
    result["source_holdout"] = "fresh frozen-policy episode after training" if fresh_source else "original classifier trajectory partition"
    return result


class DiagnosticTrainer(base.TransitH2OTrainerV7):
    output = None

    def update(self, **kwargs):
        result = super().update(**kwargs)
        if kwargs.get("online") and kwargs.get("collect_metrics"):
            write(self.output / "results/diagnostics" / f"update{self.total_updates:06d}.json", diagnose(self))
        return result

    def save_policy_artifact(self, path, **kwargs):
        result = super().save_policy_artifact(path, **kwargs)
        self.save(self.output / "ratio_checkpoint.pt")
        write(self.output / "results/ratio_final.json", diagnose(self))
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["enhancement", "short", "export", "heldout"], required=True)
    parser.add_argument("--seed", type=int, default=22101)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "enhancement":
        DiagnosticTrainer.output = args.out
        base.TransitH2OTrainerV7 = DiagnosticTrainer
        sys.argv = [sys.argv[0], "--arm", "mixed_nomodel", "--seed", str(args.seed), "--out", str(args.out)]
        if args.smoke:
            sys.argv.append("--smoke")
        base.main()
    elif args.mode == "export":
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        torch.save({key: payload[key] for key in ("config", "actor", "ratio", "total_updates")}, args.out / "ratio_export.pt")
        write(args.out / "results/export_ready.json", dict(total_updates=payload["total_updates"]))
    elif args.mode == "heldout":
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = base.TransitTrainerConfig(**payload["config"])
        from data import load_protocol
        from run_budget import block_view
        paths = load_protocol()
        dataset = Path(paths["dataset"])
        if not dataset.is_absolute():
            dataset = Path(paths["h2o_root"]) / dataset
        view = args.out / "offline_view.h5"
        block_view(dataset, view, base.P["offline_blocks"])
        replay = base.TransitReplayBufferV7(view, device="cpu", online_capacity=60000)
        trainer = base.TransitH2OTrainerV7(replay, config=config, device="cpu", seed=args.seed)
        trainer.actor.load_state_dict(payload["actor"])
        trainer.ratio.load_state_dict(payload["ratio"])
        trainer.total_updates = payload["total_updates"]
        p = dict(base.P, source_day_seed_base=P["ratio"]["validation_seeds"][args.case])
        costs = base.Costs("cpu")
        source = base.collect_day(replay, trainer, p, 0, costs)
        result = diagnose(trainer, fresh_source=True)
        result.update(source=source, phases=costs.rows, new_gradient_updates=0)
        write(args.out / "results/ratio_final.json", result)
        write(args.out / "results/ready.json", dict(validation_available=result["validation_available"]))
    else:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = base.TransitTrainerConfig(**payload["config"])
        replay = base.TransitReplayBufferV7(args.dataset, device="cpu",
                                           online_capacity=payload["online_replay"]["capacity"])
        trainer = base.TransitH2OTrainerV7(replay, config=config, device="cpu", seed=args.seed)
        trainer.load(args.checkpoint)
        write(args.out / "results/ratio_final.json", diagnose(trainer))


if __name__ == "__main__":
    main()
    print("DONE review supplement", flush=True)
