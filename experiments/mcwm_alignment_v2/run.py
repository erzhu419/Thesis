"""Aligned H2O continuation and MC-WM bus adaptation; no online target training."""

from alignment_common import HERE, P, write, seed_all, milestones, mix_model_loss
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch

from data import load_protocol
from mc import Costs
from run_control import (TransitReplayBufferV7, TransitH2OTrainerV7, TransitTrainerConfig,
                         AllLinesSimEnv, AllLinesTransitSamplerV7, TransitMDPConfigV7)
from source_mismatch import install_source_mechanism
from adaptive_model import AdaptiveModel


class ModelTrainer(TransitH2OTrainerV7):
    augmentation = None
    model_loss = None

    def _ensemble_td_loss(self, predictions, targets, weights=None):
        real = super()._ensemble_td_loss(predictions, targets, weights)
        return mix_model_loss(real, self.model_loss, self.model_fraction)

    def update(self, *, online, online_only=False, collect_metrics=True):
        self.model_loss = None
        model_metrics = {}
        if online and self.augmentation is not None:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(self.model_seed+self.total_updates)
                if self.total_updates % self.protocol["trust_update_every"] == 0:
                    with self.costs.phase("bellman_trust_update"):
                        self.augmentation.update_trust(self.actor)
                with self.costs.phase("model_training_batch"):
                    batch, weights = self.augmentation.generate(self.actor, self.protocol["model_batch_size"])
                    q = self.critic(batch["observations"], batch["actions"])
                    target = self._target(batch, backup="h2oplus")
                    self.model_loss = super()._ensemble_td_loss(q, target, weights)
                    if collect_metrics:
                        model_metrics = dict(model_td_loss=float(self.model_loss.detach()),
                                             model_weight_mean=float(weights.mean()))
        try:
            result = super().update(online=online, online_only=online_only, collect_metrics=collect_metrics)
            result.update(model_metrics)
            return result
        finally:
            self.model_loss = None


def evaluate(args, p):
    from evaluate_control import evaluate as existing_evaluate, build_scenario
    blocks = p["evaluation_blocks"]
    block = blocks[args.case % len(blocks)]
    scale = p["evaluation_demand_scales"][args.case // len(blocks)]
    case_dir = args.out / f"case_{args.case:02d}"
    (case_dir / "evaluation").mkdir(parents=True, exist_ok=True)
    policy = case_dir / "policy.pt"
    if not policy.exists():
        policy.symlink_to(args.out / "policy.pt")
    cfg, objects = build_scenario(case_dir, scale, block["demand_seed"], p["departure_jitter_seconds"])
    row = existing_evaluate(("policy", scale, block, cfg, objects, case_dir, p))
    row.update(arm=args.arm, scenario=args.scenario, training_seed=args.seed, case=args.case,
               deployment_hold_scale=p["deployment_hold_scale"])
    write(args.out / "results" / f"eval_{args.case:02d}.json", row)


def train(args, p):
    device = "cuda"
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    costs = Costs(device)
    start = time.monotonic()
    initializer = HERE / "inputs" / f"h2oplus_seed{args.seed}.pt"
    payload = torch.load(initializer, map_location="cpu", weights_only=False)
    if payload["total_updates"] != p["offline_updates_in_initializer"]:
        raise ValueError("Expected original 50000-update offline initializer")
    config = replace(TransitTrainerConfig(**payload["source_config"]), use_ratio_correction=False)
    del payload
    env_paths = load_protocol()
    dataset = Path(env_paths["dataset"])
    if not dataset.is_absolute():
        dataset = Path(env_paths["h2o_root"]) / dataset
    with costs.phase("dataset_loading"):
        replay = TransitReplayBufferV7(dataset, device=device, online_capacity=30000,
                                      max_offline_rows=10000 if args.smoke else 0)
    trainer = ModelTrainer(replay, config=config, device=device, seed=args.seed)
    trainer.model_fraction = p["model_fraction"]
    trainer.model_seed = args.seed+60000
    trainer.protocol, trainer.costs = p, costs
    trainer.load_offline_initializer(initializer)
    scenario = next(s for s in p["scenarios"] if s["name"] == args.scenario)
    seed_all(args.seed+p["source_seed_offset"])
    install_source_mechanism(scenario)
    env = AllLinesSimEnv(env_paths["source_env"], observation_schema="v7", demand_scale=scenario["od_factor"],
                        observation_speed_seed=args.seed+p["source_seed_offset"])
    if scenario["od_factor"] > 1.25:
        # The existing observation archive ends at 1.25; physical demand remains 1.5.
        env._observation_speed_profiles_by_demand[1.5] = env._observation_speed_profiles_by_demand[1.25]
    sampler = AllLinesTransitSamplerV7(env, replay, station_max_by_line=replay.station_max_by_line,
                                      contract_config=TransitMDPConfigV7(**replay.contract_config))
    write(args.out / "results/protocol.json", dict(protocol=p, arm=args.arm, scenario=scenario,
          training_seed=args.seed, config=asdict(config), offline_rows=replay.offline_size,
          target_online_training_events=0, initializer_updates=trainer.total_updates))
    events = updates = 0
    cuda_devices = [torch.cuda.current_device()]
    for endpoint in milestones(p):
        with costs.phase("source_collection"):
            # Preserve simultaneous batches, accepting at most one batch of endpoint overshoot.
            if endpoint > events:
                collected = sampler.collect(trainer.act, endpoint-events)
                events += collected["events"]
        if args.arm == "mcwm" and endpoint in p["model_refresh_events"]:
            with torch.random.fork_rng(devices=cuda_devices), costs.phase("model_refresh"):
                torch.manual_seed(args.seed+endpoint)
                if trainer.augmentation is None:
                    trainer.augmentation = AdaptiveModel(replay, p, args.out / "models", device, costs)
                else:
                    trainer.augmentation.refit(replay, events)
        desired_updates = max(0, endpoint-p["warmup_events"])
        while updates < desired_updates:
            logical_event = p["warmup_events"]+updates
            rate = config.actor_learning_rate*(p["actor_taper_multiplier"]
                   if logical_event >= p["actor_taper_event"] else 1.)
            for group in trainer.actor_optimizer.param_groups:
                group["lr"] = rate
            log = (updates+1) % 1000 == 0
            with costs.phase("policy_update"):
                stats = trainer.update(online=True, collect_metrics=log)
            updates += 1
            if log:
                row = dict(source_events=events, continuation_updates=updates,
                           elapsed_seconds=time.monotonic()-start, metrics=stats)
                write(args.out / "results/progress.json", row)
                print(json.dumps(row), flush=True)
        if endpoint in p["model_refresh_events"]:
            trainer.save_policy_artifact(args.out / f"policy_events{endpoint}.pt",
                                         extra=dict(arm=args.arm, source_events=events))
    trainer.save_policy_artifact(args.out / "policy.pt", extra=dict(arm=args.arm, seed=args.seed,
                                scenario=args.scenario, source_events=events, updates=trainer.total_updates))
    summary = dict(arm=args.arm, seed=args.seed, scenario=args.scenario, source_events=events,
                   continuation_updates=updates, total_policy_updates=trainer.total_updates,
                   total_critic_updates=trainer.total_updates, target_online_training_events=0,
                   model_history=trainer.augmentation.history if trainer.augmentation is not None else [],
                   phases=costs.rows, training_wall_seconds=time.monotonic()-start,
                   reused_offline_updates=p["offline_updates_in_initializer"],
                   peak_host_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                   peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                   peak_gpu_reserved_mb=torch.cuda.max_memory_reserved()/2**20)
    write(args.out / "results/train_summary.json", summary)
    write(args.out / "results/ready.json", summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=P["arms"], required=True)
    parser.add_argument("--scenario", choices=[s["name"] for s in P["scenarios"]], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    p = dict(P, training_seed=args.seed)
    if args.smoke:
        if "smoke" not in args.out.parts:
            parser.error("Use a separate smoke output directory")
        p.update(warmup_events=128, source_event_budget=192, actor_taper_event=160,
                 model_refresh_events=[128,160], model_updates_per_member=2, model_refit_updates=2,
                 model_ensemble_members=1, model_max_fit_rows=256, model_max_validation_rows=64)
    if args.evaluate:
        evaluate(args, p)
    elif not (args.out / "results/ready.json").exists():
        train(args, p)


if __name__ == "__main__":
    main()
