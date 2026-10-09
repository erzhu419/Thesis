"""One independently scheduled bus training cell, followed by fixed SUMO tests."""

import argparse
import json
import random
import shutil
import time
from dataclasses import replace
from pathlib import Path

import cloudpickle
import numpy as np
import torch

from data import HERE, load_protocol
from run_budget import block_view, target_splits, fit_budget_model, policy_tag
from run_control import (ROOT, AllLinesSimEnv, AllLinesTransitSamplerV7,
                         TransitReplayBufferV7, TransitH2OTrainerV7,
                         TransitMDPConfigV7, TransitTrainerConfig, extra_critic_step)
from control_model import ModelAugmentation
from source_mismatch import install_source_mechanism, restore_source_mechanism


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def baseline_critic_step(trainer, replay, method, rng, size):
    scope = "online" if method == "online" else "offline"
    batch = replay.sample(size, scope=scope, rng=rng)
    backup = "sac" if method == "online" else "iql"
    loss = trainer._ensemble_td_loss(
        trainer.critic(batch["observations"], batch["actions"]), trainer._target(batch, backup=backup))
    trainer.critic_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    trainer.critic_optimizer.step()
    trainer._soft_update_target()
    return {"extra_real_loss": float(loss.detach())}


def train_cell(args, p, out):
    torch.set_num_threads(args.threads)
    metadata = torch.load(args.initializer, map_location="cpu", weights_only=False)
    config = TransitTrainerConfig(**metadata["source_config"])
    del metadata
    online_only = args.method == "online"
    if online_only:
        config = replace(config, offline_objective="none", online_backup="sac",
                         online_fraction=1., use_ratio_correction=False)
    scenario = next(s for s in p["mismatch_scenarios"] if s["name"] == args.scenario)
    blocks = p["offline_blocks"][:args.blocks]
    source = ROOT / load_protocol()["dataset"]
    dataset = out / "offline_view.h5"
    if not online_only and not dataset.exists():
        block_view(source, dataset, blocks)
    # The shared replay constructor needs trajectory metadata; SAC never samples it.
    replay = TransitReplayBufferV7(source if online_only else dataset, device=args.device,
                                  max_offline_rows=256 if online_only else 0,
                                  online_capacity=max(p["source_event_budgets"])+1000,
                                  device_cache_observation_dtype="float32")
    if not online_only:
        replay.materialize_device_cache()
    trainer = TransitH2OTrainerV7(replay, config=config, device=args.device, seed=p["training_seed"])
    if not online_only:
        trainer.load_offline_initializer(args.initializer)
    prefix = f"blocks{args.blocks}_offline"
    if args.method == "offline":
        trainer.save_policy_artifact(out / f"{prefix}.pt", extra={"source_events": 0, "blocks": blocks})
    target_raw = target_splits(replay, blocks, p) if args.method == "mcwm" else None
    random.seed(p["source_seed"])
    np.random.seed(p["source_seed"])
    torch.manual_seed(p["source_seed"])
    sampler = None
    if args.method != "offline":
        install_source_mechanism(scenario)
        env = AllLinesSimEnv(str(load_protocol().get("source_env", ROOT / "bus_h2o/calibrated_env")),
                            observation_schema="v7", demand_scale=scenario["od_factor"],
                            observation_speed_seed=p["source_seed"])
        if scenario["name"] == "joint50":
            # The existing observation archive stops at 1.25; physical demand remains 1.5.
            env._observation_speed_profiles_by_demand[1.5] = env._observation_speed_profiles_by_demand[1.25]
        sampler = AllLinesTransitSamplerV7(env, replay, station_max_by_line=replay.station_max_by_line,
                                          contract_config=TransitMDPConfigV7(**replay.contract_config))
    augmentation = None
    aux_rng = np.random.default_rng(p["training_seed"]+10000)
    events = updates = last = 0
    elapsed = 0.
    snapshots, history, models = [], [], []
    checkpoint = out / "checkpoints/latest.pt"
    if args.resume and checkpoint.exists():
        state = trainer.load(checkpoint)
        runtime = cloudpickle.loads(state["runtime"])
        if sampler is not None:
            sampler.load_runtime_state_dict(runtime["sampler"])
        augmentation = runtime["augmentation"]
        aux_rng = runtime["aux_rng"]
        events, updates, last = state["events"], state["updates"], state["threshold"]
        elapsed = state["elapsed"]
        snapshots, history, models = state["snapshots"], state["history"], state["models"]
    started = time.monotonic()
    cuda = [torch.cuda.current_device()] if args.device.startswith("cuda") else []
    for threshold in range(last+p["collection_interval"], max(p["source_event_budgets"])+1,
                           p["collection_interval"]):
        if sampler is not None:
            stats = sampler.collect(trainer.act, threshold-events, hard_event_cap=True)
            events += stats["events"]
        if args.method == "mcwm" and threshold in p["model_refresh_events"]:
            model_run = out / f"model_events{threshold}"
            with torch.random.fork_rng(devices=cuda):
                summary = fit_budget_model(replay, target_raw, p, model_run, args.device)
                fresh = ModelAugmentation(args.device, p, replay.station_max_by_line, model_run=model_run)
            if augmentation is not None:
                fresh.trust = augmentation.trust
            augmentation = fresh
            models.append(summary)
        extra = {}
        for _ in range(p["policy_updates_per_interval"]):
            updates += 1
            metrics = trainer.update(online=args.method != "offline", online_only=online_only,
                                     collect_metrics=updates % 128 == 0)
            if updates % p["extra_critic_every"] == 0:
                with torch.random.fork_rng(devices=cuda):
                    torch.manual_seed(p["training_seed"]+20000+updates)
                    if args.method in ("offline", "online"):
                        extra = baseline_critic_step(trainer, replay, args.method, aux_rng, p["extra_batch_size"])
                    else:
                        extra = extra_critic_step(trainer, replay, aux_rng, p, augmentation)
            if metrics:
                row = {"updates": updates, "source_events": events, **metrics, **extra}
                if not all(np.isfinite(v) for v in row.values()):
                    raise RuntimeError("Non-finite cell training metric")
                history.append(row)
        if threshold in p["source_event_budgets"]:
            tag = policy_tag(args.blocks, args.method, threshold, args.scenario)
            snapshot = {"tag": tag, "method": args.method, "source_mechanism": scenario,
                        "nominal_source_events": 0 if args.method == "offline" else threshold,
                        "actual_source_events": events, "matched_update_budget_events": threshold,
                        "offline_rows_used_for_learning": 0 if online_only else replay.offline_size,
                        "offline_prefix_updates": 0 if online_only else p["offline_updates"],
                        "source_rows": replay.online_size, "policy_updates": updates,
                        "critic_updates": updates+updates//p["extra_critic_every"],
                        "elapsed_seconds": elapsed+time.monotonic()-started,
                        "model_family": augmentation.family if augmentation else None}
            trainer.save_policy_artifact(out / f"{tag}.pt", extra=snapshot)
            write_json(out / "results" / f"{tag}.json", snapshot)
            snapshots.append(snapshot)
            print(json.dumps(snapshot), flush=True)
        if threshold % 2048 == 0:
            runtime = cloudpickle.dumps({"sampler": sampler.runtime_state_dict() if sampler else None,
                                         "augmentation": augmentation, "aux_rng": aux_rng})
            trainer.save(checkpoint, extra={"runtime": runtime, "threshold": threshold,
                         "events": events, "updates": updates, "snapshots": snapshots,
                         "history": history, "models": models,
                         "elapsed": elapsed+time.monotonic()-started})
            if args.stop_after and threshold >= args.stop_after:
                return False
    summary = {"snapshots": snapshots, "history": history, "models": models}
    (out / "results/policies").mkdir(exist_ok=True)
    for tag in p["methods"]:
        shutil.copy2(out / f"{tag}.pt", out / "results/policies" / f"{tag}.pt")
    write_json(out / "results/training_summary.json", summary)
    restore_source_mechanism()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--blocks", type=int, choices=[0, 2, 6], required=True)
    parser.add_argument("--method", choices=["h2o", "mcwm", "offline", "online"], required=True)
    parser.add_argument("--scenario", default="calibrated")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    if (args.method == "online") != (args.blocks == 0):
        parser.error("Source-only online uses zero offline blocks; other methods use 2 or 6")
    p = json.loads((HERE / "budget_protocol_v2.json").read_text())
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    methods = [policy_tag(args.blocks, args.method, budget, args.scenario) for budget in p["source_event_budgets"]]
    if args.scenario != "calibrated":
        methods = methods[-1:]
    if args.method == "offline":
        methods.insert(0, f"blocks{args.blocks}_offline")
    p["methods"] = methods
    write_json(out / "protocol.json", p)
    write_json(out / "results/protocol.json", p)
    if not (out / "results/training_summary.json").exists():
        if not train_cell(args, p, out):
            return
    if not args.skip_evaluation:
        torch.cuda.empty_cache()
        from evaluate_control import main as evaluate
        evaluate(out, methods, args.threads)
        shutil.copytree(out / "evaluation", out / "results/evaluation", dirs_exist_ok=True)
        write_json(out / "results/completed.json", {"methods": methods, "evaluation_complete": True})


if __name__ == "__main__":
    main()
