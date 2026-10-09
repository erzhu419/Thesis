"""Fresh-seed bus model controls, matched learner updates, and measured costs."""

from study_common import EXPERIMENTS, P, arguments, seed_all, write
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time

sys.path.insert(0, str(EXPERIMENTS / "mcwm_bus"))
from data import load_protocol
from run_budget import (block_view, target_splits, TransitReplayBufferV7,
                        TransitH2OTrainerV7, TransitTrainerConfig, AllLinesSimEnv,
                        AllLinesTransitSamplerV7, TransitMDPConfigV7)
from model import BusModel, Normalizer, targets, metrics
from run_pilot import train, predict
from control_model import ModelAugmentation
import numpy as np
import torch


class Costs:
    def __init__(self, device):
        self.device, self.rows = device, {}

    def sync(self):
        if self.device == "cuda":
            torch.cuda.synchronize()

    @contextmanager
    def phase(self, name):
        self.sync()
        start, cpu = time.monotonic(), time.process_time()
        try:
            yield
        finally:
            self.sync()
            row = self.rows.setdefault(name, dict(wall_seconds=0., cpu_seconds=0., calls=0))
            row["wall_seconds"] += time.monotonic()-start
            row["cpu_seconds"] += time.process_time()-cpu
            row["calls"] += 1


def collect_day(replay, trainer, p, day, costs):
    seed = p["source_day_seed_base"] + day
    seed_all(seed)
    source_path = load_protocol()["source_env"]
    start = replay.online_size
    with costs.phase("source_collection"):
        env = AllLinesSimEnv(str(source_path), observation_schema="v7",
                            demand_scale=1., observation_speed_seed=seed)
        sampler = AllLinesTransitSamplerV7(env, replay, station_max_by_line=replay.station_max_by_line,
                                          contract_config=TransitMDPConfigV7(**replay.contract_config))
        # Set provenance before append() assigns the classifier holdout mask.
        sampler.episodes = day
        sampler.next_trajectory_id = day
        while not env.done:
            sampler.collect(trainer.act, 1)
    stop = replay.online_size
    if stop >= replay.online_capacity:
        raise RuntimeError("Source replay filled; complete-day provenance would be lost")
    terminal = int(replay._online_terminals[start:stop].sum())
    if terminal != 1:
        raise RuntimeError("Incomplete source running day")
    row = dict(day=day, seed=seed, transitions=stop-start, terminal_rows=terminal,
               ratio_validation_rows=int(replay._online_ratio_validation[start:stop].sum()))
    print("source " + json.dumps(row), flush=True)
    return row


def fit_models(replay, p, out, arm, device, costs):
    out.mkdir(parents=True, exist_ok=True)
    raw = target_splits(replay, p["offline_blocks"], p)
    stations = torch.as_tensor(replay.station_max_by_line, dtype=torch.float32, device=device)
    target = {k: targets(v, stations, device) for k, v in raw.items()}
    rng = np.random.default_rng(p["training_seed"]+50000)
    source, source_rows = {}, {}
    size = replay.online_size
    eligible = (replay._online_delta_t_seconds[:size] > 0) & (replay._online_terminals[:size] < .5)
    for split in ("fit", "stop"):
        day = p["source_"+split+"_day"]
        ids = np.flatnonzero(eligible & (replay._online_episode_ids[:size] == day))
        limit = p["model_max_fit_rows"] if split == "fit" else p["model_max_validation_rows"]
        ix = np.sort(rng.choice(ids, min(len(ids), limit), replace=False))
        source_rows[split] = dict(day=day, rows=len(ix))
        data = dict(s=replay._online_observations[ix], a=replay._online_actions[ix],
                    ns=replay._online_next_observations[ix], r=replay._online_rewards[ix],
                    dt=replay._online_delta_t_seconds[ix], block=np.full(len(ix), day))
        source[split] = targets(data, stations, device)
    norm = Normalizer(source["fit"]).to(device)
    families = {name: [] for name in ("source", "direct", "residual")}
    settings = dict(learning_rate=3e-4, batch_size=128,
                    updates_per_member=p["model_updates_per_member"],
                    validation_every=min(100, p["model_updates_per_member"]))
    needed = (["direct", "residual"] if arm == "calibrated_selection" else
              ["direct"] if arm == "fixed_direct" else ["residual"])
    history = {}
    for member in range(p["model_ensemble_members"]):
        seed = p["training_seed"]+50000+member
        base = None
        if "residual" in needed:
            torch.manual_seed(seed)
            base = BusModel().to(device)
            with costs.phase("source_model_fit"):
                history[f"source_{member}"] = train(base, source["fit"], source["stop"], norm,
                                                     settings, seed, "source")
            families["source"].append(base)
        for family in needed:
            torch.manual_seed(seed+100)
            model = BusModel(base if family == "residual" else None).to(device)
            with costs.phase(family+"_model_fit"):
                history[f"{family}_{member}"] = train(model, target["fit"], target["stop"], norm,
                                                       settings, seed+100, family)
            families[family].append(model)
    with costs.phase("model_calibration"):
        scores = {family: metrics(predict(families[family], target["calibrate"]),
                                   target["calibrate"], norm) for family in needed}
        selected = min(needed, key=lambda family: scores[family]["selection_loss"])
    summary = dict(selected_family=selected, calibration=scores, source_split=source_rows,
                   target_split=dict(fit=p["offline_blocks"][:-2], stop=p["offline_blocks"][-2:-1],
                                     calibrate=p["offline_blocks"][-1:]), training=history,
                   model_gradient_updates=len(history)*p["model_updates_per_member"],
                   parameters={k: sum(v.numel() for m in ms for v in m.parameters())
                               for k, ms in families.items()},
                   parameter_note="Residual entries include the frozen source base; do not sum families.")
    write(out / "calibration_decision.json", summary)
    torch.save(dict(models={k: [m.state_dict() for m in ms] for k, ms in families.items()},
                    normalizer=norm.state_dict()), out / "pilot_models.pt")
    np.savez_compressed(out / "target_fit.npz", **raw["fit"])
    return summary


def extra_step(trainer, replay, rng, p, augmentation, arm, costs):
    with costs.phase("extra_critic_update"):
        if arm in ("source_only", "offline_only"):
            batch = replay.sample(p["extra_batch_size"], rng=rng,
                                  scope="online" if arm == "source_only" else "offline")
        else:
            off, sim = replay.sample_mixed(p["extra_batch_size"], trainer.config.online_fraction, rng=rng)
            batch = trainer._concatenate_batches(off, sim)
        backup = "sac" if arm == "source_only" else "iql" if arm == "offline_only" else "h2oplus"
        loss = trainer._ensemble_td_loss(trainer.critic(batch["observations"], batch["actions"]),
                                         trainer._target(batch, backup=backup))
    if augmentation is not None:
        with costs.phase("bellman_trust_update"):
            augmentation.update_trust(trainer.actor)
        with costs.phase("model_generation"):
            generated, weights = augmentation.generate(trainer.actor, p["extra_batch_size"])
        with costs.phase("extra_critic_update"):
            model_loss = trainer._ensemble_td_loss(trainer.critic(generated["observations"], generated["actions"]),
                                                   trainer._target(generated, backup=backup), weights)
            fraction = p["model_fraction_in_extra_loss"]
            loss = (1-fraction)*loss+fraction*model_loss
    with costs.phase("extra_critic_update"):
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite extra critic loss")
        trainer.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        trainer.critic_optimizer.step()
        trainer._soft_update_target()


def inference_cost(trainer, replay, augmentation, costs):
    batch = replay.sample(128, scope="online" if replay.online_size else "offline", rng=np.random.default_rng(991))
    with torch.inference_mode():
        for _ in range(20):
            trainer.actor(batch["observations"][:1], deterministic=True)
        with costs.phase("inference_actor_1000_batch1"):
            for _ in range(1000):
                trainer.actor(batch["observations"][:1], deterministic=True)
        if augmentation is not None:
            for _ in range(20):
                augmentation.predict(batch["observations"], batch["actions"])
            with costs.phase("inference_model_100_batch128"):
                for _ in range(100):
                    augmentation.predict(batch["observations"], batch["actions"])


def main():
    args = arguments("mc")
    if args.evaluate:
        from evaluate_control import evaluate, build_scenario
        block = P["evaluation_blocks"][args.case % len(P["evaluation_blocks"])]
        scale = P["evaluation_demand_scales"][args.case // len(P["evaluation_blocks"])]
        case_dir = args.out / f"case_{args.case:02d}"
        (case_dir / "evaluation").mkdir(parents=True, exist_ok=True)
        policy = case_dir / "policy.pt"
        if not policy.exists():
            policy.symlink_to(args.out / "policy.pt")
        cfg, objects = build_scenario(case_dir, scale, block["demand_seed"], P["departure_jitter_seconds"])
        row = evaluate(("policy", scale, block, cfg, objects, case_dir, P))
        if row["sumo_version"][1] != "SUMO 1.25.0":
            raise RuntimeError("This comparison uses the existing SUMO 1.25.0 evaluation runtime")
        row.update(arm=args.arm, training_seed=args.seed, case=args.case)
        write(args.out / "results" / f"eval_{args.case:02d}.json", row)
        return
    if (args.out / "results/ready.json").exists() and (args.out / "policy.pt").exists():
        return
    if args.resume:
        raise ValueError("This study trains fresh; policy-only files are not training checkpoints")
    p = dict(P, training_seed=args.seed)
    if args.smoke:
        p.update(prefix_updates=4, adaptation_days=1, updates_per_day=4,
                 model_updates_per_member=2, model_ensemble_members=1)
    device = "cuda"
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    costs = Costs(device)
    start = time.monotonic()
    old_root = EXPERIMENTS / "mcwm_bus"
    metadata = json.loads((old_root / "prefix_config.json").read_text())
    config = replace(TransitTrainerConfig(**metadata), online_actor_update_interval=1)
    if args.arm == "source_only":
        config = replace(config, offline_objective="none", online_backup="sac",
                         online_fraction=1., use_ratio_correction=False)
    dataset = Path(load_protocol()["dataset"])
    if not dataset.is_absolute():
        dataset = Path(load_protocol()["h2o_root"]) / dataset
    with costs.phase("dataset_loading"):
        if args.arm != "source_only":
            view = args.out / "offline_view.h5"
            block_view(dataset, view, p["offline_blocks"])
            dataset = view
        replay = TransitReplayBufferV7(dataset, device=device, online_capacity=200000,
                                      max_offline_rows=256 if args.arm == "source_only" else 0)
    trainer = TransitH2OTrainerV7(replay, config=config, device=device, seed=args.seed)
    write(args.out / "results/protocol.json", dict(protocol=p, config=asdict(config), arm=args.arm,
            initialization="random; no pretrained weights", actor_updates_per_critic_update=1))
    days = []
    if args.arm == "source_only":
        for day in range(2):
            days.append(collect_day(replay, trainer, p, day, costs))
    with costs.phase("prefix_policy_updates"):
        for step in range(p["prefix_updates"]):
            trainer.update(online=args.arm == "source_only", online_only=args.arm == "source_only",
                           collect_metrics=False)
    if args.arm not in ("source_only", "offline_only"):
        for day in range(2):
            days.append(collect_day(replay, trainer, p, day, costs))
    augmentation, model_summary = None, None
    cuda_devices = [torch.cuda.current_device()]
    if args.arm in ("fixed_direct", "fixed_residual", "calibrated_selection"):
        with torch.random.fork_rng(devices=cuda_devices):
            model_summary = fit_models(replay, p, args.out / "models", args.arm, device, costs)
            with costs.phase("model_loading_and_fit_errors"):
                augmentation = ModelAugmentation(device, p, replay.station_max_by_line, args.out / "models")
    rng = np.random.default_rng(args.seed+10000)
    updates = extras = 0
    for day in range(2, 2+p["adaptation_days"]):
        if args.arm != "offline_only":
            days.append(collect_day(replay, trainer, p, day, costs))
        for step in range(p["updates_per_day"]):
            updates += 1
            with costs.phase("adaptation_policy_updates"):
                stats = trainer.update(online=args.arm != "offline_only", online_only=args.arm == "source_only",
                                       collect_metrics=updates % 512 == 0)
            if updates % p["extra_critic_every"] == 0:
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.manual_seed(args.seed+20000+updates)
                    extra_step(trainer, replay, rng, p, augmentation, args.arm, costs)
                extras += 1
            if updates % 512 == 0:
                row = dict(adaptation_updates=updates, source_rows=replay.online_size, metrics=stats)
                write(args.out / "results/progress.json", row)
                print(json.dumps(row), flush=True)
    with costs.phase("policy_export"):
        trainer.save_policy_artifact(args.out / "policy.pt", extra=dict(arm=args.arm, seed=args.seed))
    elapsed_training = time.monotonic()-start
    inference_cost(trainer, replay, augmentation, costs)
    summary = dict(arm=args.arm, training_seed=args.seed, source_days=days,
                   source_events=sum(d["transitions"] for d in days),
                   offline_rows_used=0 if args.arm == "source_only" else replay.offline_size,
                   policy_updates=p["prefix_updates"]+updates, extra_critic_updates=extras,
                   total_critic_updates=p["prefix_updates"]+updates+extras,
                   trust_updates=extras*p["qdelta_updates_per_extra_step"] if augmentation else 0,
                   model=model_summary, phases=costs.rows, training_wall_seconds=elapsed_training,
                   peak_host_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                   peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                   peak_gpu_reserved_mb=torch.cuda.max_memory_reserved()/2**20,
                   hostname=socket.gethostname(), gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                   actor_parameters=sum(v.numel() for v in trainer.actor.parameters()))
    write(args.out / "results/train_summary.json", summary)
    write(args.out / "results/ready.json", dict(arm=args.arm, training_seed=args.seed,
                                               policy_updates=summary["policy_updates"]))


if __name__ == "__main__":
    main()
