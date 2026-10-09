"""Common-prefix, matched-update bus continuation experiment."""

import argparse
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from data import HERE, load_protocol
from control_model import ModelAugmentation

ROOT = Path(load_protocol()["h2o_root"])
sys.path.insert(0, str(ROOT))
from bus_h2o.envs.all_lines_sim_env import AllLinesSimEnv
from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7
from SimpleSAC.transit_v7.replay_buffer import TransitReplayBufferV7
from SimpleSAC.transit_v7.sampler import AllLinesTransitSamplerV7
from SimpleSAC.transit_v7.trainer import TransitH2OTrainerV7, TransitTrainerConfig


def collect_source(replay, trainer, p, out):
    result_path = out / "source_replay.npz"
    if result_path.exists():
        with np.load(result_path) as data:
            for i in range(len(data["rewards"])):
                replay.append(data["observations"][i], data["actions"][i], data["rewards"][i],
                              data["next_observations"][i], data["discounts"][i],
                              terminal=data["terminals"][i], episode_id=int(data["episode_id"][i]),
                              trajectory_id=int(data["episode_id"][i]), delta_t_seconds=data["delta_t_seconds"][i])
        return
    summaries = []
    for day, spec in enumerate(p["source_days"]):
        seed = spec["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = AllLinesSimEnv(str(ROOT / "bus_h2o/calibrated_env"), observation_schema="v7",
                            demand_scale=spec["demand_scale"], observation_speed_seed=seed)
        sampler = AllLinesTransitSamplerV7(env, replay, station_max_by_line=replay.station_max_by_line,
                                          contract_config=TransitMDPConfigV7(**replay.contract_config))
        sampler.episodes = day
        sampler.next_trajectory_id = day
        start = replay.online_size
        started = time.monotonic()
        while not env.done:
            sampler.collect(trainer.act, 1)
        stop = replay.online_size
        summary = {**spec, "transitions": stop-start,
                   "terminal_rows": int(replay._online_terminals[start:stop].sum()),
                   "elapsed_seconds": time.monotonic()-started}
        if summary["terminal_rows"] != 1:
            raise RuntimeError("Source day did not end with one terminal transition")
        summaries.append(summary)
        print("source " + json.dumps(summary), flush=True)
    size = replay.online_size
    data = {key: getattr(replay, "_online_"+key)[:size] for key in (
        "observations", "actions", "rewards", "next_observations", "discounts", "terminals", "delta_t_seconds")}
    data["episode_id"] = replay._online_episode_ids[:size]
    np.savez_compressed(result_path, **data)
    (out / "source_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


def extra_critic_step(trainer, replay, rng, p, augmentation):
    off, sim = replay.sample_mixed(p["extra_batch_size"], trainer.config.online_fraction, rng=rng)
    batch = trainer._concatenate_batches(off, sim) if sim is not None else off
    real_target = trainer._target(batch, backup="h2oplus")
    real_loss = trainer._ensemble_td_loss(trainer.critic(batch["observations"], batch["actions"]), real_target)
    loss = real_loss
    summary = {"extra_real_loss": float(real_loss.detach())}
    if augmentation is not None:
        trust_metrics = augmentation.update_trust(trainer.actor)
        generated, weights = augmentation.generate(trainer.actor, p["extra_batch_size"])
        model_target = trainer._target(generated, backup="h2oplus")
        model_loss = trainer._ensemble_td_loss(
            trainer.critic(generated["observations"], generated["actions"]), model_target, weights)
        fraction = p["model_fraction_in_extra_loss"]
        loss = (1-fraction)*real_loss + fraction*model_loss
        summary.update(model_weight_mean=float(weights.mean()), model_loss=float(model_loss.detach()),
                       qdelta_td_loss=trust_metrics["td_loss"])
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite continuation loss")
    trainer.critic_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    trainer.critic_optimizer.step()
    trainer._soft_update_target()
    return summary


def main(out, device):
    p = json.loads((HERE / "control_protocol_v1.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    if (out / "training_summary.json").exists():
        raise FileExistsError("Completed control experiment already exists")
    (out / "protocol.json").write_text(json.dumps(p, indent=2) + "\n")
    torch.set_num_threads(4)
    initializer = ROOT / p["offline_initializer"]
    payload = torch.load(initializer, map_location="cpu", weights_only=False)
    config = TransitTrainerConfig(**payload["source_config"])
    config = replace(config, use_ratio_correction=False)
    del payload
    replay = TransitReplayBufferV7(ROOT / load_protocol()["dataset"], device=device,
                                   max_offline_rows=p["offline_subsample_rows"],
                                   subsample_seed=p["offline_subsample_seed"], online_capacity=60000)
    print(f"replay: {replay.offline_size} target rows, {int(replay.terminals.sum())} terminal rows", flush=True)
    trainer = TransitH2OTrainerV7(replay, config=config, device=device, seed=p["training_seed"])
    trainer.load_offline_initializer(initializer)
    trainer.save_policy_artifact(out / "offline_prefix.pt", extra={"method": "offline_prefix"})
    collect_source(replay, trainer, p, out)
    print(json.dumps(replay.materialize_device_cache()), flush=True)
    summaries = {}
    del trainer
    for method in p["methods"][1:]:
        if (out / f"{method}.json").exists():
            summaries[method] = json.loads((out / f"{method}.json").read_text())
            continue
        augmentation = ModelAugmentation(device, p, replay.station_max_by_line) if method == "mcwm_assisted" else None
        trainer = TransitH2OTrainerV7(replay, config=config, device=device, seed=p["training_seed"])
        trainer.load_offline_initializer(initializer)
        if method == "offline_continue":
            trainer.config = replace(trainer.config, online_fraction=0.)
        aux_rng = np.random.default_rng(p["training_seed"] + 10000)
        started, history, extra = time.monotonic(), [], {}
        cuda_devices = [torch.cuda.current_device()] if str(device).startswith("cuda") else []
        for step in range(1, p["continuation_updates"]+1):
            log = step % 200 == 0
            stats = trainer.update(online=True, collect_metrics=log)
            if step % p["extra_critic_every"] == 0:
                # Keep model sampling and trust fitting out of the learner RNG stream.
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.manual_seed(p["training_seed"] + 20000 + step)
                    extra = extra_critic_step(trainer, replay, aux_rng, p, augmentation)
            if log:
                row = {"step": step, "elapsed_seconds": time.monotonic()-started, **stats, **extra}
                if not all(np.isfinite(v) for v in row.values()):
                    raise RuntimeError(f"Non-finite metrics in {method}")
                history.append(row)
                print(f"{method} step={step} critic={stats['train/critic_loss']:.5f} "
                      f"actor={stats['train/actor_loss']:.5f} elapsed={row['elapsed_seconds']:.1f}s", flush=True)
        trainer.save_policy_artifact(out / f"{method}.pt", extra={"method": method, "protocol": p["name"]})
        if augmentation is not None:
            torch.save(augmentation.trust.net.state_dict(), out / "qdelta.pt")
        summary = {"method": method, "history": history, "policy_updates": p["continuation_updates"],
                   "critic_updates": p["continuation_updates"]*(1+1/p["extra_critic_every"]),
                   "source_rows_used": replay.online_size if method != "offline_continue" else 0,
                   "model_family": augmentation.family if augmentation is not None else None,
                   "elapsed_seconds": time.monotonic()-started}
        (out / f"{method}.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries[method] = summary
        del trainer, augmentation
    (out / "training_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "runs/control_v1")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args.out, args.device)
