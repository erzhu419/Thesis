"""Fresh offline prefixes and live source interaction budgets for bus control."""

import argparse
import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from data import HERE, KEYS, load_protocol
from control_model import ModelAugmentation
from model import BusModel, Normalizer, metrics, targets
from run_pilot import predict, take, train
from source_mismatch import install_source_mechanism, restore_source_mechanism
from run_control import (ROOT, AllLinesSimEnv, AllLinesTransitSamplerV7,
                         TransitReplayBufferV7, TransitH2OTrainerV7,
                         TransitMDPConfigV7, TransitTrainerConfig, extra_critic_step)


def policy_tag(count, method, budget, scenario):
    middle = "" if scenario == "calibrated" else scenario + "_"
    return f"blocks{count}_{middle}{method}_events{budget}"


def block_view(source, destination, blocks):
    """Expose complete permitted trajectories without copying the large log."""
    with h5py.File(source, "r") as original, h5py.File(destination, "w", libver="latest") as view:
        selected = np.isin(original["demand_realization_seed"][:].reshape(-1), blocks)
        rows = np.flatnonzero(selected)
        starts = np.r_[0, np.flatnonzero(np.diff(rows) != 1) + 1]
        stops = np.r_[starts[1:], len(rows)]
        for key, value in original.attrs.items():
            view.attrs[key] = value
        view.attrs["budget_blocks"] = json.dumps(blocks)
        for key, dataset in original.items():
            if dataset.shape[0] != len(selected):
                raise ValueError(f"Unexpected non-row dataset: {key}")
            layout = h5py.VirtualLayout((len(rows), *dataset.shape[1:]), dataset.dtype)
            virtual = h5py.VirtualSource(str(source), key, shape=dataset.shape)
            for start, stop in zip(starts, stops):
                layout[start:stop] = virtual[rows[start]:rows[stop-1]+1]
            view.create_virtual_dataset(key, layout)
    return int(len(rows))


def target_splits(replay, blocks, protocol):
    eligible = (replay.delta_t_seconds > 0) & (replay.terminals < .5)
    groups = ({"fit": blocks[:1], "stop": blocks[1:], "calibrate": blocks[1:]}
              if len(blocks) == 2 else
              {"fit": blocks[:-2], "stop": blocks[-2:-1], "calibrate": blocks[-1:]})
    rng = np.random.default_rng(62121)
    result = {}
    for name, allowed in groups.items():
        ids = np.flatnonzero(eligible & np.isin(replay.demand_realization_seeds, allowed))
        count = protocol["model_max_fit_rows" if name == "fit" else "model_max_validation_rows"]
        ix = np.sort(rng.choice(ids, min(count, len(ids)), replace=False))
        result[name] = dict(s=replay.observations[ix], a=replay.actions[ix],
                            ns=replay.next_observations[ix], r=replay.rewards[ix],
                            dt=replay.delta_t_seconds[ix], block=replay.demand_realization_seeds[ix])
    return result


def fit_budget_model(replay, target_raw, protocol, output, device):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    size = replay.online_size
    ids = np.flatnonzero((replay._online_delta_t_seconds[:size] > 0) &
                        (replay._online_terminals[:size] < .5))
    rng = np.random.default_rng(protocol["training_seed"] + size)
    rng.shuffle(ids)
    split = max(1, len(ids)//5)
    groups = {"stop": ids[:split], "fit": ids[split:][:protocol["model_max_fit_rows"]]}
    source = {}
    stations = torch.as_tensor(replay.station_max_by_line, dtype=torch.float32, device=device)
    for key, ix in groups.items():
        raw = dict(s=replay._online_observations[ix], a=replay._online_actions[ix],
                   ns=replay._online_next_observations[ix], r=replay._online_rewards[ix],
                   dt=replay._online_delta_t_seconds[ix], block=np.zeros(len(ix)))
        source[key] = targets(raw, stations, device)
    target = {key: targets(value, stations, device) for key, value in target_raw.items()}
    norm = Normalizer(source["fit"]).to(device)
    settings = {"learning_rate": 3e-4, "batch_size": 128, "validation_every": 100,
                "updates_per_member": protocol["model_updates_per_member"]}
    families = {key: [] for key in ("source", "direct", "residual")}
    history = {}
    for member in range(protocol["model_ensemble_members"]):
        seed = protocol["training_seed"] + member + 50000
        torch.manual_seed(seed)
        base = BusModel().to(device)
        history[f"source_{member}"] = train(base, source["fit"], source["stop"], norm,
                                             settings, seed, f"source/{member}")
        families["source"].append(base)
        for family in ("direct", "residual"):
            torch.manual_seed(seed + 100)
            model = BusModel(base if family == "residual" else None).to(device)
            history[f"{family}_{member}"] = train(model, target["fit"], target["stop"], norm,
                                                   settings, seed+100, f"{family}/{member}")
            families[family].append(model)
    calibration = {name: metrics(predict(models, target["calibrate"]), target["calibrate"], norm)
                   for name, models in families.items()}
    selected = min(("residual", "direct"), key=lambda name: calibration[name]["selection_loss"])
    summary = {"selected_family": selected, "source_rows_available": size,
               "source_boundary_fit_rows": len(groups["fit"]),
               "source_boundary_stop_rows": len(groups["stop"]),
               "target_rows": {k: len(v["s"]) for k, v in target_raw.items()},
               "calibration": calibration, "training": history,
               "elapsed_seconds": time.monotonic()-started}
    (output / "calibration_decision.json").write_text(json.dumps(summary, indent=2)+"\n")
    torch.save({"models": {name: [model.state_dict() for model in models]
                           for name, models in families.items()},
                "normalizer": norm.state_dict()}, output / "pilot_models.pt")
    np.savez_compressed(output / "target_fit.npz", **target_raw["fit"])
    return summary


def run(out, counts, device):
    p = json.loads((HERE / "budget_protocol_v2.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    methods = []
    for count in p["offline_block_counts"]:
        methods.append(f"blocks{count}_offline")
        for budget in p["source_event_budgets"]:
            methods.extend(f"blocks{count}_{method}_events{budget}" for method in ("h2o", "mcwm"))
        if count == p["mismatch_block_count"]:
            methods.extend(policy_tag(count, method, max(p["source_event_budgets"]), scenario["name"])
                           for scenario in p["mismatch_scenarios"] if scenario["name"] != "calibrated"
                           for method in ("h2o", "mcwm"))
    p["methods"] = methods
    (out / "protocol.json").write_text(json.dumps(p, indent=2)+"\n")
    torch.set_num_threads(4)
    initial_path = ROOT / json.loads((HERE / "control_protocol_v1.json").read_text())["offline_initializer"]
    # Only network/training hyperparameters are inherited, never pretrained tensors.
    config = TransitTrainerConfig(**torch.load(initial_path, map_location="cpu", weights_only=False)["source_config"])
    for count in counts or p["offline_block_counts"]:
        cell = out / f"blocks{count}"
        cell.mkdir(exist_ok=True)
        blocks = p["offline_blocks"][:count]
        dataset = cell / "offline_view.h5"
        if not dataset.exists():
            block_view(ROOT / load_protocol()["dataset"], dataset, blocks)
        replay = TransitReplayBufferV7(dataset, device=device, online_capacity=20000,
                                      device_cache_observation_dtype="float32")
        empty_online = replay.online_state_dict()
        replay.materialize_device_cache()
        print(json.dumps({"blocks": blocks, "offline_rows": replay.offline_size,
                          "offline_trajectories": len(np.unique(replay.trajectory_ids)),
                          "terminal_rows": int(replay.terminals.sum())}), flush=True)
        initializer = cell / "offline_initializer.pt"
        trainer = TransitH2OTrainerV7(replay, config=config, device=device, seed=p["training_seed"])
        if not initializer.exists():
            start = time.monotonic()
            for step in tqdm(range(p["offline_updates"]), desc=f"blocks{count} offline", mininterval=10):
                stats = trainer.update(online=False, collect_metrics=(step+1) % 1000 == 0)
                if stats and not all(np.isfinite(v) for v in stats.values()):
                    raise RuntimeError("Non-finite offline training metrics")
            trainer.save_offline_initializer(initializer, extra={"blocks": blocks, "fresh_budget_prefix": True})
            summary = {"blocks": blocks, "rows": replay.offline_size,
                       "trajectories": len(np.unique(replay.trajectory_ids)),
                       "updates": p["offline_updates"], "elapsed_seconds": time.monotonic()-start}
            (cell / "offline_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        else:
            trainer.load_offline_initializer(initializer)
        trainer.save_policy_artifact(out / f"blocks{count}_offline.pt", extra={"blocks": blocks, "source_events": 0})
        del trainer
        target_raw = target_splits(replay, blocks, p)
        scenarios = p["mismatch_scenarios"] if count == p["mismatch_block_count"] else p["mismatch_scenarios"][:1]
        for scenario, method in [(scenario, method) for scenario in scenarios for method in ("h2o", "mcwm")]:
            branch = cell / scenario["name"] / method
            branch.mkdir(parents=True, exist_ok=True)
            result_path = branch / "summary.json"
            if result_path.exists():
                continue
            replay.load_online_state_dict(empty_online)
            trainer = TransitH2OTrainerV7(replay, config=config, device=device, seed=p["training_seed"])
            trainer.load_offline_initializer(initializer)
            random.seed(p["source_seed"])
            np.random.seed(p["source_seed"])
            torch.manual_seed(p["source_seed"])
            install_source_mechanism(scenario)
            env = AllLinesSimEnv(str(ROOT / "bus_h2o/calibrated_env"), observation_schema="v7",
                                demand_scale=scenario["od_factor"], observation_speed_seed=p["source_seed"])
            sampler = AllLinesTransitSamplerV7(env, replay, station_max_by_line=replay.station_max_by_line,
                                              contract_config=TransitMDPConfigV7(**replay.contract_config))
            started = time.monotonic()
            events = updates = 0
            augmentation = None
            aux_rng = np.random.default_rng(p["training_seed"]+10000)
            history, snapshots, models = [], [], []
            for threshold in tqdm(range(p["collection_interval"], max(p["source_event_budgets"])+1,
                                        p["collection_interval"]), desc=f"blocks{count} {scenario['name']} {method}", mininterval=10):
                stats = sampler.collect(trainer.act, threshold-events, hard_event_cap=True)
                events += stats["events"]
                if method == "mcwm" and threshold in p["model_refresh_events"]:
                    model_run = branch / f"model_events{threshold}"
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.startswith("cuda") else []):
                        model_summary = fit_budget_model(replay, target_raw, p, model_run, device)
                        refreshed = ModelAugmentation(device, p, replay.station_max_by_line, model_run=model_run)
                    if augmentation is not None:
                        refreshed.trust = augmentation.trust
                    augmentation = refreshed
                    models.append(model_summary)
                for _ in range(p["policy_updates_per_interval"]):
                    updates += 1
                    metrics_row = trainer.update(online=True, collect_metrics=updates % 128 == 0)
                    if updates % p["extra_critic_every"] == 0:
                        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.startswith("cuda") else []):
                            torch.manual_seed(p["training_seed"]+20000+updates)
                            extra = extra_critic_step(trainer, replay, aux_rng, p, augmentation)
                    if metrics_row:
                        history.append({"policy_updates": updates, "source_events": events,
                                        "elapsed_seconds": time.monotonic()-started, **metrics_row, **extra})
                if threshold in p["source_event_budgets"]:
                    tag = policy_tag(count, method, threshold, scenario["name"])
                    snapshot = {"method": method, "nominal_source_events": threshold,
                                "source_mechanism": scenario,
                                "actual_source_events": events, "source_rows": replay.online_size,
                                "source_terminal_rows": int(replay._online_terminals[:replay.online_size].sum()),
                                "policy_updates": updates, "critic_updates": updates+updates//p["extra_critic_every"],
                                "elapsed_seconds": time.monotonic()-started,
                                "model_family": augmentation.family if augmentation else None}
                    trainer.save_policy_artifact(out / f"{tag}.pt", extra=snapshot)
                    (out / f"{tag}.json").write_text(json.dumps(snapshot, indent=2)+"\n")
                    snapshots.append(snapshot)
                    print(json.dumps(snapshot), flush=True)
            result_path.write_text(json.dumps({"history": history, "snapshots": snapshots,
                                               "models": models}, indent=2)+"\n")
            del trainer, augmentation, sampler, env
            restore_source_mechanism()
        del replay
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "runs/budget_v2")
    parser.add_argument("--blocks", nargs="+", type=int, choices=[2, 6])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.out.resolve(), args.blocks, args.device)
