"""Fixed-0.25 MC-WM continuation on the formal three-demand source schedule."""

import argparse
from dataclasses import asdict, replace
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import resource
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=range(6201, 6206))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--h2o-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-env", type=Path, required=True)
    parser.add_argument("--mcwm-root", type=Path)
    parser.add_argument("--thesis-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.seed != 6201 or args.smoke or args.device != "cpu":
        raise ValueError("v105 is the fixed original6201 CPU reconstruction, without smoke")
    args.out = args.out.resolve()
    if args.out.exists():
        raise FileExistsError("Do not overwrite or retry the fixed reconstruction: " + str(args.out))
    if (args.out / "results/ready.json").exists():
        print("Already complete:", args.out, flush=True)
        return
    for name in ("h2o_root", "dataset", "source_env", "mcwm_root"):
        value = getattr(args, name)
        if value is not None:
            os.environ["THESIS_" + name.upper()] = str(value.resolve())
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    original = args.thesis_root.resolve() / "experiments/mcwm_seed_confirmation_v13"
    sys.path.insert(0, str(original))
    spec = importlib.util.spec_from_file_location("unified_mcwm_v13", original / "run.py")
    implementation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(implementation)
    import numpy as np
    import torch

    base = implementation.base
    write = implementation.write
    p = dict(implementation.P, name="thesis-mcwm-three-demand-alignment-20261006",
        arm="fixed0250", training_seed=args.seed, relative_out=str(args.out),
        source_seed_offset=0, source_demand_scales=[0.75, 1.0, 1.25],
        source_scenario="formal_calibrated_three_demand", smoke=args.smoke)
    if args.smoke:
        # Exercise source collection, model fit/refit, normalized trust, taper,
        # gradient updates and actor export without changing the 24,000-s MDP.
        p.update(warmup_events=128, source_event_budget=192, actor_taper_event=160,
            model_refresh_events=[128, 160], model_updates_per_member=2,
            model_refit_updates=2, model_ensemble_members=1,
            model_max_fit_rows=256, model_max_validation_rows=64,
            context_window_physical_rows=8)
    torch.set_num_threads(1)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.reset_peak_memory_stats()
    costs, started = base.Costs(args.device), time.monotonic()
    payload = torch.load(args.initializer, map_location="cpu", weights_only=False)
    extra = payload.get("extra", {})
    initializer_seed = extra.get("source_training_seed")
    if initializer_seed != args.seed:
        raise ValueError(f"Initializer seed {initializer_seed!r} differs from {args.seed}")
    if payload["total_updates"] != 50000 or payload["online_critic_updates"] != 0:
        raise ValueError("Expected the formal 50,000-update offline-only IQL initializer")
    config = replace(base.TransitTrainerConfig(**payload["source_config"]),
        use_ratio_correction=False)
    del payload
    with costs.phase("dataset_loading"):
        replay = base.TransitReplayBufferV7(args.dataset, device=args.device,
            online_capacity=30000, max_offline_rows=10000 if args.smoke else 0)
    learner = implementation.MappingTrainer(replay, config=config,
        device=args.device, seed=args.seed)
    learner.configure_context(p, "calibrated", None)
    learner.model_fraction, learner.model_seed, learner.costs = (
        p["model_fraction"], args.seed + 60000, costs)
    learner.load_offline_initializer(args.initializer)
    contract = base.TransitMDPConfigV7(**replay.contract_config)

    # Match formal train_transit_v3: preserve the initializer's RNG streams
    # around source construction and use the training seed for speed profiles.
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if args.device == "cuda" else None
    try:
        env = base.AllLinesSimEnv(str(args.source_env), observation_schema="v7",
            episode_seconds=contract.horizon_seconds,
            passenger_arrival_end_seconds=18000.0, demand_scale=0.75,
            observation_speed_seed=args.seed)
        sampler = base.AllLinesTransitSamplerV7(env, replay,
            station_max_by_line=replay.station_max_by_line, contract_config=contract,
            demand_scale_schedule=tuple(p["source_demand_scales"]))
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng)
    # No install_source_mechanism: original calibrated travel times follow
    # the current episode's demand scale, exactly as in formal H2O+/SAC.
    write(args.out / "results/protocol.json", dict(protocol=p, config=asdict(config),
        seed=args.seed, initializer=str(args.initializer.resolve()),
        initializer_extra={key: extra.get(key) for key in (
            "phase", "source_method", "source_run_name", "source_training_seed",
            "offline_updates_completed")}, offline_rows=replay.offline_size,
        dataset=str(args.dataset.resolve()), source_env=str(args.source_env.resolve()),
        h2o_root=str(args.h2o_root.resolve()), horizon_seconds=env.episode_seconds,
        passenger_arrival_end_seconds=env.passenger_arrival_end_seconds,
        observation_speed_seed=args.seed, source_demand_scales=list(sampler.demand_scale_schedule),
        target_online_training_events=0,
        comparison_scope="Whole controllers using the same source generation rules, demand schedule and nominal event budget; realized coverage is recorded; not a model-only ablation",
        actor_schedule="MC-WM v13: 5k warmup, direct mixed-state actor updates, step LR x0.25 from 15k events",
        runtime=dict(host=platform.node(), python=platform.python_version(),
            torch=torch.__version__, device=args.device)))
    print(json.dumps(dict(stage="initialized", seed=args.seed, offline_rows=replay.offline_size,
        seconds=time.monotonic()-started, device=args.device)), flush=True)
    events = updates = 0
    last_collection = {}
    gpu_devices = [torch.cuda.current_device()] if args.device == "cuda" else []
    for endpoint in base.milestones(p):
        with costs.phase("source_collection"):
            if endpoint > events:
                last_collection = sampler.collect(learner.act, endpoint-events,
                    hard_event_cap=endpoint == p["source_event_budget"])
                events += last_collection["events"]
        learner.observe_completed()
        if endpoint in p["model_refresh_events"]:
            with torch.random.fork_rng(devices=gpu_devices), costs.phase("model_refresh"):
                torch.manual_seed(args.seed + endpoint)
                if learner.augmentation is None:
                    learner.augmentation = implementation.previous.previous.ContextModel(
                        replay, p, args.out / "models", args.device, costs)
                else:
                    learner.augmentation.refit(replay, events)
        while updates < max(0, endpoint-p["warmup_events"]):
            logical_event = p["warmup_events"] + updates
            rate = config.actor_learning_rate * (
                p["actor_taper_multiplier"] if logical_event >= p["actor_taper_event"] else 1.0)
            for group in learner.actor_optimizer.param_groups:
                group["lr"] = rate
            logged = (updates+1) % (16 if args.smoke else 1000) == 0
            with costs.phase("policy_update"):
                stats = learner.update(online=True, collect_metrics=logged)
            updates += 1
            if logged:
                row = dict(source_events=events, continuation_updates=updates,
                    elapsed_seconds=time.monotonic()-started, metrics=stats,
                    demand_scale_episode_counts=sampler.demand_scale_episode_counts)
                write(args.out / "results/progress.json", row)
                print(json.dumps(row), flush=True)
    underfill = p["source_event_budget"] - events
    if underfill < 0 or (underfill and not (
            last_collection.get("hard_event_cap_reached")
            and last_collection.get("blocked_simultaneous_batch_size", 0) > underfill)):
        raise RuntimeError("Source collection differs from the formal atomic-batch event cap")
    if updates != p["source_event_budget"] - p["warmup_events"]:
        raise RuntimeError("Continuation update budget did not close")
    # The formal 25k source budget may end before the third daily episode.
    # Record actual visits; do not extend the budget to force all demand levels.
    learner.save_policy_artifact(args.out / "policy.pt", extra=dict(arm="fixed0250",
        seed=args.seed, training_seed=args.seed, contract_config=asdict(contract),
        scenario="formal_calibrated_three_demand", source_events=events,
        updates=learner.total_updates, smoke=args.smoke))
    learner.save_context()
    result = dict(seed=args.seed, smoke=args.smoke, source_events=events,
        source_event_cap=p["source_event_budget"], source_event_underfill=underfill,
        continuation_updates=updates, total_policy_updates=learner.total_updates,
        reused_offline_updates=50000, target_online_training_events=0,
        real_td_coefficient=p["real_td_coefficient"], model_td_coefficient=learner.coefficient,
        normalized_model_trust=True, model_history=learner.augmentation.history,
        source_demand_scales=list(sampler.demand_scale_schedule),
        demand_scale_episode_counts=sampler.demand_scale_episode_counts,
        phases=costs.rows, training_wall_seconds=time.monotonic()-started,
        peak_host_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/2**20 if args.device == "cuda" else 0.)
    write(args.out / "results/train_summary.json", result)
    write(args.out / "results/ready.json", result)
    print(json.dumps(result), flush=True)
    sys.path.insert(0, str(args.thesis_root.resolve()))
    from experiments.mcwm_full_state_recovery_v105_20261029.verify import finish_recovery
    recovery = finish_recovery(learner, args, result)
    if recovery['status'] != 'complete':
        raise RuntimeError('v105 recovery gate failed; inspect recovery_summary.json; no retry')


if __name__ == "__main__":
    main()
