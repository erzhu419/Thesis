"""Fresh IQL prefixes, matched bus continuations and unchanged SUMO evaluation."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import resource
import time
import torch

from confirmation_common import (HERE, P, FIXED, CONTEXTS, module, write, smoke_protocol,
    initializer_path, initializer_folder, bundle_path, policy_folder)

previous = module("confirmation_mapping", HERE.parent / "mcwm_context_mapping_v11/run.py")
for implementation in (previous, previous.previous):
    implementation.HERE, implementation.FIXED, implementation.bundle_path = HERE, FIXED, bundle_path
MappingTrainer = previous.MappingTrainer
base = previous.previous.base
evaluate_existing = previous.evaluate


class NoModelTrainer(MappingTrainer):
    def _ensemble_td_loss(self, predictions, targets, weights=None):
        return base.TransitH2OTrainerV7._ensemble_td_loss(self, predictions, targets, weights)


def load_data(device, smoke):
    paths = base.load_protocol()
    dataset = Path(paths["dataset"])
    if not dataset.is_absolute():
        dataset = Path(paths["h2o_root"]) / dataset
    replay = base.TransitReplayBufferV7(dataset, device=device, online_capacity=30000,
        max_offline_rows=10000 if smoke else 0)
    return paths, replay


def initialize(args, p):
    folder = initializer_folder(args.seed, args.smoke)
    if (folder / "ready.json").exists():
        return
    torch.set_num_threads(1)
    started = time.monotonic()
    template = HERE.parent / "mcwm_alignment_v2/inputs/h2oplus_seed6201.pt"
    payload = torch.load(template, map_location="cpu", weights_only=False)
    config = base.TransitTrainerConfig(**payload["source_config"])
    del payload
    _, replay = load_data("cpu", args.smoke)
    learner = base.TransitH2OTrainerV7(replay, config=config, device="cpu", seed=args.seed)
    for update in range(p["offline_updates_in_initializer"]):
        stats = learner.update(online=False, collect_metrics=(update+1) % 1000 == 0)
        if (update+1) % 1000 == 0:
            progress = dict(seed=args.seed, offline_updates=update+1,
                elapsed_seconds=time.monotonic()-started, metrics=stats)
            write(folder / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    learner.save_offline_initializer(initializer_path(args.seed, args.smoke),
        extra=dict(seed=args.seed, fresh_initialization=True, offline_updates=learner.total_updates,
            template_usage="Hyperparameters only; no pretrained tensor or RNG state loaded"))
    write(folder / "ready.json", dict(seed=args.seed, fresh_initialization=True,
        offline_updates=learner.total_updates, online_events=replay.online_size,
        wall_seconds=time.monotonic()-started, offline_rows=replay.offline_size,
        config=asdict(config), runtime=runtime(), smoke=args.smoke))


def train(args, p):
    torch.set_num_threads(1)
    started = time.monotonic()
    costs = base.Costs("cpu")
    init = initializer_path(args.seed, args.smoke)
    init_row = json.loads((init.parent / "ready.json").read_text())
    assert init_row["seed"] == args.seed and init_row["fresh_initialization"]
    assert init_row["offline_updates"] == p["offline_updates_in_initializer"] and init_row["online_events"] == 0
    payload = torch.load(init, map_location="cpu", weights_only=False)
    config = replace(base.TransitTrainerConfig(**payload["source_config"]), use_ratio_correction=False)
    del payload
    with costs.phase("dataset_loading"):
        paths, replay = load_data("cpu", args.smoke)
    learner_type = NoModelTrainer if args.arm == "h2o_matched" else MappingTrainer
    learner = learner_type(replay, config=config, device="cpu", seed=args.seed)
    bundle = None if args.arm in FIXED else json.loads(bundle_path(args.seed).read_text())
    if bundle is not None:
        assert bundle["consumer_seed"] == args.seed and bundle["provider_seed"] != args.seed
    learner.configure_context(p, args.scenario, bundle)
    learner.model_fraction, learner.model_seed, learner.costs = p["model_fraction"], args.seed+60000, costs
    learner.load_offline_initializer(init)
    scene = next(s for s in p["scenarios"] if s["name"] == args.scenario)
    base.seed_all(args.seed+p["source_seed_offset"])
    base.install_source_mechanism(scene)
    env = base.AllLinesSimEnv(paths["source_env"], observation_schema="v7", demand_scale=scene["od_factor"],
        observation_speed_seed=args.seed+p["source_seed_offset"])
    if env.demand_scale > 1.25:
        env._observation_speed_profiles_by_demand[env.demand_scale] = env._observation_speed_profiles_by_demand[1.25]
    sampler = base.AllLinesTransitSamplerV7(env, replay, station_max_by_line=replay.station_max_by_line,
        contract_config=base.TransitMDPConfigV7(**replay.contract_config))
    write(args.out / "results/protocol.json", dict(protocol=p, arm=args.arm, scenario=scene,
        training_seed=args.seed, config=asdict(config), offline_rows=replay.offline_size,
        initializer_seed=args.seed, fresh_initialization=True, context_provider=bundle,
        target_online_training_events=0))
    events = updates = 0
    model_enabled = args.arm != "h2o_matched"
    for endpoint in base.milestones(p):
        with costs.phase("source_collection"):
            if endpoint > events:
                events += sampler.collect(learner.act, endpoint-events)["events"]
        with costs.phase("context_observation"):
            learner.observe_completed()
        if model_enabled and endpoint in p["model_refresh_events"]:
            with torch.random.fork_rng(devices=[]), costs.phase("model_refresh"):
                torch.manual_seed(args.seed+endpoint)
                if learner.augmentation is None:
                    learner.augmentation = previous.previous.ContextModel(replay, p, args.out / "models", "cpu", costs)
                else:
                    learner.augmentation.refit(replay, events)
        while updates < max(0, endpoint-p["warmup_events"]):
            logical = p["warmup_events"]+updates
            rate = config.actor_learning_rate*(p["actor_taper_multiplier"] if logical >= p["actor_taper_event"] else 1.)
            for group in learner.actor_optimizer.param_groups:
                group["lr"] = rate
            logged = (updates+1) % 1000 == 0
            with costs.phase("policy_update"):
                stats = learner.update(online=True, collect_metrics=logged)
            updates += 1
            if logged:
                progress = dict(source_events=events, continuation_updates=updates,
                    elapsed_seconds=time.monotonic()-started, model_coefficient=learner.coefficient, metrics=stats)
                write(args.out / "results/progress.json", progress)
                print(json.dumps(progress), flush=True)
    learner.save_policy_artifact(args.out / "policy.pt", extra=dict(arm=args.arm, seed=args.seed,
        scenario=args.scenario, source_events=events, updates=learner.total_updates))
    learner.save_context()
    result = dict(arm=args.arm, seed=args.seed, scenario=args.scenario, initializer_seed=args.seed,
        fresh_initialization=True, source_events=events, continuation_updates=updates,
        total_policy_updates=learner.total_updates, total_critic_updates=learner.total_updates,
        target_online_training_events=0, model_history=learner.augmentation.history if model_enabled else [],
        phases=costs.rows, training_wall_seconds=time.monotonic()-started,
        reused_offline_updates=p["offline_updates_in_initializer"], real_td_coefficient=p["real_td_coefficient"],
        final_model_coefficient=learner.coefficient, context_blocks=len(learner.blocks.blocks),
        context_provider_seed=None if bundle is None else bundle["provider_seed"],
        peak_host_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, peak_gpu_allocated_mb=0.)
    write(args.out / "results/train_summary.json", result)
    write(args.out / "results/ready.json", result)


def runtime():
    return dict(host=platform.node(), torch_version=torch.__version__, cuda_version=torch.version.cuda,
        gpu_model=None, device="cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=P["arms"])
    parser.add_argument("--scenario", choices=CONTEXTS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--case", type=int, default=0, choices=range(20))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    p = dict(P, training_seed=args.seed, arm=args.arm,
        real_td_coefficient=1. if args.arm == "h2o_matched" else P["real_td_coefficient"])
    if args.smoke:
        p = smoke_protocol(p)
    if args.initialize:
        initialize(args, p)
    else:
        if args.arm is None or args.scenario is None or args.out is None:
            parser.error("Continuation/evaluation requires arm, scenario and output")
        args.out = args.out.resolve()
        args.out.mkdir(parents=True, exist_ok=True)
        p["relative_out"] = str(args.out.relative_to(HERE))
        if args.evaluate:
            source = policy_folder(args.scenario, args.arm, args.seed, args.smoke) / "policy.pt"
            if source.parent != args.out and not (args.out / "policy.pt").exists():
                (args.out / "policy.pt").symlink_to(source)
            evaluate_existing(args, p)
        elif not (args.out / "results/ready.json").exists():
            assert args.arm in P["training_arms"]
            write(args.out / "results/runtime.json", runtime())
            train(args, p)
    print("DONE fresh-seed bus comparison", flush=True)


if __name__ == "__main__":
    main()
