"""Fresh matched controls and causal adaptive TD weights using the v7 learner."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import resource
import time
import numpy as np
import torch

from control_common import HERE, P, FIXED, CONTEXTS, module, write, bundle_path, smoke_protocol

previous = module("context_control_training", HERE.parent / "mcwm_context_v7/run.py")
previous.HERE, previous.FIXED, previous.bundle_path = HERE, FIXED, bundle_path
original = module("context_control_original", HERE.parent / "mcwm_context_v7/context.py")
window = module("context_control_window", HERE.parent / "mcwm_context_filter_v9/filters.py")


class AdaptiveTrainer(previous.ContextTrainer):
    def configure_context(self, p, scenario, bundle):
        super().configure_context(p, scenario, bundle)
        self.output = HERE / p["relative_out"] / "results"
        self.episode_index = 0
        self.feature_indices = list(range(8)) if self.arm == "recognized_full" else [0, 1, 2]
        if self.arm in FIXED:
            self.coefficient = FIXED[self.arm]
            return
        key = "recognizer" if self.arm == "recognized_full" else "physical_recognizer"
        model = bundle[key]
        coefficients = [bundle["coefficients"][c] for c in model["classes"]]
        if self.arm == "physical_window4_fallback":
            self.context_filter = window.WindowFilter(model, coefficients, p["window_blocks"],
                bundle["global_coefficient"], p["fallback_score_margin"])
        else:
            self.context_filter = original.ContextFilter(model, coefficients, p)
        self.coefficient = self.context_filter.coefficient

    def observe_completed(self):
        def observe(block):
            if self.context_filter is not None:
                features = np.asarray(block["features"])[self.feature_indices]
                self.coefficient = self.context_filter.observe(features)
                accepted = self.context_filter.decision[1] if self.arm == "physical_window4_fallback" else True
                self.trace.append(dict(end_source_row=block["end_source_row"], episode_index=self.episode_index,
                    belief=self.context_filter.belief.tolist(), coefficient=self.coefficient,
                    context_accepted=accepted))

        def reset():
            self.episode_index += 1
            if self.context_filter is not None:
                self.context_filter.reset()
                self.coefficient = self.context_filter.coefficient

        self.blocks.consume(self.replay_buffer, observe, reset)


previous.ContextTrainer = AdaptiveTrainer
ContextTrainer, ContextModel, base = AdaptiveTrainer, previous.ContextModel, previous.base


def train(args, p):
    device = "cpu" if args.smoke else p["training_device"]
    torch.set_num_threads(1)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    costs, started = base.Costs(device), time.monotonic()
    initializer = HERE.parent / "mcwm_alignment_v2/inputs" / f"h2oplus_seed{args.seed}.pt"
    payload = torch.load(initializer, map_location="cpu", weights_only=False)
    assert payload["total_updates"] == p["offline_updates_in_initializer"]
    config = replace(base.TransitTrainerConfig(**payload["source_config"]), use_ratio_correction=False)
    del payload
    paths = base.load_protocol()
    dataset = Path(paths["dataset"])
    if not dataset.is_absolute():
        dataset = Path(paths["h2o_root"]) / dataset
    with costs.phase("dataset_loading"):
        replay = base.TransitReplayBufferV7(dataset, device=device, online_capacity=30000,
            max_offline_rows=10000 if args.smoke else 0)
    learner = ContextTrainer(replay, config=config, device=device, seed=args.seed)
    bundle = None if args.arm in FIXED else json.loads(bundle_path(args.seed, args.smoke).read_text())
    if bundle is not None and not args.smoke:
        assert bundle["provider_seed"] != args.seed and bundle["consumer_seed"] == args.seed
    learner.configure_context(p, args.scenario, bundle)
    learner.model_fraction, learner.model_seed, learner.costs = p["model_fraction"], args.seed+60000, costs
    learner.load_offline_initializer(initializer)
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
        target_online_training_events=0, context_provider=bundle))
    events = updates = 0
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    for endpoint in base.milestones(p):
        with costs.phase("source_collection"):
            if endpoint > events:
                events += sampler.collect(learner.act, endpoint-events)["events"]
        with costs.phase("context_observation"):
            learner.observe_completed()
        if endpoint in p["model_refresh_events"]:
            with torch.random.fork_rng(devices=devices), costs.phase("model_refresh"):
                torch.manual_seed(args.seed+endpoint)
                if learner.augmentation is None:
                    learner.augmentation = ContextModel(replay, p, args.out / "models", device, costs)
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
        if endpoint in p["model_refresh_events"]:
            learner.save_policy_artifact(args.out / f"policy_events{endpoint}.pt",
                extra=dict(arm=args.arm, source_events=events))
    learner.save_policy_artifact(args.out / "policy.pt", extra=dict(arm=args.arm, seed=args.seed,
        scenario=args.scenario, source_events=events, updates=learner.total_updates))
    learner.save_context()
    result = dict(arm=args.arm, seed=args.seed, scenario=args.scenario, source_events=events,
        continuation_updates=updates, total_policy_updates=learner.total_updates,
        total_critic_updates=learner.total_updates, target_online_training_events=0,
        model_history=learner.augmentation.history, phases=costs.rows,
        training_wall_seconds=time.monotonic()-started, reused_offline_updates=p["offline_updates_in_initializer"],
        real_td_coefficient=p["real_td_coefficient"], final_model_coefficient=learner.coefficient,
        context_blocks=len(learner.blocks.blocks), context_provider_seed=None if bundle is None else bundle["provider_seed"],
        peak_host_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/2**20 if device == "cuda" else 0.)
    write(args.out / "results/train_summary.json", result)
    write(args.out / "results/ready.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=P["arms"])
    parser.add_argument("--scenario", required=True, choices=CONTEXTS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--case", type=int, default=10, choices=range(10, 20))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    p = dict(P, arm=args.arm, training_seed=args.seed, relative_out=str(args.out.relative_to(HERE)))
    if args.smoke:
        assert "smoke" in args.out.parts
        p = smoke_protocol(p)
    if args.evaluate:
        previous.evaluate(args, p)
    elif not (args.out / "results/ready.json").exists():
        device = "cpu" if args.smoke else P["training_device"]
        gpu = torch.cuda.get_device_name(0) if device == "cuda" else None
        write(args.out / "results/runtime.json", dict(host=platform.node(), torch_version=torch.__version__,
            cuda_version=torch.version.cuda, gpu_model=gpu, device=device))
        train(args, p)
    print("DONE adaptive context control study", flush=True)
