"""Fresh objective-specific prefixes and matched H2O+/MC-WM continuations."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch

from cross_common import HERE, EXPERIMENTS, P, module, write, initializer_folder
from rewards import canonical_headway_reward

previous = module("cross_confirmation", EXPERIMENTS / "mcwm_seed_confirmation_v13/run.py")
base = previous.base
# The stable v7 trainer uses fixed coefficients, without the rejected recognizer.
from trainer import ContextTrainer, ContextModel, usage


class ObjectiveReplay(base.TransitReplayBufferV7):
    def __init__(self, *args, objective="passenger", **kwargs):
        super().__init__(*args, **kwargs)
        self.objective = objective
        self.target_scale = self.contract_config["scheduled_headway_scale_seconds"]
        if objective == "headway":
            for start in range(0, self.offline_size, 8192):
                stop = min(start+8192, self.offline_size)
                self.rewards[start:stop] = canonical_headway_reward(self.next_observations[start:stop],
                    self.terminals[start:stop], self.target_scale)

    def append(self, observation, action, reward, next_observation, discount, **kwargs):
        if self.objective == "headway":
            reward = canonical_headway_reward(next_observation, kwargs.get("terminal", 0.), self.target_scale)
        return super().append(observation, action, reward, next_observation, discount, **kwargs)


def protocol(args):
    p = dict(previous.P, name=P["name"], training_seed=args.seed, arm=args.arm,
        objective=args.objective, real_td_coefficient=.75 if args.arm == "fixed0250" else 1.,
        relative_out=str(args.out.relative_to(HERE)))
    return previous.smoke_protocol(p) if args.smoke else p


def load_data(args):
    paths = base.load_protocol()
    dataset = Path(paths["dataset"])
    if not dataset.is_absolute():
        dataset = Path(paths["h2o_root"]) / dataset
    replay = ObjectiveReplay(dataset, objective=args.objective, device="cpu", online_capacity=30000,
        max_offline_rows=10000 if args.smoke else 0)
    return paths, replay


def prefix(args):
    return initializer_folder(args.objective, args.seed, args.smoke) / "initializer.pt"


def initialize(args, p):
    out = prefix(args).parent
    if (out / "ready.json").exists():
        return
    started = time.monotonic()
    payload = torch.load(EXPERIMENTS / "mcwm_alignment_v2/inputs/h2oplus_seed6201.pt",
                         map_location="cpu", weights_only=False)
    config = base.TransitTrainerConfig(**payload["source_config"])
    del payload
    if args.objective == "headway":
        config = replace(config, reward_scale=P["headway_reward_scale_mc"])
    _, replay = load_data(args)
    learner = base.TransitH2OTrainerV7(replay, config=config, device="cpu", seed=args.seed)
    for update in range(p["offline_updates_in_initializer"]):
        logged = (update+1) % 1000 == 0
        metrics = learner.update(online=False, collect_metrics=logged)
        if logged:
            row = dict(objective=args.objective, seed=args.seed, offline_updates=update+1,
                elapsed_seconds=time.monotonic()-started, metrics=metrics)
            write(out / "progress.json", row)
            print(json.dumps(row), flush=True)
    learner.save_offline_initializer(prefix(args), extra=dict(objective=args.objective,
        fresh_initialization=True, seed=args.seed, template_usage="Hyperparameters only"))
    write(out / "ready.json", dict(objective=args.objective, seed=args.seed,
        offline_updates=learner.total_updates, online_events=replay.online_size, config=asdict(config),
        fresh_initialization=True, wall_seconds=time.monotonic()-started))


def train(args, p):
    if (args.out / "results/ready.json").exists():
        return
    started, costs = time.monotonic(), base.Costs("cpu")
    ready = json.loads((prefix(args).parent / "ready.json").read_text())
    assert ready["objective"] == args.objective and ready["seed"] == args.seed
    payload = torch.load(prefix(args), map_location="cpu", weights_only=False)
    config = replace(base.TransitTrainerConfig(**payload["source_config"]),
                     use_ratio_correction=args.arm == "h2oplus")
    del payload
    with costs.phase("dataset_loading"):
        paths, replay = load_data(args)
    model_enabled = args.arm == "fixed0250"
    learner = (ContextTrainer if model_enabled else base.TransitH2OTrainerV7)(
        replay, config=config, device="cpu", seed=args.seed)
    if model_enabled:
        learner.configure_context(p, args.scenario, None)
        learner.coefficient, learner.output = .25, args.out / "results"
        learner.model_fraction, learner.model_seed, learner.costs = p["model_fraction"], args.seed+60000, costs
    learner.load_offline_initializer(prefix(args))
    scene = next(s for s in p["scenarios"] if s["name"] == args.scenario)
    base.seed_all(args.seed+p["source_seed_offset"])
    base.install_source_mechanism(scene)
    env = base.AllLinesSimEnv(paths["source_env"], observation_schema="v7", demand_scale=scene["od_factor"],
        observation_speed_seed=args.seed+p["source_seed_offset"])
    if env.demand_scale > 1.25:
        env._observation_speed_profiles_by_demand[env.demand_scale] = env._observation_speed_profiles_by_demand[1.25]
    sampler = base.AllLinesTransitSamplerV7(env, replay, station_max_by_line=replay.station_max_by_line,
        contract_config=base.TransitMDPConfigV7(**replay.contract_config))
    write(args.out / "results/protocol.json", dict(protocol=p, crossover_protocol=P,
        requested_objective=args.requested_objective, config=asdict(config), offline_rows=replay.offline_size,
        initializer=str(prefix(args)), target_online_training_events=0))
    events = updates = 0
    for endpoint in base.milestones(p):
        with costs.phase("source_collection"):
            if endpoint > events:
                events += sampler.collect(learner.act, endpoint-events)["events"]
        if model_enabled:
            learner.observe_completed()
        if model_enabled and endpoint in p["model_refresh_events"]:
            with torch.random.fork_rng(devices=[]), costs.phase("model_refresh"):
                torch.manual_seed(args.seed+endpoint)
                if learner.augmentation is None:
                    learner.augmentation = ContextModel(replay, p, args.out / "models", "cpu", costs)
                else:
                    learner.augmentation.refit(replay, events)
        while updates < max(0, endpoint-p["warmup_events"]):
            logical = p["warmup_events"]+updates
            rate = config.actor_learning_rate*(p["actor_taper_multiplier"] if logical >= p["actor_taper_event"] else 1.)
            for group in learner.actor_optimizer.param_groups:
                group["lr"] = rate
            logged = (updates+1) % 1000 == 0
            with costs.phase("policy_update"):
                metrics = learner.update(online=True, collect_metrics=logged)
            updates += 1
            if logged:
                row = dict(source_events=events, continuation_updates=updates, objective=args.objective,
                    elapsed_seconds=time.monotonic()-started, metrics=metrics)
                write(args.out / "results/progress.json", row)
                print(json.dumps(row), flush=True)
    learner.save_policy_artifact(args.out / "policy.pt", extra=dict(arm=args.arm, seed=args.seed,
        objective=args.objective, scenario=args.scenario, source_events=events))
    row = dict(arm=args.arm, seed=args.seed, scenario=args.scenario, objective=args.objective,
        source_events=events, continuation_updates=updates, offline_updates=ready["offline_updates"],
        ratio_correction=config.use_ratio_correction, reward_scale=config.reward_scale,
        model_coefficient=.25 if model_enabled else 0., target_online_training_events=0,
        phases=costs.rows, training_wall_seconds=time.monotonic()-started,
        peak_host_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        model_history=learner.augmentation.history if model_enabled else [])
    write(args.out / "results/train_summary.json", row)
    write(args.out / "results/ready.json", row)


def evaluate(args, p):
    if (args.out / "results" / f"eval_{args.case:02d}.json").exists():
        return
    import evaluate_control as control
    from bus_h2o.mdp_contract_v7 import raw_state_to_canonical_v7
    import h5py
    paths = base.load_protocol()
    dataset = Path(paths["dataset"])
    if not dataset.is_absolute():
        dataset = Path(paths["h2o_root"])/dataset
    with h5py.File(dataset, "r") as data:
        stations = np.asarray(json.loads(data.attrs["station_max_by_line"]), np.float32)
        config = base.TransitMDPConfigV7(**json.loads(data.attrs["config"]))
    samples = []
    original = control.completion_safety_action

    def record(raw, action, mode, contract):
        canonical = raw_state_to_canonical_v7(raw, stations, config)
        local = canonical[5:].reshape(16, 14)[int(round(float(canonical[2])*15))]
        tf, tb = float(local[5])*config.scheduled_headway_scale_seconds, float(local[6])*config.scheduled_headway_scale_seconds
        f, b = float(local[3])*tf, float(local[4])*tb
        present = local[11:13] > .5
        deviations = [abs(h-target) for h,target,flag in zip((f,b), (tf,tb), present) if flag]
        if deviations:
            samples.append((float(canonical_headway_reward(canonical, target_scale=config.scheduled_headway_scale_seconds)),
                sum(deviations)/len(deviations), abs(f-b) if present.all() else None))
        return original(raw, action, mode, contract)

    control.completion_safety_action = record
    try:
        evaluation = dict(p, evaluation_blocks=p["test_blocks"])
        if args.smoke:
            evaluation["evaluation_horizon_seconds"] = 1200
        usage.evaluate(args, evaluation)
    finally:
        control.completion_safety_action = original
    path = args.out / "results" / f"eval_{args.case:02d}.json"
    row = json.loads(path.read_text())
    row.update(arm=args.arm, objective=args.objective, training_seed=args.seed, split="test",
        headway_ridge_mean=float(np.mean([s[0] for s in samples])),
        headway_plan_mae_seconds=float(np.mean([s[1] for s in samples])),
        headway_imbalance_seconds=float(np.mean([s[2] for s in samples if s[2] is not None])))
    write(path, row)


def run(args):
    torch.set_num_threads(1)
    p = protocol(args)
    if args.initialize:
        initialize(args, p)
    elif args.evaluate:
        evaluate(args, p)
    else:
        train(args, p)
