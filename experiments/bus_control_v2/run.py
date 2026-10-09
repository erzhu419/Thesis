"""Fresh matched H2O+/MC-WM bus cells; completed holding/welfare cells stay frozen."""

import argparse
from dataclasses import asdict, replace
import math
import os
from pathlib import Path
import random
import resource
import time

import numpy as np
import torch

from station_common import P, HERE, action_dim, data_folder, policy_folder, write
from dataset import Replay, prepare_holding, collect_dual, merge_dual
from networks import Actor, bindings
from SimpleSAC.transit_v3.trainer import TransitH2OTrainer, TransitTrainerConfig


class StationTrainer(TransitH2OTrainer):
    @property
    def alpha(self):
        return super().alpha.clamp_max(.6)

    def _ensemble_td_loss(self, predictions, targets, weights=None):
        loss = super()._ensemble_td_loss(predictions, targets, weights)
        model = getattr(self, "model_loss", None)
        return loss if model is None else .75*loss + .25*model

    def update(self, *, online, online_only=False, collect_metrics=True):
        self.model_loss = None
        if online and getattr(self, "models", None) is not None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed + self.total_updates + 60000)
                if self.total_updates % 4 == 0:
                    self.models.update_trust(self.actor)
                batch, weights = self.models.generate(self.actor, 128)
                predicted = self.critic(batch["observations"], batch["actions"])
                self.model_loss = super()._ensemble_td_loss(predicted, self._target(batch, backup="h2oplus"), weights)
        try:
            result = super().update(online=online, online_only=online_only, collect_metrics=collect_metrics)
            with torch.no_grad():
                self.log_alpha.clamp_(max=math.log(.6))
            if collect_metrics and self.model_loss is not None:
                result["model_td_loss"] = float(self.model_loss.detach())
            return result
        finally:
            self.model_loss = None


def config(mode):
    return TransitTrainerConfig(ensemble_size=5, hidden_dim=P["hidden_dim"], batch_size=P["batch_size"],
        reward_scale=1., target_entropy=-float(action_dim(mode)), initial_temperature=.1,
        offline_objective="iql", online_backup="h2oplus", mixed_backup_lambda=.5,
        gradient_clip_norm=1., activation="relu", offline_cql_alpha=0., online_cql_alpha=0., ratio_min_weight=.1,
        ratio_max_weight=5., bootstrap_probability=1., online_actor_batch_mode="latest")


def initializer(args):
    return HERE / ("smoke_inputs" if args.smoke else "inputs") / args.mode / args.objective / f"seed{args.seed}" / "initializer.pt"


def probe(learner, replay):
    action = learner.act(replay.offline["observations"][:512], deterministic=True)
    holds = 30*(action[:, 0]+1)
    return dict(mean_hold_seconds=float(holds.mean()), near_zero_fraction=float(np.mean(holds < 1)),
        near_sixty_fraction=float(np.mean(holds > 59)),
        mean_speed_ratio=float((1+.2*action[:, 1]).mean()) if action.shape[1] == 2 else 1.)


def initialize(args):
    path = initializer(args)
    if (path.parent / "ready.json").exists():
        return
    started = time.monotonic()
    replay = Replay(args.mode, args.objective, args.smoke)
    learner = StationTrainer(replay, config=config(args.mode), seed=args.seed, **bindings(args.mode))
    learner.seed = args.seed
    updates = 32 if args.smoke else P["offline_updates"]
    for step in range(updates):
        logged = (step+1) % 1000 == 0 or step+1 == updates
        stats = learner.update(online=False, collect_metrics=logged)
        if logged:
            row = dict(offline_updates=step+1, wall_seconds=time.monotonic()-started, metrics=stats)
            write(path.parent / "progress.json", row)
            print(row, flush=True)
    learner.save(path, extra=dict(mode=args.mode, objective=args.objective, seed=args.seed,
                                observation_dim=15+action_dim(args.mode), version=P["name"]),
                 include_online_replay=False)
    write(path.parent / "ready.json", dict(mode=args.mode, objective=args.objective, seed=args.seed,
        offline_updates=updates, offline_rows=replay.offline_size, reward_mean=replay.mean,
        reward_std=replay.std, config=asdict(learner.config), probe=probe(learner, replay),
        wall_seconds=time.monotonic()-started))


def train(args):
    from environments import ControlledSimEnv, StationSampler, install_source
    from world_model import StationModels
    out = policy_folder(args.mode, args.objective, args.scenario, args.arm, args.seed, args.smoke)
    if (out / "ready.json").exists():
        return
    started = time.monotonic()
    replay = Replay(args.mode, args.objective, args.smoke)
    learner = StationTrainer(replay, config=config(args.mode), seed=args.seed, **bindings(args.mode))
    learner.seed = args.seed
    learner.load(initializer(args))
    learner.config = replace(learner.config, use_ratio_correction=args.arm == "h2oplus")
    scene = next(s for s in P["scenarios"] if s["name"] == args.scenario)
    random.seed(args.seed+P["source_seed_offset"])
    np.random.seed(args.seed+P["source_seed_offset"])
    install_source(scene)
    env = ControlledSimEnv(os.environ["THESIS_SOURCE_ENV"], observation_schema="v7",
        demand_scale=scene["od_factor"], observation_speed_seed=args.seed+P["source_seed_offset"])
    if env.demand_scale > 1.25:
        env._observation_speed_profiles_by_demand[env.demand_scale] = env._observation_speed_profiles_by_demand[1.25]
    sampler = StationSampler(env, replay, args.mode, replay.stations)
    learner.models = None
    budget, warmup = (512, 256) if args.smoke else (P["source_events"], P["warmup_events"])
    refresh = [warmup] if args.smoke else P["model_refresh_events"]
    events = updates = 0
    write(out / "protocol.json", dict(protocol=P, mode=args.mode, objective=args.objective,
        arm=args.arm, seed=args.seed, scene=scene, config=asdict(learner.config),
        reward_mean=replay.mean, reward_std=replay.std, offline_rows=replay.offline_size,
        target_online_training_events=0))
    while events < budget:
        next_stop = min([x for x in refresh+[budget] if x > events] + [events+P["collection_interval"]])
        events += sampler.collect(learner.act, next_stop-events)
        if args.arm == "mcwm" and any(events >= x and x not in getattr(learner, "refreshed", []) for x in refresh):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(args.seed+events)
                if learner.models is None:
                    learner.models = StationModels(replay, args.seed, out / "models", args.smoke)
                else:
                    learner.models.refit(events)
            learner.refreshed = [x for x in refresh if x <= events]
        while updates < max(0, min(events, budget)-warmup):
            logical = updates+warmup
            for group in learner.actor_optimizer.param_groups:
                group["lr"] = 3e-4*(.25 if logical >= 15000 else 1.)
            logged = (updates+1) % 1000 == 0 or updates+1 == budget-warmup
            stats = learner.update(online=True, collect_metrics=logged)
            updates += 1
            if logged:
                row = dict(source_events=events, updates=updates, metrics=stats, probe=probe(learner, replay),
                           wall_seconds=time.monotonic()-started)
                write(out / "progress.json", row)
                print(row, flush=True)
    torch.save(dict(actor=learner.actor.state_dict(), mode=args.mode, objective=args.objective,
                    hidden_dim=P["hidden_dim"], seed=args.seed, arm=args.arm), out / "policy.pt")
    if learner.models is not None:
        learner.models.save()
    write(out / "ready.json", dict(mode=args.mode, objective=args.objective, seed=args.seed,
        arm=args.arm, scenario=args.scenario, source_events=events, updates=updates,
        settled_source_transitions=replay.online_size, probe=probe(learner, replay),
        training_wall_seconds=time.monotonic()-started,
        peak_host_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        target_online_training_events=0))


def evaluate(args):
    from evaluate import evaluate_policy
    evaluate_policy(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("holding-data", "dual-data", "merge", "initialize", "train", "evaluate"))
    parser.add_argument("--mode", choices=("holding", "dual"), default="dual")
    parser.add_argument("--objective", choices=("headway", "passenger"), default="headway")
    parser.add_argument("--arm", choices=P["arms"]+["zero", "daganzo_duet"], default="h2oplus")
    parser.add_argument("--scenario", choices=[s["name"] for s in P["scenarios"]], default="calibrated")
    parser.add_argument("--seed", type=int, default=P["training_seeds"][0])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == "holding-data":
        prepare_holding(args.smoke)
    elif args.phase == "dual-data":
        collect_dual(args.shard, args.smoke)
    elif args.phase == "merge":
        merge_dual(args.smoke)
    else:
        globals()[args.phase](args)


if __name__ == "__main__":
    main()
