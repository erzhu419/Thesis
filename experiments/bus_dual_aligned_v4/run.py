"""Matched dual-control learners on the restored chronological contract."""

import argparse
from dataclasses import asdict, replace
import json
import os
import random
import resource
import time

import numpy as np
import torch

from aligned_common import P, HERE, EXPERIMENTS, policy_folder, write
from chronological import Replay, collect, merge, Sampler
from dual_networks import bindings
from SimpleSAC.transit_v3.trainer import TransitH2OTrainer, TransitTrainerConfig


class Trainer(TransitH2OTrainer):
    def _ensemble_td_loss(self,predictions,targets,weights=None):
        observed = super()._ensemble_td_loss(predictions,targets,weights)
        model = getattr(self,"model_loss",None)
        return observed if model is None else (1-P["model_fraction"])*observed+P["model_fraction"]*model

    def update(self,*,online,online_only=False,collect_metrics=True):
        self.model_loss = None
        if online and getattr(self,"models",None) is not None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed+self.total_updates+60000)
                if self.total_updates%4==0:
                    self.models.update_trust(self.actor)
                batch,weights = self.models.generate(self.actor,128)
                self.model_loss = super()._ensemble_td_loss(self.critic(batch["observations"],batch["actions"]),
                    self._target(batch,backup="h2oplus"),weights)
                self.model_weight = float(weights.detach().mean())
        try:
            result = super().update(online=online,online_only=online_only,collect_metrics=collect_metrics)
            if "train/hybrid_gate_entropy" in result:
                result["train/gate_and_zero_speed_entropy"] = result.pop("train/hybrid_gate_entropy")
                result["train/weighted_positive_joint_entropy"] = result.pop("train/hybrid_weighted_positive_entropy")
            if collect_metrics and self.model_loss is not None:
                result.update(model_td_loss=float(self.model_loss.detach()),model_weight_mean=self.model_weight)
            return result
        finally:
            self.model_loss = None


def config(objective):
    value = json.loads((EXPERIMENTS / "mcwm_bus/prefix_config.json").read_text())
    value.update(reward_scale=P["reward_scales"][objective],target_entropy=-1.,use_ratio_correction=False)
    return TransitTrainerConfig(**value)


def initializer(args):
    return HERE / ("smoke_inputs" if args.smoke else "inputs") / args.objective / f"seed{args.seed}" / "initializer.pt"


def probe(learner,replay):
    s = replay.offline["observations"][:512]
    action = learner.act(s,deterministic=True)
    return dict(mean_hold_seconds=float((30*(action[:,0]+1)).mean()),exact_zero_fraction=float((action[:,0]==-1).mean()),
                mean_speed_ratio=float((1+.2*action[:,1]).mean()))


def initialize(args):
    path = initializer(args)
    if (path.parent / "ready.json").exists():
        return
    started = time.monotonic()
    replay = Replay(args.objective,args.smoke)
    learner = Trainer(replay,config=config(args.objective),seed=args.seed,**bindings())
    updates = 32 if args.smoke else P["offline_updates"]
    for step in range(updates):
        logged = (step+1)%1000==0 or step+1==updates
        stats = learner.update(online=False,collect_metrics=logged)
        if logged:
            row = dict(offline_updates=step+1,wall_seconds=time.monotonic()-started,metrics=stats,probe=probe(learner,replay))
            write(path.parent / "progress.json",row)
            print(json.dumps(row),flush=True)
    learner.save(path,extra=dict(objective=args.objective,seed=args.seed,version=P["name"]),include_online_replay=False)
    write(path.parent / "ready.json",dict(objective=args.objective,seed=args.seed,offline_updates=updates,
        offline_rows=replay.offline_size,config=asdict(learner.config),probe=probe(learner,replay),wall_seconds=time.monotonic()-started))


def train(args):
    from environments import ControlledSimEnv,install_source
    from station_common import P as STATION_P
    from dual_model import Models
    out = policy_folder(args.objective,args.scene,args.arm,args.seed,args.smoke)
    if (out / "ready.json").exists():
        return
    started = time.monotonic()
    replay = Replay(args.objective,args.smoke)
    learner = Trainer(replay,config=config(args.objective),seed=args.seed,**bindings())
    learner.seed = args.seed
    learner.load(initializer(args))
    learner.config = replace(learner.config,use_ratio_correction=args.arm=="h2oplus")
    scene = next(s for s in STATION_P["scenarios"] if s["name"]==args.scene)
    random.seed(args.seed+30000)
    np.random.seed(args.seed+30000)
    install_source(scene)
    env = ControlledSimEnv(os.environ["THESIS_SOURCE_ENV"],observation_schema="v7",demand_scale=scene["od_factor"],
        observation_speed_seed=args.seed+30000)
    if env.demand_scale>1.25:
        env._observation_speed_profiles_by_demand[env.demand_scale] = env._observation_speed_profiles_by_demand[1.25]
    sampler = Sampler(env,replay)
    learner.models = None
    budget,warmup = (512,256) if args.smoke else (P["source_events"],P["warmup_events"])
    refresh = [warmup] if args.smoke else P["model_refresh_events"]
    write(out / "protocol.json",dict(protocol=P,objective=args.objective,scene=scene,arm=args.arm,seed=args.seed,
        config=asdict(learner.config),offline_rows=replay.offline_size,target_online_training_events=0))
    events,updates,refreshed = 0,0,[]
    while events<budget:
        stop = min([x for x in refresh+[budget] if x>events]+[events+P["collection_interval"]])
        events += sampler.collect(learner.act,stop-events)
        if args.arm=="mcwm" and any(events>=x and x not in refreshed for x in refresh):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(args.seed+events)
                if learner.models is None:
                    learner.models = Models(replay,args.seed,out / "models",args.smoke)
                else:
                    learner.models.refit(events)
            refreshed = [x for x in refresh if x<=events]
        while updates<max(0,min(events,budget)-warmup):
            logical = updates+warmup
            for group in learner.actor_optimizer.param_groups:
                group["lr"] = learner.config.actor_learning_rate*(.25 if logical>=15000 else 1.)
            logged = (updates+1)%1000==0 or updates+1==budget-warmup
            stats = learner.update(online=True,collect_metrics=logged)
            updates += 1
            if logged:
                row = dict(source_events=events,updates=updates,metrics=stats,probe=probe(learner,replay),wall_seconds=time.monotonic()-started)
                write(out / "progress.json",row)
                print(json.dumps(row),flush=True)
    torch.save(dict(actor=learner.actor.state_dict(),objective=args.objective,arm=args.arm,seed=args.seed,
                    hidden_dim=P["hidden_dim"],state_dim=245,action_dim=2),out / "policy.pt")
    if learner.models is not None:
        learner.models.save()
    write(out / "ready.json",dict(objective=args.objective,arm=args.arm,seed=args.seed,source_scene=args.scene,
        source_events=events,updates=updates,settled_source_transitions=replay.online_size,probe=probe(learner,replay),
        training_wall_seconds=time.monotonic()-started,peak_host_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        target_online_training_events=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",required=True,choices=("collect","merge","initialize","train","evaluate"))
    parser.add_argument("--objective",choices=P["objectives"],default="passenger")
    parser.add_argument("--arm",choices=P["arms"]+["daganzo_duet","zero"],default="mcwm")
    parser.add_argument("--scene",choices=P["scenes"],default="calibrated")
    parser.add_argument("--seed",type=int,default=P["training_seeds"][0])
    parser.add_argument("--shard",type=int,default=0)
    parser.add_argument("--case",type=int,default=0)
    parser.add_argument("--layer",choices=P["executors"],default=P["primary_executor"])
    parser.add_argument("--smoke",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(P["threads"] if args.phase in ("initialize","train") else 1)
    if args.phase=="collect":
        collect(args.shard,args.smoke)
    elif args.phase=="merge":
        merge(args.smoke)
    elif args.phase=="evaluate":
        from evaluate_dual import evaluate
        evaluate(args)
    else:
        globals()[args.phase](args)
    print("DONE aligned dual bus",vars(args),flush=True)


if __name__ == "__main__":
    main()
