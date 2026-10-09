"""New same-vehicle headway learning; no changes to completed campaigns."""

import argparse
from dataclasses import replace
import importlib.util
import json
import sys
import torch

from headway_common import HERE,P,folder,init_folder,write
spec = importlib.util.spec_from_file_location("headway_original_training",HERE.parent / "bus_dual_aligned_v4/run.py")
original = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = original
spec.loader.exec_module(original)
from vehicle_data import Replay,Sampler,collect,merge
from reward_model import Models
import dual_model

OriginalTrainer,old_config = original.Trainer,original.config


class Trainer(OriginalTrainer):
    def load(self,*args,**kwargs):
        result = super().load(*args,**kwargs)
        self.config = replace(self.config,gradient_clip_norm=P["gradient_clip_norm"])
        return result


def run(args):
    if args.phase in ("collect","merge"):
        globals()[args.phase](args.shard,args.smoke) if args.phase=="collect" else merge(args.smoke)
        return
    original.Replay,original.Sampler,original.Trainer = Replay,Sampler,Trainer
    original.config = lambda objective: replace(old_config(objective),gradient_clip_norm=P["gradient_clip_norm"])
    original.HERE = HERE
    original.initializer = lambda a: init_folder(a.seed,a.smoke) / "initializer.pt"
    destination = folder(args.scene,args.arm,args.seed,args.smoke)
    original.policy_folder = lambda *a,**k: destination
    dual_model.Models = Models
    Models.raw = args.arm=="mc_raw"
    args.objective = "headway"
    if args.phase=="initialize":
        original.initialize(args)
    elif args.phase=="evaluate":
        from evaluate_headway import evaluate
        evaluate(args)
    else:
        arm = args.arm
        out = folder(args.scene,arm,args.seed,args.smoke)
        if arm.startswith("mc_"):
            args.arm = "mcwm"
        original.train(args)
        path = out / "ready.json"
        r = json.loads(path.read_text())
        r.update(arm=arm,experiment=P["name"],transition_contract="same-vehicle next station",
                 reward_estimator="raw" if Models.raw else "state_reconstruction",discount=P["discount"])
        write(path,r)
    print("DONE MCWM headway alignment",flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",choices=("collect","merge","initialize","train","evaluate"),required=True)
    parser.add_argument("--scene",choices=P["scenes"],default=P["scenes"][0])
    parser.add_argument("--arm",choices=P["arms"]+["daganzo_duet","zero"],default="mc_state")
    parser.add_argument("--seed",type=int,default=P["training_seeds"][0])
    parser.add_argument("--shard",type=int,default=0)
    parser.add_argument("--case",type=int,default=0)
    parser.add_argument("--layer",choices=P["executors"],default="guarded")
    parser.add_argument("--smoke",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args)
