"""Reuse the dual evaluator, recording both hard and historical rewards."""

import json
import numpy as np

from headway_common import HERE,P,folder,state_reward,write
import evaluate_dual as old


def evaluate(args):
    arm = args.arm
    out = folder(args.scene,arm,args.seed,args.smoke) if arm in P["arms"] else HERE / ("rule_smoke" if args.smoke else "rules") / arm
    old.policy_folder = lambda *a,**k: out
    old.HERE = HERE
    if arm in P["arms"]:
        args.arm = "mcwm" if arm.startswith("mc_") else arm
    rewards = []
    original_reward = old.ridge_of_state

    def record(states):
        rewards.extend(state_reward(states).tolist())
        return original_reward(states)

    old.ridge_of_state = record
    try:
        old.evaluate(args)
    finally:
        old.ridge_of_state = original_reward
    path = out / args.layer / f"case{args.case:02d}.json"
    r = json.loads(path.read_text())
    r.update(arm=arm,experiment=P["name"],headway_metric="Hard ridge unchanged; historical tanh metric additionally reported")
    if rewards:
        r["headway_historical_mean"] = float(np.mean(rewards))
    write(path,r)
