"""Original RE-SAC update and reward; add one native-scale speed coordinate."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from actions import controlled_bus_class, physical_action
from common import EXPERIMENTS, P, folder, module, write

old = module("dual_legacy_reward", EXPERIMENTS / "reward_cross_v1/legacy.py")
core = old.core
from env import sim
from gym.spaces import Box

sim.Bus = controlled_bus_class(sim.Bus)
BaseEnv = old.ObjectiveEnv


class DualEnv(BaseEnv):
    mode = "dual"
    smoke = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = Box(0., 60., shape=(2 if self.mode == "dual" else 1,))

    def step(self, action, **kwargs):
        state, reward, done = super().step(action, **kwargs)
        return state, reward, done or (self.smoke and self.current_time >= 3600)


class MeasuredPolicy(old.MeasuredPolicy):
    def __init__(self, policy):
        super().__init__(policy)
        self.commands = []

    def get_action(self, state, deterministic=True):
        action = super().get_action(state, deterministic)
        self.commands.append(physical_action(action, "resac"))
        return action


def run(args):
    DualEnv.mode, DualEnv.smoke, DualEnv.objective = args.mode, args.smoke, args.objective
    old.ObjectiveEnv = DualEnv
    core.env_bus = DualEnv
    old.P = dict(old.P, episodes=P["episodes"])
    out = folder(args.kind, args.mode, args.objective, args.arm, args.seed, args.smoke)
    if args.phase == "train":
        payload = SimpleNamespace(**vars(args), out=out, evaluate=False, requested_objective=args.objective)
        old.run(payload)
        record = json.loads((out / "results/ready.json").read_text())
        record.update(kind=args.kind, mode=args.mode, action_dim=2 if args.mode == "dual" else 1,
                      policy_action_units=P["resac_action"])
        write(out / "results/ready.json", record)
        return
    path = out / "results" / f"eval_{args.case:02d}.json"
    if path.exists():
        return
    scene = P["legacy_scenes"][args.case//3]
    eval_seed = P["evaluation_seeds"][args.kind][args.case % 3]
    if args.arm == "fixed_speed":
        dimension = 2 if args.mode == "dual" else 1
        controller = SimpleNamespace(get_action=lambda *values, **kw: np.array([0., 60.][:dimension], np.float32))
    else:
        config = core.TrainConfig(seed=args.seed, max_episodes=P["episodes"], role="specialist", fixed_mode="normal")
        env = core.make_environment(config)
        env.reset()
        controller = core.load_controller(out / "bundle/controller.pt", env).policy
    measured = MeasuredPolicy(controller)
    evaluate_module = __import__("evaluate_pair")
    evaluate_module.HORIZON = 3600 if args.smoke else P["horizon_seconds"][args.kind]
    row = old.old.evaluate(measured, args.arm, scene, eval_seed)
    commands = np.asarray(measured.commands)
    row.update(kind=args.kind, mode=args.mode, objective=args.objective, arm=args.arm,
        training_seed=args.seed, evaluation_seed=eval_seed, case=args.case,
        decisions=len(measured.samples),
        headway_imbalance_seconds=float(np.mean([x[0] for x in measured.samples])),
        headway_plan_mae_seconds=float(np.mean([x[1] for x in measured.samples])),
        completion_rate=row["arrived_passengers"]/row["generated_passengers"],
        mean_hold_seconds=float(commands[:, 0].mean()), mean_speed_ratio=float(commands[:, 1].mean()),
        action_dim=2 if args.mode == "dual" else 1)
    row.pop("windows", None)
    write(path, row)
