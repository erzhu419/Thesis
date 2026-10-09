"""Original RE-SAC runner with an optional cooperative passenger-time reward."""

from dataclasses import asdict
import json
import time
import torch

from cross_common import HERE, EXPERIMENTS, P, module, write
from rewards import PassengerLedger, counts

old = module("cross_legacy", EXPERIMENTS / "enhancement_v1/legacy.py")
core = old.core
original_env = core.env_bus


class ObjectiveEnv(original_env):
    objective = "headway"

    def reset(self, *args, **kwargs):
        self.passenger_ledger = PassengerLedger()
        return super().reset(*args, **kwargs)

    def step(self, action, **kwargs):
        before = self.current_time
        waiting, onboard = counts([self])
        state, reward, done = super().step(action, **kwargs)
        self.passenger_ledger.advance(waiting, onboard, self.current_time-before)
        if self.objective == "passenger":
            reward = {key: self.passenger_ledger.settle(key) for key in reward}
        return state, reward, done


class MeasuredPolicy:
    def __init__(self, policy):
        self.policy, self.samples = policy, []

    def get_action(self, state, deterministic=True):
        self.samples.append((abs(float(state[4]-state[5])),
                             .5*(abs(float(state[4])-360)+abs(float(state[5])-360))))
        return self.policy.get_action(state, deterministic=deterministic)


def run(args):
    torch.set_num_threads(1)
    config = core.TrainConfig(seed=args.seed, max_episodes=1 if args.smoke else P["episodes"],
        role="specialist", fixed_mode="normal", checkpoint_interval=10,
        beta_ood=.01 if args.arm == "resac" else 0., beta=-2. if args.arm == "resac" else 0.)
    if args.evaluate:
        scene = P["legacy_scenes"][args.case // len(P["evaluation_seeds"])]
        seed = P["evaluation_seeds"][args.case % len(P["evaluation_seeds"])]
        env = core.make_environment(config)
        env.reset()
        learner = core.load_controller(args.out / "bundle/controller.pt", env)
        policy = MeasuredPolicy(learner.policy)
        row = old.evaluate(policy, args.arm, scene, seed)
        row.update(arm=args.arm, training_seed=args.seed, objective=args.objective, case=args.case,
            headway_imbalance_seconds=sum(s[0] for s in policy.samples)/len(policy.samples),
            headway_plan_mae_seconds=sum(s[1] for s in policy.samples)/len(policy.samples),
            completion_rate=row["arrived_passengers"]/row["generated_passengers"])
        write(args.out / "results" / f"eval_{args.case:02d}.json", row)
        print(json.dumps({k:v for k,v in row.items() if k != "windows"}), flush=True)
        return
    if (args.out / "results/ready.json").exists():
        return
    ObjectiveEnv.objective = args.objective
    core.env_bus = ObjectiveEnv
    write(args.out / "results/protocol.json", dict(protocol=P, objective=args.objective,
        requested_objective=args.requested_objective, arm=args.arm, config=asdict(config)))
    started = time.monotonic()
    core.train(config, args.out, args.out / "bundle", resume=False)
    row = dict(objective=args.objective, arm=args.arm, seed=args.seed,
        episodes=config.max_episodes, training_wall_seconds=time.monotonic()-started)
    write(args.out / "results/ready.json", row)
