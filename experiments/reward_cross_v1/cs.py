"""Keep the published multi-line learner; select its transition reward."""

from contextlib import redirect_stdout
from dataclasses import asdict
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import torch

from cross_common import HERE, EXPERIMENTS, P, module, write
from rewards import PassengerLedger, counts

old = module("cross_cs", EXPERIMENTS / "enhancement_v1/cs.py")
# Use the same resident, frozen learner and environment as the earlier campaign.
from scripts.test_multiline_convergence import run_episode_multiline


class ObjectiveEnv(old.ObservedEnv):
    objective = "headway"

    def reset(self):
        self.passenger_ledger = PassengerLedger()
        self.windows = {}
        return super().reset()

    def step(self, actions, **kwargs):
        before = self.current_time
        waiting, onboard = counts(list(self.line_map.values()))
        state, reward, done = super().step(actions, **kwargs)
        self.passenger_ledger.advance(waiting, onboard, self.current_time-before)
        if self.objective == "passenger":
            # The legacy reward dictionary retains stale keys on idle ticks.
            reward = {line: {bus: self.passenger_ledger.settle((line, bus))
                             for bus, observations in buses.items() if observations}
                      for line, buses in state.items()}
        return state, reward, done


class MeasuredPolicy(old.ObservedPolicy):
    def __init__(self, agent, env):
        super().__init__(agent, env)
        self.plan_deviation_sum = 0.

    def select_action(self, state, deterministic=False):
        self.plan_deviation_sum += .5*(abs(float(state[5]-state[8]))+abs(float(state[6]-state[8])))*600
        return super().select_action(state, deterministic=deterministic)


def run(args):
    torch.set_num_threads(1)
    old.seed_all(args.seed)
    config = old.configuration("nau_bc0" if args.arm == "csbapr" else "mlp_bc0", "cpu")
    config.max_episodes = 1 if args.smoke else P["episodes"]
    agent = old.CSBAPRAgent(15, 1, config)
    from csbapr.networks.critic import EnsembleQNet
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed+100000)
        critic = EnsembleQNet(15, 1, 64, 3)
    agent.critic.load_state_dict(critic.state_dict())
    agent.target_critic.load_state_dict(critic.state_dict())
    source = old.ROOT / "offline-sumo/env/calibrated_env"
    ObjectiveEnv.objective = args.objective
    with tempfile.TemporaryDirectory(prefix="thesis-reward-cross-") as directory:
        directory = Path(directory)
        shutil.copy2(source / "config.json", directory / "config.json")
        os.symlink(source / "data", directory / "data", target_is_directory=True)
        with redirect_stdout(StringIO()):
            env = ObjectiveEnv(str(directory))
        # Stop at the shared observation horizon before legacy cleanup erases metrics.
        for line in env.line_map.values():
            line.max_time = max(line.max_time, old.PROTOCOL["horizon_seconds"]+line.time_step)
        if args.evaluate:
            payload = torch.load(args.out / "policy.pt", map_location="cpu", weights_only=False)
            agent.actor.load_state_dict(payload["actor"])
            agent.actor.eval()
            scene = list(P["cs_scenes"])[args.case // len(P["evaluation_seeds"])]
            seed = P["evaluation_seeds"][args.case % len(P["evaluation_seeds"])]
            old.seed_all(seed)
            env.set_ood_schedule(P["cs_scenes"][scene])
            policy = MeasuredPolicy(agent, env)
            started = time.monotonic()
            with redirect_stdout(StringIO()):
                rewards, decisions, _ = run_episode_multiline(env, policy, deterministic=True, train=False)
            people = [p for line in env.line_map.values() for station in line.stations for p in station.total_passenger]
            count = len(people)
            action_count = sum(w["decisions"] for w in env.windows.values())
            row = dict(arm=args.arm, training_seed=args.seed, evaluation_seed=seed, scene=scene,
                objective=args.objective, case=args.case, reward=sum(rewards.values()), decisions=decisions,
                horizon_seconds=env.current_time, generated_passengers=count,
                arrived_passengers=sum(p.arrived for p in people),
                completion_rate=sum(p.arrived for p in people)/count,
                wait_passenger_seconds=env.passenger_ledger.wait,
                ride_passenger_seconds=env.passenger_ledger.ride,
                generalized_seconds_per_passenger=(2*env.passenger_ledger.wait+env.passenger_ledger.ride)/count,
                headway_imbalance_seconds=sum(w["imbalance_sum"] for w in env.windows.values())/action_count,
                headway_plan_mae_seconds=policy.plan_deviation_sum/action_count,
                mean_hold_seconds=sum(w["hold_sum"] for w in env.windows.values())/action_count,
                windows=env.windows, elapsed_seconds=time.monotonic()-started)
            write(args.out / "results" / f"eval_{args.case:02d}.json", row)
            print(json.dumps({k:v for k,v in row.items() if k != "windows"}), flush=True)
            return
        if (args.out / "results/ready.json").exists():
            return
        write(args.out / "results/protocol.json", dict(protocol=P, objective=args.objective,
            requested_objective=args.requested_objective, arm=args.arm, config=asdict(config)))
        history, started = [], time.monotonic()
        for episode in range(config.max_episodes):
            old.seed_all(args.seed*1000+episode)
            with redirect_stdout(StringIO()):
                rewards, decisions, updates = run_episode_multiline(env, agent, config=config)
            row = dict(episode=episode+1, reward=sum(rewards.values()), decisions=decisions, updates=updates,
                global_passenger_seconds=2*env.passenger_ledger.wait+env.passenger_ledger.ride,
                horizon_seconds=env.current_time, elapsed_seconds=time.monotonic()-started)
            history.append(row)
            write(args.out / "results/progress.json", row)
            print(json.dumps(row), flush=True)
        torch.save(dict(actor=agent.actor.state_dict(), config=asdict(config), objective=args.objective), args.out / "policy.pt")
        write(args.out / "results/train_summary.json", dict(history=history, objective=args.objective,
            training_wall_seconds=time.monotonic()-started))
        write(args.out / "results/ready.json", dict(arm=args.arm, seed=args.seed, objective=args.objective,
            episodes=config.max_episodes))
