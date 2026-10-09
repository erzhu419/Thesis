"""Repeated ensemble sizes and operational evaluation of frozen reward policies."""

from review_common import HERE, EXPERIMENTS, P, write, distribution
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time

sys.path.insert(0, str(EXPERIMENTS / "legacy_bus"))
import evaluate_pair as pair
from train_representation import core
import numpy as np
import torch


class OperationalEnv(pair.ObservedEnv):
    latest = None

    def __init__(self, scene):
        super().__init__(scene)
        OperationalEnv.latest = self
        self.departures, self.completed_trips = [], []
        self.trip_starts, self.previous_finishes = {}, {}
        self.turnaround_seconds = []

    def launch_bus(self, trip):
        super().launch_bus(trip)
        bus = next(b for b in self.bus_all if b.on_route and b.trip_id == trip.launch_turn)
        self.trip_starts[(bus.bus_id, bus.trip_id)] = float(self.current_time)
        self.departures.append(float(self.current_time-trip.launch_time))
        if bus.bus_id in self.previous_finishes:
            self.turnaround_seconds.append(float(self.current_time-self.previous_finishes[bus.bus_id]))

    def step(self, action, **kwargs):
        active = [(b, b.trip_id) for b in self.bus_all if b.on_route]
        result = super().step(action, **kwargs)
        for bus, trip in active:
            if not bus.on_route and bus.back_to_terminal_time is not None:
                start = self.trip_starts[(bus.bus_id, trip)]
                self.completed_trips.append(float(bus.back_to_terminal_time-start))
                self.previous_finishes[bus.bus_id] = float(bus.back_to_terminal_time)
        return result


class MeasuredPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.front, self.back, self.latency = [], [], []

    def get_action(self, state, deterministic=True):
        env = OperationalEnv.latest
        bus = next(b for b in env.bus_all if b.bus_id == int(state[0]))
        if bus.forward_bus:
            self.front.append(float(state[4])-360.)
        if bus.backward_bus:
            self.back.append(float(state[5])-360.)
        started = time.perf_counter()
        result = self.policy.get_action(state, deterministic=deterministic)
        self.latency.append(1000*(time.perf_counter()-started))
        return result


def evaluate(controller, arm, scene, seed):
    env = core.make_environment(core.TrainConfig(seed=seed, max_episodes=100,
                                                role="specialist", fixed_mode="normal"))
    env.reset()
    trainer = core.load_controller(controller, env)
    policy = MeasuredPolicy(trainer.policy)
    pair.ObservedEnv = OperationalEnv
    row = pair.evaluate(policy, arm, scene, seed)
    observed = OperationalEnv.latest
    row["operational"] = dict(
        front_plan_error_seconds=distribution(policy.front),
        back_plan_error_seconds=distribution(policy.back),
        front_plan_absolute_error_seconds=distribution(np.abs(policy.front)),
        back_plan_absolute_error_seconds=distribution(np.abs(policy.back)),
        departure_lateness_seconds=distribution(observed.departures),
        completed_trip_duration_seconds=distribution(observed.completed_trips),
        terminal_turnaround_seconds=distribution(observed.turnaround_seconds),
        policy_inference_ms=distribution(policy.latency),
        launched_trips=len(observed.departures), completed_trips=len(observed.completed_trips),
        observations="Headway errors exclude missing-neighbour defaults. Departure lateness is relative to the input timetable; the simulator launches at its first eligible step. No stop-arrival timetable exists. Trip duration is not round-trip duration.")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--size", type=int, choices=P["ensemble"]["sizes"], default=10)
    parser.add_argument("--arm", default="ensemble")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--controller", type=Path)
    parser.add_argument("--case", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.smoke and "smoke" not in args.out.parts:
        parser.error("Use a separate smoke directory")
    if args.case is not None:
        p = P["ensemble" if args.arm == "ensemble" else "reward"]
        seed = p["evaluation_seeds"][args.case % len(p["evaluation_seeds"])]
        scene = p["scenes"][args.case // len(p["evaluation_seeds"])]
        if args.smoke:
            pair.HORIZON = 1200
        row = evaluate(args.controller or args.out / "bundle/controller.pt", args.arm, scene, seed)
        row.update(training_seed=args.seed, ensemble_size=args.size)
        write(args.out / "results" / f"eval{args.case:02d}.json", row)
        return
    config = core.TrainConfig(seed=args.seed, max_episodes=1 if args.smoke else P["ensemble"]["episodes"],
                              role="specialist", fixed_mode="normal", ensemble_size=args.size)
    write(args.out / "results/protocol.json", asdict(config))
    if not (args.out / "results/ready.json").exists():
        start = time.monotonic()
        core.train(config, args.out, args.out / "bundle", resume=False)
        row = dict(training_wall_seconds=time.monotonic()-start, ensemble_size=args.size,
                   seed=args.seed, episodes=config.max_episodes,
                   device="cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            row.update(peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                       peak_gpu_reserved_mb=torch.cuda.max_memory_reserved()/2**20)
        write(args.out / "results/ready.json", row)


if __name__ == "__main__":
    main()
    print("DONE review supplement", flush=True)
