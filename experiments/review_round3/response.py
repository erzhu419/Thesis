"""Conditional same-vehicle transition responses to fixed holding interventions."""

from review_common import HERE, EXPERIMENTS, P, module, write, distribution
import argparse
from collections import defaultdict
from pathlib import Path
import os
import random
import sys
import time
import numpy as np

# Both platforms use the already deployed v7 observation contract.
base = module("round3_response_eval", EXPERIMENTS / "sumo_eval/evaluate_control.py")
from bus_h2o.mdp_contract_v7 import RAW_EVENT_STATE_INDEX as I


def cases():
    return [dict(domain=domain, hold=hold, demand=demand, seed=seed)
            for domain in P["response"]["domains"] for hold in P["response"]["holds_seconds"]
            for demand in P["response"]["demands"] for seed in P["response"]["seeds"]]


class Responses:
    def __init__(self):
        self.pending, self.rows = {}, []
        self.decisions = defaultdict(int)

    def observe(self, key, raw, hold):
        def value(name):
            return float(raw[I[name]])
        groups = ["all"]
        low = value("onboard_passengers") <= .25*value("bus_capacity")
        missing = value("backward_bus_present") < .5
        late = value("sim_time_seconds") >= P["response"]["late_seconds"]
        if low:
            groups.append("low_load")
        if missing:
            groups.append("no_following_bus")
        if late:
            groups.append("late")
        if low and missing:
            groups.append("low_load_no_following_bus")
        for group in groups:
            self.decisions[group] += 1
        current = dict(time=value("sim_time_seconds"), station=value("station_index"),
                       onboard=value("onboard_passengers"), waiting=value("eligible_stranded_passengers"),
                       front=value("forward_headway_seconds"), front_present=value("forward_bus_present") > .5,
                       target=value("target_forward_headway_seconds"), groups=groups, hold=hold)
        previous = self.pending.get(key)
        if previous is not None and current["time"] > previous["time"] and current["station"] > previous["station"]:
            self.rows.append(dict(groups=previous["groups"], hold=previous["hold"],
                 next_event_seconds=current["time"]-previous["time"],
                 onboard_change=current["onboard"]-previous["onboard"], next_waiting=current["waiting"],
                 next_front_absolute_plan_error=abs(current["front"]-current["target"]) if current["front_present"] else None))
        self.pending[key] = current

    def summary(self):
        result = {}
        for group, count in self.decisions.items():
            rows = [r for r in self.rows if group in r["groups"]]
            result[group] = dict(decisions=count, completed_same_vehicle_transitions=len(rows),
                **{key: distribution([r[key] for r in rows if r[key] is not None]) for key in
                   ("next_event_seconds", "onboard_change", "next_waiting", "next_front_absolute_plan_error")})
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    case = cases()[args.case]
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(case["seed"])
    np.random.seed(case["seed"])
    horizon = 1200 if args.smoke else P["response"]["horizon_seconds"]
    response = Responses()
    start = time.monotonic()
    if case["domain"] == "sumo":
        cfg, objects = base.build_scenario(args.out, case["demand"], case["seed"], 120)
        env = base.AllLinesSumoEnv(str(base.SUMO_ROOT / "online_control"),
               str(base.ROOT / "bus_h2o/network_data/a_sorted_busline_edge.xml"),
               episode_seconds=horizon, sumo_seed=case["seed"]+1000, sumo_cfg=cfg,
               passenger_obj_xml=objects, observation_schema="v5", sumo_backend="libsumo")
        try:
            env.reset()
            env.step_to_event({})
            while not env.done:
                actions = {}
                for event in sorted(env._current_events, key=base.event_sort_key):
                    terminal = base.is_terminal_event(env, event)
                    hold = 0. if terminal else case["hold"]
                    if not terminal:
                        response.observe((event.line_id, event.bus_id), base.raw_event_observation(env, event.bus_id), hold)
                    actions[event.bus_id] = [hold, 1.]
                env.step_to_event(actions)
                env._bridge.consume_completed_trip_passenger_times()
            actual_horizon = env._bridge.current_time
        finally:
            env.close()
    else:
        from bus_h2o.envs.all_lines_sim_env import AllLinesSimEnv
        from SimpleSAC.transit_v7.sampler import _active_event_batch, _terminal_passenger_time
        env = AllLinesSimEnv(os.environ["THESIS_SOURCE_ENV"], observation_schema="v7",
                            demand_scale=case["demand"], observation_speed_seed=case["seed"], episode_seconds=horizon)
        env.reset()
        env.step_to_event({})
        while not env.done:
            actions = {}
            for key, raw in _active_event_batch(env.state):
                response.observe(key, raw, case["hold"])
                actions[key] = [case["hold"], 1.]
            env.step_to_event(actions)
        actual_horizon = _terminal_passenger_time(env).time_seconds
    if actual_horizon != horizon:
        raise RuntimeError(f"Incomplete response day: {actual_horizon} != {horizon}")
    write(args.out / "results/response.json", dict(case=case, horizon=horizon, groups=response.summary(),
          elapsed_seconds=time.monotonic()-start,
          estimand="Conditional same-vehicle next-decision response, not a common microscopic-state causal comparison"))
    write(args.out / "results/ready.json", dict(case=case, horizon=horizon))


if __name__ == "__main__":
    main()
    print("DONE review supplement", flush=True)
