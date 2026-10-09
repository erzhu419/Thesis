"""Additional holding and speed control without changing resident environments."""

import types

import numpy as np

from station_common import P
from bus_h2o.envs.all_lines_sim_env import AllLinesSimEnv
from sim_core.bus import Bus, LINE_SIGNAL_CONFIG


def controlled_travel_steps(runtime, overhead, speed, signal_config, segments):
    total = max(float(runtime) - overhead, 1.)
    expected_signal_delay = (signal_config["n_signals"] / max(segments, 1)
                             * signal_config["cycle"] * (1 - signal_config["green_frac"])**2 / 2)
    fixed = min(expected_signal_delay, total - 1.)
    return max(int(round((total - fixed) / speed + fixed)), 1)


def install_source(scene):
    import sim_core.sim as simulation
    import random

    class ControlledBus(Bus):
        def exchange_passengers(self, current_time, debug, station=None):
            super().exchange_passengers(current_time, debug, station)
            self.holding_time *= scene["dwell_factor"]
            self._base_stop_duration = self.holding_time

        def _advance_on_route(self):
            route = self.current_route
            if self._empirical_runtime_calibrated and route.runtime_calibration is not None:
                if self._calibrated_travel_steps_remaining is None:
                    self._segment_start_time = getattr(self, "_current_sim_time", self.launch_time)
                    runtime = route.sample_runtime_seconds(self._segment_start_time, demand_scale=1.)
                    signals = LINE_SIGNAL_CONFIG[self.line_id_str]
                    duration = controlled_travel_steps(runtime, route.runtime_fixed_overhead_seconds,
                        scene["speed_factor"] * getattr(self, "_speed_ratio", 1.), signals, self._n_route_segments)
                    extra = np.random.poisson(scene["extra_signal_fraction"] * signals["n_signals"] / max(self._n_route_segments, 1))
                    for _ in range(extra):
                        if random.random() > signals["green_frac"]:
                            duration += int(round(random.uniform(0, signals["cycle"] * (1-signals["green_frac"]))))
                    self._calibrated_travel_steps_remaining = max(duration, 1)
            super()._advance_on_route()

    simulation.Bus = ControlledBus


class ControlledSimEnv(AllLinesSimEnv):
    def _nested_actions(self, actions):
        nested = {line: {bus: None for bus in range(env.max_agent_num)} for line, env in self.line_map.items()}
        for (line, bus), value in actions.items():
            if value is not None:
                hold, speed = map(float, value)
                if not (0 <= hold <= 60 and .8 <= speed <= 1.2):
                    raise ValueError("invalid bus control command")
                nested[line][int(bus)] = [hold, speed]
        return nested


def enable_sumo_speed(env):
    env._ensure_bridge()
    bridge = env._bridge
    apply_hold = bridge.apply_action

    def apply_action(self, event, command):
        hold, speed = map(float, command)
        if not (0 <= hold <= 60 and .8 <= speed <= 1.2):
            raise ValueError("invalid bus control command")
        apply_hold(event, [hold, 1.])
        from sumo_env.rl_bridge import traci
        traci.vehicle.setSpeedFactor(event.bus_id, speed)
        event.metadata["executed_speed_ratio"] = speed

    bridge.apply_action = types.MethodType(apply_action, bridge)


class StationSampler:
    def __init__(self, env, replay, mode, stations):
        self.env, self.replay, self.mode, self.stations = env, replay, mode, stations
        self.pending, self.last = {}, {}
        self.active, self.episodes = False, 0

    def collect(self, policy, count):
        from station_common import neutral_action, observation, physical_action, state_headway_reward
        from bus_h2o.passenger_time import aggregate_passenger_time_snapshots
        events = 0
        while events < count:
            if not self.active:
                self.pending.clear()
                self.last.clear()
                self.env.reset()
                self.env.step_to_event({})
                self.episodes += 1
                self.active = True
            commands = {}
            for key, history in sorted(self.env.state.items()):
                if not history:
                    continue
                raw = np.asarray(history[-1], np.float32).reshape(-1)
                line = self.env.line_map[key[0]]
                bus = next(b for b in line.bus_all if b.bus_id == key[1])
                identity = (*key, int(bus.trip_id))
                snapshot = aggregate_passenger_time_snapshots(x.passenger_time_ledger.snapshot() for x in self.env.line_map.values())
                totals = (snapshot.accrued_wait_seconds, snapshot.accrued_in_vehicle_seconds)
                state = observation(raw, self.last.get(identity, neutral_action(self.mode)), self.stations)
                if identity in self.pending:
                    previous = self.pending.pop(identity)
                    if raw[1] > previous["station"]:
                        welfare = -(2 * (totals[0]-previous["totals"][0]) + totals[1]-previous["totals"][1])
                        self.replay.append(previous["state"], previous["action"], state,
                            float(state_headway_reward(state[None])[0]), welfare, raw[2]-previous["time"])
                action = policy(state[None], deterministic=False)[0]
                self.last[identity] = action.copy()
                self.pending[identity] = dict(state=state, action=action.copy(), totals=totals,
                    station=float(raw[1]), time=float(raw[2]))
                commands[key] = physical_action(action)
                events += 1
            self.env.step_to_event(commands)
            if self.env.done:
                self.active = False
        return events
