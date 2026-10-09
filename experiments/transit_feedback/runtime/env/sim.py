import json
import time
import numpy as np
import pandas as pd
import copy
import os
import sys
import pygame

# Ensure the project root (parent of this env package) is on sys.path so that
# `import env.*` works even when running env/sim.py directly.
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.timetable import Timetable
from env.bus import Bus
from env.route import Route
from env.station import Station
from env.visualize import visualize
from gym.spaces.box import Box
from gym.spaces import MultiDiscrete


class env_bus(object):

    def __init__(self, path, debug=False, render=False, route_sigma=1.5):
        self.path = path
        self.route_sigma = route_sigma
        sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'r') as f:
            args = json.load(f)
        self.args = args
        self.effective_trip_num = 264
        
        self.time_step = args["time_step"]
        self.passenger_update_freq = args["passenger_state_update_freq"]
        # read data, multi-index used here
        self.od = pd.read_excel(os.path.join(path, "data/passenger_OD.xlsx"), index_col=[1, 0])
        self.station_set = pd.read_excel(os.path.join(path, "data/stop_news.xlsx"))
        self.routes_set = pd.read_excel(os.path.join(path, "data/route_news.xlsx"))
        # Ensure hourly columns use datetime.time objects so downstream lookups work
        time_cols = self.routes_set.columns[5:]
        rename_map = {}
        for col in time_cols:
            if isinstance(col, str):
                try:
                    rename_map[col] = pd.to_datetime(col).time()
                except ValueError:
                    continue
        if rename_map:
            self.routes_set = self.routes_set.rename(columns=rename_map)
        self.timetable_set = pd.read_excel(os.path.join(path, "data/time_table.xlsx"))
        # Truncate the original timetable by first 50 trips to reduce the calculation pressure
        self.timetable_set = self.timetable_set.sort_values(by=['launch_time', 'direction'])[:self.effective_trip_num].reset_index(drop=True)
        # add index for timetable
        self.timetable_set['launch_turn'] = range(self.timetable_set.shape[0])
        self.max_agent_num = 25

        self.visualizer = visualize(self)
        # Allow disabling automatic plotting when simulation ends
        self.enable_plot = True
        self.dispatch_mode = 'legacy'
        self.timetable_base_headway = 360.0
        self.timetable_min_headway = 180.0
        self.timetable_max_headway = 600.0
        self.timetable_min_notice = 60.0
        self.timetable_max_shift_abs = None
        self.use_lower_context = False
        self.lower_context_dim = 0

        # Set effective station and time period
        self.effective_station_name = sorted(set([self.od.index[i][0] for i in range(self.od.shape[0])]))
        self.effective_period = sorted(list(set([self.od.index[i][1] for i in range(self.od.shape[0])])))

        self.action_space = Box(0, 60, shape=(1,))

        if debug:
            self.summary_data = pd.DataFrame(columns=['bus_id', 'station_id', 'trip_id', 'abs_dis', 'forward_headway',
                                                  'backward_headway', 'headway_diff', 'time'])
            self.summary_reward = pd.DataFrame(columns=['bus_id', 'station_id', 'trip_id', 'forward_headway',
                                                    'backward_headway', 'reward', 'time'])

        self.stations = self.set_stations()
        self.routes = self.set_routes()
        self.timetables = self.set_timetables()

        self.state_dim = 8 + self.lower_context_dim + len(self.routes)//2  # +1 for headway_dev

        # TransitDuet: upper policy callback, cost tracking
        self._upper_policy_callback = None  # Set by runner
        self._peak_concurrent = 0

    def _ensure_agent_slot(self, bus_id):
        if not hasattr(self, 'state'):
            return
        self.state.setdefault(bus_id, [])
        self.reward.setdefault(bus_id, 0)
        self.cost.setdefault(bus_id, 0.0)
        if hasattr(self, 'action_dict'):
            self.action_dict.setdefault(bus_id, 0.0)

    def configure_control(self, dispatch_mode='legacy', lower_context=False,
                          timetable=None):
        """Configure optional TransitDuet control features before training."""
        self.dispatch_mode = dispatch_mode
        self.use_lower_context = bool(lower_context)
        self.lower_context_dim = 5 if self.use_lower_context else 0
        self.state_dim = 8 + self.lower_context_dim + len(self.routes)//2
        timetable = timetable or {}
        self.timetable_base_headway = float(timetable.get('base_headway', 360.0))
        self.timetable_min_headway = float(timetable.get('min_headway', 180.0))
        self.timetable_max_headway = float(timetable.get('max_headway', 600.0))
        self.timetable_min_notice = float(timetable.get('min_notice', 60.0))
        max_shift_abs = timetable.get('max_shift_abs', None)
        self.timetable_max_shift_abs = (
            None if max_shift_abs is None else float(max_shift_abs))

    @property
    def bus_in_terminal(self):
        return [bus for bus in self.bus_all if not bus.on_route]

    # @property
    # def bus_on_route(self):
    #     return [bus for bus in self.bus_all if bus.on_route]

    def set_timetables(self):
        return [Timetable(self.timetable_set['launch_time'][i], self.timetable_set['launch_turn'][i], self.timetable_set['direction'][i]) for i in range(self.timetable_set.shape[0])]

    def set_routes(self):
        return [
            Route(
                self.routes_set['route_id'][i],
                self.routes_set['start_stop'][i],
                self.routes_set['end_stop'][i],
                self.routes_set['distance'][i],
                self.routes_set['V_max'][i],
                self.routes_set.iloc[i, 5:],
                sigma=self.route_sigma
            )
            for i in range(self.routes_set.shape[0])
        ]

    def set_stations(self):
        station_concat = pd.concat([self.station_set, self.station_set[::-1][1:]]).reset_index()
        total_station = []
        for idx, station in station_concat.iterrows():
            # station type is 0 if Terminal else 1
            station_type = 1 if station['stop_name'] not in ['Terminal_up', 'Terminal_down'] else 0

            direction = False if idx >= station_concat.shape[0] / 2 else True
            od = None
            if station['stop_name'] in self.effective_station_name:
                od = self.od.loc[station['stop_name'], station['stop_name']:] if direction else self.od.loc[station['stop_name'], :station['stop_name']]
                # To reduce the OD value in False direction stations in ['X13','X14','X15'] because too many passengers stuck cause the overwhelming
                if station['stop_name'] in ['X13','X14','X15'] and not direction:
                    od *= 0.4

                od.index = od.index.map(str)
                od = od.to_dict(orient='index')

            total_station.append(Station(station_type, station['stop_id'], station['stop_name'], direction, od))

        return total_station

    # return default state and reward
    def reset(self):

        self.current_time = 0

        # initialize station, routes and timetables
        self.stations = self.set_stations()
        self.routes = self.set_routes()
        self.timetables = self.set_timetables()

        # Episode-level demand stochasticity:
        # - Per-hour multiplier: uniform(0.7, 1.3) → demand intensity varies
        # - Peak hour shift: ±30min (0 or 1 hour granularity in OD)
        # Stored on env so station_update can use it
        demand_noise = getattr(self, 'demand_noise', 0.0)
        if demand_noise > 0:
            # Per-hour demand multipliers (14 hours: 6:00-19:00)
            self._demand_multipliers = {
                h: np.clip(np.random.normal(1.0, demand_noise), 0.3, 2.0)
                for h in range(6, 20)
            }
            # Random peak shift: shift peak demand pattern by ±1 hour
            self._peak_shift = np.random.choice([-1, 0, 0, 0, 1])
        else:
            self._demand_multipliers = None
            self._peak_shift = 0

        # initial list of bus on route
        self.bus_id = 0
        self.bus_all = []
        self.route_state = []

        # self.state is combine with route_state, which contains the route.speed_limit of each route, station_state, which
        # contains the station.waiting_passengers of each station and bus_state, which is bus.obs for each bus.
        self.state = {key: [] for key in range(self.max_agent_num)}
        self.reward = {key: 0 for key in range(self.max_agent_num)}
        self.cost = {key: 0.0 for key in range(self.max_agent_num)}
        self.done = False
        self._peak_concurrent = 0
        self._cached_measurement = None
        self._cached_episode_metrics = None
        self._dispatch_rewards = {}
        self._last_dispatch_trip = {}
        # Track last dispatch time per direction for headway enforcement
        self._last_dispatch_time = {True: -9999, False: -9999}  # direction → time
        self._init_dynamic_timetable()

        self.action_dict = {key: None for key in list(range(self.max_agent_num))}

    def _init_dynamic_timetable(self):
        for tt in self.timetables:
            tt.baseline_launch_time = float(getattr(tt, 'baseline_launch_time', tt.launch_time))
            tt.planned_launch_time = float(tt.baseline_launch_time)
            tt.actual_launch_time = None
            tt._actual_launch_time = None
            tt.planned_headway = self.timetable_base_headway
            tt.target_headway = self.timetable_base_headway
            tt.dispatch_lateness = 0.0
            tt.planned_shift = 0.0
            tt._upper_queried = False
            tt.launched = False
        if self.dispatch_mode != 'timetable':
            return
        first_by_dir = {}
        for tt in sorted(self.timetables, key=lambda x: (x.baseline_launch_time, x.direction)):
            if tt.direction not in first_by_dir:
                first_by_dir[tt.direction] = tt
        for tt in self.timetables:
            if first_by_dir.get(tt.direction) is tt:
                tt.planned_launch_time = float(tt.baseline_launch_time)
            else:
                tt.planned_launch_time = None

    def initialize_state(self, render=False):
        def count_non_empty_sublist(lst):
            return sum(1 for sublist in lst if sublist)

        while count_non_empty_sublist(list(self.state.values())) == 0:
            self.state, self.reward, self.cost, _ = self.step(self.action_dict, render=render)

        return self.state, self.reward, self.done

    def launch_bus(self, trip):
        # Trip set(self.timetable) contain both direction trips. So we have to make sure the direction and launch time
        # is satisfied before the trip launched.
        # If there is no more appropriate bus in terminal, create a new bus, then add it to all_bus list.
        if len(list(filter(lambda i: i.direction == trip.direction, self.bus_in_terminal))) == 0:
            # cause bus.next_station， current_route and effective station & routes is defined by @property, so no initialize here
            launch_time = getattr(trip, 'actual_launch_time', None)
            if launch_time is None:
                launch_time = getattr(trip, '_actual_launch_time', trip.launch_time)
            bus = Bus(self.bus_id, trip.launch_turn, launch_time, trip.direction, self.routes, self.stations)
            self.bus_all.append(bus)
            self._ensure_agent_slot(bus.bus_id)
            self.bus_id += 1
        else:
            # if there is bus in terminal and also the direction is satisfied, then we reuse the bus to relaunch one of
            # them, which has the earliest arrived time to terminal.
            bus = sorted(list(filter(lambda i: i.direction == trip.direction, self.bus_in_terminal)), key=lambda bus: bus.back_to_terminal_time)[0]
            launch_time = getattr(trip, 'actual_launch_time', None)
            if launch_time is None:
                launch_time = getattr(trip, '_actual_launch_time', trip.launch_time)
            bus.reset_bus(trip.launch_turn, launch_time)
            # in drive() function, we set bus.on_route = False when it finished a trip. Here we set it to True because
            # the iteration in drive(), we just update the state of those bus which on routes
            bus.on_route = True
            self._ensure_agent_slot(bus.bus_id)

    def step(self, action, debug=False, render=False, episode = 0):
        if self.dispatch_mode == 'timetable':
            self._dispatch_timetable_trips()
        else:
            self._dispatch_legacy_trips()
        # route
        route_state = []
        # update route speed limit by freq
        if self.current_time % self.args['route_state_update_freq'] == 0:
            for route in self.routes:
                route.route_update(self.current_time, self.effective_period)
                route_state.append(route.speed_limit)
            self.route_state = route_state
        # update waiting passengers of every station every second
        # station_state = []
        if self.current_time % self.passenger_update_freq == 0:
            for station in self.stations:
                station.station_update(self.current_time, self.stations, self.passenger_update_freq,
                                       demand_multipliers=self._demand_multipliers,
                                       peak_shift=self._peak_shift)
            # station_state.append(len(station.waiting_passengers))
        # update bus state
        for bus in self.bus_all:
            if bus.on_route:
                self._ensure_agent_slot(bus.bus_id)
                bus.reward = None
                bus.obs = []
                bus.cost = None
                target_hw = self._get_target_headway_for_bus(bus)
                lower_context = self._build_lower_context_for_bus(bus)
                bus.drive(self.current_time, action.get(bus.bus_id, 0.0), self.bus_all,
                          debug=debug, target_headway=target_hw,
                          lower_context=lower_context)

        self.state_bus_list = state_bus_list = list(filter(lambda x: len(x.obs) != 0, self.bus_all))
        self.reward_list = reward_list = list(filter(lambda x: x.reward is not None, self.bus_all))

        if len(state_bus_list) != 0:
            # state_bus_list = sorted(state_bus_list, key=lambda x: x.bus_id)
            for i in range(len(state_bus_list)):
                self._ensure_agent_slot(state_bus_list[i].bus_id)
                # print('return state is ', state_bus_list[i].obs, ' for bus: ', state_bus_list[i].bus_id, 'at time:', self.current_time)
                # if len(self.state[state_bus_list[i].bus_id]) < 2:
                self.state[state_bus_list[i].bus_id].append(state_bus_list[i].obs)
                # if state_bus_list[i].last_station.station_id not in [0,1,21,22]:
                #     print(1)
                # else:
                #     self.state[state_bus_list[i].bus_id][0] = self.state[state_bus_list[i].bus_id][1]
                #     self.state[state_bus_list[i].bus_id][1] = state_bus_list[i].obs
                # if state_bus_list[i].bus_id == 0:
                #     print(state_bus_list[i].obs[-1], 'bus_id: ', state_bus_list[i].obs[0], ', station_id: ', state_bus_list[i].obs[1], ', trip_id: ', state_bus_list[i].obs[2])
                #     print('return state is ', state_bus_list[i].obs, ' for bus: ', state_bus_list[i].bus_id,
                #           'at time: ', self.current_time)
                # if len(self.state[state_bus_list[i].bus_id]) > 2:
                #     print(1)
                # if debug:
                #     new_data = [state_bus_list[i].obs[0], state_bus_list[i].obs[1], state_bus_list[i].obs[2],
                #                 state_bus_list[i].obs[4]*1000, state_bus_list[i].obs[6] * 60, state_bus_list[i].obs[7]*60,
                #                 state_bus_list[i].obs[6] * 60 - state_bus_list[i].obs[7] * 60, self.current_time]
                #     self.summary_data.loc[len(self.summary_data)] = new_data
        if len(reward_list) != 0:
            # reward_list = sorted(reward_list, key=lambda x: x.bus_id)
            for i in range(len(reward_list)):
                self._ensure_agent_slot(reward_list[i].bus_id)
                # if reward_list[i].bus_id == 0:
                #     print('return reward is: ', reward_list[i].reward, ' for bus: ', reward_list[i].bus_id, ' at time:', self.current_time)
                # if (reward_list[i].last_station.station_id != 22 and reward_list[i].direction != 0) and \
                #         (reward_list[i].last_station.station_id != 1 and reward_list[i].direction != 1):
                # if len(self.reward[reward_list[i].bus_id]) > 1:
                #     print(2)
                self.reward[reward_list[i].bus_id] = reward_list[i].reward

        # TransitDuet: collect cost
        self.cost_list = [bus for bus in self.bus_all if bus.cost is not None]
        for bus in self.cost_list:
            self._ensure_agent_slot(bus.bus_id)
            self.cost[bus.bus_id] = bus.cost

        # Track peak concurrent vehicles for measurement_vector
        concurrent = sum(1 for bus in self.bus_all if bus.on_route)
        self._peak_concurrent = max(self._peak_concurrent, concurrent)


        self.current_time += self.time_step
        if sum([trip.launched for trip in self.timetables]) == len(self.timetables) and sum([bus.on_route for bus in self.bus_all]) == 0:
            self.done = True
            # Cache measurement_vector BEFORE clearing data
            self._cached_measurement = self._compute_measurement_vector()
            self._cached_episode_metrics = self._compute_episode_metrics()
            if not debug:
                for bus in self.bus_all:
                    bus.trajectory.clear()
                    bus.trajectory_dict.clear()
                    del bus.trajectory
                    del bus.trajectory_dict
                for station in self.stations:
                    station.waiting_passengers = np.array([])
                    station.total_passenger.clear()
        else:
            self.done = False

        if self.done and debug:
            self.summary_data = self.summary_data.sort_values(['bus_id', 'time'])

            output_dir = os.path.join(self.path, 'pic')
            os.makedirs(output_dir, exist_ok=True)
            if self.enable_plot:
                self.visualizer.plot(episode)

            self.summary_data.to_csv(os.path.join(output_dir, 'summary_data.csv'))
            self.summary_reward = self.summary_reward.sort_values(['bus_id', 'time'])
            self.summary_reward.to_csv(os.path.join(self.path, 'pic', 'summary_reward.csv'))

        if render and self.current_time % 1 == 0:
            self.visualizer.render()
            time.sleep(0.05)  # Add a delay to slow down the rendering

        return self.state, self.reward, self.cost, self.done

    # ---- TransitDuet helper methods ----

    def _can_launch_trip(self):
        n_fleet = getattr(self, '_n_fleet_target', 25)
        buffer = getattr(self, '_fleet_buffer', 3)
        concurrent = sum(1 for bus in self.bus_all if bus.on_route)
        return concurrent < n_fleet + buffer

    def _mark_trip_launched(self, trip):
        trip.launched = True
        trip.actual_launch_time = float(self.current_time)
        trip._actual_launch_time = float(self.current_time)
        planned = getattr(trip, 'planned_launch_time', trip.baseline_launch_time)
        if planned is None:
            planned = trip.baseline_launch_time
        trip.dispatch_lateness = max(0.0, float(self.current_time) - float(planned))
        trip.planned_shift = float(planned) - float(trip.baseline_launch_time)
        self.launch_bus(trip)
        self._last_dispatch_time[trip.direction] = float(self.current_time)

    def _dispatch_legacy_trips(self):
        # Enumerate trips in timetables, if current_time >= launch_time then launch.
        for trip in self.timetables:
            if trip.launch_time <= self.current_time and not trip.launched:
                if self._upper_policy_callback is not None:
                    # Call upper policy ONCE per trip (when it first becomes eligible)
                    if not hasattr(trip, '_upper_queried') or not trip._upper_queried:
                        s_upper = self._build_upper_state(trip)
                        result = self._upper_policy_callback(s_upper, trip)
                        trip.target_headway = float(result)
                        trip.planned_headway = float(result)
                        trip._upper_queried = True
                        self._compute_dispatch_proxy_reward(trip)

                    # v2g mode: if trip has _delta_t, use direct time offset
                    if hasattr(trip, '_delta_t'):
                        effective_launch = trip._original_launch + trip._delta_t
                        if self.current_time < effective_launch:
                            continue
                    else:
                        # v1 mode: headway enforcement
                        actual_gap = self.current_time - self._last_dispatch_time[trip.direction]
                        if actual_gap < trip.target_headway:
                            continue

                if not self._can_launch_trip():
                    continue
                self._mark_trip_launched(trip)

    def _dispatch_timetable_trips(self):
        for trip in self.timetables:
            if trip.launched:
                continue
            planned = getattr(trip, 'planned_launch_time', None)
            if planned is None or self.current_time < planned:
                continue
            if not self._can_launch_trip():
                continue
            self._mark_trip_launched(trip)
            self._compute_dispatch_proxy_reward(trip)
            self._plan_next_same_direction_trip(trip)

    def _next_same_direction_trip(self, trip):
        candidates = [
            tt for tt in self.timetables
            if (not tt.launched
                and tt.direction == trip.direction
                and tt.launch_turn != trip.launch_turn
                and tt.baseline_launch_time > trip.baseline_launch_time)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda x: x.baseline_launch_time)

    def _plan_next_same_direction_trip(self, current_trip):
        next_trip = self._next_same_direction_trip(current_trip)
        if next_trip is None:
            return
        if getattr(next_trip, 'planned_launch_time', None) is not None:
            return

        if self._upper_policy_callback is not None:
            s_upper = self._build_upper_state(next_trip)
            planned_headway = float(self._upper_policy_callback(s_upper, next_trip))
            next_trip._upper_queried = True
        else:
            planned_headway = self.timetable_base_headway

        planned_headway = float(np.clip(
            planned_headway, self.timetable_min_headway, self.timetable_max_headway))
        last_actual = float(getattr(current_trip, 'actual_launch_time', self.current_time))
        earliest_feasible = max(
            last_actual + self.timetable_min_headway,
            float(self.current_time) + self.timetable_min_notice)
        planned_launch = max(last_actual + planned_headway, earliest_feasible)

        if self.timetable_max_shift_abs is not None:
            baseline = float(next_trip.baseline_launch_time)
            planned_launch = float(np.clip(
                planned_launch,
                baseline - self.timetable_max_shift_abs,
                baseline + self.timetable_max_shift_abs))
            planned_launch = max(planned_launch, earliest_feasible)

        planned_headway = planned_launch - last_actual
        next_trip.planned_headway = planned_headway
        next_trip.target_headway = planned_headway
        next_trip.planned_launch_time = planned_launch
        next_trip.planned_shift = planned_launch - float(next_trip.baseline_launch_time)

    def _compute_dispatch_proxy_reward(self, current_trip):
        """
        Compute per-dispatch proxy reward for the PREVIOUS same-direction dispatch.
        Measures what happened between the last dispatch and this one:
          - headway regularity of on-route buses (negative CV)
          - fleet utilization penalty (over N_fleet)
          - passenger wait proxy (waiting passengers at stations)

        Stored in self._dispatch_rewards[trip_id] for runner to collect.
        """
        if not hasattr(self, '_dispatch_rewards'):
            self._dispatch_rewards = {}
        if not hasattr(self, '_last_dispatch_trip'):
            self._last_dispatch_trip = {}

        direction = current_trip.direction
        prev_trip_id = self._last_dispatch_trip.get(direction, None)
        self._last_dispatch_trip[direction] = current_trip.launch_turn

        if prev_trip_id is None:
            return  # first dispatch in this direction, no reward yet

        # Headway regularity: negative CV of active buses' forward headways
        active_buses = [b for b in self.bus_all
                        if b.on_route and b.direction == direction
                        and b.forward_headway > 0]
        if len(active_buses) >= 2:
            hws = np.array([b.forward_headway for b in active_buses])
            hw_cv = hws.std() / max(hws.mean(), 1.0)
            r_regularity = -min(hw_cv, 2.0)  # in [-2, 0]
        else:
            r_regularity = 0.0

        # Fleet penalty: penalize exceeding N_fleet
        concurrent = sum(1 for b in self.bus_all if b.on_route)
        n_fleet = getattr(self, '_n_fleet_target', 12)
        r_fleet = -max(0.0, concurrent - n_fleet) / n_fleet  # in [-1, 0] roughly

        # Waiting passengers proxy (normalized)
        total_waiting = sum(len(s.waiting_passengers) for s in self.stations)
        r_wait = -min(total_waiting / 500.0, 2.0)  # in [-2, 0]

        # Combined proxy reward (normalized to roughly [-1, 0])
        proxy_reward = 0.5 * r_regularity + 0.3 * r_fleet + 0.2 * r_wait

        self._dispatch_rewards[prev_trip_id] = float(proxy_reward)

    def _get_target_headway_for_bus(self, bus):
        """Look up the target headway from the timetable that launched this bus."""
        for tt in self.timetables:
            if tt.launch_turn == bus.trip_id:
                return float(getattr(tt, 'planned_headway', tt.target_headway))
        return self.timetable_base_headway

    def _timetable_for_bus(self, bus):
        for tt in self.timetables:
            if tt.launch_turn == bus.trip_id:
                return tt
        return None

    def _build_lower_context_for_bus(self, bus):
        if not self.use_lower_context:
            return []
        tt = self._timetable_for_bus(bus)
        if tt is None:
            planned_hw = self.timetable_base_headway
            prev_gap = self.timetable_base_headway
            next_gap = self.timetable_base_headway
        else:
            planned_hw = float(getattr(tt, 'planned_headway', self.timetable_base_headway))
            same_dir = sorted(
                [x for x in self.timetables if x.direction == tt.direction],
                key=lambda x: x.baseline_launch_time)
            idx = same_dir.index(tt)
            if idx > 0 and same_dir[idx - 1].planned_launch_time is not None:
                prev_gap = float(tt.planned_launch_time or tt.baseline_launch_time) - \
                    float(same_dir[idx - 1].planned_launch_time)
            else:
                prev_gap = planned_hw
            if idx + 1 < len(same_dir) and same_dir[idx + 1].planned_launch_time is not None:
                next_gap = float(same_dir[idx + 1].planned_launch_time) - \
                    float(tt.planned_launch_time or tt.baseline_launch_time)
            else:
                next_gap = planned_hw
        station_count = max(len(getattr(bus, 'effective_station', [])) - 1, 1)
        station_phase = float(getattr(bus.last_station, 'station_id', 0)) / station_count
        return [
            float(getattr(bus, 'last_applied_action', 0.0)) / 60.0,
            planned_hw / 600.0,
            prev_gap / 600.0,
            next_gap / 600.0,
            station_phase,
        ]

    def _build_upper_state(self, trip):
        """
        Build upper policy state vector at each dispatch event.
        Returns: np.array of shape (5,)
            [hour_norm, demand_norm, fleet_norm, prev_headway_norm, unhealthy_rate]
        """
        hour = 6 + self.current_time // 3600

        # Aggregate current-period passenger demand across all stations
        effective_period_str = f"{min(int(hour), 19):02}:00:00"
        total_demand = 0.0
        for s in self.stations:
            if s.od is not None:
                period_data = s.od.get(effective_period_str, {})
                if isinstance(period_data, dict):
                    total_demand += sum(period_data.values())

        fleet_on_route = sum(1 for bus in self.bus_all if bus.on_route)

        # Actual headway from previous same-direction trip
        same_dir_launched = [tt for tt in self.timetables
                             if tt.launched and tt.direction == trip.direction]
        if len(same_dir_launched) >= 2:
            last_two = sorted(
                same_dir_launched,
                key=lambda t: getattr(t, 'actual_launch_time', t.launch_time))[-2:]
            prev_actual_headway = (
                getattr(last_two[1], 'actual_launch_time', last_two[1].launch_time)
                - getattr(last_two[0], 'actual_launch_time', last_two[0].launch_time))
        else:
            prev_actual_headway = 360.0

        # Fraction of on-route buses currently holding (signal of fleet pressure)
        from env.bus import BusState
        holding_count = sum(1 for bus in self.bus_all
                            if bus.on_route and bus.state == BusState.HOLDING)
        holding_ratio = holding_count / max(fleet_on_route, 1)

        return np.array([
            hour / 24.0,
            total_demand / 1000.0,
            fleet_on_route / self.max_agent_num,
            prev_actual_headway / 600.0,
            holding_ratio,
        ], dtype=np.float32)

    def _compute_measurement_vector(self):
        """
        Compute measurement z(π) from live data. Must be called BEFORE cleanup.
        Returns: np.array [avg_wait_min, peak_fleet, headway_cv]
        """
        # z[0]: average passenger wait time (minutes)
        total_wait, pax_count = 0.0, 0
        for s in self.stations:
            for p in s.total_passenger:
                if hasattr(p, 'boarding_time') and p.boarding_time is not None:
                    total_wait += (p.boarding_time - p.appear_time)
                    pax_count += 1
        avg_wait = (total_wait / max(pax_count, 1)) / 60.0

        # z[1]: peak concurrent fleet size
        peak_fleet = self._peak_concurrent

        # z[2]: headway coefficient of variation (replaces bunching_rate)
        headways = [bus.forward_headway for bus in self.bus_all
                    if hasattr(bus, 'forward_headway') and bus.forward_headway > 0]
        if len(headways) >= 2:
            hw_arr = np.array(headways)
            headway_cv = float(hw_arr.std() / max(hw_arr.mean(), 1.0))
        else:
            headway_cv = 0.0

        return np.array([avg_wait, peak_fleet, headway_cv])

    def _compute_episode_metrics(self):
        z = self._compute_measurement_vector()
        all_passengers = []
        for s in self.stations:
            all_passengers.extend(s.total_passenger)
        arrived = [
            p for p in all_passengers
            if getattr(p, 'arrive_time', None) is not None
            and getattr(p, 'boarding_time', None) is not None
        ]
        if arrived:
            onboard = np.array([p.arrive_time - p.boarding_time for p in arrived], dtype=float)
            total = np.array([p.arrive_time - p.appear_time for p in arrived], dtype=float)
            onboard_min = float(onboard.mean() / 60.0)
            total_min = float(total.mean() / 60.0)
        else:
            onboard_min = 0.0
            total_min = 0.0

        holds = [
            float(a) for bus in self.bus_all
            for a in getattr(bus, 'applied_actions', [])
        ]
        hold_arr = np.array(holds, dtype=float) if holds else np.array([], dtype=float)

        actual_headways = []
        planned_headways = []
        for direction in [True, False]:
            trips = [
                tt for tt in self.timetables
                if tt.direction == direction and getattr(tt, 'actual_launch_time', None) is not None
            ]
            trips.sort(key=lambda x: x.actual_launch_time)
            actual_headways.extend([
                trips[i + 1].actual_launch_time - trips[i].actual_launch_time
                for i in range(len(trips) - 1)
            ])
            planned = [
                tt for tt in self.timetables
                if tt.direction == direction and getattr(tt, 'planned_launch_time', None) is not None
            ]
            planned.sort(key=lambda x: x.planned_launch_time)
            planned_headways.extend([
                planned[i + 1].planned_launch_time - planned[i].planned_launch_time
                for i in range(len(planned) - 1)
            ])

        def mean_std_cv(values):
            if not values:
                return 0.0, 0.0, 0.0
            arr = np.asarray(values, dtype=float)
            mean = float(arr.mean())
            std = float(arr.std())
            return mean, std, float(std / max(mean, 1.0))

        plan_mean, plan_std, plan_cv = mean_std_cv(planned_headways)
        actual_mean, actual_std, actual_cv = mean_std_cv(actual_headways)

        shifts = [
            float(getattr(tt, 'planned_shift', 0.0))
            for tt in self.timetables
            if getattr(tt, 'planned_launch_time', None) is not None
        ]
        lateness = [
            float(getattr(tt, 'dispatch_lateness', 0.0))
            for tt in self.timetables if tt.launched
        ]
        shifts_arr = np.asarray(shifts, dtype=float) if shifts else np.array([], dtype=float)
        late_arr = np.asarray(lateness, dtype=float) if lateness else np.array([], dtype=float)

        return {
            'avg_wait_min': float(z[0]),
            'peak_fleet': float(z[1]),
            'headway_cv': float(z[2]),
            'avg_holding_sec': float(hold_arr.mean()) if hold_arr.size else 0.0,
            'total_holding_sec': float(hold_arr.sum()) if hold_arr.size else 0.0,
            'avg_onboard_time_min': onboard_min,
            'avg_total_passenger_time_min': total_min,
            'n_arrived_passengers': len(arrived),
            'planned_dispatch_headway_mean': plan_mean,
            'planned_dispatch_headway_std': plan_std,
            'planned_dispatch_headway_cv': plan_cv,
            'actual_dispatch_headway_mean': actual_mean,
            'actual_dispatch_headway_std': actual_std,
            'actual_dispatch_headway_cv': actual_cv,
            'planned_shift_mean': float(shifts_arr.mean()) if shifts_arr.size else 0.0,
            'planned_shift_std': float(shifts_arr.std()) if shifts_arr.size else 0.0,
            'dispatch_lateness_mean': float(late_arr.mean()) if late_arr.size else 0.0,
            'dispatch_lateness_max': float(late_arr.max()) if late_arr.size else 0.0,
        }

    @property
    def episode_metrics(self):
        if hasattr(self, '_cached_episode_metrics') and self._cached_episode_metrics is not None:
            return self._cached_episode_metrics
        return self._compute_episode_metrics()

    @property
    def measurement_vector(self):
        """
        ApproPO long-term measurement z(π), computed at episode end.
        Returns: np.array [avg_wait_min, peak_fleet, headway_cv]
        """
        if hasattr(self, '_cached_measurement') and self._cached_measurement is not None:
            return self._cached_measurement
        return self._compute_measurement_vector()

    # ---- v2: per-trip holding feedback support ----

    def get_completed_trip_holdings(self):
        """
        Collect applied holding actions for all buses that have completed trips
        (off-route) since last call.

        Returns: dict {trip_id: [action_1, action_2, ...]}
        """
        result = {}
        for bus in self.bus_all:
            if not bus.on_route and hasattr(bus, 'applied_actions') and bus.applied_actions:
                # Bus completed its trip; collect actions keyed by trip_id
                result[bus.trip_id] = list(bus.applied_actions)
        return result

    def get_direction_holding_stats(self, direction, n_recent=5):
        """
        Get holding statistics from recent completed trips in a given direction.

        Returns: dict with rolling_mean, rolling_std (of per-trip mean holdings)
        """
        completed = []
        for bus in self.bus_all:
            if not bus.on_route and bus.direction == direction:
                if hasattr(bus, 'applied_actions') and bus.applied_actions:
                    completed.append({
                        'trip_id': bus.trip_id,
                        'mean': float(np.mean(bus.applied_actions)),
                        'n_stops': len(bus.applied_actions),
                    })
        # Sort by trip_id (proxy for time order), take last n_recent
        completed.sort(key=lambda x: x['trip_id'])
        recent = completed[-n_recent:] if len(completed) > n_recent else completed

        if not recent:
            return {'rolling_mean': 0.0, 'rolling_std': 0.0, 'n_trips': 0}
        means = [c['mean'] for c in recent]
        return {
            'rolling_mean': float(np.mean(means)),
            'rolling_std': float(np.std(means)) if len(means) > 1 else 0.0,
            'n_trips': len(recent),
        }

    def _build_upper_state_v2(self, trip):
        """
        Build enriched upper state for v2 coupling.

        Returns: np.array of shape (upper_state_dim_v2,)
            [hour_norm, demand_norm, fleet_norm, gap_to_prev_norm,
             holding_ratio, holding_mean_same_dir, holding_std_same_dir,
             holding_mean_other_dir, scheduled_headway_norm, direction]
        """
        hour = 6 + self.current_time // 3600

        # Demand
        effective_period_str = f"{min(int(hour), 19):02}:00:00"
        total_demand = 0.0
        for s in self.stations:
            if s.od is not None:
                period_data = s.od.get(effective_period_str, {})
                if isinstance(period_data, dict):
                    total_demand += sum(period_data.values())

        fleet_on_route = sum(1 for bus in self.bus_all if bus.on_route)

        # Actual headway gap to previous dispatch in same direction
        actual_gap = self.current_time - self._last_dispatch_time.get(trip.direction, -9999)
        if actual_gap > 9000:
            actual_gap = 360.0  # first trip

        # Holding ratio (fraction of active buses currently holding)
        from env.bus import BusState
        holding_count = sum(1 for bus in self.bus_all
                            if bus.on_route and bus.state == BusState.HOLDING)
        holding_ratio = holding_count / max(fleet_on_route, 1)

        # v2 KEY: holding statistics from recent trips (lower → upper feedback)
        same_dir_stats = self.get_direction_holding_stats(trip.direction, n_recent=5)
        other_dir_stats = self.get_direction_holding_stats(not trip.direction, n_recent=5)

        # Scheduled/planned headway used by the lower controller
        scheduled_hw = getattr(
            trip, 'planned_headway',
            trip.target_headway if hasattr(trip, 'target_headway') else 360.0)

        # v2k: include current fleet budget in state (for Pareto-aware policy)
        n_fleet_norm = getattr(self, '_n_fleet_target', 12) / 20.0

        return np.array([
            hour / 24.0,                                   # [0] time of day
            total_demand / 1000.0,                         # [1] demand level
            fleet_on_route / self.max_agent_num,           # [2] fleet utilization
            actual_gap / 600.0,                            # [3] gap to prev dispatch
            holding_ratio,                                 # [4] fraction of fleet holding
            same_dir_stats['rolling_mean'] / 60.0,         # [5] mean holding, same dir
            same_dir_stats['rolling_std'] / 60.0,          # [6] std holding, same dir
            other_dir_stats['rolling_mean'] / 60.0,        # [7] mean holding, other dir
            scheduled_hw / 600.0,                          # [8] base scheduled headway
            float(trip.direction),                         # [9] direction (0 or 1)
            n_fleet_norm,                                  # [10] v2k: fleet budget
        ], dtype=np.float32)

    def set_timetable_from_planner(self, headway_params):
        """
        Upper layer writes target_headway into timetable.

        Args:
            headway_params: dict {'peak': float, 'off_peak': float, 'transition': float}
                            OR list[float] of per-trip target headways
        """
        if isinstance(headway_params, dict):
            for tt in self.timetables:
                hour = 6 + tt.launch_time // 3600
                if 7 <= hour <= 9 or 17 <= hour <= 19:
                    tt.target_headway = headway_params['peak']
                elif 9 < hour < 17:
                    tt.target_headway = headway_params['off_peak']
                else:
                    tt.target_headway = headway_params['transition']
        else:
            for tt, hw in zip(self.timetables, headway_params):
                tt.target_headway = float(hw)


if __name__ == '__main__':
    debug = True
    render = False
    num_runs = 1

    env_dir = Path(__file__).resolve().parent
    env = env_bus(str(env_dir), debug=debug)
    env.enable_plot = True
    actions = {key: 0. for key in list(range(env.max_agent_num))}

    all_events = []
    cumulative_time = 0

    for run_idx in range(1, num_runs + 1):
        env.reset()
        while not env.done:
            state, reward, cost, done = env.step(action=actions, debug=debug,
                                           render=render, episode=run_idx)

        events = env.visualizer.extract_bunching_events()
        cumulative_time += env.current_time
        all_events.extend(events)

    # Only quit pygame if it was initialized
    if pygame.get_init():
        pygame.quit()

    if all_events:
        df = pd.DataFrame(all_events).sort_values(['time'])
        output_dir = os.path.join(env.path, 'pic')
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, f'all_bunching_records_{num_runs}.csv'), index=False)
        # env.visualizer.plot_bunching_events(all_events, exp=str(num_runs))

    print('Total simulation time:', cumulative_time)
