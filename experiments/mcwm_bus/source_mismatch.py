"""Source-only physical interventions; original simulator files remain unchanged."""

import random

import numpy as np


def travel_steps(runtime, overhead, speed_factor, added_signals, green_fraction, cycle):
    moving = max(int(round(runtime))-overhead, 1) / speed_factor
    delay = 0.
    for _ in range(np.random.poisson(added_signals)) if added_signals > 0 else ():
        if random.random() > green_fraction:
            delay += random.uniform(0, cycle*(1-green_fraction))
    return max(int(round(moving+delay)), 1)


def install_source_mechanism(scenario):
    import sim_core.sim as simulation
    from sim_core.bus import Bus, LINE_SIGNAL_CONFIG

    class SourceBus(Bus):
        def exchange_passengers(self, current_time, debug, station=None):
            super().exchange_passengers(current_time, debug, station)
            self.holding_time *= scenario["dwell_factor"]
            self._base_stop_duration = self.holding_time

        def _advance_on_route(self):
            if not scenario["empirical_runtime"]:
                self._empirical_runtime_calibrated = False
            route = self.current_route
            if self._empirical_runtime_calibrated and route.runtime_calibration is not None:
                if self._calibrated_travel_steps_remaining is None:
                    self._segment_start_time = getattr(self, "_current_sim_time", self.launch_time)
                    runtime = route.sample_runtime_seconds(self._segment_start_time, demand_scale=1.0)
                    signals = LINE_SIGNAL_CONFIG[self.line_id_str]
                    additional = scenario["extra_signal_fraction"]*signals["n_signals"]/max(self._n_route_segments, 1)
                    self._calibrated_travel_steps_remaining = travel_steps(
                        runtime, route.runtime_fixed_overhead_seconds, scenario["speed_factor"],
                        additional, signals["green_frac"], signals["cycle"])
            super()._advance_on_route()

    simulation.Bus = SourceBus


def restore_source_mechanism():
    import sim_core.sim as simulation
    from sim_core.bus import Bus
    simulation.Bus = Bus
