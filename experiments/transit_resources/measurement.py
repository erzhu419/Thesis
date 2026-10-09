"""Observe full-trip resources without changing bus decisions or random draws."""

import math


class ResourceMeter:
    def __init__(self):
        self.completed = set()
        self.trip_seconds = 0.0
        self.active_seconds = 0.0
        self.passenger_seconds = 0.0
        self.capacity_seconds = 0.0
        self.kilometres = 0.0
        self.holding_seconds = 0.0
        self.holding_decisions = 0
        self.direction_lengths = {}

    def sample(self, env, dt):
        for bus in env.bus_all:
            if bus.on_route:
                self.active_seconds += dt
                self.passenger_seconds += len(bus.passengers) * dt
                self.capacity_seconds += bus.capacity * dt
            elif bus.trip_id not in self.completed:
                self.completed.add(bus.trip_id)
                self.trip_seconds += bus.back_to_terminal_time - bus.launch_time
                # Bus direction flips on arrival at the terminal.
                self.kilometres += self.direction_lengths[not bool(bus.direction)] / 1000
                self.holding_seconds += sum(bus.applied_actions)
                self.holding_decisions += len(bus.applied_actions)

    def results(self, env):
        expected = {trip.launch_turn for trip in env.timetables}
        assert env.done and self.completed == expected, "Missing completed trip"
        assert math.isclose(self.trip_seconds, self.active_seconds, abs_tol=1e-6), (
            "Vehicle reuse or time alignment error", self.trip_seconds, self.active_seconds)
        assert self.capacity_seconds > 0 and self.holding_decisions > 0
        return {
            "completed_trips": len(self.completed),
            "vehicles_used": len(env.bus_all),
            "vehicle_hours": self.trip_seconds / 3600,
            "vehicle_hours_integral": self.active_seconds / 3600,
            "service_km": self.kilometres,
            "time_weighted_load": self.passenger_seconds / self.capacity_seconds,
            "passenger_hours_on_vehicles": self.passenger_seconds / 3600,
            "full_day_holding_mean_sec": self.holding_seconds / self.holding_decisions,
            "full_day_holding_vehicle_hours": self.holding_seconds / 3600,
            "holding_decisions": self.holding_decisions,
        }


def measured_environment(base):
    class MeasuredEnvironment(base):
        def reset(self, *args, **kwargs):
            self.resource_meter = ResourceMeter()
            result = super().reset(*args, **kwargs)
            midpoint = round(len(self.routes) / 2)
            self.resource_meter.direction_lengths = {
                True: sum(route.distance for route in self.routes[:midpoint]),
                False: sum(route.distance for route in self.routes[midpoint:]),
            }
            return result

        def step(self, *args, **kwargs):
            before = self.current_time
            result = super().step(*args, **kwargs)
            self.resource_meter.sample(self, self.current_time - before)
            return result

    return MeasuredEnvironment
