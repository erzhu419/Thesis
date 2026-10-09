"""Native policy units and per-vehicle speed control, without shared route edits."""

import numpy as np


def physical_action(raw, kind):
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)
    if raw.size not in (1, 2) or not np.isfinite(raw).all():
        raise ValueError("Invalid bus policy action")
    if kind == "resac":
        hold = float(np.clip(raw[0], 0, 60))
        speed = .8 + float(np.clip(raw[1], 0, 60))/150 if raw.size == 2 else 1.
    else:
        hold = 30*(float(np.clip(raw[0], -1, 1))+1)
        speed = 1+.2*float(np.clip(raw[1], -1, 1)) if raw.size == 2 else 1.
    return np.array([hold, speed], np.float32)


class VehicleRoute:
    def __init__(self, route, vehicle):
        self.route, self.vehicle = route, vehicle

    @property
    def speed_limit(self):
        return self.route.speed_limit*self.vehicle.control_speed_ratio

    def __getattr__(self, name):
        return getattr(self.route, name)


def controlled_bus_class(original):
    class ControlledBus(original):
        control_speed_ratio = 1.

        @property
        def current_route(self):
            route = super().current_route
            return route if self.control_speed_ratio == 1. else VehicleRoute(route, self)

        def drive(self, current_time, action, bus_all, debug):
            command = physical_action([0.] if action is None else action, "resac")
            # Latch only when the native state machine accepts a station action.
            if self.in_station and self.holding and not self.dwelling and self.held and self.holding_time <= 1:
                self.control_speed_ratio = float(command[1])
            return super().drive(current_time, float(command[0]), bus_all, debug)

        def bus_update(self, *args, **kwargs):
            self.control_speed_ratio = 1.
            return super().bus_update(*args, **kwargs)

        def reset_bus(self, *args, **kwargs):
            self.control_speed_ratio = 1.
            return super().reset_bus(*args, **kwargs)

    return ControlledBus
