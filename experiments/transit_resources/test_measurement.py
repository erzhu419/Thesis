"""Detect lost trips when a physical bus is reused, and terminal double counting."""

import unittest
from types import SimpleNamespace as NS

from measurement import ResourceMeter


class MeasurementTests(unittest.TestCase):
    def test_reused_vehicle_and_terminal_wait(self):
        meter = ResourceMeter()
        meter.direction_lengths = {True: 1000.0, False: 1500.0}
        bus = NS(on_route=True, trip_id=0, direction=True, launch_time=10,
                 passengers=[1, 2], capacity=50, applied_actions=[1, 0])
        env = NS(bus_all=[bus], done=False,
                 timetables=[NS(launch_turn=0), NS(launch_turn=1)])
        meter.sample(env, 1)
        meter.sample(env, 1)
        bus.on_route = False
        bus.direction = False
        bus.back_to_terminal_time = 12
        meter.sample(env, 1)
        meter.sample(env, 1)
        bus.trip_id = 1
        bus.launch_time = 14
        bus.on_route = True
        bus.passengers = [1]
        bus.applied_actions = [2, 3, 0]
        meter.sample(env, 1)
        bus.on_route = False
        bus.direction = True
        bus.back_to_terminal_time = 15
        env.done = True
        meter.sample(env, 1)
        result = meter.results(env)
        self.assertEqual(result["completed_trips"], 2)
        self.assertEqual(result["vehicles_used"], 1)
        self.assertEqual(result["service_km"], 2.5)
        self.assertAlmostEqual(result["vehicle_hours"], 3 / 3600)
        self.assertAlmostEqual(result["time_weighted_load"], 5 / 150)
        self.assertAlmostEqual(result["full_day_holding_mean_sec"], 6 / 5)


if __name__ == "__main__":
    unittest.main()
