"""Check paired inputs, equal-time ordering and unfinished-trip accounting."""

from types import SimpleNamespace
import unittest

import numpy as np

from cohorts import ArrivalFlow, install_arrivals, passenger_class, passenger_times, summarize_passengers


class Passenger:
    def __init__(self, time, station, destination):
        self.appear_time = time
        self.boarding_time = self.arrive_time = None
        self.appear_station, self.destination_station = station, destination


class CohortTests(unittest.TestCase):
    def test_baseline_is_identical_across_scenes_and_global_rng_draws(self):
        normal = ArrivalFlow(78101, 3, "normal", 10, 20, 49)
        shock = ArrivalFlow(78101, 3, "burst50", 10, 20, 49)
        for time in range(30):
            before, absent = normal.draw([0.4, 0.8], 1, time)
            np.random.random(17)
            after, extra = shock.draw([0.4, 0.8], 1, time)
            np.testing.assert_array_equal(before, after)
            self.assertFalse(absent.any())
            if not 10 <= time < 20:
                self.assertFalse(extra.any())
            else:
                self.assertGreater(extra.sum(), 0)

    def test_equal_time_cohorts_are_interleaved(self):
        station = SimpleNamespace(waiting_passengers=[], total_passenger=[])
        line = SimpleNamespace(_pax_cache_hour=6, _pax_flat_rates=np.array([10.0]),
                               _pax_flat_map=[(station, object())])
        flow = ArrivalFlow(78101, 0, "burst50", 0, 20, 49)
        install_arrivals(line, passenger_class(Passenger), flow)
        line._batch_passenger_arrival(0, 1)
        labels = [p.cohort for p in station.total_passenger]
        self.assertIn("baseline", labels)
        self.assertIn("extra", labels)
        self.assertNotEqual(labels, sorted(labels))

    def test_clipped_passenger_times_and_cohort_partition(self):
        tagged = passenger_class(Passenger)
        waiting = tagged(10, None, None, "baseline")
        completed = tagged(10, None, None, "baseline")
        completed.boarding_time, completed.arrive_time = 20, 50
        onboard = tagged(20, None, None, "extra")
        onboard.boarding_time = 40
        self.assertEqual(passenger_times(waiting, 100), (90, 0))
        self.assertEqual(passenger_times(completed, 100), (10, 30))
        self.assertEqual(passenger_times(onboard, 100), (20, 60))
        groups = summarize_passengers([waiting, completed, onboard], 100, 10, 20)
        self.assertEqual(groups["baseline"]["passengers"], 2)
        self.assertEqual(groups["baseline_during"]["passengers"], 2)
        self.assertEqual(groups["all"]["mean_wait_seconds"], 40)
        self.assertEqual(groups["all"]["mean_generalized_seconds"], 110)
        self.assertEqual(groups["baseline"]["completion_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
