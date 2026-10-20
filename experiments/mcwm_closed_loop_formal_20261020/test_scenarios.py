"""Checks that detect off-clock interventions and missing accrued passenger time."""

from types import SimpleNamespace
import unittest

from scenarios import (LANE_IDS, ScenarioRecorder, is_direct_interval,
                       prediction_period, UPSTREAM_STOP, DOWNSTREAM_STOP)


class FakeAPI:
    def __init__(self):
        self.now = 1.
        self.speed = {lane: 13.89 for lane in LANE_IDS}
        self.changes = []
        self.simulation = SimpleNamespace(getTime=lambda: self.now)
        self.lane = SimpleNamespace(getMaxSpeed=lambda lane: self.speed[lane],
            setMaxSpeed=self.set_speed)

    def set_speed(self, lane, value):
        self.speed[lane] = value
        self.changes.append((self.now, lane, value))


def ledger(time=0., wait=10., ride=20., waiting=2, onboard=3):
    value = SimpleNamespace(time_seconds=time, accrued_wait_seconds=wait,
        accrued_in_vehicle_seconds=ride, waiting_passengers=waiting,
        in_vehicle_passengers=onboard)
    return SimpleNamespace(snapshot=lambda: value), value


class ScenarioTests(unittest.TestCase):
    def test_native_clock_applies_exactly_once_and_stable_does_not_mutate(self):
        for name, expected in (("stable", 0), ("local_speed_drop", 4)):
            api = FakeAPI()
            clock, snapshot = ledger()
            hook = ScenarioRecorder(api, clock, name)
            for now in (8999., 9000., 9001., 15000.):
                api.now, snapshot.time_seconds = now, now-1
                hook.step(0)
            self.assertEqual(len(api.changes), expected)
            if expected:
                self.assertEqual(hook.applied_at, 9000.)
                self.assertTrue(all(t == 9000. for t, _, _ in api.changes))
                self.assertAlmostEqual(hook.lane_caps[0]["observed_mps"], 9.0285)
            else:
                self.assertIsNone(hook.applied_at)

    def test_tick_projection_keeps_ledger_unmodified_and_counts_unfinished_people(self):
        api = FakeAPI()
        clock, snapshot = ledger(time=299., wait=100., ride=200.)
        api.now = 299.
        hook = ScenarioRecorder(api, clock, "stable")
        api.now = 300.
        hook.step(0)
        point = hook.metrics[-1]
        self.assertEqual(point["time_seconds"], 300.)
        self.assertEqual(point["accrued_wait_seconds"], 102.)
        self.assertEqual(point["accrued_in_vehicle_seconds"], 203.)
        self.assertEqual(point["accrued_generalized_passenger_seconds"], 407.)
        self.assertEqual(snapshot.accrued_wait_seconds, 100.)
        self.assertEqual(snapshot.time_seconds, 299.)
        snapshot.time_seconds = 300.
        snapshot.accrued_wait_seconds = 102.
        snapshot.accrued_in_vehicle_seconds = 203.
        self.assertEqual(hook.capture_final()["accrued_generalized_passenger_seconds"], 407.)
        self.assertEqual(len(hook.metrics), 2)

    def test_direct_labels_and_prediction_period_use_actual_issue_time(self):
        self.assertTrue(is_direct_interval("102X", UPSTREAM_STOP, DOWNSTREAM_STOP))
        self.assertFalse(is_direct_interval("7S", UPSTREAM_STOP, DOWNSTREAM_STOP))
        self.assertEqual(prediction_period(8999.), "before")
        self.assertEqual(prediction_period(9000.), "early_after")
        self.assertEqual(prediction_period(12000.), "middle_after")
        self.assertEqual(prediction_period(15000.), "late_after")


if __name__ == "__main__":
    unittest.main()
