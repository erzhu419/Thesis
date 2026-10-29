"""Synthetic service-risk contracts; no simulation, fitting or remote data."""
from copy import deepcopy
import unittest

from experiments.mcwm_service_risk_contract_v126_20261029.interface import (
    can_release, overrun_target, risk_inputs,
)


def record_fixture():
    return dict(
        context=dict(meta=dict(source_group="fixed/stable", sequence=8,
                               observed_at=1000., receiver="R", station="S")),
        selection=dict(sequence=8, window_start=1000., window_end=1060.,
                       feedback_available_at=1090., packet_deadline=1200.),
        action=dict(applied_at=1000., hold_seconds=5.2),
        labels=dict(available_at=1060., receiver_departure=dict(
            value=1006., right_censored=False, observed_until=1060.)))


def observation_fixture(record, count=2, complete=True):
    return dict(meta=deepcopy(record["context"]["meta"]),
        receiver=dict(onboard_count=10, current_leg_dest_count=count,
            current_stage0_complete=complete, estimated_service_ready_time=998.,
            estimated_service_ready_age_seconds=2.),
        people=[], capture=dict(status="complete" if complete else "missing"))


class ServiceRiskInterfaceTests(unittest.TestCase):
    def test_clock_deadline_boundary_and_observed_late_departure(self):
        record = record_fixture()
        before = deepcopy(record)
        target = overrun_target(record)
        self.assertEqual(target["value"], 0)  # D == ceil(1000 + 5.2).
        self.assertEqual(target["clock_deadline"], 1006.)
        self.assertEqual(target["window_end"], 1060.)
        self.assertEqual(target["feedback_available_at"], 1090.)
        self.assertEqual(target["packet_deadline"], 1200.)
        self.assertFalse(target["departure_time_censored"])
        self.assertTrue(target["reason"])
        self.assertEqual(record, before)
        record["labels"]["receiver_departure"]["value"] = 1003.
        self.assertEqual(overrun_target(record)["value"], 0)
        record["labels"]["receiver_departure"]["value"] = 1007.
        self.assertEqual(overrun_target(record)["value"], 1)

    def test_censored_equal_window_deadline_is_known_but_missing_coverage_is_not(self):
        record = record_fixture()
        record["action"]["hold_seconds"] = 60.
        record["labels"]["receiver_departure"].update(value=None, right_censored=True)
        before = deepcopy(record)
        target = overrun_target(record)
        self.assertEqual(target["clock_deadline"], target["window_end"])
        self.assertEqual(target["value"], 1)
        self.assertTrue(target["departure_time_censored"])
        self.assertEqual(record, before)

        record["full_day_departure"] = 1065.
        record["labels"]["future_boarding_count"] = 7
        record["labels"]["full_day_departure"] = 1065.
        with_future = deepcopy(record)
        self.assertEqual(overrun_target(record), target)
        self.assertEqual(record, with_future)
        record["labels"]["receiver_departure"]["observed_until"] = 1059.
        self.assertIsNone(overrun_target(record)["value"])
        record["labels"]["receiver_departure"]["observed_until"] = None
        self.assertIsNone(overrun_target(record)["value"])
        del record["labels"]["receiver_departure"]["observed_until"]
        self.assertIsNone(overrun_target(record)["value"])

    def test_release_requires_original_packet_strictly_before_issue(self):
        record = record_fixture()
        target = overrun_target(record)
        for issue_time in (1007., 1036., 1090., 1200.):
            with self.subTest(issue_time=issue_time):
                self.assertFalse(can_release(target, issue_time))
        self.assertTrue(can_release(target, 1201.))
        record["labels"]["receiver_departure"].update(
            value=None, right_censored=True, observed_until=None)
        unknown = overrun_target(record)
        self.assertIsNone(unknown["value"])
        self.assertFalse(can_release(unknown, 1201.))

    def test_risk_inputs_are_only_current_count_and_hold_with_exact_root_binding(self):
        record = record_fixture()
        observation = observation_fixture(record, count=0)
        before = deepcopy((record, observation))
        self.assertEqual(risk_inputs(record, observation), [0, 5.2])
        self.assertEqual((record, observation), before)
        record["labels"]["receiver_departure"]["value"] = 1007.
        self.assertEqual(overrun_target(record)["value"], 1)  # Complete n=0 does not force no risk.
        observation["receiver"]["estimated_service_ready_time"] = 2000.
        observation["receiver"]["estimated_service_ready_age_seconds"] = -1000.
        record["full_day_departure"] = 2000.
        self.assertEqual(risk_inputs(record, observation), [0, 5.2])
        observation["receiver"]["current_leg_dest_count"] = None
        self.assertIsNone(risk_inputs(record, observation))
        observation["receiver"].update(current_leg_dest_count=2, current_stage0_complete=False)
        self.assertIsNone(risk_inputs(record, observation))

        for field, value in (("source_group", "fixed/local_speed_drop"), ("sequence", 9),
                             ("observed_at", 1001.), ("receiver", "other"), ("station", "other")):
            with self.subTest(mismatched_field=field):
                mismatched = observation_fixture(record)
                mismatched["meta"][field] = value
                with self.assertRaises(ValueError):
                    risk_inputs(record, mismatched)


if __name__ == "__main__":
    unittest.main()
