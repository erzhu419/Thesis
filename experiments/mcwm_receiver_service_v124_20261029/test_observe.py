"""Receiver-service getter fixtures; no native environment or training."""
from copy import deepcopy
from types import SimpleNamespace as NS
import unittest

from experiments.mcwm_receiver_service_v124_20261029.observe import capture_receiver_service


class Bridge:
    def __init__(self):
        self.pending_events = {("L", "R", "S"): NS(
            assigned_action=None, metadata={"service_completion_time": 8.})}

    @property
    def passenger_obj_dic(self):
        raise AssertionError("Permanent journey intent is not a current-stage input")


class Native:
    def __init__(self):
        self.clock = 10.
        self.members = ["current_target", "future_target"]
        self.stages = {
            "current_target": NS(type=3, line="L", destStop="S"),
            "future_target": NS(type=3, line="L", destStop="D"),
        }
        self.future_destinations = {"current_target": "D", "future_target": "S"}
        self.member_error = False
        self.stage_errors = set()
        self.calls = []
        self.simulation = NS(getTime=self.get_time)
        self.vehicle = NS(getPersonIDList=self.get_members)
        self.person = NS(getStage=self.get_stage)

    def get_time(self):
        self.calls.append(("getTime",))
        return self.clock

    def get_members(self, trip):
        self.calls.append(("getPersonIDList", trip))
        if trip != "R":
            raise AssertionError("Only the current receiver may be queried")
        if self.member_error:
            raise RuntimeError("Membership unavailable")
        return list(self.members)

    def get_stage(self, person, index):
        self.calls.append(("getStage", person, index))
        if index != 0:
            raise AssertionError("Future stages are not observable inputs")
        if person in self.stage_errors:
            raise RuntimeError("Current stage unavailable")
        return self.stages[person]

    def physical_state(self):
        return deepcopy((self.clock, self.members, self.stages, self.future_destinations))


traci = None


def issued():
    return dict(trip_id="R", line_id="L", station_id="S", sequence=6, started_at=10.)


class ReceiverServiceTests(unittest.TestCase):
    def setUp(self):
        global traci
        traci = Native()
        self.native, self.bridge = traci, Bridge()

    def capture(self):
        return capture_receiver_service(self.bridge, issued(), source_group="fixed-root")

    def test_current_leg_not_future_or_permanent_destination(self):
        result = self.capture()
        self.assertEqual(result["capture"]["status"], "complete")
        self.assertEqual(result["receiver"]["onboard_count"], 2)
        self.assertEqual(result["receiver"]["current_leg_dest_count"], 1)
        self.assertTrue(result["receiver"]["current_stage0_complete"])
        people = {row["person_id"]: row["current_stage0"] for row in result["people"]}
        self.assertEqual(people["current_target"], dict(known=True, type=3, line="L", destStop="S"))
        self.assertEqual(people["future_target"]["destStop"], "D")
        self.assertTrue(all(call[2] == 0 for call in self.native.calls if call[0] == "getStage"))

    def test_complete_zero_is_distinct_from_partial_or_failed_reads(self):
        self.native.members = []
        empty = self.capture()
        self.assertEqual(empty["receiver"]["onboard_count"], 0)
        self.assertEqual(empty["receiver"]["current_leg_dest_count"], 0)
        self.assertTrue(empty["receiver"]["current_stage0_complete"])
        self.assertEqual(empty["capture"]["status"], "complete")
        self.assertEqual(empty["people"], [])

        self.native.members = ["current_target", "future_target"]
        self.native.stages["future_target"] = NS(type=3, line=" ", destStop="S")
        partial = self.capture()
        self.assertEqual(partial["receiver"]["onboard_count"], 2)
        self.assertIsNone(partial["receiver"]["current_leg_dest_count"])
        self.assertFalse(partial["receiver"]["current_stage0_complete"])
        self.assertNotEqual(partial["capture"]["status"], "complete")
        self.assertTrue(partial["capture"]["missing"])
        known = [row for row in partial["people"] if row["person_id"] == "current_target"]
        self.assertTrue(known[0]["current_stage0"]["known"])

        self.native.stage_errors.add("future_target")
        failed_stage = self.capture()
        self.assertIsNone(failed_stage["receiver"]["current_leg_dest_count"])
        self.assertFalse(failed_stage["receiver"]["current_stage0_complete"])
        self.assertTrue(failed_stage["capture"]["errors"])
        self.assertTrue(any(row["person_id"] == "current_target" and row["current_stage0"]["known"]
                            for row in failed_stage["people"]))

        self.native.member_error = True
        failed_members = self.capture()
        self.assertIsNone(failed_members["receiver"]["onboard_count"])
        self.assertIsNone(failed_members["receiver"]["current_leg_dest_count"])
        self.assertFalse(failed_members["receiver"]["current_stage0_complete"])
        self.assertEqual(failed_members["people"], [])
        self.assertTrue(failed_members["capture"]["errors"])

    def test_getters_preserve_state_and_require_pre_action_clock(self):
        before = self.native.physical_state()
        pending = deepcopy(self.bridge.pending_events)
        result = self.capture()
        self.assertEqual(self.native.physical_state(), before)
        self.assertEqual(self.bridge.pending_events, pending)
        self.assertEqual(result["meta"], dict(sequence=6, observed_at=10., receiver="R",
                                              station="S", source_group="fixed-root"))
        self.assertTrue(result["capture"]["checks"]["clock_unchanged"])
        self.assertTrue(result["capture"]["checks"]["pending_event_unassigned"])
        self.assertEqual({call[0] for call in self.native.calls},
                         {"getTime", "getPersonIDList", "getStage"})
        self.native.clock = 11.
        with self.assertRaises(ValueError):
            self.capture()
        self.native.clock = 10.
        self.bridge.pending_events[("L", "R", "S")].assigned_action = 0.
        with self.assertRaises(ValueError):
            self.capture()

    def test_elapsed_estimate_does_not_erase_current_alighting_demand(self):
        self.native.members = ["current_target"]
        result = self.capture()
        self.assertEqual(result["receiver"]["estimated_service_ready_time"], 8.)
        self.assertEqual(result["receiver"]["estimated_service_ready_age_seconds"], 2.)
        self.assertEqual(result["receiver"]["current_leg_dest_count"], 1)
        self.assertTrue(result["receiver"]["current_stage0_complete"])
        self.assertEqual(result["capture"]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
