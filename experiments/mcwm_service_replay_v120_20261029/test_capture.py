"""Generalized getter fixtures; no native runtime or simulation steps."""
from copy import deepcopy
from types import SimpleNamespace as NS
import unittest

from experiments.mcwm_service_replay_v120_20261029.capture import capture_context
from experiments.mcwm_service_contract_v119_20261029.features import bind_action


class Bridge:
    def __init__(self):
        self.bus_obj_dic = dict.fromkeys(("R", "F", "empty", "unrelated"))
        self.pending_events = {("L", "R", "S"): NS(assigned_action=None, metadata={"service_completion_time": 8.})}

    @property
    def passenger_obj_dic(self):
        raise AssertionError("Static journey intent must never be read")


class Native:
    def __init__(self):
        self.clock, self.receiver_missing, self.stage_missing, self.unknown = 10., False, False, False
        self.stage_indices = []
        self.plans = dict(R=["S", "D"], F=["S", "D"], empty=["T", "S"], unrelated=["D"])
        self.members = dict(R=["r"], F=["p"], empty=[], unrelated=[])
        self.simulation = NS(getTime=lambda: self.clock)
        self.vehicle = NS(getIDList=lambda: [trip for trip in self.plans if trip != "R" or not self.receiver_missing],
            getStops=lambda trip, limit: [NS(stoppingPlaceID=stop) for stop in self.plans[trip]],
            getPersonIDList=lambda trip: self.members[trip], getPersonCapacity=lambda trip: 50.,
            getLaneID=lambda trip: None if self.unknown and trip == "F" else "lane",
            getLanePosition=lambda trip: 25. if trip == "R" else 10.,
            getSpeed=lambda trip: None if self.unknown and trip == "F" else 1.,
            isAtBusStop=lambda trip: trip == "R")
        self.person = NS(getStage=self.stage)

    def stage(self, pid, index):
        self.stage_indices.append(index)
        if index != 0:
            raise AssertionError("Future stage access")
        if self.stage_missing:
            raise RuntimeError("Unknown current stage")
        return NS(type=3, line="F", destStop="D")


traci = None


def issued():
    return dict(trip_id="R", line_id="L", station_id="S", sequence=6, started_at=10.)


class CaptureTests(unittest.TestCase):
    def test_no_diagnostic_zero_cohort_and_absent_feeders_are_valid(self):
        global traci
        traci, bridge = Native(), Bridge()
        result = capture_context(bridge, issued(), source_group="new-day")
        self.assertEqual(result["capture"]["status"], "complete")
        self.assertEqual([f["trip_id"] for f in result["feeders"]], ["F", "empty"])
        self.assertTrue(all(f["cohort_ids"] == [] and f["x"]["current_leg_dest_count"] == 0 for f in result["feeders"]))
        self.assertEqual(result["receiver"]["onboard"], 1.)
        bound = bind_action(result, dict(trip_id="R", stop_id="S", accepted=True, applied_at=10., hold_seconds=0.))
        self.assertEqual(bound["action"]["hold_seconds"], 0.)
        traci.plans.update(F=["D"], empty=["T"])
        none = capture_context(bridge, issued(), source_group="new-day")
        self.assertEqual(none["feeders"], [])
        self.assertEqual(none["capture"]["status"], "complete")
        self.assertTrue(all(index == 0 for index in traci.stage_indices))

    def test_passive_unknown_and_missing_receiver_do_not_become_zero(self):
        global traci
        traci, bridge = Native(), Bridge()
        traci.stage_missing = traci.unknown = True
        before = deepcopy((traci.clock, traci.plans, traci.members, bridge.pending_events[("L", "R", "S")].assigned_action))
        result = capture_context(bridge, issued(), source_group="g")
        after = (traci.clock, traci.plans, traci.members, bridge.pending_events[("L", "R", "S")].assigned_action)
        self.assertEqual(before, after)
        feeder = result["feeders"][0]
        self.assertEqual(feeder["cohort_ids"], [])
        for key in ("current_leg_dest_count", "feeder_speed", "same_lane", "gap_m"):
            self.assertIsNone(feeder["x"][key])
        self.assertEqual(result["capture"]["status"], "missing")
        self.assertTrue(result["capture"]["checks"]["clock_unchanged"])
        self.assertEqual(result["capture"]["errors"][0]["field"], "p/current_stage0")
        traci.receiver_missing = True
        absent = capture_context(bridge, issued(), source_group="g")
        self.assertIsNone(absent["receiver"]["onboard"])
        self.assertIsNone(absent["receiver"]["capacity"])
        self.assertFalse(absent["capture"]["checks"]["receiver_native_present"])
        self.assertIn("receiver_native_vehicle", absent["capture"]["missing"])
        traci.clock = 11.
        with self.assertRaises(ValueError):
            capture_context(bridge, issued(), source_group="g")
        traci.clock = 10.
        bridge.pending_events[("L", "R", "S")].assigned_action = 0.
        with self.assertRaises(ValueError):
            capture_context(bridge, issued(), source_group="g")


if __name__ == "__main__":
    unittest.main()
