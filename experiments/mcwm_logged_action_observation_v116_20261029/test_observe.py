"""Fake current-state reads only; no SUMO, learner or optimizer."""
from copy import deepcopy
import random
import sys
from types import SimpleNamespace
import unittest

from observe import capture


class CurrentPeople(dict):
    def __iter__(self):
        raise AssertionError("Must not enumerate preloaded future passenger demand")


class Bridge:
    pass


def fixture():
    global traci
    clock = [100.]
    stops = {"R": ["T", "END"], "F1": ["T", "ELSE"], "F2": ["BEFORE", "T"],
             "UNRELATED": ["OTHER"], "UNBORN": ["T"]}
    memberships = {"R": ["receiver_person"], "F1": ["target", "second_leg"],
                   "F2": ["another"], "UNRELATED": ["unrelated"]}
    stages = {"receiver_person": SimpleNamespace(type=3, line="R", destStop="END"),
        "target": SimpleNamespace(type=3, line="F1 F1_ALT", destStop="T"),
        "second_leg": SimpleNamespace(type=3, line="F1", destStop="ELSE"),
        "another": SimpleNamespace(type=3, line="F2", destStop="T")}
    stage_calls = []

    def stage(person, index):
        stage_calls.append((person, index))
        if index != 0:
            raise AssertionError("Future stages are outside the information contract")
        return stages[person]

    traci = SimpleNamespace(simulation=SimpleNamespace(getTime=lambda: clock[0]),
        vehicle=SimpleNamespace(getIDList=lambda: ["R", "F1", "F2", "UNRELATED", "background_car"],
            getStops=lambda bus, limit: [SimpleNamespace(stoppingPlaceID=s) for s in stops[bus]],
            getLine=lambda bus: bus, getPersonIDList=lambda bus: memberships[bus],
            getPersonCapacity=lambda bus: 50., getRoadID=lambda bus: "E", getLaneID=lambda bus: "E_0",
            getLanePosition=lambda bus: 7., getDistance=lambda bus: 1000., getSpeed=lambda bus: 0. if bus == "R" else 3.,
            isAtBusStop=lambda bus: bus == "R"), person=SimpleNamespace(getStage=stage))
    bridge = Bridge()
    bridge.bus_obj_dic = {bus: SimpleNamespace() for bus in stops}
    # Deliberately different old ledger counts: this is allowed at issue time.
    snapshot = SimpleNamespace(time_seconds=100., waiting_passengers=99, in_vehicle_passengers=700,
                               accrued_wait_seconds=500., accrued_in_vehicle_seconds=900.)
    bridge.passenger_time_ledger = SimpleNamespace(snapshot=lambda: snapshot)
    bridge.pending_events = {("R", "R", "T"): SimpleNamespace(assigned_action=None,
        sim_time=99., metadata={"service_completion_time": 99.})}
    bridge.passenger_obj_dic = CurrentPeople(
        receiver_person=SimpleNamespace(transfer_stop_id_s="", end_stop_id_s="END"),
        target=SimpleNamespace(transfer_stop_id_s="T", end_stop_id_s="END"),
        second_leg=SimpleNamespace(transfer_stop_id_s="T", end_stop_id_s="ELSE"),
        another=SimpleNamespace(transfer_stop_id_s="", end_stop_id_s="T"),
        unborn_person=SimpleNamespace(transfer_stop_id_s="T", end_stop_id_s="END"))
    issued = dict(trip_id="R", line_id="R", station_id="T", started_at=100., sequence=12)
    return bridge, issued, clock, stage_calls


class ObserveTests(unittest.TestCase):
    def test_current_stage_all_feeders_and_separate_planned_intent(self):
        bridge, issued, _, stage_calls = fixture()
        before = deepcopy(vars(bridge.passenger_time_ledger.snapshot()))
        result = capture(bridge, issued, diagnostic_person_id="target")
        summary = result["summary"]
        self.assertTrue(summary["all_pass"])
        self.assertEqual(summary["issue"]["event_clock"], 99.)
        self.assertEqual(summary["issue"]["observed_clock"], 100.)
        self.assertEqual(summary["coverage"]["related_feeder_vehicles"], 2)
        self.assertEqual(summary["coverage"]["current_leg_dest_receiver_observed"], 2)
        self.assertEqual(summary["coverage"]["immediate_current_leg_dest_receiver_observed"], 1)
        self.assertEqual(summary["coverage"]["planned_receiver_compatible_observed"], 1)
        self.assertEqual(set(stage_calls), {("receiver_person", 0), ("target", 0), ("second_leg", 0), ("another", 0)})
        second = result["full"]["vehicles"][0]["people"][1]
        self.assertEqual(second["person_id"], "second_leg")
        self.assertFalse(second["additional_intent"]["first_leg_transfer_at_receiver"])
        self.assertIsNone(second["additional_intent"]["next_leg_permitted_lines"])
        self.assertEqual(vars(bridge.passenger_time_ledger.snapshot()), before)

    def test_getter_time_or_rng_mutation_is_not_called_passive(self):
        global traci
        for mutation in ("time", "rng"):
            with self.subTest(mutation=mutation):
                bridge, issued, clock, _ = fixture()
                saved = random.getstate()
                def speed(bus):
                    if mutation == "time":
                        clock[0] += 1
                    else:
                        random.random()
                    return 0.
                traci.vehicle.getSpeed = speed
                try:
                    summary = capture(bridge, issued, diagnostic_person_id="target")["summary"]
                finally:
                    random.setstate(saved)
                self.assertEqual(summary["status"], "HOLD")
                key = "native_clock_unchanged" if mutation == "time" else "python_rng_unchanged"
                self.assertFalse(summary["checks"][key])

    def test_missing_current_stage_retains_partial_counts_and_fails_gate(self):
        bridge, issued, _, _ = fixture()
        original = traci.person.getStage
        def stage(person, index):
            if person == "another":
                raise RuntimeError("Current stage unavailable")
            return original(person, index)
        traci.person.getStage = stage
        summary = capture(bridge, issued, diagnostic_person_id="target")["summary"]
        self.assertEqual(summary["status"], "HOLD")
        self.assertTrue(summary["counts_are_partial"])
        self.assertEqual(summary["coverage"]["current_stage_known"], 2)
        self.assertFalse(summary["checks"]["required_reads_complete"])

    def test_optional_location_missing_is_reported_but_missing_target_is_not_ready(self):
        bridge, issued, _, _ = fixture()
        def road(bus):
            if bus == "F1":
                raise RuntimeError("Road location unavailable")
            return "E"
        traci.vehicle.getRoadID = road
        result = capture(bridge, issued, diagnostic_person_id="target")
        self.assertTrue(result["summary"]["all_pass"])
        self.assertGreater(result["summary"]["error_count"], 0)
        self.assertIsNone(result["full"]["vehicles"][0]["location"]["road_id"])
        missing = capture(bridge, issued, diagnostic_person_id="not_onboard")["summary"]
        self.assertEqual(missing["status"], "HOLD")
        self.assertFalse(missing["checks"]["diagnostic_current_onboard_stage_observed"])


if __name__ == "__main__":
    unittest.main()
