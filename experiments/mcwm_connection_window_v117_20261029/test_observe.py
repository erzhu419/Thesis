"""Small fake native API fixtures: no SUMO, learner, or optimizer."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

from experiments.mcwm_connection_window_v117_20261029.observe import WindowObserver, WindowComplete


class Bridge:
    def __init__(self, api):
        self.api = api
        self.wait, self.ride = 100., 200.
        self.passenger_time_ledger = NS(snapshot=lambda: NS(time_seconds=api.clock,
            waiting_passengers=len(api.waiting()), in_vehicle_passengers=sum(len(api.members(t)) for t in ("R", "F", "G")),
            accrued_wait_seconds=self.wait, accrued_in_vehicle_seconds=self.ride))
        # A closure intentionally exercises the scenario-installed method shape.
        def advance():
            self.wait += len(api.waiting())
            self.ride += sum(len(api.members(t)) for t in ("R", "F", "G"))
            api.clock += 1
        self._advance_one_step = advance

    def apply_action(self, event, action):
        try:
            self.api.vehicle.setStopParameter(event.bus_id, 0, "duration", str(action))
        except RuntimeError:
            pass  # Native bridge also preserves the assigned action on rejection.
        event.assigned_action = action


class FakeAPI:
    def __init__(self, *, exchange=True, reject=False):
        self.clock, self.exchange, self.reject = 10., exchange, reject
        self.setter_calls = 0
        self.simulation = NS(getTime=lambda: self.clock)
        self.vehicle = NS(getIDList=lambda: ("R", "F", "G"), getPersonIDList=self.members,
            getStops=lambda trip, limit: [NS(stoppingPlaceID="S" if self.at_stop(trip) or trip != "R" else "D")],
            isAtBusStop=self.at_stop, getRoadID=lambda trip: "edge", getLaneID=lambda trip: "lane",
            getLanePosition=lambda trip: self.clock, getSpeed=lambda trip: 0. if self.at_stop(trip) else 1.,
            setStopParameter=self.set_stop)
        self.person = NS(getIDList=lambda: ("r", "p", "q"), getVehicle=self.person_vehicle, getStage=self.stage)
        self.busstop = NS(getPersonIDs=lambda station: self.waiting())

    def at_stop(self, trip):
        return (trip == "R" and (not self.exchange or self.clock < 13)) or (trip == "F" and self.exchange and self.clock >= 11)

    def members(self, trip):
        if trip == "R":
            return ["r", "p"] if self.exchange and self.clock >= 13 else ["r"]
        if trip == "F":
            return [] if self.exchange and self.clock >= 12 else ["p"]
        return ["q"]

    def waiting(self):
        return ["p"] if self.exchange and self.clock == 12 else []

    def person_vehicle(self, pid):
        return next((trip for trip in ("R", "F", "G") if pid in self.members(trip)), "")

    def stage(self, pid, index):
        assert index == 0
        vehicle = self.person_vehicle(pid)
        return NS(type=3 if vehicle else 2, line=vehicle, destStop="D" if vehicle == "R" else "S")

    def set_stop(self, *args):
        self.setter_calls += 1
        if self.reject:
            raise RuntimeError("rejected test command")


def root_capture():
    def person(pid, trip):
        return dict(person_id=pid, actual_onboard_trip=trip,
            current_stage0=dict(known=True, type=3, line=trip, destStop="D" if trip == "R" else "S"),
            additional_intent=dict(source="additional_preloaded_journey_intent", next_leg_permitted_lines=None))
    return dict(summary=dict(all_pass=True, issue=dict(observed_clock=10., trip_id="R", station_id="S")),
        vehicles=[dict(trip_id=trip, receiver=trip == "R", next_stop_matches=trip == "F",
            people=[person(pid, trip)]) for trip, pid in (("R", "r"), ("F", "p"), ("G", "q"))])


class ObservationTests(unittest.TestCase):
    def run_window(self, path, *, exchange=True, reject=False):
        global traci
        traci = FakeAPI(exchange=exchange, reject=reject)
        bridge = Bridge(traci)
        original_advance = bridge._advance_one_step
        original_setter = traci.vehicle.setStopParameter
        observer = WindowObserver(bridge, path, root_capture(), end_clock=13.)
        observer.issue(dict(trip_id="R", station_id="S", sequence=7, proposed_a=[0.1], a=[0.2]))
        event = NS(bus_id="R", stop_id="S", assigned_action=None, metadata={"service_completion_time": 9.})
        bridge.apply_action(event, 2.5)
        self.assertIs(traci.vehicle.setStopParameter, original_setter)
        bridge._advance_one_step()
        bridge._advance_one_step()
        with self.assertRaises(WindowComplete):
            bridge._advance_one_step()
        self.assertEqual(traci.clock, 13.)
        result = observer.finish()
        observer.close()
        self.assertIs(bridge._advance_one_step, original_advance)
        self.assertNotIn("apply_action", bridge.__dict__)
        return result, traci

    def test_exact_post_ledger_stop_and_actual_id_exchanges(self):
        with tempfile.TemporaryDirectory() as directory:
            result, api = self.run_window(Path(directory))
            self.assertTrue(result["all_pass"])
            self.assertEqual(result["global_window"], dict(wait=1., ride=8., cost=10., unit="passenger_seconds"))
            self.assertEqual(result["receiver_departure_time"], 13.)
            p, q = result["cohort_outcomes"]
            self.assertEqual((p["feeder_exit_time"], p["alighted_at_target_time"], p["receiver_board_time"]), (12., 12., 13.))
            self.assertEqual(p["end_state"]["current_stage0"]["line"], "R")
            self.assertIsNone(q["receiver_board_time"])
            self.assertEqual((result["cohort_size"], result["near_cohort_size"], result["cohort_boarded"]), (2, 1, 1))
            self.assertEqual(result["command_window_person_seconds"], 2.5)
            self.assertEqual(result["at_target_stop_person_seconds"], 3.)
            self.assertEqual(result["receiver_initial_rider_ride_seconds"], 3.)
            rows = [json.loads(line) for line in (Path(directory)/"window_ticks.jsonl").read_text().splitlines()]
            self.assertEqual([r["clock"] for r in rows], [10., 11., 12., 13.])
            self.assertEqual(rows[2]["exchanges"][0]["alighted_ids"], ["p"])
            self.assertEqual(rows[3]["exchanges"][0]["boarded_ids"], ["p"])
            root = json.loads((Path(directory)/"window_root.json").read_text())
            self.assertIsNone(root["cohort"]["p"]["root"]["additional_intent"]["next_leg_permitted_lines"])
            self.assertEqual(api.setter_calls, 1)

    def test_unobserved_events_are_censored_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _ = self.run_window(Path(directory), exchange=False)
            self.assertTrue(result["all_pass"])
            self.assertIsNone(result["receiver_departure_time"])
            self.assertTrue(result["receiver_departure_right_censored"])
            self.assertTrue(all(p["receiver_board_right_censored"] for p in result["cohort_outcomes"]))
            self.assertTrue(all(p["arrival_time"] is None and p["right_censored"] for p in result["feeder_arrivals"]))
            self.assertEqual(result["cohort_boarded"], 0)

    def test_rejected_setter_retains_original_behavior_and_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            result, api = self.run_window(Path(directory), reject=True)
            self.assertFalse(result["all_pass"])
            self.assertEqual(result["root_command"]["hold_seconds"], 2.5)
            self.assertFalse(result["root_command"]["accepted"])
            self.assertEqual(result["root_command"]["setter_succeeded"], 0)
            self.assertEqual(result["command_window_person_seconds"], 0.)
            self.assertEqual(api.setter_calls, 1)


if __name__ == "__main__":
    unittest.main()
