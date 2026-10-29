"""Synthetic prefix/suffix accounting checks; no native runtime or optimizer."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest

from experiments.mcwm_connection_full_day_v118_20261029.accounting import Accounting


class Ledger:
    def __init__(self):
        self.time_seconds = self.accrued_wait_seconds = self.accrued_in_vehicle_seconds = 0.
        self.waiting_passengers = self.in_vehicle_passengers = 0

    def advance_to(self, time):
        delta = time-self.time_seconds
        self.accrued_wait_seconds += delta*self.waiting_passengers
        self.accrued_in_vehicle_seconds += delta*self.in_vehicle_passengers
        self.time_seconds = time

    def snapshot(self):
        return NS(**vars(self))


class Native:
    def __init__(self):
        self.now = 0.
        self.simulation = NS(getTime=lambda: self.now)
        self.vehicle = NS(getIDList=lambda: ["R", "F"], getPersonIDList=self.onboard,
            isAtBusStop=lambda trip: self.now < 2, getSpeed=lambda trip: 0. if self.now < 2 else 5.,
            getStopState=lambda trip: int(self.now < 2),
            getStops=lambda trip, n: [NS(stoppingPlaceID="S")], setStopParameter=lambda *args: None)
        self.busstop = NS(getPersonIDs=lambda stop: ["w", "target"] if self.now == 2 else ["w"])

    def onboard(self, trip):
        if trip == "F":
            return ["target"] if self.now < 2 else []
        return (["r"] if self.now < 4 else []) + (["target"] if self.now >= 3 else [])

    def simulationStep(self):
        self.now += 1.


traci = None


class Bridge:
    def __init__(self):
        self.passenger_time_ledger = Ledger()
        self.active_bus_ids = {"R", "F"}
        self.bus_obj_dic = {trip: NS(belong_line_id_s=trip,
            just_server_stop_data_d={"S": [0., .5]}, arriver_stop_time_d={"S": 0.},
            depart_stop_time_d={}) for trip in ("R", "F")}
        self.stop_obj_dic, self.fleet_obj_dic = {"S": object()}, {}
        self.departed_passenger_times = dict(r=0., target=0., w=0.)
        self.completed_passenger_times = {}

    def _advance_one_step(self):
        ledger = self.passenger_time_ledger
        ledger.waiting_passengers = len(traci.busstop.getPersonIDs("S"))
        ledger.in_vehicle_passengers = sum(len(traci.onboard(trip)) for trip in ("R", "F"))
        traci.simulationStep()
        ledger.advance_to(traci.now)
        if traci.now == 4:
            self.completed_passenger_times["r"] = 4.

    def apply_action(self, event, action):
        traci.vehicle.setStopParameter(event.bus_id, 0, "duration", str(action))
        event.assigned_action = action


def capture():
    return dict(summary=dict(all_pass=True, issue=dict(observed_clock=1.)), vehicles=[
        dict(receiver=True, trip_id="R", people=[dict(person_id="r")]),
        dict(receiver=False, trip_id="F", people=[dict(person_id="target")])])


class AccountingTests(unittest.TestCase):
    def run_day(self, root, *, detach):
        global traci
        traci, bridge = Native(), Bridge()
        observer = Accounting(bridge, root/"parent")
        try:
            bridge._advance_one_step()
            binding = observer.bind_root(capture(), diagnostic_person_id="target")
            self.assertEqual(binding["whole_network"], dict(W=1., R=2., G=4.))
            people, visits, previous, total = observer.people, observer.visits, observer.previous_onboard, observer.total
            if detach:
                observer.flush()
                manifest = observer.detach(root/"child")
                self.assertEqual(manifest["prefix_clock"], 1.)
                self.assertIs(observer.people, people)
                self.assertIs(observer.visits, visits)
                self.assertIs(observer.previous_onboard, previous)
                self.assertIs(observer.total, total)
            observer.issue(dict(sequence=0, trip_id="R", station_id="S"))
            bridge.apply_action(NS(bus_id="R", line_id="R", stop_id="S", metadata={"service_completion_time": .5}), .25)
            for _ in range(3):
                bridge._advance_one_step()
            result = observer.finish()
            self.assertTrue(result["all_pass"], result)
            return result, observer.out
        finally:
            observer.close()

    def test_prefix_and_suffix_preserve_exact_full_day_accounts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            split, out = self.run_day(root/"split", detach=True)
            continuous, _ = self.run_day(root/"continuous", detach=False)
            for key in ("whole_network", "groups", "root_receiver_riders", "diagnostic_target"):
                self.assertEqual(split[key], continuous[key])
            self.assertEqual(split["whole_network"]["full_day"], dict(W=5., R=7., G=17.))
            self.assertEqual(split["whole_network"]["after_root"], dict(W=4., R=5., G=13.))
            self.assertEqual(len((root/"split"/"parent"/"ticks.jsonl").read_text().splitlines()), 1)
            self.assertEqual(len((out/"ticks.jsonl").read_text().splitlines()), 3)
            self.assertEqual(split["root_receiver_riders"][0]["full_day"], dict(W=0., R=4., G=4.))
            self.assertEqual(split["root_receiver_riders"][0]["after_root"], dict(W=0., R=3., G=3.))
            self.assertEqual(split["root_receiver_riders"][0]["completion_time"], 4.)

    def test_incomplete_costs_disjoint_groups_and_native_endpoint(self):
        with TemporaryDirectory() as directory:
            result, out = self.run_day(Path(directory), detach=True)
            target = result["diagnostic_target"]
            self.assertFalse(target["completed"])
            self.assertIsNone(target["completion_time"])
            self.assertEqual(target["full_day"], dict(W=1., R=3., G=5.))
            self.assertEqual(target["after_root"], dict(W=1., R=2., G=4.))
            self.assertTrue(target["native_onboard_at_end"])
            self.assertEqual(result["groups"]["other_passengers"]["full_day"], dict(W=4., R=0., G=8.))
            self.assertEqual(sum(g["passengers"] for g in result["groups"].values()), 3)
            for period in ("full_day", "before_root", "after_root"):
                for field in ("W", "R", "G"):
                    self.assertEqual(sum(g[period][field] for g in result["groups"].values()),
                                     result["whole_network"][period][field])
            endpoint = json.loads((out/"native_endpoint.json").read_text())
            self.assertEqual(endpoint["waiting_ids"], ["w"])
            self.assertEqual(endpoint["onboard_ids"], ["target"])
            self.assertEqual(result["whole_network"]["native_onboard_at_end"], 1)
            self.assertEqual(result["whole_network"]["ledger_onboard_at_end"], 2)


if __name__ == "__main__":
    unittest.main()
