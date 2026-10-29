"""Deterministic fake-native checks of passive timing and exact accounting."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from observer import Observer, phases


class Ledger:
    def __init__(self):
        self.time_seconds = self.accrued_wait_seconds = self.accrued_in_vehicle_seconds = 0.
        self.waiting_passengers = self.in_vehicle_passengers = 0

    def advance_to(self, time):
        dt = time-self.time_seconds
        self.accrued_wait_seconds += self.waiting_passengers*dt
        self.accrued_in_vehicle_seconds += self.in_vehicle_passengers*dt
        self.time_seconds = time

    def set_counts(self, time, waiting, onboard):
        self.advance_to(time)
        self.waiting_passengers, self.in_vehicle_passengers = waiting, onboard

    def snapshot(self):
        return SimpleNamespace(**vars(self))


class Native:
    def __init__(self):
        self.now, self.step_calls = 0., 0
        self.reject_command = False
        self.simulation = SimpleNamespace(getTime=lambda: self.now)
        self.vehicle = SimpleNamespace(getPersonIDList=lambda bus: self.onboard(),
            isAtBusStop=lambda bus: self.now < 2, getSpeed=lambda bus: 8. if self.now == 2 else 0.,
            getStopState=lambda bus: 1 if self.now < 2 else 0,
            getStops=lambda bus, n: [SimpleNamespace(stoppingPlaceID="S")],
            getIDList=lambda: ["trip"] if self.now < 4 else [],
            setStopParameter=self.set_stop)
        self.busstop = SimpleNamespace(getPersonIDs=lambda stop: ["p2"] if self.now < 2 else [])

    def onboard(self):
        return (["p1"] if self.now < 2 else ["p1", "p2"]) if self.now < 4 else []

    def simulationStep(self):
        self.step_calls += 1
        self.now += 1.

    def set_stop(self, *args):
        if self.reject_command:
            raise ValueError("synthetic native rejection")


traci = None


class Bridge:
    def __init__(self):
        self.passenger_time_ledger = Ledger()
        self.active_bus_ids = {"trip"}
        self.bus_obj_dic = {"trip": SimpleNamespace(belong_line_id_s="L",
            just_server_stop_data_d={"S": [0., .5]}, arriver_stop_time_d={"S": 0.},
            depart_stop_time_d={}, bus_state_s="Stop")}
        self.stop_obj_dic, self.fleet_obj_dic = {"S": object()}, {"trip": SimpleNamespace(fleet_id="fleet")}
        self.departed_passenger_times = dict(p1=0., p2=0., p3=0.)
        self.completed_passenger_times = dict(p1=4., p2=4.)
        self.commands = []

    def _advance_one_step(self):
        now = traci.now
        self.passenger_time_ledger.set_counts(now, int(now < 2), len(traci.onboard()))
        bus = self.bus_obj_dic["trip"]
        if now == 2:
            bus.bus_state_s = "Edge"
            bus.depart_stop_time_d["S"] = now
        traci.simulationStep()
        self.passenger_time_ledger.advance_to(traci.now)

    def apply_action(self, event, hold):
        self.commands.append((traci.now, hold))
        try:
            traci.vehicle.setStopParameter(event.bus_id, 0, "duration", str(hold))
        except ValueError:
            pass  # Same continuation behavior as the production TraCIException handler.
        event.assigned_action = hold
        event.metadata["predicted_departure_time_seconds"] = .5+hold
        self.bus_obj_dic["trip"].just_server_stop_data_d["S"][1] = .5+hold


class ObserverTests(unittest.TestCase):
    def test_phase_boundaries_use_actual_apply_and_keep_protection_separate(self):
        parts = phases(0., 4., at_stop=True, speed=0., ready=.5, applied_at=1., hold=.25)
        self.assertEqual(parts, [(0., .5, "service", "base_service"),
            (.5, 1., "traffic_or_other", "protected_wait"),
            (1., 1.25, "holding", "applied_command"),
            (1.25, 4., "traffic_or_other", "after_hold")])
        self.assertEqual(phases(3., 4., at_stop=False, speed=0., ready=None,
            applied_at=None, hold=None), [(3., 4., "traffic_or_other", "off_stop")])

    def test_native_hook_identity_exchange_counts_and_ledger(self):
        global traci
        baseline_native, baseline = Native(), Bridge()
        traci = baseline_native
        event = lambda: SimpleNamespace(bus_id="trip", line_id="L", stop_id="S",
            metadata=dict(service_completion_time=.5))
        baseline._advance_one_step()
        baseline.apply_action(event(), .25)
        for _ in range(3):
            baseline._advance_one_step()
        expected_ledger, expected_commands = vars(baseline.passenger_time_ledger).copy(), baseline.commands[:]

        native, bridge = Native(), Bridge()
        traci = native
        with TemporaryDirectory() as folder:
            recorder = Observer(bridge, folder)
            original_step = native.simulationStep
            try:
                bridge._advance_one_step()
                recorder.issue(dict(sequence=0, trip_id="trip", vehicle_id="fleet", line_id="L", station_id="S",
                    a=[-.99], proposed_a=[-.9], started_at=1.))
                bridge.apply_action(event(), .25)
                for _ in range(3):
                    bridge._advance_one_step()
                result = recorder.finish()
                self.assertEqual(vars(bridge.passenger_time_ledger), expected_ledger)
                self.assertEqual(bridge.commands, expected_commands)
                self.assertEqual(native.step_calls, 4)
                self.assertEqual(native.simulationStep, original_step)
                self.assertEqual(result["status"], "complete", result)
                self.assertEqual(result["passenger_totals"], dict(wait=2., ride=6.,
                    ride_phases=dict(service=.5, holding=.25, moving=2., traffic_or_other=3.25)))
                people = {row["passenger_id"]: row for row in map(json.loads,
                    (Path(folder)/"passengers.jsonl").read_text().splitlines())}
                self.assertEqual(set(people), {"p1", "p2", "p3"})
                self.assertEqual([people[p]["ride"] for p in ("p1", "p2", "p3")], [4., 2., 0.])
                visits = list(map(json.loads, (Path(folder)/"visits.jsonl").read_text().splitlines()))
                self.assertEqual(visits[0]["departure_time"], 2.)
                self.assertEqual(visits[0]["service_ready_time"], .5)
                self.assertEqual(visits[0]["action_time"], 1.)
                exchanges = list(map(json.loads, (Path(folder)/"exchanges.jsonl").read_text().splitlines()))
                self.assertEqual(sum(len(row["boarded_ids"]) for row in exchanges), 2)
                self.assertEqual(sum(len(row["alighted_ids"]) for row in exchanges), 2)
                command = json.loads((Path(folder)/"commands.jsonl").read_text())
                self.assertEqual(command["sequence"], 0)
                self.assertTrue(command["accepted"])
            finally:
                recorder.close()

    def test_rejected_original_setter_is_reported_without_changing_continuation(self):
        global traci
        native, bridge = Native(), Bridge()
        traci = native
        with TemporaryDirectory() as folder:
            recorder = Observer(bridge, folder)
            try:
                bridge._advance_one_step()
                recorder.issue(dict(sequence=9, trip_id="trip", station_id="S"))
                event = SimpleNamespace(bus_id="trip", line_id="L", stop_id="S",
                    metadata=dict(service_completion_time=.5))
                bridge.apply_action(event, .1)
                native.reject_command = True
                original_setter = native.vehicle.setStopParameter
                bridge.apply_action(event, .5)  # No new policy issue: must not inherit seq9.
                self.assertEqual(native.vehicle.setStopParameter, original_setter)
                bridge._advance_one_step()
                result = recorder.finish()
                self.assertEqual(result["rejected_commands"], 1)
                self.assertEqual(result["passenger_totals"]["ride_phases"]["holding"], 0.)
                rows = list(map(json.loads, (Path(folder)/"commands.jsonl").read_text().splitlines()))
                self.assertEqual([row["sequence"] for row in rows], [9, None])
                self.assertFalse(rows[1]["accepted"])
                self.assertEqual(rows[1]["setter_called"], 1)
                self.assertEqual(rows[1]["setter_succeeded"], 0)
                self.assertIn("synthetic native rejection", rows[1]["setter_errors"][0])
            finally:
                recorder.close()
            self.assertNotIn("_advance_one_step", bridge.__dict__)
            self.assertNotIn("apply_action", bridge.__dict__)

    def test_fractional_phases_do_not_change_exact_integer_ride_accounting(self):
        global traci
        native, bridge = Native(), Bridge()
        traci = native
        with TemporaryDirectory() as folder:
            recorder = Observer(bridge, folder)
            try:
                bridge._advance_one_step()
                event = SimpleNamespace(bus_id="trip", line_id="L", stop_id="S",
                    metadata=dict(service_completion_time=.5))
                bridge.apply_action(event, .123456789)
                bridge._advance_one_step()
                result = recorder.finish()
                self.assertEqual(result["passenger_totals"]["ride"], 2.)
                self.assertTrue(result["checks"]["passenger_totals_exact"])
                self.assertTrue(result["checks"]["phase_totals_reconcile"])
            finally:
                recorder.close()


if __name__ == "__main__":
    unittest.main()
