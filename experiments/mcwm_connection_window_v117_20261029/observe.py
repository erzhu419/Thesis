"""Passive, censored native observation of one fixed connection window."""
import json
from pathlib import Path
import sys

from experiments.mcwm_logged_action_observation_v116_20261029.observe import _guard, _checks


class WindowComplete(Exception):
    """The original native step and passenger ledger reached the window end."""


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


class WindowObserver:
    def __init__(self, bridge, out, root_capture, *, end_clock):
        self.bridge = bridge
        self.api = sys.modules[type(bridge).__module__].traci
        self.out, self.end_clock = Path(out), float(end_clock)
        issue = root_capture["summary"]["issue"]
        self.start_clock = float(issue["observed_clock"])
        self.receiver, self.station = issue["trip_id"], issue["station_id"]
        vehicles = root_capture["vehicles"]
        receiver = next(v for v in vehicles if v["receiver"])
        self.root_riders = {p["person_id"] for p in receiver["people"]}
        self.cohort = {}
        for vehicle in vehicles:
            if vehicle["receiver"]:
                continue
            for person in vehicle["people"]:
                if person["current_stage0"]["known"] and person["current_stage0"]["destStop"] == self.station:
                    self.cohort[person["person_id"]] = dict(root=person,
                        initial_feeder=vehicle["trip_id"], near=vehicle["next_stop_matches"],
                        feeder_exit_time=None, alighted_at_target_time=None,
                        receiver_board_time=None, receiver_board_at_target=None)
        self.feeder_ids = sorted({p["initial_feeder"] for p in self.cohort.values()})
        self.trips = [self.receiver] + self.feeder_ids
        self.person_ids = sorted(self.root_riders | set(self.cohort))
        self.arrivals = {trip: None for trip in self.feeder_ids}
        self.departure = None
        self.exchanges, self.commands, self.root_commands = [], [], []
        self.issues = {}
        self.errors = []
        self.checks = dict(root_capture_passed=root_capture["summary"]["all_pass"],
                           getters_unchanged=True, one_second_steps=True)
        self.steps, self.stopped = 0, False
        self.rider_ride = {pid: 0. for pid in self.root_riders}
        self.at_target_stop_person_seconds = 0.
        self.command_window_person_seconds = 0.
        self.root_rider_command_window_person_seconds = 0.
        self.writers = {name: (self.out/(name+".jsonl")).open("x")
                        for name in ("window_ticks", "actions", "commands")}
        self.previous = self._sample()
        self.root_ledger = self.previous["ledger"]
        self.checks["root_clock_exact"] = self.previous["clock"] == self.start_clock
        expected = {v["trip_id"]: sorted(p["person_id"] for p in v["people"])
                    for v in vehicles if v["trip_id"] in self.trips}
        self.checks["root_onboard_exact"] = all(self.previous["vehicles"][trip]["onboard"] == ids
                                                for trip, ids in expected.items())
        self.checks["receiver_at_target_at_root"] = self.previous["vehicles"][self.receiver]["at_target_stop"]
        self.arrival_left_censored = {trip: self.previous["vehicles"][trip]["at_target_stop"]
                                      for trip in self.feeder_ids}
        self._write("window_ticks", dict(self.previous, root=True))
        with (self.out/"window_root.json").open("x") as handle:
            json.dump(dict(issue=issue, cohort=self.cohort, receiver_initial_riders=sorted(self.root_riders),
                           feeder_ids=self.feeder_ids, end_clock=self.end_clock), handle, allow_nan=False)
        self._owned = {name: name in bridge.__dict__ for name in ("_advance_one_step", "apply_action")}
        self._advance, self._apply = bridge._advance_one_step, bridge.apply_action

        def advance():
            result = self._advance()
            current = self._sample()
            self._record_tick(current)
            if current["clock"] >= self.end_clock:
                self.stopped = current["clock"] == self.end_clock
                if not self.stopped:
                    raise RuntimeError("Native step overshot the fixed observation endpoint")
                raise WindowComplete()
            return result

        def apply(event, action):
            if str(event.bus_id) not in self.trips:
                return self._apply(event, action)
            now = float(self.api.simulation.getTime())
            setter = self.api.vehicle.setStopParameter
            calls = dict(called=0, succeeded=0, errors=[])

            def observed_setter(*args, **kwargs):
                calls["called"] += 1
                try:
                    result = setter(*args, **kwargs)
                except Exception as error:
                    calls["errors"].append(type(error).__name__ + ": " + str(error))
                    raise
                calls["succeeded"] += 1
                return result

            self.api.vehicle.setStopParameter = observed_setter
            try:
                result = self._apply(event, action)
            finally:
                self.api.vehicle.setStopParameter = setter
            key = (str(event.bus_id), str(event.stop_id))
            command = dict(trip_id=key[0], station_id=key[1], sequence=self.issues.pop(key, None),
                applied_at=now, hold_seconds=float(event.assigned_action),
                service_ready=event.metadata.get("service_completion_time"),
                accepted=calls["called"] == calls["succeeded"] == 1,
                setter_called=calls["called"], setter_succeeded=calls["succeeded"], setter_errors=calls["errors"])
            self.commands.append(command)
            if key == (self.receiver, self.station) and now == self.start_clock:
                self.root_commands.append(command)
            self._write("commands", command)
            return result

        bridge._advance_one_step, bridge.apply_action = advance, apply

    def _write(self, name, row):
        self.writers[name].write(json.dumps(clean(row), separators=(",", ":"), allow_nan=False)+"\n")

    def _sample(self):
        before = _guard(self.bridge, self.api)
        active = set(map(str, self.api.vehicle.getIDList()))
        people_active = set(map(str, self.api.person.getIDList()))
        vehicles, people = {}, {}
        for trip in self.trips:
            if trip not in active:
                vehicles[trip] = dict(active=False, onboard=[], at_target_stop=False, at_stop=False,
                    next_stop=None, road_id=None, lane_id=None, lane_position=None, speed=None)
                continue
            stops = self.api.vehicle.getStops(trip, 1)
            next_stop = stops[0].stoppingPlaceID if stops else None
            at_stop = bool(self.api.vehicle.isAtBusStop(trip))
            vehicles[trip] = dict(active=True, onboard=sorted(map(str, self.api.vehicle.getPersonIDList(trip))),
                at_stop=at_stop, at_target_stop=at_stop and next_stop == self.station, next_stop=next_stop,
                road_id=self.api.vehicle.getRoadID(trip), lane_id=self.api.vehicle.getLaneID(trip),
                lane_position=float(self.api.vehicle.getLanePosition(trip)), speed=float(self.api.vehicle.getSpeed(trip)))
        waiting = sorted(map(str, self.api.busstop.getPersonIDs(self.station)))
        for pid in self.person_ids:
            present = pid in people_active
            stage = None
            if present and pid in self.cohort:
                try:
                    current = self.api.person.getStage(pid, 0)
                    stage = {key: getattr(current, key) for key in ("type", "line", "destStop")}
                except Exception as error:
                    self.errors.append(dict(clock=before["clock"], person_id=pid, field="current_stage0",
                                            error=type(error).__name__))
            people[pid] = dict(native_active=present,
                vehicle=str(self.api.person.getVehicle(pid)) if present else None,
                waiting_at_target=pid in waiting, current_stage0=stage)
        after = _guard(self.bridge, self.api)
        self.checks["getters_unchanged"] &= all(_checks(before, after, before["clock"]).values())
        return dict(clock=after["clock"], ledger=after["ledger"], vehicles=vehicles,
                    people=people, target_waiting_ids=waiting)

    def issue(self, event):
        self.issues[(str(event["trip_id"]), str(event["station_id"]))] = int(event["sequence"])
        self._write("actions", event)

    def _record_tick(self, current):
        previous, end = self.previous, current["clock"]
        start = previous["clock"]
        dt = end-start
        self.checks["one_second_steps"] &= dt == 1.
        self.steps += 1
        receiver = previous["vehicles"][self.receiver]
        for pid in self.root_riders:
            if previous["people"][pid]["vehicle"]:
                self.rider_ride[pid] += dt
        if receiver["at_target_stop"]:
            self.at_target_stop_person_seconds += len(receiver["onboard"])*dt
            if len(self.root_commands) == 1 and self.root_commands[0]["accepted"]:
                command = self.root_commands[0]
                ready = command["service_ready"]
                if ready is not None:
                    left = max(start, command["applied_at"], float(ready))
                    right = min(end, command["applied_at"]+command["hold_seconds"])
                    duration = max(0., right-left)
                    self.command_window_person_seconds += duration*len(receiver["onboard"])
                    self.root_rider_command_window_person_seconds += duration*len(set(receiver["onboard"]) & self.root_riders)
        tick_exchanges = []
        for trip in self.trips:
            old, new = previous["vehicles"][trip], current["vehicles"][trip]
            if trip == self.receiver and old["at_target_stop"] and not new["at_target_stop"] and self.departure is None:
                self.departure = end
            if trip in self.arrivals and not old["at_target_stop"] and new["at_target_stop"] and self.arrivals[trip] is None:
                self.arrivals[trip] = end
            boarded, alighted = sorted(set(new["onboard"])-set(old["onboard"])), sorted(set(old["onboard"])-set(new["onboard"]))
            if boarded or alighted:
                event = dict(interval_start=start, observed_at=end, trip_id=trip,
                    at_target_stop_before=old["at_target_stop"], at_target_stop_after=new["at_target_stop"],
                    boarded_ids=boarded, alighted_ids=alighted)
                self.exchanges.append(event)
                tick_exchanges.append(event)
            for pid, outcome in self.cohort.items():
                if trip == outcome["initial_feeder"] and pid in alighted and outcome["feeder_exit_time"] is None:
                    outcome["feeder_exit_time"] = end
                    if old["at_target_stop"] or new["at_target_stop"] or pid in current["target_waiting_ids"]:
                        outcome["alighted_at_target_time"] = end
                if trip == self.receiver and pid in boarded and outcome["receiver_board_time"] is None:
                    outcome["receiver_board_time"] = end
                    outcome["receiver_board_at_target"] = old["at_target_stop"] or new["at_target_stop"]
        self._write("window_ticks", dict(current, exchanges=tick_exchanges))
        self.previous = current

    def finish(self):
        for writer in self.writers.values():
            writer.flush()
        end = self.previous["ledger"]
        wait = end["accrued_wait_seconds"]-self.root_ledger["accrued_wait_seconds"]
        ride = end["accrued_in_vehicle_seconds"]-self.root_ledger["accrued_in_vehicle_seconds"]
        self.checks.update(exact_window_end=self.stopped and self.previous["clock"] == self.end_clock,
            ledger_endpoint=end["time_seconds"] == self.end_clock,
            native_endpoint=float(self.api.simulation.getTime()) == self.end_clock,
            required_current_stage_reads=not self.errors,
            root_command_accepted=len(self.root_commands) == 1 and self.root_commands[0]["accepted"])
        outcomes = [dict(person_id=pid, initial_feeder=p["initial_feeder"], near=p["near"],
            feeder_exit_time=p["feeder_exit_time"], alighted_at_target_time=p["alighted_at_target_time"],
            receiver_board_time=p["receiver_board_time"], receiver_board_at_target=p["receiver_board_at_target"],
            receiver_board_right_censored=p["receiver_board_time"] is None,
            end_state=self.previous["people"][pid]) for pid, p in sorted(self.cohort.items())]
        summary = dict(status="complete" if all(self.checks.values()) else "HOLD", all_pass=all(self.checks.values()),
            checks=self.checks, start_clock=self.start_clock, end_clock=self.previous["clock"], window_steps=self.steps,
            receiver=self.receiver, station=self.station, cohort_size=len(self.cohort), near_cohort_size=sum(p["near"] for p in self.cohort.values()),
            receiver_initial_onboard=len(self.root_riders), receiver_departure_time=self.departure,
            receiver_departure_right_censored=self.departure is None,
            feeder_arrivals=[dict(trip_id=trip, arrival_time=self.arrivals[trip], left_censored=self.arrival_left_censored[trip],
                right_censored=self.arrivals[trip] is None and not self.arrival_left_censored[trip]) for trip in self.feeder_ids],
            cohort_outcomes=outcomes, cohort_boarded=sum(p["receiver_board_time"] is not None for p in self.cohort.values()),
            near_cohort_boarded=sum(p["near"] and p["receiver_board_time"] is not None for p in self.cohort.values()),
            global_window=dict(wait=wait, ride=ride, cost=2*wait+ride, unit="passenger_seconds"),
            receiver_initial_rider_ride_seconds=sum(self.rider_ride.values()),
            at_target_stop_person_seconds=self.at_target_stop_person_seconds,
            command_window_person_seconds=self.command_window_person_seconds,
            receiver_initial_rider_command_window_person_seconds=self.root_rider_command_window_person_seconds,
            root_command=self.root_commands[0] if len(self.root_commands) == 1 else None,
            receiver_boarding_events=sum(len(e["boarded_ids"]) for e in self.exchanges if e["trip_id"] == self.receiver),
            errors=self.errors, paths=dict(ticks="window_ticks.jsonl", actions="actions.jsonl", commands="commands.jsonl", root="window_root.json"),
            semantics="Native changes observed at t occurred in (t-1,t]; same-tick exchange and departure order is unresolved. None is unobserved/censored, not zero. Current stage0 is observed per tick; root additional intent is separate. Occupancy uses pre-step membership. Command-window person-seconds are descriptive, not causal removable cost, and are already included in global ledger cost. This is a bounded nonterminal window.")
        with (self.out/"observer_summary.json").open("x") as handle:
            json.dump(summary, handle, separators=(",", ":"), allow_nan=False)
        return summary

    def close(self):
        for name, value in (("_advance_one_step", self._advance), ("apply_action", self._apply)):
            if self._owned[name]:
                setattr(self.bridge, name, value)
            else:
                delattr(self.bridge, name)
        for writer in self.writers.values():
            writer.close()
