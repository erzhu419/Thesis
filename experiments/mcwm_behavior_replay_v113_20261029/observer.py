"""Passive native-tick passenger accounting for the unchanged v111 run."""

from collections import defaultdict
import json
import math
from pathlib import Path


PHASES = ("service", "holding", "moving", "traffic_or_other")
WINDOWS = {"0_9000": (0., 9000.), "9000_12000": (9000., 12000.),
           "12000_18300": (12000., 18300.), "18300_24000": (18300., 24000.),
           "12600_12900": (12600., 12900.)}


def totals():
    return dict(wait=0., ride=0., ride_phases=dict.fromkeys(PHASES, 0.))


def account():
    return dict(**totals(), periods={name: totals() for name in WINDOWS})


def add(row, start, end, *, waiting=False, phase=None, count=1, phase_only=False):
    field, duration = ("wait" if waiting else "ride"), (end-start)*count
    if not phase_only:
        row[field] += duration
    if phase is not None:
        row["ride_phases"][phase] += duration
    for name, (low, high) in WINDOWS.items():
        amount = max(0., min(end, high)-max(start, low))*count
        if not phase_only:
            row["periods"][name][field] += amount
        if phase is not None:
            row["periods"][name]["ride_phases"][phase] += amount


def phases(start, end, *, at_stop, speed, ready, applied_at, hold, command_accepted=True):
    """Partition this tick; only an active applied command counts as holding."""
    if not at_stop:
        return [(start, end, "moving" if speed > 0 else "traffic_or_other", "off_stop")]
    cuts = {start, end}
    for point in (ready, applied_at, None if applied_at is None or hold is None else applied_at+hold):
        if point is not None and start < point < end:
            cuts.add(point)
    parts = []
    ordered = sorted(cuts)
    for left, right in zip(ordered, ordered[1:]):
        middle = (left+right)/2
        if ready is not None and middle < ready:
            phase, reason = "service", "base_service"
        elif (command_accepted and ready is not None and applied_at is not None and hold is not None
              and middle >= max(ready, applied_at) and middle < applied_at+hold):
            phase, reason = "holding", "applied_command"
        else:
            phase = "traffic_or_other"
            reason = ("base_ready_unknown" if ready is None else
                      "rejected_command" if applied_at is not None and not command_accepted else
                      "protected_wait" if applied_at is None or middle < applied_at else "after_hold")
        parts.append((left, right, phase, reason))
    return parts


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


class Observer:
    def __init__(self, bridge, out):
        self.bridge, self.out = bridge, Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        names = ("actions", "commands", "ticks", "exchanges", "waiting_changes", "visit_events")
        self.writers = {name: (self.out / (name+".jsonl")).open("x") for name in names}
        self.people, self.visits, self.current = {}, [], {}
        self.previous_onboard, self.previous_waiting = {}, set()
        self.visit_counts, self.issue_sequence = defaultdict(int), {}
        self.total, self.reasons = account(), defaultdict(float)
        self.start = self.end = None
        self.ticks = self.issues = self.commands = 0
        self.rejected_commands = 0
        self.checks = dict(count_ids_exact=True, unique_onboard_ids=True,
            waiting_onboard_disjoint=True, ledger_exact=True, tick_contiguous=True)
        self.maximum_count_gap = self.maximum_ledger_gap = 0.
        self._advance, self._apply = bridge._advance_one_step, bridge.apply_action
        self._advance_owned = "_advance_one_step" in bridge.__dict__
        self._apply_owned = "apply_action" in bridge.__dict__
        self.finished = self.closed = False

        def advance():
            # reset chooses libsumo in this actual method's module, possibly
            # imported as sumo_env rather than bus_h2o.sumo_env.
            api = self._advance.__func__.__globals__["traci"]
            native_step = api.simulationStep
            captured = []

            def step(*args, **kwargs):
                sample = self._sample(api)
                result = native_step(*args, **kwargs)
                captured.append((sample, float(api.simulation.getTime())))
                return result

            api.simulationStep = step
            try:
                result = self._advance()
            finally:
                api.simulationStep = native_step
            for sample, end in captured:
                self._integrate(sample, end)
            if captured:
                self._check_ledger()
            return result

        def apply(event, action):
            api = self._advance.__func__.__globals__["traci"]
            now = float(api.simulation.getTime())
            native_setter = api.vehicle.setStopParameter
            setter = dict(called=0, succeeded=0, errors=[])

            def set_stop(*args, **kwargs):
                setter["called"] += 1
                try:
                    value = native_setter(*args, **kwargs)
                except Exception as error:
                    setter["errors"].append(type(error).__name__+": "+str(error))
                    raise
                setter["succeeded"] += 1
                return value

            api.vehicle.setStopParameter = set_stop
            try:
                result = self._apply(event, action)
            finally:
                api.vehicle.setStopParameter = native_setter
            bus_id = str(event.bus_id)
            visit = self.current.get(bus_id)
            if visit is None or visit["stop_id"] != str(event.stop_id):
                raise RuntimeError("Applied command has no observed matching stop visit")
            hold = float(event.assigned_action)
            accepted = setter["called"] == setter["succeeded"] == 1
            visit.update(action_time=now, command_hold_seconds=hold,
                         command_accepted=accepted,
                         service_ready_time=float(event.metadata["service_completion_time"]))
            self.commands += 1
            self.rejected_commands += not accepted
            self._write("commands", dict(trip_id=bus_id, vehicle_id=visit["vehicle_id"],
                line_id=str(event.line_id), stop_id=str(event.stop_id), visit_index=visit["visit_index"],
                sequence=self.issue_sequence.pop((bus_id, str(event.stop_id), visit["visit_index"]), None), applied_at=now,
                requested_additional_hold_seconds=hold,
                accepted=accepted, setter_called=setter["called"], setter_succeeded=setter["succeeded"],
                setter_errors=setter["errors"],
                service_ready_time=visit["service_ready_time"],
                bridge_predicted_departure=event.metadata.get("predicted_departure_time_seconds")))
            return result

        bridge._advance_one_step, bridge.apply_action = advance, apply

    def _write(self, name, row):
        self.writers[name].write(json.dumps(clean(row), separators=(",", ":"), allow_nan=False)+"\n")

    def _person(self, person):
        person = str(person)
        if person not in self.people:
            self.people[person] = account()
        return self.people[person]

    def _new_visit(self, bus_id, stop_id, now, bus):
        self.visit_counts[bus_id] += 1
        fleet = self.bridge.fleet_obj_dic.get(bus_id)
        visit = dict(trip_id=bus_id, vehicle_id=str(fleet.fleet_id) if fleet else bus_id,
            line_id=str(bus.belong_line_id_s), stop_id=stop_id, visit_index=self.visit_counts[bus_id],
            arrival_time=now if stop_id is not None else None, service_ready_time=None,
            legacy_arrival_time=bus.arriver_stop_time_d.get(stop_id), legacy_departure_time=None,
            action_time=None, command_hold_seconds=None, command_accepted=None, departure_time=None,
            segment_start=now, segment_end=None, boarded_ids=[], alighted_ids=[], **account())
        self.visits.append(visit)
        self.current[bus_id] = visit
        self._write("visit_events", dict(event="arrival" if stop_id is not None else "route_start",
            time=now, trip_id=bus_id, line_id=visit["line_id"], stop_id=stop_id,
            visit_index=visit["visit_index"]))
        return visit

    def _visit(self, api, bus_id, now, at_stop):
        bus = self.bridge.bus_obj_dic[bus_id]
        visit = self.current.get(bus_id)
        if at_stop:
            stops = api.vehicle.getStops(bus_id, 1)
            stop_id = str(stops[0].stoppingPlaceID) if stops else None
            if stop_id is None:
                raise RuntimeError("Native stopped bus has no physical stopping place")
            if visit is None or visit["stop_id"] != stop_id or visit["departure_time"] is not None:
                if visit is not None:
                    visit["segment_end"] = now
                visit = self._new_visit(bus_id, stop_id, now, bus)
            if visit["service_ready_time"] is None:
                timing = bus.just_server_stop_data_d.get(stop_id)
                if timing is not None:
                    visit["service_ready_time"] = float(timing[1])
        elif visit is None:
            visit = self._new_visit(bus_id, None, now, bus)
        elif visit["stop_id"] is not None and visit["departure_time"] is None:
            visit["departure_time"] = now
            visit["legacy_departure_time"] = bus.depart_stop_time_d.get(visit["stop_id"])
            self._write("visit_events", dict(event="departure", time=now, trip_id=bus_id,
                line_id=visit["line_id"], stop_id=visit["stop_id"], visit_index=visit["visit_index"]))
        return visit

    def _sample(self, api, *, final=False):
        now = float(api.simulation.getTime())
        active = set(self.bridge.active_bus_ids)
        if final:
            active &= set(api.vehicle.getIDList())
        waiting_list = [str(person) for stop in sorted(self.bridge.stop_obj_dic)
                        for person in api.busstop.getPersonIDs(stop)]
        waiting = set(waiting_list)
        entered, left = sorted(waiting-self.previous_waiting), sorted(self.previous_waiting-waiting)
        if entered or left:
            self._write("waiting_changes", dict(time=now, entered=entered, left=left))
        self.previous_waiting = waiting
        vehicles, all_onboard, observed = [], [], {}
        for bus_id in sorted(active):
            bus_id = str(bus_id)
            ids = set(map(str, api.vehicle.getPersonIDList(bus_id)))
            observed[bus_id] = ids
            all_onboard.extend(ids)
            at_stop = bool(api.vehicle.isAtBusStop(bus_id))
            speed = float(api.vehicle.getSpeed(bus_id))
            native_stop_state = int(api.vehicle.getStopState(bus_id))
            visit = self._visit(api, bus_id, now, at_stop)
            before = self.previous_onboard.get(bus_id, set())
            boarded, alighted = sorted(ids-before), sorted(before-ids)
            if boarded or alighted:
                visit["boarded_ids"].extend(boarded)
                visit["alighted_ids"].extend(alighted)
                self._write("exchanges", dict(time=now, trip_id=bus_id, line_id=visit["line_id"],
                    stop_id=visit["stop_id"] if at_stop else None, visit_index=visit["visit_index"],
                    boarded_ids=boarded, alighted_ids=alighted, observed_at_stop=at_stop))
            vehicles.append(dict(trip_id=bus_id, line_id=visit["line_id"], visit_index=visit["visit_index"],
                stop_id=visit["stop_id"] if at_stop else None, at_stop=at_stop, speed=speed,
                native_stop_state=native_stop_state,
                ids=ids, visit=visit))
        for bus_id in self.previous_onboard.keys()-observed.keys():
            visit = self.current[bus_id]
            ids = sorted(self.previous_onboard[bus_id])
            if ids:
                visit["alighted_ids"].extend(ids)
                self._write("exchanges", dict(time=now, trip_id=bus_id, line_id=visit["line_id"],
                    stop_id=None, visit_index=visit["visit_index"], boarded_ids=[], alighted_ids=ids,
                    observed_at_stop=False, reason="vehicle_no_longer_active"))
            visit["segment_end"] = now
            self._write("visit_events", dict(event="route_end", time=now, trip_id=bus_id,
                line_id=visit["line_id"], stop_id=visit["stop_id"], visit_index=visit["visit_index"]))
        self.previous_onboard = observed
        self.checks["unique_onboard_ids"] &= len(all_onboard) == len(set(all_onboard))
        self.checks["waiting_onboard_disjoint"] &= not bool(waiting.intersection(all_onboard))
        if not final:
            snapshot = self.bridge.passenger_time_ledger.snapshot()
            gap = max(abs(len(waiting)-snapshot.waiting_passengers),
                      abs(len(all_onboard)-snapshot.in_vehicle_passengers), len(waiting_list)-len(waiting))
            self.maximum_count_gap = max(self.maximum_count_gap, gap)
            self.checks["count_ids_exact"] &= gap == 0
        return dict(time=now, waiting=waiting, vehicles=vehicles)

    def _integrate(self, sample, end):
        start = sample["time"]
        if end <= start:
            raise RuntimeError("Native tick must advance time")
        if self.start is None:
            self.start = start
        self.checks["tick_contiguous"] &= self.end is None or self.end == start
        self.end, self.ticks = end, self.ticks+1
        for person in sample["waiting"]:
            add(self._person(person), start, end, waiting=True)
        add(self.total, start, end, waiting=True, count=len(sample["waiting"]))
        tick_vehicles = []
        for vehicle in sample["vehicles"]:
            visit, ids = vehicle["visit"], vehicle["ids"]
            for person in ids:
                add(self._person(person), start, end)
            add(visit, start, end, count=len(ids))
            add(self.total, start, end, count=len(ids))
            segments = phases(start, end, at_stop=vehicle["at_stop"], speed=vehicle["speed"],
                ready=visit["service_ready_time"], applied_at=visit["action_time"], hold=visit["command_hold_seconds"],
                command_accepted=visit["command_accepted"] is not False)
            for low, high, phase, reason in segments:
                for person in ids:
                    add(self._person(person), low, high, phase=phase, phase_only=True)
                add(visit, low, high, phase=phase, count=len(ids), phase_only=True)
                add(self.total, low, high, phase=phase, count=len(ids), phase_only=True)
                if phase == "traffic_or_other":
                    self.reasons[reason] += (high-low)*len(ids)
            tick_vehicles.append(dict(trip_id=vehicle["trip_id"], line_id=vehicle["line_id"],
                stop_id=vehicle["stop_id"], visit_index=vehicle["visit_index"], at_stop=vehicle["at_stop"],
                speed=vehicle["speed"], native_stop_state=vehicle["native_stop_state"],
                onboard=len(ids), segments=segments))
        self._write("ticks", dict(start=start, end=end, waiting=len(sample["waiting"]),
            in_vehicle=sum(len(v["ids"]) for v in sample["vehicles"]), vehicles=tick_vehicles))

    def _check_ledger(self):
        snapshot = self.bridge.passenger_time_ledger.snapshot()
        gap = max(abs(self.total["wait"]-snapshot.accrued_wait_seconds),
                  abs(self.total["ride"]-snapshot.accrued_in_vehicle_seconds))
        self.maximum_ledger_gap = max(self.maximum_ledger_gap, gap)
        self.checks["ledger_exact"] &= gap == 0

    def issue(self, event):
        bus_id = str(event["trip_id"])
        visit = self.current.get(bus_id)
        if visit is not None:
            self.issue_sequence[(bus_id, str(event["station_id"]), visit["visit_index"])] = int(event["sequence"])
        row = dict(event, visit_index=None if visit is None else visit["visit_index"])
        self._write("actions", row)
        self.issues += 1

    def finish(self):
        if self.finished:
            raise RuntimeError("Observer finish is single-use")
        api = self._advance.__func__.__globals__["traci"]
        self._sample(api, final=True)  # Endpoint changes only; no extra integration.
        self._check_ledger()
        departed = self.bridge.departed_passenger_times
        completed = self.bridge.completed_passenger_times
        self.checks["observed_people_departed"] = set(self.people).issubset(departed)
        for person in departed:
            self._person(person)
        passenger_path, visit_path = self.out / "passengers.jsonl", self.out / "visits.jsonl"
        with passenger_path.open("x") as handle:
            for person, row in sorted(self.people.items()):
                handle.write(json.dumps(dict(passenger_id=person, departure_time=departed.get(person),
                    completion_time=completed.get(person), completed=person in completed, **row),
                    separators=(",", ":"), allow_nan=False)+"\n")
        with visit_path.open("x") as handle:
            for visit in self.visits:
                if visit["segment_end"] is None:
                    visit["segment_end"] = self.end
                handle.write(json.dumps(visit, separators=(",", ":"), allow_nan=False)+"\n")
        passenger_wait = sum(row["wait"] for row in self.people.values())
        passenger_ride = sum(row["ride"] for row in self.people.values())
        visit_ride = sum(row["ride"] for row in self.visits)
        self.checks["passenger_totals_exact"] = passenger_wait == self.total["wait"] and passenger_ride == self.total["ride"]
        self.checks["visit_ride_exact"] = visit_ride == self.total["ride"]
        self.checks["phase_totals_reconcile"] = math.isclose(sum(self.total["ride_phases"].values()),
            self.total["ride"], rel_tol=0., abs_tol=1e-6)
        snapshot = self.bridge.passenger_time_ledger.snapshot()
        ledger = {key: getattr(snapshot, key) for key in ("time_seconds", "waiting_passengers",
            "in_vehicle_passengers", "accrued_wait_seconds", "accrued_in_vehicle_seconds")}
        errors = [name for name, value in self.checks.items() if not value]
        result = dict(status="complete" if not errors else "failed", checks=self.checks, errors=errors,
            ledger_exact=self.checks["ledger_exact"], start=self.start, end=self.end, ticks=self.ticks,
            issues=self.issues, commands=self.commands, passengers=len(self.people), visits=len(self.visits),
            rejected_commands=self.rejected_commands,
            passenger_totals={key: self.total[key] for key in ("wait", "ride", "ride_phases")},
            periods=self.total["periods"], ledger=ledger, maximum_count_gap=self.maximum_count_gap,
            maximum_ledger_gap=self.maximum_ledger_gap, traffic_or_other_reasons=dict(self.reasons),
            paths={name: name+".jsonl" for name in (*self.writers, "passengers", "visits")},
            semantics="Native tick-start memberships integrated over actual dt, matching the original ledger. Phase is tick-start physical stop/speed plus original service-ready and actual command time; traffic_or_other includes protected and post-command dwell. Visits own the following road segment. ID changes are observed native-tick exchanges; no causal holding effect is inferred.")
        with (self.out / "observer_summary.json").open("x") as handle:
            json.dump(result, handle, separators=(",", ":"), allow_nan=False)
        for writer in self.writers.values():
            writer.flush()
        self.finished = True
        return result

    def close(self):
        if self.closed:
            return
        if self._advance_owned:
            self.bridge._advance_one_step = self._advance
        else:
            del self.bridge._advance_one_step
        if self._apply_owned:
            self.bridge.apply_action = self._apply
        else:
            del self.bridge.apply_action
        for writer in self.writers.values():
            writer.close()
        self.closed = True
