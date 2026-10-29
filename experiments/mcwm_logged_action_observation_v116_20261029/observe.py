"""Read one current native connection state before applying its issued action."""
import json
import math
import random
import sys

import numpy as np


LEDGER_FIELDS = ("time_seconds", "waiting_passengers", "in_vehicle_passengers",
                 "accrued_wait_seconds", "accrued_in_vehicle_seconds")


def _guard(bridge, api):
    snapshot = bridge.passenger_time_ledger.snapshot()
    torch = sys.modules.get("torch")
    return dict(clock=float(api.simulation.getTime()),
        ledger={key: float(getattr(snapshot, key)) for key in LEDGER_FIELDS},
        python=random.getstate(), numpy=np.random.get_state(),
        torch_module=torch,
        torch=None if torch is None else torch.get_rng_state().clone())


def _checks(before, after, issue_clock):
    left, right = before["numpy"], after["numpy"]
    same_torch = before["torch_module"] is after["torch_module"]
    return dict(native_clock_unchanged=before["clock"] == after["clock"],
        observed_at_issue=before["clock"] == after["clock"] == issue_clock,
        ledger_unchanged=before["ledger"] == after["ledger"],
        python_rng_unchanged=before["python"] == after["python"],
        numpy_rng_unchanged=left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:],
        torch_module_unchanged=same_torch,
        torch_rng_unchanged=same_torch and (before["torch_module"] is None
            or before["torch_module"].equal(before["torch"], after["torch"])))


def capture(bridge, issued, *, diagnostic_person_id):
    """Return a <=16KB summary and complete server-resident detail.

    Current stage0 is distinct from additional preloaded whole-journey intent.
    No future stage, realized future event, simulation step or setter is read.
    """
    api = sys.modules[type(bridge).__module__].traci
    before = _guard(bridge, api)
    trip, line, station = (str(issued[key]) for key in ("trip_id", "line_id", "station_id"))
    issue_clock = float(issued["started_at"])
    errors, vehicles, all_members = [], [], set()
    native_event = bridge.pending_events.get((line, trip, station))

    def read(label, function, *, required=False):
        try:
            return function()
        except Exception as error:
            errors.append(dict(field=label, error=type(error).__name__, required=required))
            return None

    def finite(value, label):
        if value is None:
            return None
        value = float(value)
        if not math.isfinite(value):
            errors.append(dict(field=label, error="nonfinite"))
            return None
        return value

    def plan(bus):
        values = read(bus+"/stop_plan", lambda: api.vehicle.getStops(bus, 0))
        if values is None:
            return None
        stops = [getattr(value, "stoppingPlaceID", None) for value in values]
        if any(not isinstance(stop, str) or not stop for stop in stops):
            errors.append(dict(field=bus+"/stop_plan", error="missing_stop_id"))
            return None
        return stops

    def person(person_id, bus, bus_line, receiver_remaining):
        stage = read(person_id+"/current_stage0", lambda: api.person.getStage(person_id, 0), required=True)
        kind, permitted, destination = (getattr(stage, key, None) for key in ("type", "line", "destStop"))
        known = kind == 3 and isinstance(permitted, str) and bool(permitted.strip()) and isinstance(destination, str) and bool(destination)
        if not known:
            errors.append(dict(field=person_id+"/current_stage0", error="unsupported_or_missing", required=True))
        current = dict(type=kind, line=permitted, destStop=destination, known=known,
                       source="native_current_stage0")
        # Access only this currently onboard person's record, never all demand.
        saved = bridge.passenger_obj_dic.get(person_id)
        extra = dict(source="additional_preloaded_journey_intent", available=False,
            transfer_stop=None, final_stop=None, first_leg_transfer_at_receiver=None,
            receiver_remaining_route_compatible=None, next_leg_permitted_lines=None)
        if saved is not None:
            transfer, final = saved.transfer_stop_id_s, saved.end_stop_id_s
            available = isinstance(transfer, str) and isinstance(final, str) and bool(final)
            first_leg = bool(known and transfer == destination == station and final != station)
            extra.update(available=available, transfer_stop=transfer, final_stop=final,
                first_leg_transfer_at_receiver=first_leg if available and known else None,
                receiver_remaining_route_compatible=(first_leg and final in receiver_remaining)
                    if available and known and receiver_remaining is not None else None)
        return dict(person_id=person_id, actual_onboard_trip=bus, actual_onboard_line=bus_line,
                    current_stage0=current, additional_intent=extra)

    native_ids = read("active_native_ids", api.vehicle.getIDList, required=True)
    active = sorted(str(bus) for bus in (native_ids or ()) if str(bus) in bridge.bus_obj_dic)
    receiver_plan = plan(trip) if trip in active else None
    downstream = None
    if receiver_plan is not None and station in receiver_plan:
        downstream = receiver_plan[receiver_plan.index(station)+1:]

    # Same-line buses are also counted; no target-person or outcome-based filter.
    for bus in active:
        stops = receiver_plan if bus == trip else plan(bus)
        if bus != trip and stops is not None and station not in stops:
            continue
        if stops is None:
            continue
        actual_line = read(bus+"/line", lambda: api.vehicle.getLine(bus))
        members = read(bus+"/onboard_membership", lambda: api.vehicle.getPersonIDList(bus), required=True)
        if members is None:
            continue
        members = list(map(str, members))
        if len(set(members)) != len(members) or all_members.intersection(members):
            errors.append(dict(field=bus+"/onboard_membership", error="duplicate_membership", required=True))
        all_members.update(members)
        capacity = finite(read(bus+"/capacity", lambda: api.vehicle.getPersonCapacity(bus)), bus+"/capacity")
        if capacity is not None and capacity <= 0:
            errors.append(dict(field=bus+"/capacity", error="nonpositive"))
        location = dict(
            road_id=read(bus+"/road", lambda: api.vehicle.getRoadID(bus)),
            lane_id=read(bus+"/lane", lambda: api.vehicle.getLaneID(bus)),
            lane_position=finite(read(bus+"/lane_position", lambda: api.vehicle.getLanePosition(bus)), bus+"/lane_position"),
            route_distance=finite(read(bus+"/distance", lambda: api.vehicle.getDistance(bus)), bus+"/distance"),
            speed=finite(read(bus+"/speed", lambda: api.vehicle.getSpeed(bus)), bus+"/speed"),
            at_stop=read(bus+"/at_stop", lambda: api.vehicle.isAtBusStop(bus)))
        people = [person(pid, bus, actual_line, downstream) for pid in members]
        counts = dict(onboard=len(people),
            current_stage_known=sum(p["current_stage0"]["known"] for p in people),
            current_leg_dest_receiver_observed=sum(p["current_stage0"]["known"] and
                p["current_stage0"]["destStop"] == station for p in people),
            additional_intent_available=sum(p["additional_intent"]["available"] for p in people),
            planned_first_leg_transfers_observed=sum(p["additional_intent"]["first_leg_transfer_at_receiver"] is True for p in people),
            planned_receiver_compatible_observed=sum(p["additional_intent"]["receiver_remaining_route_compatible"] is True for p in people))
        vehicles.append(dict(trip_id=bus, line_id=actual_line, receiver=bus == trip,
            location=location, capacity=capacity, free_seats=None if capacity is None else max(0., capacity-len(people)),
            planned_stops=stops, planned_current_or_next_stop=stops[0] if stops else None,
            next_stop_matches=bool(stops and stops[0] == station), counts=counts, people=people))

    receiver = next((v for v in vehicles if v["receiver"]), None)
    feeders = [v for v in vehicles if not v["receiver"]]
    after = _guard(bridge, api)
    checks = _checks(before, after, issue_clock)
    checks.update(receiver_native_active=trip in active,
        receiver_native_identity=receiver is not None and receiver["line_id"] == line,
        receiver_at_declared_stop=receiver is not None and receiver["location"]["at_stop"] is True
            and receiver["planned_current_or_next_stop"] == station,
        native_event_before_apply=native_event is not None and native_event.assigned_action is None)
    service_ready = None if native_event is None else native_event.metadata.get("service_completion_time")
    if service_ready is not None:
        service_ready = finite(service_ready, "receiver/service_ready")
    checks["required_reads_complete"] = not any(error.get("required", False) for error in errors)

    def compact(vehicle):
        return {key: value for key, value in vehicle.items() if key not in ("people", "planned_stops")}

    diagnostic = [dict(vehicle=compact(v), person=p) for v in vehicles for p in v["people"]
                  if p["person_id"] == diagnostic_person_id]
    checks["diagnostic_current_onboard_stage_observed"] = (len(diagnostic) == 1
        and diagnostic[0]["person"]["current_stage0"]["known"])
    status = "complete" if all(checks.values()) else "HOLD"
    positive = [v for v in feeders if v["counts"]["current_leg_dest_receiver_observed"] > 0]
    keys = ("onboard", "current_stage_known", "current_leg_dest_receiver_observed",
            "additional_intent_available", "planned_first_leg_transfers_observed", "planned_receiver_compatible_observed")
    counts = {key: sum(v["counts"][key] for v in feeders) for key in keys}
    counts.update(native_active_buses=len(active), related_feeder_vehicles=len(feeders),
        next_stop_matches_vehicles=sum(v["next_stop_matches"] for v in feeders),
        vehicles_with_current_leg_dest_receiver=len(positive),
        immediate_current_leg_dest_receiver_observed=sum(v["counts"]["current_leg_dest_receiver_observed"]
            for v in feeders if v["next_stop_matches"]))
    summary = dict(status=status, all_pass=all(checks.values()), checks=checks,
        issue=dict(sequence=int(issued["sequence"]), trip_id=trip, line_id=line, station_id=station,
            issue_clock=issue_clock, event_clock=None if native_event is None else float(native_event.sim_time),
            observed_clock=before["clock"]),
        receiver=None if receiver is None else dict(compact(receiver), service_ready_time=service_ready,
            service_ready_now=None if service_ready is None else before["clock"] >= service_ready),
        coverage=counts, counts_are_partial=bool(errors),
        positive_feeders=[dict(trip_id=v["trip_id"], line_id=v["line_id"],
            next_stop_matches=v["next_stop_matches"], counts=v["counts"]) for v in positive[:12]],
        positive_feeder_rows_total=len(positive), positive_feeder_rows_returned=min(12, len(positive)),
        diagnostic_person_id=diagnostic_person_id, diagnostic_matches=diagnostic,
        torch_rng_checked=before["torch_module"] is not None,
        guard=dict(before={k: before[k] for k in ("clock", "ledger")},
                   after={k: after[k] for k in ("clock", "ledger")}),
        error_count=len(errors), errors=errors[:8],
        semantics="Counts cover actual onboard people of all active vehicles whose remaining native stop plan contains the receiver stop. Current-stage destinations are planned, not realized arrivals. Additional whole-journey intent does not supply next-leg permission; no ETA is inferred. Ledger counts may lag current native membership by one tick.")
    full = dict(summary=summary, vehicles=vehicles, errors=errors,
                active_bus_ids=active, receiver_remaining_stops=downstream)
    if len(json.dumps(summary, separators=(",", ":"), allow_nan=False).encode()) > 16000:
        raise RuntimeError("Observation summary exceeds the fixed16KB export bound; full capture must stay resident")
    return dict(summary=summary, full=full)
