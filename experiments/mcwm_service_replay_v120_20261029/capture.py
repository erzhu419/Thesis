"""Compact passive native contexts for every replayed service decision."""
import sys

from experiments.mcwm_service_contract_v119_20261029.features import build_context, number


def capture_context(bridge, issued, *, source_group):
    """Return a v119 context plus explicit read/coverage metadata.

    Empty cohorts and absent nearby feeders are valid observations. Missing
    native measurements remain unknown. A clock mismatch or an already applied
    event is a temporal contract error, so no context is returned in that case.
    """
    api = sys.modules[type(bridge).__module__].traci
    before = float(api.simulation.getTime())
    issue_clock = float(issued["started_at"])
    if before != issue_clock:
        raise ValueError("Context clock differs from the issued action clock")
    receiver_id, line, station = (str(issued[name]) for name in ("trip_id", "line_id", "station_id"))
    native_event = bridge.pending_events.get((line, receiver_id, station))
    if native_event is None or native_event.assigned_action is not None:
        raise ValueError("Context requires the actual pending event before application")
    errors, missing = [], []

    def read(label, getter):
        try:
            value = getter()
        except Exception as error:
            errors.append(dict(field=label, error=type(error).__name__))
            value = None
        if value is None:
            missing.append(label)
        return value

    def numeric(label, getter):
        value = read(label, getter)
        value = number(value)
        if value is None and label not in missing:
            missing.append(label)
        return value

    def plan(trip):
        stops = read(trip+"/remaining_stops", lambda: api.vehicle.getStops(trip, 0))
        if stops is None:
            return None
        values = [getattr(stop, "stoppingPlaceID", None) for stop in stops]
        if any(not isinstance(stop, str) or not stop for stop in values):
            missing.append(trip+"/remaining_stop_identity")
            return None
        return values

    def location(trip):
        return dict(lane_id=read(trip+"/lane_id", lambda: api.vehicle.getLaneID(trip)),
            lane_position=numeric(trip+"/lane_position", lambda: api.vehicle.getLanePosition(trip)),
            speed=numeric(trip+"/speed", lambda: api.vehicle.getSpeed(trip)),
            at_stop=read(trip+"/at_stop", lambda: api.vehicle.isAtBusStop(trip)))

    native_ids = read("native_vehicle_ids", api.vehicle.getIDList)
    active = sorted(str(trip) for trip in (native_ids or ()) if str(trip) in bridge.bus_obj_dic)
    receiver_present = receiver_id in active
    receiver_plan = plan(receiver_id) if receiver_present else None
    receiver_ids = read(receiver_id+"/onboard_ids", lambda: api.vehicle.getPersonIDList(receiver_id)) if receiver_present else None
    capacity = numeric(receiver_id+"/capacity", lambda: api.vehicle.getPersonCapacity(receiver_id)) if receiver_present else None
    receiver_location = location(receiver_id) if receiver_present else {}
    if not receiver_present:
        missing.append("receiver_native_vehicle")
    receiver_at_target = (receiver_location.get("at_stop") is True and bool(receiver_plan)
                          and receiver_plan[0] == station)
    if receiver_present and not receiver_at_target:
        missing.append("receiver_at_declared_stop")
    onboard = None if receiver_ids is None else len(receiver_ids)
    receiver = dict(trip_id=receiver_id, receiver=True, capacity=capacity,
        counts=dict(onboard=onboard), free_seats=None if capacity is None or onboard is None else max(0., capacity-onboard),
        location=receiver_location, planned_stops=receiver_plan, people=[])
    vehicles = [receiver]
    plans_complete = native_ids is not None
    for trip in active:
        if trip == receiver_id:
            continue
        stops = plan(trip)
        if stops is None:
            plans_complete = False
            continue
        if station not in stops:
            continue
        members = read(trip+"/onboard_ids", lambda: api.vehicle.getPersonIDList(trip))
        people = None if members is None else []
        for pid in members or ():
            pid = str(pid)
            stage = read(pid+"/current_stage0", lambda: api.person.getStage(pid, 0))
            kind, permitted, destination = (getattr(stage, key, None) for key in ("type", "line", "destStop"))
            known = (kind == 3 and isinstance(permitted, str) and bool(permitted.strip())
                     and isinstance(destination, str) and bool(destination))
            if not known and pid+"/current_stage0" not in missing:
                missing.append(pid+"/current_stage0")
            people.append(dict(person_id=pid, current_stage0=dict(known=known, type=kind,
                                                                line=permitted, destStop=destination)))
        vehicles.append(dict(trip_id=trip, receiver=False, planned_stops=stops,
            counts=dict(onboard=None if members is None else len(members)), location=location(trip), people=people))
    after = float(api.simulation.getTime())
    if after != before or native_event.assigned_action is not None:
        raise RuntimeError("Native context getters changed time or the pending action")
    ready = number(native_event.metadata.get("service_completion_time"))
    if ready is None:
        missing.append("receiver_service_ready_time")
    full = dict(summary=dict(issue=dict(trip_id=receiver_id, station_id=station,
        sequence=int(issued["sequence"]), issue_clock=issue_clock, observed_clock=before),
        checks=dict(native_event_before_apply=True), receiver=dict(service_ready_time=ready)), vehicles=vehicles)
    context = build_context(full, source_group=source_group)
    context["capture"] = dict(status="complete" if not missing else "missing",
        native_clock_before=before, native_clock_after=after,
        checks=dict(clock_unchanged=before == after, pending_event_unassigned=True,
            receiver_native_present=receiver_present, receiver_at_declared_stop=receiver_at_target,
            feeder_selection_complete=plans_complete), missing=missing, errors=errors)
    return context
