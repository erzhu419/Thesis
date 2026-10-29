"""Observe receiver current-leg destinations before its unchanged action."""
import math
import sys


def capture_receiver_service(bridge, issued, *, source_group):
    """Return current observations and provenance, without advancing the runtime.

    Count zero requires complete membership and current-stage reads. The count
    describes onboard current-leg destinations, not a remaining service time.
    The bridge's service_completion_time is retained explicitly as an estimate.
    """
    api = sys.modules[type(bridge).__module__].traci
    before = float(api.simulation.getTime())
    if before != float(issued['started_at']):
        raise ValueError('Receiver observation must match the issued action clock')
    receiver, line, station = (str(issued[key]) for key in ('trip_id', 'line_id', 'station_id'))
    event = bridge.pending_events.get((line, receiver, station))
    if event is None or event.assigned_action is not None:
        raise ValueError('Receiver observation requires the pending, unapplied action')
    missing, errors, people = [], [], []

    def read(field, getter):
        try:
            value = getter()
        except Exception as error:
            errors.append(dict(field=field, error=type(error).__name__))
            value = None
        if value is None:
            missing.append(field)
        return value

    members = read('receiver/onboard_ids', lambda: api.vehicle.getPersonIDList(receiver))
    complete = members is not None
    matches = 0
    if members is not None:
        for person in members:
            person = str(person)
            field = person+'/current_stage0'
            stage = read(field, lambda: api.person.getStage(person, 0))
            kind, permitted, destination = (getattr(stage, key, None) for key in ('type', 'line', 'destStop'))
            known = (kind == 3 and isinstance(permitted, str) and bool(permitted.strip())
                     and isinstance(destination, str) and bool(destination))
            if not known and field not in missing:
                missing.append(field)
            complete = complete and known
            matches += int(known and destination == station)
            people.append(dict(person_id=person, current_stage0=dict(
                known=known, type=kind, line=permitted, destStop=destination)))

    ready = event.metadata.get('service_completion_time')
    ready = None if ready is None else float(ready)
    if ready is not None and not math.isfinite(ready):
        ready = None
    if ready is None:
        missing.append('receiver/estimated_service_ready_time')
    after = float(api.simulation.getTime())
    if after != before or event.assigned_action is not None:
        raise RuntimeError('Receiver getters changed time or the pending action')
    return dict(
        meta=dict(source_group=source_group, sequence=int(issued['sequence']),
                  observed_at=before, receiver=receiver, station=station),
        receiver=dict(onboard_count=None if members is None else len(members),
            current_leg_dest_count=matches if complete else None,
            current_stage0_complete=complete,
            estimated_service_ready_time=ready,
            estimated_service_ready_age_seconds=None if ready is None else before-ready),
        people=people,
        capture=dict(status='complete' if not missing else 'missing',
            native_clock_before=before, native_clock_after=after,
            checks=dict(clock_unchanged=before == after, pending_event_unassigned=event.assigned_action is None),
            missing=missing, errors=errors))
