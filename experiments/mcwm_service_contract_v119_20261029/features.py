"""Current native service features, independent of outcomes and learning code."""
from copy import deepcopy
import math


def number(value):
    """Preserve unknown native measurements; never replace them with zero."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def build_context(full_capture, *, source_group):
    """Read the v116 full capture at its action-before issue clock.

    A feeder's x contains numeric measurements only. Trip/person identifiers
    and the common-root source group are provenance, not model inputs. The
    current-stage cohort is unrelated to additional whole-journey intent.
    """
    summary = full_capture["summary"]
    issue = summary["issue"]
    observed = number(issue["observed_clock"])
    if observed is None or observed != number(issue["issue_clock"]):
        raise ValueError("Native context must be observed at the issued action clock")
    if summary["checks"]["native_event_before_apply"] is not True:
        raise ValueError("Native context must precede action application")
    receiver_id, station = issue["trip_id"], issue["station_id"]
    receivers = [vehicle for vehicle in full_capture["vehicles"]
                 if vehicle["receiver"] and vehicle["trip_id"] == receiver_id]
    if len(receivers) != 1:
        raise ValueError("The declared native receiver must be present exactly once")
    receiver = receivers[0]
    ready = number(summary["receiver"].get("service_ready_time"))
    receiver_fields = dict(capacity=number(receiver.get("capacity")),
        onboard=number(receiver.get("counts", {}).get("onboard")),
        free_seats=number(receiver.get("free_seats")),
        service_ready_age_seconds=None if ready is None else observed-ready)
    receiver_location = receiver.get("location", {})
    feeders = []
    for feeder in full_capture["vehicles"]:
        if feeder["receiver"]:
            continue
        plan = feeder.get("planned_stops")
        if plan is None or station not in plan:
            continue
        people = feeder.get("people")
        cohort = [person["person_id"] for person in (people or [])
                  if person.get("current_stage0", {}).get("known") is True
                  and person["current_stage0"].get("destStop") == station]
        stages_complete = people is not None and all(
            person.get("current_stage0", {}).get("known") is True for person in people)
        location = feeder.get("location", {})
        receiver_lane, feeder_lane = receiver_location.get("lane_id"), location.get("lane_id")
        same_lane = receiver_lane == feeder_lane if receiver_lane and feeder_lane else None
        receiver_position, feeder_position = number(receiver_location.get("lane_position")), number(location.get("lane_position"))
        gap = (receiver_position-feeder_position
               if same_lane is True and receiver_position is not None and feeder_position is not None else None)
        x = {"receiver_"+key: value for key, value in receiver_fields.items()}
        x.update(feeder_onboard=number(feeder.get("counts", {}).get("onboard")),
            feeder_speed=number(location.get("speed")), same_lane=same_lane, gap_m=gap,
            feeder_next_stop_matches=plan[0] == station,
            current_leg_dest_count=len(cohort) if stages_complete else None)
        at_stop = location.get("at_stop")
        feeders.append(dict(trip_id=feeder["trip_id"], cohort_ids=cohort,
            at_target_stop_at_issue=None if at_stop is None else bool(at_stop and plan[0] == station),
            current_stage0_complete=stages_complete, x=x))
    return dict(meta=dict(receiver=receiver_id, station=station, sequence=issue["sequence"],
        observed_at=observed, source_group=source_group), receiver=receiver_fields, feeders=feeders)


def bind_action(context, command):
    """Attach an accepted v117/v118 command without modifying the context."""
    if command.get("accepted") is not True:
        raise ValueError("Only an accepted native command can be attached")
    stops = [command[name] for name in ("stop_id", "station_id") if name in command]
    if (command.get("trip_id") != context["meta"]["receiver"] or not stops
            or any(stop != context["meta"]["station"] for stop in stops)):
        raise ValueError("Native command must belong to the observed receiver and station")
    applied_at = number(command.get("applied_at"))
    if applied_at is None or applied_at != context["meta"]["observed_at"]:
        raise ValueError("Command application must match the observed action clock")
    fields = [name for name in ("hold_seconds", "requested_additional_hold_seconds") if name in command]
    if not fields:
        raise ValueError("Native command has no recorded holding duration")
    values = [number(command[name]) for name in fields]
    hold = values[0]
    if hold is None or hold < 0 or any(value != hold for value in values):
        raise ValueError("Recorded holding duration must be finite, nonnegative and unambiguous")
    result = deepcopy(context)
    result["action"] = dict(hold_seconds=hold, applied_at=applied_at)
    return result
