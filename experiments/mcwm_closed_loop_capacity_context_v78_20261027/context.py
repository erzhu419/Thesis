"""Read a next-stop capacity context from current native state only.

No collection hook is installed by this module. Its fake-API tests establish
interface semantics, not native joint-snapshot coverage. Event time may precede
issue time by one tick; the existing state and execution transform stay intact.
"""
import math


def observe_context(api, *, trip_id, line_id, station_index, route_stops,
                    issue_clock, event_clock):
    """Return counts and a signed queue/seat margin; never return person IDs.

    Only stage index 0 is read. A planned current-leg destination is visible
    information; actual later alighting, arrivals and queues are never queried.
    A partial observation cannot supply a context value and leaves ``z=None``.
    """
    reasons = []
    stages = {group: dict(read=0, driving=0, missing=0, unsupported=0)
              for group in ('onboard', 'queue')}
    result = dict(observed=False, z=None, reasons=reasons,
        clocks=dict(issue=issue_clock, event=event_clock, obs_start=None,
                    obs_end=None, atomic_at_issue=False),
        next_stop=None, capacity=None, actual_line=None,
        counts=dict(onboard=None, next_queue=None, release_next=None,
                    compatible_next_queue=None, seats_after_release=None),
        stage_counts=stages)

    def read_clock(label):
        try:
            return float(api.simulation.getTime())
        except Exception:
            reasons.append(label+'_clock_missing')
            return None

    result['clocks']['obs_start'] = read_clock('start')
    next_index = station_index+1
    if station_index < 0 or station_index >= len(route_stops):
        reasons.append('station_index_not_in_route')
    elif next_index == len(route_stops):
        reasons.append('terminal_no_next_stop')
    else:
        result['next_stop'] = route_stops[next_index]
        if not result['next_stop']:
            reasons.append('next_stop_missing')

    if result['next_stop'] is not None and 'next_stop_missing' not in reasons:
        try:
            capacity = float(api.vehicle.getPersonCapacity(trip_id))
            if math.isfinite(capacity) and capacity > 0:
                result['capacity'] = capacity
            else:
                reasons.append('capacity_unsupported')
        except Exception:
            reasons.append('capacity_missing')
        try:
            result['actual_line'] = api.vehicle.getLine(trip_id)
            if result['actual_line'] != line_id:
                reasons.append('vehicle_line_mismatch')
        except Exception:
            reasons.append('vehicle_line_missing')

        memberships = {}
        for group, getter in (
            ('onboard', lambda: api.vehicle.getPersonIDList(trip_id)),
            ('queue', lambda: api.busstop.getPersonIDs(result['next_stop'])),
        ):
            try:
                memberships[group] = tuple(getter())
                key = 'onboard' if group == 'onboard' else 'next_queue'
                result['counts'][key] = len(memberships[group])
            except Exception:
                reasons.append(group+'_membership_missing')

        for group, people in memberships.items():
            matched = 0
            complete = (group == 'onboard' or
                        result['actual_line'] == line_id)
            for person in people:
                stages[group]['read'] += 1
                try:
                    stage = api.person.getStage(person, 0)
                except Exception:
                    stages[group]['missing'] += 1
                    reasons.append(group+'_stage_missing')
                    complete = False
                    continue
                if getattr(stage, 'type', None) != 3:
                    stages[group]['unsupported'] += 1
                    reasons.append(group+'_stage_not_driving')
                    complete = False
                    continue
                stages[group]['driving'] += 1
                if group == 'onboard':
                    destination = getattr(stage, 'destStop', None)
                    if not isinstance(destination, str) or not destination:
                        stages[group]['unsupported'] += 1
                        reasons.append('onboard_destination_missing')
                        complete = False
                    else:
                        matched += destination == result['next_stop']
                else:
                    lines = getattr(stage, 'line', None)
                    tokens = lines.split() if isinstance(lines, str) else []
                    if not tokens or any(token in ('ANY', '*') for token in tokens):
                        stages[group]['unsupported'] += 1
                        reasons.append('queue_line_condition_unsupported')
                        complete = False
                    else:
                        matched += result['actual_line'] in tokens
            key = 'release_next' if group == 'onboard' else 'compatible_next_queue'
            if complete:
                result['counts'][key] = matched

    result['clocks']['obs_end'] = read_clock('end')
    clocks = result['clocks']
    clocks['atomic_at_issue'] = (
        clocks['obs_start'] is not None
        and clocks['obs_start'] == clocks['obs_end'] == issue_clock)
    if not clocks['atomic_at_issue']:
        reasons.append('observation_not_atomic_at_issue')
    result['reasons'] = list(dict.fromkeys(reasons))
    counts = result['counts']
    if not result['reasons']:
        counts['seats_after_release'] = max(
            0., result['capacity']-counts['onboard']+counts['release_next'])
        result['z'] = (counts['compatible_next_queue']
                       - counts['seats_after_release'])/result['capacity']
        result['observed'] = True
    return result
