"""Build as-of-time service labels from complete, resident native event logs.

One row remains one receiver-root/action, regardless of the number of passengers.
No environment, model, policy or optimizer is imported.
"""
import math


def feedback_deadline(mature_at, protocol):
    """First existing packet deadline after feedback delay, or no legal packet."""
    if mature_at is None:
        return None
    first, last = protocol['feedback_first_deadline'], protocol['feedback_last_deadline']
    step = protocol['feedback_interval_seconds']
    release = mature_at + protocol['feedback_delay_seconds']
    deadline = first + max(0, math.ceil((release-first)/step))*step
    return deadline if deadline <= last else None


def mature_labels(sample, evidence, *, available_at):
    """Evidence uses native observation times, not predicted departure metadata.

    evidence: start, end, continuous, visit_events, exchanges. Complete coverage
    is required to turn an absent boarding into zero at receiver departure.
    Event timestamps label the observed tick endpoint, not sub-tick order.
    """
    meta, action = sample['meta'], sample['action']
    start = meta['observed_at']
    if not evidence['continuous'] or evidence['start'] != start:
        raise ValueError('Complete observation since the pre-action root is required')
    if not start <= available_at <= evidence['end']:
        raise ValueError('Label clock is outside the observed interval')
    receiver, station = meta['receiver'], meta['station']
    visits = [e for e in evidence['visit_events'] if start < e['time'] <= available_at]
    exchanges = [e for e in evidence['exchanges'] if start < e['time'] <= available_at]
    departures = [e['time'] for e in visits if e['event'] == 'departure'
                  and e['trip_id'] == receiver and e['stop_id'] == station]
    departure = min(departures) if departures else None
    visit_end = departure if departure is not None else available_at
    receiver_events = [e for e in exchanges if e['trip_id'] == receiver
                       and e['stop_id'] == station and e['time'] <= visit_end]
    boarded = {}
    for event in receiver_events:
        for person in event['boarded_ids']:
            boarded.setdefault(person, event['time'])

    def event_time(value):
        return dict(value=value, seconds_after_issue=None if value is None else value-start,
                    mature_at=value, right_censored=value is None,
                    observed_until=available_at)

    def visit_count(value):
        return dict(value=value if departure is not None else None,
                    observed_count_lower_bound=value, mature_at=departure,
                    right_censored=departure is None)

    feeders = []
    all_cohort = set()
    for feeder in sample['feeders']:
        trip, cohort = feeder['trip_id'], set(feeder['cohort_ids'])
        all_cohort.update(cohort)
        arrivals = [e['time'] for e in visits if e['event'] == 'arrival'
                    and e['trip_id'] == trip and e['stop_id'] == station]
        alighted = {}
        for event in exchanges:
            if event['trip_id'] == trip and event['stop_id'] == station:
                for person in cohort.intersection(event['alighted_ids']):
                    alighted.setdefault(person, event['time'])
        people = []
        for person in sorted(cohort):
            board = boarded.get(person)
            boarding = event_time(board)
            boarding['closed_without_event'] = board is None and departure is not None
            if boarding['closed_without_event']:
                boarding.update(right_censored=False, mature_at=departure)
            people.append(dict(person_id=person, alighting=event_time(alighted.get(person)),
                boarded_this_visit=dict(value=1 if board is not None else
                    (0 if departure is not None else None),
                    mature_at=board if board is not None else departure,
                    right_censored=board is None and departure is None),
                boarding=boarding))
        arrival = event_time(min(arrivals) if arrivals else None)
        # A feeder already at the station has no identifiable future arrival time.
        arrival['left_censored'] = bool(feeder.get('at_target_stop_at_issue', False))
        if arrival['left_censored']:
            arrival.update(value=None, seconds_after_issue=None, mature_at=None,
                           right_censored=False)
        feeders.append(dict(trip_id=trip, cohort_size=len(cohort), arrival=arrival,
            cohort_definition_complete=feeder['current_stage0_complete'],
            cohort_boarded_count=visit_count(len(cohort.intersection(boarded))), people=people))
    residual = None if departure is None else departure-action['applied_at']-action['hold_seconds']
    return dict(available_at=available_at, receiver_departure=event_time(departure),
        receiver_post_issue_boarded_count=visit_count(len(boarded)),
        cohort_boarded_count=visit_count(len(all_cohort.intersection(boarded))),
        departure_residual_seconds=dict(value=residual, mature_at=departure,
            meaning='Observed departure minus (applied_at + hold); not isolated boarding service time.'),
        feeders=feeders,
        semantics='Native times are tick endpoints. Count zero closes this receiver visit, '
                  'not the person journey. Stage0 destination does not certify transfer intent.')
