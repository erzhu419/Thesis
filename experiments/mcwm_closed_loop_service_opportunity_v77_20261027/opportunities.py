"""Read observed boarding-versus-queue contrasts without exporting person IDs."""
from collections import Counter, defaultdict
from statistics import median


DIRECTIONS = (
    ('fresh_wait_baseline_boarded', 'fresh', 'baseline'),
    ('baseline_wait_fresh_boarded', 'baseline', 'fresh'),
)


def _key(row):
    return tuple(row[name] for name in ('bus_id', 'line_id', 'station_id', 'occurrence'))


def _physical(row):
    return (row is not None and row['observed_at'] == row['clock_seconds']
            and row.get('queued_passenger_ids') is not None
            and row.get('onboard_passenger_ids') is not None)


def _ids(row, field):
    return set(row[field])


def _summary(values):
    values = sorted(values)
    return dict(count=len(values), min=values[0] if values else None,
                max=values[-1] if values else None,
                mean=sum(values)/len(values) if values else None,
                median=median(values) if values else None)


def _visits(rows):
    arrivals, departures = {}, {}
    for row in rows:
        if row['kind'] == 'arrival':
            arrivals[_key(row)] = row
        elif row['kind'] == 'departure':
            departures[_key(row)] = row
    return {key: dict(departure=row, arrival=arrivals.get(key))
            for key, row in departures.items()}


def _queued_before(visit, person):
    arrival, departure = visit['arrival'], visit['departure']
    return (_physical(arrival)
            and arrival['clock_seconds'] <= departure['clock_seconds']
            and person in _ids(arrival, 'queued_passenger_ids')
            and person not in _ids(arrival, 'onboard_passenger_ids'))


def _service(visit):
    departure, arrival = visit['departure'], visit['arrival']
    return dict(bus_id=departure['bus_id'], line_id=departure['line_id'],
                station_id=departure['station_id'], occurrence=departure['occurrence'],
                departure_clock=departure['clock_seconds'],
                departure_observed_at=departure['observed_at'],
                arrival_clock=None if arrival is None else arrival['clock_seconds'],
                arrival_physical=_physical(arrival),
                capacity=departure.get('capacity'),
                free_capacity=departure.get('free_capacity'),
                eligible_waiting=departure.get('eligible_waiting'),
                waiting=len(departure['queued_passenger_ids']),
                onboard=len(departure['onboard_passenger_ids']),
                native_board_count=departure.get('native_board_count'),
                native_alight_count=departure.get('native_alight_count'),
                native_stranded_count=departure.get('native_stranded_count'))


def _recovery_index(visits):
    result = defaultdict(list)
    for visit in visits.values():
        departure = visit['departure']
        if not _physical(departure):
            continue
        for person in _ids(departure, 'onboard_passenger_ids'):
            if _queued_before(visit, person):
                result[person, departure['station_id']].append(visit)
    for values in result.values():
        values.sort(key=lambda visit: (visit['departure']['clock_seconds'],
                                      _key(visit['departure'])))
    return result


def waiting_opportunities(fresh_services, baseline_services, lo=12000, hi=16200):
    """Use each arm's own event tick; retain only its first in-window contrast.

    Complete existing logs may supply earlier context and later recovery. Neither
    later truth nor final passenger metadata selects or upgrades a contrast.
    """
    visits = dict(fresh=_visits(fresh_services), baseline=_visits(baseline_services))
    recovery = {arm: _recovery_index(rows) for arm, rows in visits.items()}
    common = visits['fresh'].keys() & visits['baseline'].keys()
    pairs, audit = [], Counter()
    for key in common:
        pair = {arm: rows[key] for arm, rows in visits.items()}
        clocks = [visit['departure']['clock_seconds'] for visit in pair.values()]
        if not all(_physical(visit['departure']) for visit in pair.values()):
            audit['matched_departures_missing_or_lagged_inventory'] += 1
            continue
        in_window = [lo <= clock < hi for clock in clocks]
        audit['physical_matched_departures'] += 1
        if any(in_window) and not all(in_window):
            audit['cross_window_boundary_departures'] += 1
        if all(in_window):
            audit['both_departures_in_window'] += 1
        pairs.append((max(clocks), key, pair, all(in_window), max(clocks) < lo))
    pairs.sort(key=lambda row: (row[0], row[1]))

    result = {}
    for name, waiting_arm, boarded_arm in DIRECTIONS:
        raw, verified_raw = 0, 0
        before, selected = set(), {}
        for contrast_clock, key, pair, in_window, before_window in pairs:
            waiting, boarded = pair[waiting_arm], pair[boarded_arm]
            intersection = (_ids(waiting['departure'], 'queued_passenger_ids')
                            & _ids(boarded['departure'], 'onboard_passenger_ids'))
            for person in sorted(intersection):
                identity = person, key[2]
                if before_window:
                    before.add(identity)
                if not in_window:
                    continue
                raw += 1
                verified = all(_queued_before(visit, person) for visit in pair.values())
                verified_raw += verified
                if identity in selected:
                    continue
                selected[identity] = dict(person=person, station=key[2], line=key[1],
                    clock=contrast_clock, pair=pair, verified=verified,
                    observed_before_window=identity in before)

        totals = Counter()
        stations, lines = defaultdict(Counter), defaultdict(Counter)
        deltas, bracket_lows, bracket_highs, cases = [], [], [], []
        for index, (identity, item) in enumerate(selected.items(), 1):
            waiting, boarded = item['pair'][waiting_arm], item['pair'][boarded_arm]
            departed = waiting['departure']['clock_seconds']
            later = next((visit for visit in recovery[waiting_arm].get(identity, ())
                          if visit['departure']['clock_seconds'] > departed), None)
            verified = item['verified']
            category = 'verified_board_vs_wait' if verified else 'weak_context'
            totals[category] += 1
            context = ('observed_before_window' if item['observed_before_window']
                       else 'first_observed_in_window')
            totals[context] += 1
            totals['recovered_with_new_boarding_evidence' if later is not None
                   else 'right_censored_no_recovery_observed'] += 1
            for group in (stations[item['station']], lines[item['line']]):
                group['unique_person_stop'] += 1
                group[category] += 1
                group[context] += 1
                group['recovery_observed'] += later is not None
            served_bracket = ([boarded['arrival']['clock_seconds'],
                               boarded['departure']['clock_seconds']]
                              if verified else None)
            later_bracket = ([later['arrival']['clock_seconds'],
                              later['departure']['clock_seconds']]
                             if later is not None else None)
            delta, bounds = None, None
            if verified and later is not None:
                delta = later_bracket[1]-served_bracket[1]
                bounds = [later_bracket[0]-served_bracket[1],
                          later_bracket[1]-served_bracket[0]]
                deltas.append(delta)
                bracket_lows.append(bounds[0])
                bracket_highs.append(bounds[1])
            if len(cases) < 5:
                cases.append(dict(case_id=f'{name}-{index:03d}', category=category,
                    station_id=item['station'], line_id=item['line'],
                    observed_context=context, contrast_clock=item['clock'],
                    fresh_service=_service(item['pair']['fresh']),
                    baseline_service=_service(item['pair']['baseline']),
                    served_boarding_bracket=served_bracket,
                    waiting_arm_recovery_service=None if later is None else _service(later),
                    waiting_arm_boarding_bracket=later_bracket,
                    boarding_time_difference_bounds=bounds,
                    departure_observed_time_delta=delta,
                    right_censored=later is None))
        last_observed = max((row['departure']['observed_at']
                             for row in visits[waiting_arm].values()
                             if _physical(row['departure'])), default=None)
        group_rows = lambda groups, field: [dict({field: key}, **dict(counts))
            for key, counts in sorted(groups.items(),
                key=lambda value: (-value[1]['unique_person_stop'], value[0]))[:6]]
        result[name] = dict(waiting_arm=waiting_arm, boarded_arm=boarded_arm,
            raw_intersection_memberships=raw,
            raw_verified_memberships=verified_raw,
            raw_weak_context_memberships=raw-verified_raw,
            unique_person_stop=len(selected),
            unique_people=len({person for person, station in selected}),
            counts=dict(totals),
            departure_observed_time_delta=_summary(deltas),
            boarding_difference_lower_bounds=_summary(bracket_lows),
            boarding_difference_upper_bounds=_summary(bracket_highs),
            recovery_search_last_observed_departure=last_observed,
            station_groups=group_rows(stations, 'station_id'),
            line_groups=group_rows(lines, 'line_id'), cases=cases)
    return dict(window=dict(lo=lo, hi=hi, interval='[lo,hi)'),
        matched_departure_identities=len(common), audit=dict(audit), directions=result,
        interpretation='Observed queue-versus-onboard contrasts; not a denied-boarding count '
            'or a decomposition of passenger-ledger cost.',
        limitations=[
            'Each direction retains only the first in-window contrast per person and station; '
            'later waiting episodes at that station cannot be separated reliably.',
            'Both departures must lie in the window; boundary-straddling services are reported separately.',
            'A verified contrast requires both native arrivals to observe the person queued and not onboard. '
            'Missing or lagged arrival observations leave weak context, never inferred boarding.',
            'Boarding intervals and departure-observation differences are not exact waiting durations; '
            'a later same-station boarding can belong to another episode.',
            'Absent later new-boarding evidence is right-censored in the existing log, not proof of no boarding.',
            'No individual line compatibility is inferred from aggregate eligible_waiting or final metadata.',
            'Event membership totals, unique people, and boarding-delay intervals are distinct quantities '
            'and must not be added to passenger costs.'])
