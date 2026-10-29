import json
from opportunities import waiting_opportunities


def visit(bus, arrival, departure, aq=(), ao=(), dq=(), do=(), station='s',
          line='l', occurrence=0):
    base = dict(bus_id=bus, line_id=line, station_id=station, occurrence=occurrence,
                capacity=2, free_capacity=0, eligible_waiting=1)
    return [dict(base, kind=kind, clock_seconds=clock, observed_at=clock,
                 queued_passenger_ids=list(queue), onboard_passenger_ids=list(onboard))
            for kind, clock, queue, onboard in
            [('arrival', arrival, aq, ao), ('departure', departure, dq, do)]]


def direction(fresh, baseline, **kwargs):
    return waiting_opportunities(fresh, baseline, **kwargs)['directions'][
        'fresh_wait_baseline_boarded']


def test_verified_contrast_and_same_station_new_boarding_brackets():
    fresh = visit('b', 12001, 12010, aq=['p'], dq=['p'])
    baseline = visit('b', 12002, 12008, aq=['p'], do=['p'])
    fresh += visit('next', 12100, 12110, aq=['p'], do=['p'], line='other')
    out = direction(fresh, baseline)
    assert out['counts']['verified_board_vs_wait'] == 1
    assert out['counts']['recovered_with_new_boarding_evidence'] == 1
    case = out['cases'][0]
    assert case['served_boarding_bracket'] == [12002, 12008]
    assert case['waiting_arm_boarding_bracket'] == [12100, 12110]
    assert case['boarding_time_difference_bounds'] == [92, 108]
    assert out['departure_observed_time_delta']['mean'] == 102
    assert '"p"' not in json.dumps(out)


def test_missing_prior_queue_is_weak_not_denied_boarding_or_recovery():
    fresh = visit('b', 12001, 12010, aq=['p'], dq=['p'])
    baseline = visit('b', 12002, 12008, ao=['p'], do=['p'])
    fresh += visit('next', 12100, 12110, ao=['p'], do=['p'])
    out = direction(fresh, baseline)
    assert out['counts']['weak_context'] == 1
    assert out['counts']['right_censored_no_recovery_observed'] == 1
    assert out['cases'][0]['served_boarding_bracket'] is None
    assert out['departure_observed_time_delta']['count'] == 0


def test_person_station_dedup_and_preexisting_window_context():
    fresh, baseline = [], []
    for bus, arrival, departure in [('early', 11900, 11910),
                                     ('first', 12001, 12010), ('second', 12100, 12110)]:
        fresh += visit(bus, arrival, departure, aq=['p'], dq=['p'])
        baseline += visit(bus, arrival, departure, aq=['p'], do=['p'])
    out = direction(fresh, baseline)
    assert out['raw_intersection_memberships'] == 2
    assert out['raw_verified_memberships'] == 2
    assert out['unique_people'] == out['unique_person_stop'] == 1
    assert out['counts']['observed_before_window'] == 1
    assert out['cases'][0]['fresh_service']['bus_id'] == 'first'


def test_lagged_departure_unscored_and_lagged_arrival_not_verified():
    fresh = visit('b', 12001, 12010, aq=['p'], dq=['p'])
    baseline = visit('b', 12001, 12010, aq=['p'], do=['p'])
    fresh[0]['observed_at'] += 1
    assert direction(fresh, baseline)['counts']['weak_context'] == 1
    fresh[1]['observed_at'] += 1
    out = waiting_opportunities(fresh, baseline)
    assert out['audit']['matched_departures_missing_or_lagged_inventory'] == 1
    assert out['directions']['fresh_wait_baseline_boarded']['unique_person_stop'] == 0


def test_window_boundaries_excluded_without_cross_arm_repair():
    fresh = visit('start', 11998, 11999, aq=['p'], dq=['p'])
    baseline = visit('start', 11998, 12001, aq=['p'], do=['p'])
    fresh += visit('end', 16198, 16200, aq=['q'], dq=['q'])
    baseline += visit('end', 16198, 16200, aq=['q'], do=['q'])
    out = waiting_opportunities(fresh, baseline)
    assert out['audit']['cross_window_boundary_departures'] == 1
    assert out['directions']['fresh_wait_baseline_boarded']['unique_person_stop'] == 0


def test_first_weak_contrast_not_replaced_by_later_verified_episode():
    fresh = visit('first', 12001, 12010, aq=['p'], dq=['p'])
    baseline = visit('first', 12001, 12010, ao=['p'], do=['p'])
    fresh += visit('later', 12100, 12110, aq=['p'], dq=['p'])
    baseline += visit('later', 12100, 12110, aq=['p'], do=['p'])
    out = direction(fresh, baseline)
    assert out['unique_person_stop'] == 1
    assert out['raw_verified_memberships'] == 1
    assert out['counts'].get('verified_board_vs_wait', 0) == 0
    assert out['counts']['weak_context'] == 1
