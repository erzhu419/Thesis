"""Fake-API interface tests, not evidence of native snapshot coverage."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location('capacity_context_v78',
    Path(__file__).with_name('context.py'))
CONTEXT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTEXT)


def fake_api(*, onboard=(), queue=(), stages=None, clocks=(101., 101.)):
    """Expose exactly the read-only interface available at issue time."""
    calls, clock_values = [], iter(clocks)
    stage_values = {} if stages is None else stages

    def time():
        calls.append(('clock',))
        return next(clock_values)

    def get_stage(person, index):
        calls.append(('stage', person, index))
        if index != 0:
            raise AssertionError('Only the current native stage is permitted')
        stage = stage_values[person]
        if isinstance(stage, Exception):
            raise stage
        return stage

    def get_queue(stop):
        calls.append(('queue', stop))
        return queue

    return SimpleNamespace(
        simulation=SimpleNamespace(getTime=time),
        vehicle=SimpleNamespace(getPersonIDList=lambda trip: onboard,
            getPersonCapacity=lambda trip: 50., getLine=lambda trip: 'L'),
        person=SimpleNamespace(getStage=get_stage),
        busstop=SimpleNamespace(getPersonIDs=get_queue), calls=calls)


def observe(api, **kwargs):
    args = dict(trip_id='L_trip', line_id='L', station_index=0,
        route_stops=('S0', 'S1', 'S2'), issue_clock=101., event_clock=100.)
    args.update(kwargs)
    return CONTEXT.observe_context(api, **args)


def driving(destination, lines='L'):
    return SimpleNamespace(type=3, destStop=destination, line=lines)


def test_current_second_leg_destination_and_exact_capacity_margin():
    # Legacy transfer/end fields are deliberately absent. Current stage wins.
    api = fake_api(onboard=('second_leg', 'through'), queue=('yes', 'no'),
        stages=dict(second_leg=driving('S1'), through=driving('S2'),
                    yes=driving('S2', 'OTHER L'), no=driving('S2', 'OTHER')))
    result = observe(api)
    assert result['observed'] and result['reasons'] == []
    assert result['counts'] == dict(onboard=2, next_queue=2, release_next=1,
        compatible_next_queue=1, seats_after_release=49.)
    assert result['z'] == (1-49)/50
    assert all(call[2] == 0 for call in api.calls if call[0] == 'stage')


def test_next_stop_uses_route_index_and_empty_memberships_are_observed():
    api = fake_api()
    result = observe(api, station_index=1)
    assert result['next_stop'] == 'S2'
    assert ('queue', 'S2') in api.calls
    assert result['observed'] and result['z'] == -1.
    assert result['counts']['release_next'] == result['counts']['compatible_next_queue'] == 0


def test_terminal_is_missing_context_without_reading_memberships():
    api = fake_api()
    result = observe(api, station_index=2)
    assert not result['observed'] and result['z'] is None
    assert result['reasons'] == ['terminal_no_next_stop']
    assert result['counts']['onboard'] is None
    assert not any(call[0] in ('queue', 'stage') for call in api.calls)


@pytest.mark.parametrize('group,stage,reason', [
    ('onboard', RuntimeError('removed native person'), 'onboard_stage_missing'),
    ('onboard', driving(''), 'onboard_destination_missing'),
    ('queue', SimpleNamespace(type=2), 'queue_stage_not_driving'),
    ('queue', driving('S2', ''), 'queue_line_condition_unsupported'),
])
def test_missing_or_unsupported_current_stage_cannot_become_zero(group, stage, reason):
    api = fake_api(onboard=('p',) if group == 'onboard' else (),
                   queue=('p',) if group == 'queue' else (), stages={'p': stage})
    result = observe(api)
    assert not result['observed'] and result['z'] is None
    assert reason in result['reasons']
    count = 'release_next' if group == 'onboard' else 'compatible_next_queue'
    assert result['counts'][count] is None


def test_read_only_interface_has_no_simulation_or_legacy_dependency():
    # This fake has no start/step/load/reset APIs or legacy passenger objects.
    person_id = 'private_person_id_1'
    api = fake_api(queue=(person_id,), stages={person_id: driving('S2', 'L')})
    before = dict(vars(api.person))
    result = observe(api)
    assert result['observed'] and api.person.__dict__ == before
    assert [call[0] for call in api.calls] == ['clock', 'queue', 'stage', 'clock']
    assert person_id not in str(result)


@pytest.mark.parametrize('clocks', [(101., 102.), (100., 100.)])
def test_event_clock_lag_is_allowed_but_nonatomic_or_old_observation_is_missing(clocks):
    result = observe(fake_api(clocks=clocks))
    assert result['clocks']['event'] == 100.
    assert not result['observed'] and result['z'] is None
    assert result['reasons'] == ['observation_not_atomic_at_issue']
    assert observe(fake_api())['observed']
