"""Verify the actual tick cutoff and strict current-event identity."""
from types import SimpleNamespace

import pytest

from experiments.mcwm_closed_loop_native_context_v79_20261027 import run


def test_limit_keeps_completed_tick_and_ledger_without_marking_terminal():
    ledger = SimpleNamespace(time_seconds=16199.)
    bridge = SimpleNamespace(passenger_time_ledger=SimpleNamespace(snapshot=lambda: ledger),
        done=False, steps=0, horizon_seconds=24000)
    def original():
        ledger.time_seconds += .5
        bridge.steps += 1
    bridge._advance_one_step = original
    limit = run.NativeTimeLimit(bridge)
    bridge._advance_one_step()
    assert ledger.time_seconds == 16199.5 and bridge.steps == 1
    with pytest.raises(run.DiagnosticLimitReached):
        bridge._advance_one_step()
    assert ledger.time_seconds == 16200. and bridge.steps == 2 and limit.reached
    assert not bridge.done and bridge.horizon_seconds == 24000
    limit.close()
    assert bridge._advance_one_step is original


def test_limit_reports_real_overshoot_instead_of_relabeling_endpoint():
    ledger = SimpleNamespace(time_seconds=16199.5)
    def original():
        ledger.time_seconds += 1.
    bridge = SimpleNamespace(passenger_time_ledger=SimpleNamespace(snapshot=lambda: ledger),
        _advance_one_step=original)
    limit = run.NativeTimeLimit(bridge)
    with pytest.raises(RuntimeError, match='exceeded'):
        bridge._advance_one_step()
    assert ledger.time_seconds == 16200.5 and not limit.reached
    limit.close()


def test_match_uses_native_trip_line_stop_not_physical_vehicle_or_nearest_time():
    wrong_trip = SimpleNamespace(bus_id='fleet7', line_id='7X', stop_id='s1', sim_time=12000)
    right = SimpleNamespace(bus_id='trip7', line_id='7X', stop_id='s1', sim_time=11999.5, stop_idx=4)
    wrong_stop = SimpleNamespace(bus_id='trip7', line_id='7X', stop_id='s2', sim_time=12000)
    env = SimpleNamespace(_current_events=[wrong_trip, right, wrong_stop])
    issue = dict(trip_id='trip7', vehicle_id='fleet7', line_id='7X', station_id='s1', started_at=12000)
    assert run.native_event(env, issue) is right
    assert run.native_event(env, issue).stop_idx == 4


def test_current_event_missing_or_ambiguous_is_not_guessed():
    event = SimpleNamespace(bus_id='trip7', line_id='7X', stop_id='s1')
    issue = dict(trip_id='trip7', line_id='7X', station_id='s1')
    for events in ([], [event, event]):
        with pytest.raises(RuntimeError, match='uniquely'):
            run.native_event(SimpleNamespace(_current_events=events), issue)
