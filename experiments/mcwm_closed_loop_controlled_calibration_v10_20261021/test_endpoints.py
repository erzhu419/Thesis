"""Detect endpoint leakage, invented clocks, and slope fitting outside its budget."""

from pathlib import Path
import json
from types import SimpleNamespace
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.endpoints import (
    ControlledResponseReadout, ENDPOINT_NAMES, ResponseEndpointCollector,
)
from experiments.mcwm_closed_loop_v1.stream import EI


def snapshot(now, *, wait=None, ride=None):
    return SimpleNamespace(time_seconds=float(now), waiting_passengers=2,
        in_vehicle_passengers=3, accrued_wait_seconds=2*now if wait is None else wait,
        accrued_in_vehicle_seconds=3*now if ride is None else ride)


def issue(sequence=0, line="7X", *, now=0., hold=0., vehicle="A", trip="T", station=13):
    event = np.zeros(16, np.float32)
    values = dict(station_index=station, sim_time_seconds=now,
        bus_capacity=50, onboard_passengers=10, eligible_stranded_passengers=5,
        forward_headway_seconds=300, backward_headway_seconds=300,
        target_forward_headway_seconds=300, target_backward_headway_seconds=300)
    for key, value in values.items():
        event[EI[key]] = value
    s = np.zeros(229, np.float32)
    s[1] = 1/16
    return dict(sequence=sequence, line_id=line, vehicle_id=vehicle, trip_id=trip,
        station_id=f"S{station}", event=event, s=s, raw_s=s.copy(),
        a=np.array([-1.+hold/30.], np.float32), started_at=float(now))


def transition(saved, end, *, next_vehicle="B"):
    return SimpleNamespace(sequence=saved["sequence"], s=saved["s"].copy(),
        a=saved["a"].copy(), completed_at=float(end), dt=float(end)-saved["started_at"],
        next_vehicle_id=next_vehicle, terminal=False)


def test_three_real_endpoints_remain_distinct_until_last_endpoint_plus_delay():
    collector = ResponseEndpointCollector(delay_seconds=30)
    issued = issue()
    context = {"memory_features": np.arange(18, dtype=float)}
    row = collector.register(issued, snapshot(0), context, hold_seconds=0.)
    # Cross-bus same-time global closure is a valid zero target, not the
    # next-stop duration or the 300-second system passenger cost.
    global_row = transition(issued, 0)
    collector.observe_global(global_row, snapshot(0))
    collector.observe_global(global_row, snapshot(0))
    collector.observe_global(transition(issue(1, now=10), 20), snapshot(20))
    collector.observe_tick(snapshot(300))
    collector.observe_global(transition(issue(2, now=310), 320), snapshot(320))
    collector.observe_event(issue(2, now=410, station=14), snapshot(410))
    assert row.status == "complete" and row.global_next is global_row
    assert row.endpoints["global_next"]["dt"] == 0
    assert row.endpoints["global_next"]["cost"] == 0
    assert row.endpoints["same_vehicle_next"]["dt"] == 410
    assert row.endpoints["same_vehicle_next"]["cost"] == 2870
    assert row.endpoints["fixed300"]["dt"] == 300
    assert row.endpoints["fixed300"]["cost"] == 2100
    assert row.endpoints["fixed300"]["global_events"] == 2
    assert row.endpoints["same_vehicle_next"]["global_events"] == 3
    assert row.completed_at == 410 and row.available_at == 440
    assert collector.release(410) == []
    collector.observe_tick(snapshot(439))
    assert collector.release() == []
    collector.observe_tick(snapshot(440))
    assert collector.release() == [row] and collector.release() == []
    issued["s"][0] = 99
    context["memory_features"][0] = 99
    assert row.s[0] == 0 and row.context["memory_features"][0] == 0


def test_trip_restart_and_missed_exact_clock_preserve_missing_outcomes():
    collector = ResponseEndpointCollector()
    saved = issue()
    row = collector.register(saved, snapshot(0))
    collector.observe_global(transition(saved, 1), snapshot(1))
    collector.observe_event(issue(1, now=50, trip="new", station=14), snapshot(50))
    collector.observe_tick(snapshot(301))
    assert row.status == "missing" and collector.release() == []
    assert row.endpoints["same_vehicle_next"]["reason"] == "vehicle_started_another_trip"
    assert row.endpoints["fixed300"]["reason"] == "exact_native_endpoint_clock_not_observed"
    assert "cost" not in row.endpoints["fixed300"]
    unfinished = collector.register(issue(2, now=301, vehicle="C"), snapshot(301), role="heldout")
    missing = collector.finish(330)
    assert missing == [row, unfinished]
    assert unfinished.endpoints["global_next"]["status"] == "missing"
    assert collector.summary()["missing"] == 2


def test_native_detect_then_tick_then_deliver_uses_delivery_ledger_at_both_ends():
    collector = ResponseEndpointCollector()
    issued = issue(now=601.)
    issued['event'][EI['sim_time_seconds']] = 600.
    row = collector.register(issued, snapshot(601.), hold_seconds=0.)
    collector.observe_global(transition(issued, 610.), snapshot(610.))
    detected = issue(1, now=1011., station=14)
    collector.observe_event(detected, snapshot(1012.))
    actual = row.endpoints['same_vehicle_next']
    assert actual['dt'] == 411.  # issue601 -> delivered1012, not detection1011
    assert actual['cost'] == 7*411
    assert actual['event_detected_at'] == 1011.
    assert actual['event_delivered_at'] == 1012.
    assert actual['delivery_lag_seconds'] == row.summary()['issue_delivery_lag_seconds'] == 1.


def complete_record(sequence, line, hold, *, role="train"):
    collector = ResponseEndpointCollector()
    saved = issue(sequence, line, hold=hold)
    row = collector.register(saved, snapshot(0), role=role, hold_seconds=hold)
    collector.observe_global(transition(saved, 1), snapshot(1, wait=2, ride=3))
    collector.observe_event(issue(sequence+100, line, now=100+hold, station=14),
        snapshot(100+hold, wait=100+2*hold, ride=200+5*hold))
    collector.observe_tick(snapshot(300, wait=300+10*hold, ride=600+hold))
    collector.observe_tick(snapshot(330))
    assert collector.release() == [row]
    return row


def test_small_response_fit_uses_budgeted_controlled_labels_and_actual_action_slope():
    model = ControlledResponseReadout(ridge=.01)
    records = [complete_record(i, line, hold) for i, (line, hold) in enumerate(
        [(line, hold) for line in ("7X", "102X", "705X") for hold in (0., 7.5)])]
    assert model.predict_issue(issue())["status"] == "unsupported"
    model.accept(records[:1], now=330)
    model.refit()
    assert model.predict_issue(issue())["status"] == "unsupported"
    model.accept(records[1:], now=330)
    model.refit()
    prediction = model.predict_issue(issue())
    assert prediction["status"] == "supported"
    assert abs(prediction["gains"]["global_next"]) < 1e-10
    assert prediction["gains"]["same_vehicle_next"] < -60
    assert prediction["gains"]["fixed300"] < -150
    assert model.summary()["selected_issue_labels"] == 6  # not 6*3 endpoints
    assert model.summary()["design"]["rank"] == 6
    assert model.summary()["design"]["columns"] == 13
    assert all(v["zero_holding"] == v["high_holding"] == 1 for v in model.coverage().values())
    # Native finalization writes these real fitted counts, not just fixtures.
    json.dumps(model.summary(), allow_nan=False)
    json.dumps([r.summary() for r in records], allow_nan=False)
    errors = model.chosen_outcome_errors(records[0], prediction)
    assert errors["chosen_action"] == "low" and set(errors["endpoints"]) == set(ENDPOINT_NAMES)
    with pytest.raises(ValueError, match="another training slot"):
        model.accept(records[:1], now=330)
    with pytest.raises(ValueError, match="Held-out"):
        model.accept([complete_record(99, "7X", 0, role="heldout")], now=330)


def test_zero_duration_global_path_is_analytic_and_never_used_for_same_vehicle():
    model = ControlledResponseReadout()
    model.accept([complete_record(i, "7X", hold) for i, hold in enumerate((0., 7.5))], now=330)
    model.refit()
    selected = issue()
    selected["s"][1] = 2/16
    prediction = model.predict_issue(selected)
    assert prediction["low"]["endpoints"]["global_next"]["zero_duration_analytic"]
    assert prediction["gains"]["global_next"] == 0
    assert prediction["gains"]["same_vehicle_next"] < 0


def test_unreleased_or_future_response_cannot_fit_and_wrong_issue_clock_is_rejected():
    collector = ResponseEndpointCollector()
    with pytest.raises(ValueError, match="issue clock"):
        collector.register(issue(now=10), snapshot(9))
    row = complete_record(1, "7X", 0)
    with pytest.raises(ValueError, match="released"):
        ControlledResponseReadout().accept([row], now=329)
    row.released = False
    with pytest.raises(ValueError, match="released"):
        ControlledResponseReadout().accept([row], now=330)
    with pytest.raises(ValueError, match="observed native clock"):
        collector.release(100)


def test_global_fit_excludes_zero_dt_and_checks_its_own_action_support():
    low, high = [complete_record(i, "7X", hold) for i, hold in enumerate((0., 7.5))]
    high.endpoints["global_next"].update(dt=0., wait=0., ride=0., cost=0., zero_duration=True)
    model = ControlledResponseReadout()
    model.accept([low, high], now=330)
    model.refit()
    prediction = model.predict_issue(issue())
    assert prediction["status"] == "supported"
    assert prediction["endpoint_status"]["same_vehicle_next"] == "supported"
    assert prediction["endpoint_status"]["global_next"] == "unsupported"
    assert prediction["gains"]["global_next"] is None
    assert prediction["raw_global_next_gain"] is None
    assert model.summary()["endpoint_designs"]["global_next"]["rows"] == 1
    assert model.summary()["endpoint_designs"]["same_vehicle_next"]["rows"] == 2
    errors = model.chosen_outcome_errors(low, prediction)
    assert errors["endpoints"]["global_next"]["status"] == "unsupported"
    selected = issue()
    selected["s"][1] = 2/16
    analytic = model.predict_issue(selected)
    assert analytic["endpoint_status"]["global_next"] == "analytic_zero_duration"
    assert analytic["gains"]["global_next"] == 0
    assert analytic["raw_global_next_gain"] is None
