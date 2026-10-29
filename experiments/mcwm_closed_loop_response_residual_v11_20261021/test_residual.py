"""Check observable baselines, fixed clocks, sample budget and endpoint support."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.test_endpoints import (
    complete_record, issue,
)
from experiments.mcwm_closed_loop_response_residual_v11_20261021.residual import (
    ResidualResponseReadout, endpoint_baseline,
)


def observable_context(*, wait=1000., ride=2000., nonholding=100., low_dt=2., high_dt=3.):
    return dict(memory_features=np.arange(18, dtype=float), nonholding_seconds=nonholding,
        issue_ledger=dict(waiting_passengers=wait, in_vehicle_passengers=ride),
        global_baselines=dict(low=dict(dt=low_dt, raw_dt=999., cost=8000.),
                              high=dict(dt=high_dt, raw_dt=998., cost=12000.)))


def baseline_record(sequence, line, hold, *, role="train", context=None):
    row = complete_record(sequence, line, hold, role=role)
    row.context = observable_context() if context is None else deepcopy(context)
    row.issue_ledger.update(row.context["issue_ledger"])
    for endpoint in row.endpoints:
        row.endpoints[endpoint].update(endpoint_baseline(row, endpoint))
    return row


def fitted_model():
    model = ResidualResponseReadout(ridge=1.)
    records = [baseline_record(i, line, hold, context=observable_context(
        wait=1000.+100*i, ride=2000.+50*i, nonholding=100.+10*i))
        for i, (line, hold) in enumerate((line, hold)
            for line in ("7X", "102X", "705X") for hold in (0., 7.5))]
    model.accept(records, now=330)
    model.refit()
    return model, records


def test_global_system_counts_and_physical_clock_supply_issue_time_baselines():
    row = baseline_record(0, "7X", 7.5)
    # These are global counts, rather than the event's ten onboard passengers.
    assert endpoint_baseline(row, "same_vehicle_next") == dict(
        dt=107.5, wait=107500., ride=215000., cost=430000.)
    assert endpoint_baseline(row, "fixed300") == dict(
        dt=300., wait=300000., ride=600000., cost=1200000.)
    # Positive global uses actual unchanged world decoder dt, never raw 998.
    assert endpoint_baseline(row, "global_next")["cost"] == 12000.
    # Future response values cannot change an issue-time baseline.
    for endpoint in row.endpoints.values():
        endpoint.update(dt=99999., wait=99999., ride=99999., cost=99999.)
    assert endpoint_baseline(row, "same_vehicle_next")["dt"] == 107.5
    assert endpoint_baseline(row, "global_next")["dt"] == 3.


def test_zero_residual_prior_preserves_varying_system_levels_without_refitting_clock():
    model, records = fitted_model()
    context = observable_context(wait=4321., ride=8765., nonholding=200.)
    prediction = model.predict_issue(issue(), context)
    assert prediction["low"]["endpoints"]["fixed300"]["dt"] == 300.
    assert prediction["high"]["endpoints"]["fixed300"]["dt"] == 300.
    assert prediction["high_minus_low"]["fixed300"]["dt"] == 0.
    assert prediction["gains"]["fixed300"] == 0.
    assert prediction["low"]["endpoints"]["fixed300"]["cost"] == (2*4321+8765)*300
    assert prediction["gains"]["same_vehicle_next"] == -(2*4321+8765)*7.5
    assert model.readouts["fixed300"].shape == (13, 2)
    assert np.count_nonzero(model.readouts["fixed300"]) == 0
    assert model.summary()["selected_issue_labels"] == 6
    assert model.summary()["residual_targets"]["fixed300"] == ["wait", "ride"]
    json.dumps(model.summary(), allow_nan=False)
    json.dumps(prediction, allow_nan=False)
    assert model.chosen_outcome_errors(records[0], prediction)["chosen_action"] == "low"


def test_fits_action_response_residual_without_using_future_duration_as_cost_offset():
    model = ResidualResponseReadout(ridge=.00001)
    records = [baseline_record(i, "7X", hold) for i, hold in enumerate((0., 7.5))]
    for row in records:
        # Response clock can differ from the issue physical clock. Cost labels
        # deliberately retain the issue baseline plus a small controlled effect.
        row.endpoints["same_vehicle_next"]["dt"] += 40.
        for endpoint in row.endpoints:
            row.endpoints[endpoint]["wait"] += 2*row.hold_seconds
            row.endpoints[endpoint]["ride"] += row.hold_seconds
            row.endpoints[endpoint]["cost"] = 2*row.endpoints[endpoint]["wait"]+row.endpoints[endpoint]["ride"]
    model.accept(records, now=330)
    model.refit()
    context = observable_context()
    prediction = model.predict_issue(issue(), context)
    assert prediction["low"]["endpoints"]["same_vehicle_next"]["dt"] > 139.9
    # Future 40 seconds are not secretly multiplied into W/B baseline labels.
    assert abs(prediction["low"]["endpoints"]["same_vehicle_next"]["cost"]-400000.) < .01
    assert abs(prediction["gains"]["fixed300"]+37.5) < .001
    assert prediction["high_minus_low"]["fixed300"]["dt"] == 0
    before = deepcopy(prediction)
    for row in records:
        row.endpoints["same_vehicle_next"]["dt"] += 100000.
        row.endpoints["fixed300"]["wait"] += 10000000.
    assert model.predict_issue(issue(), context) == before


def test_heldout_future_or_duplicate_issue_cannot_increase_training_budget():
    model, records = fitted_model()
    original = model.predict_issue(issue(), observable_context())
    heldout = baseline_record(99, "7X", 0., role="heldout")
    with pytest.raises(ValueError, match="Held-out"):
        model.accept([heldout], now=330)
    with pytest.raises(ValueError, match="another training slot"):
        model.accept(records[:1], now=330)
    future = baseline_record(100, "7X", 0.)
    with pytest.raises(ValueError, match="released"):
        model.accept([future], now=329)
    assert len(model.records) == 6
    assert model.predict_issue(issue(), observable_context()) == original


def test_zero_global_is_excluded_from_fit_but_not_delayed_endpoint_support():
    model = ResidualResponseReadout()
    low, high = [baseline_record(i, "7X", hold) for i, hold in enumerate((0., 7.5))]
    high.endpoints["global_next"].update(dt=0., wait=0., ride=0., cost=0.)
    model.accept([low, high], now=330)
    model.refit()
    chosen = issue()
    prediction = model.predict_issue(chosen, observable_context())
    assert prediction["endpoint_status"]["global_next"] == "unsupported"
    assert prediction["raw_global_next_gain"] is None
    assert prediction["endpoint_status"]["same_vehicle_next"] == "supported"
    assert model.summary()["endpoint_designs"]["global_next"]["rows"] == 1
    assert model.summary()["endpoint_designs"]["same_vehicle_next"]["rows"] == 2
    chosen["s"][1] = 2/16
    analytic = model.predict_issue(chosen, observable_context())
    assert analytic["endpoint_status"]["global_next"] == "analytic_zero_duration"
    assert analytic["gains"]["global_next"] == 0.
    assert analytic["raw_global_next_gain"] is None
    # Analytic global is unrelated to the same-car physical prediction.
    assert analytic["low"]["endpoints"]["same_vehicle_next"]["dt"] == 100.


def test_supported_zero_batch_retains_raw_before_exact_override():
    model, _ = fitted_model()
    chosen = issue()
    chosen["s"][1] = 2/16
    context = observable_context()
    context["global_baselines"]["low"].update(dt=0., raw_dt=2.)
    context["global_baselines"]["high"].update(dt=0., raw_dt=3.)
    prediction = model.predict_issue(chosen, context)
    assert prediction["raw_global_next_gain"] == -4000.
    assert prediction["gains"]["global_next"] == 0.
    assert prediction["low"]["endpoints"]["global_next"]["dt"] == 0.
    context["global_baselines"]["low"]["raw_dt"] = -2.
    assert model.baseline_predict_issue(chosen, context)["low"]["raw_global_next"]["dt"] == 0.


def test_missing_issue_baseline_stays_unsupported_and_baseline_error_is_reportable():
    model, _ = fitted_model()
    context = observable_context()
    context["nonholding_seconds"] = None
    del context["global_baselines"]
    prediction = model.predict_issue(issue(), context)
    assert prediction["endpoint_status"]["same_vehicle_next"] == "missing_issue_baseline"
    assert prediction["endpoint_status"]["global_next"] == "missing_issue_baseline"
    assert prediction["gains"]["global_next"] is None
    assert prediction["gains"]["same_vehicle_next"] is None
    assert prediction["low"]["endpoints"]["fixed300"]["dt"] == 300.
    baseline = ResidualResponseReadout().baseline_predict_issue(issue(), observable_context())
    assert baseline["status"] == "supported"
    assert baseline["low"]["endpoints"]["same_vehicle_next"]["dt"] == 100.
    assert baseline["high"]["endpoints"]["same_vehicle_next"]["dt"] == 107.5
    json.dumps(baseline, allow_nan=False)
