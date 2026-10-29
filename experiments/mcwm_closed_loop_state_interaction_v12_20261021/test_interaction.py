"""Check that state-dependent action contrasts preserve clocks and label scope."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.endpoints import response_design
from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.test_endpoints import issue
from experiments.mcwm_closed_loop_response_residual_v11_20261021.residual import endpoint_baseline
from experiments.mcwm_closed_loop_response_residual_v11_20261021.test_residual import (
    baseline_record, observable_context,
)
from experiments.mcwm_closed_loop_state_interaction_v12_20261021.interaction import (
    INTERACTION_FEATURE_NAMES, INTERACTION_NAMES, StateInteractionReadout, state_interaction_design,
)
from experiments.mcwm_closed_loop_v1.stream import EI


def trained_model():
    model = StateInteractionReadout(ridge=1.)
    records = []
    for line in ("7X", "102X", "705X"):
        for deviation in (-.75, -.25, .25, .75):
            for hold in (0., 7.5):
                row = baseline_record(len(records), line, hold)
                row.event[EI["forward_headway_seconds"]] = 300.*(1.+deviation)
                u = 2.*hold/7.5-1.
                for endpoint in row.endpoints.values():
                    endpoint["wait"] += u*deviation*10000.
                    endpoint["ride"] += u*deviation*5000.
                    endpoint["cost"] = 2.*endpoint["wait"]+endpoint["ride"]
                records.append(row)
    model.accept(records, now=330.)
    model.refit()
    return model, records


def test_original_columns_and_issue_time_shared_interactions_are_exact():
    row = baseline_record(0, "102X", 7.5)
    row.event[EI["forward_headway_seconds"]] = 600.
    x = state_interaction_design(row)
    assert len(x) == 19 and len(INTERACTION_FEATURE_NAMES) == 19
    assert np.array_equal(x[:13], response_design(row))
    assert np.array_equal(x[13:], np.concatenate((x[6:10], [4., 1.])))
    low = state_interaction_design(row, hold_seconds=0.)
    assert np.array_equal(low[:3], x[:3])
    assert np.array_equal(low[6:13], x[6:13])
    assert np.array_equal(low[13:], -x[13:])
    assert np.array_equal(low[3:6], -x[3:6])


def test_actual_fit_allows_same_route_gain_to_reverse_with_observable_state():
    model, records = trained_model()
    states = [issue(line="7X"), issue(line="7X")]
    states[0]["event"][EI["forward_headway_seconds"]] = 75.
    states[1]["event"][EI["forward_headway_seconds"]] = 525.
    predictions = [model.predict_issue(state, observable_context()) for state in states]
    assert predictions[0]["gains"]["fixed300"] > 10000.
    assert predictions[1]["gains"]["fixed300"] < -10000.
    assert model.readouts["fixed300"].shape == (19, 2)
    assert model.summary()["selected_issue_labels"] == len(records) == 24
    assert model.summary()["design"]["columns"] == 19
    assert model.summary()["feature_names"] == list(INTERACTION_FEATURE_NAMES)
    assert model.summary()["shared_state_interaction_columns"] == list(INTERACTION_NAMES)
    contrast = model.summary()["contrast_design"]
    assert contrast["columns"] == 19 and contrast["rank"] <= 9
    assert 0 < contrast["rank_added_beyond_original13"] <= 6
    assert model.summary()["endpoint_contrast_designs"]["fixed300"]["rows"] == len(records)
    assert model.summary()["perline_state_ranges"]["7X"]["states"]["forward_headway_deviation"] == dict(min=-.75, max=.75)
    json.dumps(model.summary(), allow_nan=False)


def test_fixed300_clock_and_physical_baselines_are_unchanged():
    model, _ = trained_model()
    context = observable_context(wait=4321., ride=8765., nonholding=200.)
    prediction = model.predict_issue(issue(), context)
    baseline = model.baseline_predict_issue(issue(), context)
    for candidate in ("low", "high"):
        assert prediction[candidate]["endpoints"]["fixed300"]["dt"] == 300.
        assert baseline[candidate]["endpoints"]["fixed300"]["dt"] == 300.
        assert baseline[candidate]["endpoints"]["fixed300"]["cost"] == (2.*4321+8765)*300.
    assert prediction["high_minus_low"]["fixed300"]["dt"] == 0.
    assert baseline["gains"]["same_vehicle_next"] == -(2.*4321+8765)*7.5


def test_contrast_components_sum_in_physical_units_and_level_terms_cancel():
    model, _ = trained_model()
    state = issue(line="705X")
    state["event"][EI["forward_headway_seconds"]] = 450.
    context = observable_context()
    prediction = model.predict_issue(state, context)
    diagnostic = model.contrast_diagnostic(state, context)
    for name, result in diagnostic["endpoints"].items():
        for key in ("dt", "wait", "ride", "cost"):
            assert result["level"][key] == 0.
            components = result["baseline"][key]+result["line_holding"][key]
            components += sum(value[key] for value in result["state_interactions"].values())
            assert components == pytest.approx(result["raw_total"][key], abs=1e-8)
            assert result["total"][key] == pytest.approx(-prediction["high_minus_low"][name][key], abs=1e-8)
        assert result["total"]["cost"] == pytest.approx(prediction["gains"][name], abs=1e-8)
        assert result["fixed_weight_no_state_interactions"]["cost"] == pytest.approx(
            result["baseline"]["cost"]+result["line_holding"]["cost"])
    assert diagnostic["endpoints"]["fixed300"]["total"]["dt"] == 0.
    json.dumps(diagnostic, allow_nan=False)


def test_future_response_mutation_cannot_change_features_baselines_or_saved_predictions():
    model, records = trained_model()
    row = records[0]
    features = state_interaction_design(row).copy()
    baselines = {name: endpoint_baseline(row, name) for name in row.endpoints}
    prediction = model.predict_issue(issue(), observable_context())
    for outcome in row.endpoints.values():
        outcome.update(dt=99999., wait=9999999., ride=8888888., cost=28888886.)
    assert np.array_equal(state_interaction_design(row), features)
    assert {name: endpoint_baseline(row, name) for name in row.endpoints} == baselines
    assert model.predict_issue(issue(), observable_context()) == prediction


def test_raw_zero_duration_contrast_is_retained_before_exact_zero_override():
    model, _ = trained_model()
    state = issue()
    state["s"][1] = 2/16
    state["event"][EI["forward_headway_seconds"]] = 450.
    context = observable_context()
    context["global_baselines"]["low"].update(dt=0., raw_dt=2.)
    context["global_baselines"]["high"].update(dt=0., raw_dt=3.)
    prediction = model.predict_issue(state, context)
    diagnostic = model.contrast_diagnostic(state, context)["endpoints"]["global_next"]
    assert prediction["gains"]["global_next"] == 0.
    assert diagnostic["total"] == dict(dt=0., wait=0., ride=0., cost=0.)
    assert diagnostic["raw_total"]["cost"] == pytest.approx(prediction["raw_global_next_gain"])
    assert diagnostic["raw_total"]["cost"] != 0.
    for key in diagnostic["total"]:
        assert diagnostic["raw_total"][key]+diagnostic["analytic_adjustment"][key] == 0.


def test_heldout_duplicate_unreleased_and_missing_action_support_remain_distinct():
    model, records = trained_model()
    heldout = baseline_record(99, "7X", 0., role="heldout")
    with pytest.raises(ValueError, match="Held-out"):
        model.accept([heldout], now=330.)
    with pytest.raises(ValueError, match="another training slot"):
        model.accept(records[:1], now=330.)
    future = baseline_record(100, "7X", 0.)
    with pytest.raises(ValueError, match="released"):
        model.accept([future], now=329.)
    single = StateInteractionReadout()
    single.accept([deepcopy(records[0])], now=330.)
    single.refit()
    prediction = single.predict_issue(issue(), observable_context())
    assert prediction["endpoint_status"]["fixed300"] == "unsupported"
    assert prediction["gains"]["fixed300"] is None
    assert single.summary()["endpoint_designs"]["fixed300"]["columns"] == 19
