"""Verify block cross-fitting, contrast separation and retained native clocks."""

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
from experiments.mcwm_closed_loop_response_residual_v11_20261021.test_residual import baseline_record, observable_context
from experiments.mcwm_closed_loop_separated_contrast_v13_20261021.separated import (
    ACTION_NAMES, BACKGROUND_NAMES, COST_RATE_ACTION_COLUMN, FEATURE_NAMES,
    SeparatedContrastReadout, assignment_fold, separated_design,
)
from experiments.mcwm_closed_loop_state_interaction_v12_20261021.interaction import INTERACTION_NAMES
from experiments.mcwm_closed_loop_v1.stream import EI


def fitted_model(background_scale=0.):
    rows = []
    for line in ("7X", "102X", "705X"):
        for block, deviation in enumerate(np.linspace(-.8, .8, 8)):
            context = observable_context(wait=1000.+150.*(block % 3),
                ride=2000.+90.*(block % 5), nonholding=100.+5.*(block % 2))
            context["assignment_block_index"] = block
            for hold in (0., 7.5):
                row = baseline_record(len(rows), line, hold, context=context)
                row.event[EI["forward_headway_seconds"]] = 300.*(1.+deviation)
                u = 2.*hold/7.5-1.
                background = background_scale*(3.+deviation)
                for endpoint in row.endpoints.values():
                    endpoint["wait"] += background+u*deviation*10000.
                    endpoint["ride"] += background*.4+u*deviation*5000.
                    endpoint["cost"] = 2.*endpoint["wait"]+endpoint["ride"]
                rows.append(row)
    model = SeparatedContrastReadout(ridge=1.)
    model.accept(rows, now=330.)
    model.refit()
    return model, rows


def test_background_is_action_free_and_assignment_blocks_stay_in_one_fold():
    model, rows = fitted_model()
    assert len(BACKGROUND_NAMES) == 12 and len(ACTION_NAMES) == 9 and len(FEATURE_NAMES) == 21
    for low, high in zip(rows[::2], rows[1::2]):
        zl, tl = separated_design(low)
        zh, th = separated_design(high)
        assert np.array_equal(zl, zh)
        assert np.array_equal(tl, -th)
        assert np.array_equal(zl[:10], response_design(low)[[0, 1, 2, 6, 7, 8, 9, 10, 11, 12]])
        assert assignment_fold(low) == assignment_fold(high)
    assert all(v["fold_counts"] == {"0": 24, "1": 24}
               for v in model.summary()["nuisance_diagnostics"].values())
    assert model.readouts["fixed300"].shape == (21, 2)


def test_state_dependent_contrast_reverses_and_background_levels_do_not_pollute_it():
    plain, _ = fitted_model()
    shifted, _ = fitted_model(background_scale=300000.)
    queries = [issue(), issue()]
    for query, deviation in zip(queries, (-.75, .75)):
        query["event"][EI["forward_headway_seconds"]] = 300.*(1.+deviation)
    p = [plain.predict_issue(q, observable_context()) for q in queries]
    s = [shifted.predict_issue(q, observable_context()) for q in queries]
    assert p[0]["gains"]["fixed300"] > 10000.
    assert p[1]["gains"]["fixed300"] < -10000.
    assert np.allclose(plain.readouts["fixed300"][12:], shifted.readouts["fixed300"][12:], atol=1e-11)
    for a, b in zip(p, s):
        assert a["gains"]["fixed300"] == pytest.approx(b["gains"]["fixed300"], abs=1e-7)
        assert b["low"]["endpoints"]["fixed300"]["cost"] > a["low"]["endpoints"]["fixed300"]["cost"]+100000.


def test_no_cost_rate_arm_refits_remaining_action_and_background_weights():
    model, _ = fitted_model(background_scale=1000.)
    prediction = model.predict_issue(issue(), observable_context(), omit_cost_rate=True)
    diagnostic = model.contrast_diagnostic(issue(), observable_context(), omit_cost_rate=True)
    assert diagnostic["arm"] == "no_global_cost_rate_refit"
    for endpoint in model.readouts:
        full, no_cost = model.readouts[endpoint], model.no_cost_readouts[endpoint]
        assert np.array_equal(no_cost[12+COST_RATE_ACTION_COLUMN], np.zeros(no_cost.shape[1]))
        assert np.linalg.norm(full[12+COST_RATE_ACTION_COLUMN]) > 0.
        assert not np.array_equal(full[:12], no_cost[:12])
        assert diagnostic["endpoints"][endpoint]["state_interactions"][INTERACTION_NAMES[4]] == dict(dt=0., wait=0., ride=0., cost=0.)
        assert diagnostic["endpoints"][endpoint]["total"]["cost"] == pytest.approx(prediction["gains"][endpoint])


def test_complete_contrast_decomposition_and_known_fixed300_clock():
    model, _ = fitted_model()
    query = issue(line="705X")
    query["event"][EI["forward_headway_seconds"]] = 450.
    context = observable_context(wait=4321., ride=8765., nonholding=200.)
    for omit in (False, True):
        prediction = model.predict_issue(query, context, omit_cost_rate=omit)
        diagnostic = model.contrast_diagnostic(query, context, omit_cost_rate=omit)
        for endpoint, result in diagnostic["endpoints"].items():
            for key in ("dt", "wait", "ride", "cost"):
                assert result["level"][key] == 0.
                components = result["baseline"][key]+result["line_holding"][key]
                components += sum(item[key] for item in result["state_interactions"].values())
                assert components == pytest.approx(result["raw_total"][key], abs=1e-8)
                assert result["total"][key] == pytest.approx(-prediction["high_minus_low"][endpoint][key], abs=1e-8)
                assert result["raw_fixed_weight_no_global_cost_rate"][key] == pytest.approx(
                    result["raw_total"][key]-result["state_interactions"][INTERACTION_NAMES[4]][key])
            assert result["raw_fixed_weight_no_state_interactions"]["cost"] == pytest.approx(
                result["baseline"]["cost"]+result["line_holding"]["cost"])
        for candidate in ("low", "high"):
            assert prediction[candidate]["endpoints"]["fixed300"]["dt"] == 300.
        assert prediction["high_minus_low"]["fixed300"]["dt"] == 0.
    baseline = model.baseline_predict_issue(query, context)
    assert baseline["gains"]["same_vehicle_next"] == -(2.*4321.+8765.)*7.5
    assert baseline["gains"]["fixed300"] == 0.


def test_zero_global_retains_raw_gain_and_has_exact_analytic_zero():
    model, _ = fitted_model()
    query = issue()
    query["s"][1] = 2/16
    context = observable_context()
    context["global_baselines"]["low"].update(dt=0., raw_dt=2.)
    context["global_baselines"]["high"].update(dt=0., raw_dt=3.)
    for omit in (False, True):
        prediction = model.predict_issue(query, context, omit_cost_rate=omit)
        result = model.contrast_diagnostic(query, context, omit_cost_rate=omit)["endpoints"]["global_next"]
        assert prediction["gains"]["global_next"] == 0.
        assert result["raw_total"]["cost"] == pytest.approx(prediction["raw_global_next_gain"])
        assert result["raw_total"]["cost"] != 0.
        for key in result["total"]:
            assert result["total"][key] == 0.
            assert result["raw_total"][key]+result["analytic_adjustment"][key] == 0.


def test_one_fold_remains_unsupported_despite_two_action_coverage():
    model = SeparatedContrastReadout()
    rows = [baseline_record(i, "7X", hold) for i, hold in enumerate((0., 7.5))]
    for row in rows:
        row.context["assignment_block_index"] = 0
    model.accept(rows, now=330.)
    model.refit()
    result = model.predict_issue(issue(), observable_context())
    assert result["gains"]["fixed300"] is None
    assert model.summary()["nuisance_diagnostics"]["fixed300"]["status"] == "requires_two_assignment_folds"
    assert model.baseline_predict_issue(issue(), observable_context())["gains"]["fixed300"] == 0.


def test_each_nuisance_model_excludes_its_heldout_blocks_and_future_labels():
    model, rows = fitted_model()
    before = deepcopy(model.summary()["nuisance_diagnostics"]["fixed300"]["fold_models"])
    for row in rows:
        if assignment_fold(row) == 0:
            row.endpoints["fixed300"]["wait"] += 100000.
    model.refit()
    after = model.summary()["nuisance_diagnostics"]["fixed300"]["fold_models"]
    assert before[0] == after[0]
    assert before[1] != after[1]
    query, context = issue(), observable_context()
    saved = model.predict_issue(query, context)
    features = separated_design(rows[0])
    base = endpoint_baseline(rows[0], "fixed300")
    rows[0].endpoints["fixed300"]["wait"] += 10000000.
    assert all(np.array_equal(a, b) for a, b in zip(features, separated_design(rows[0])))
    assert base == endpoint_baseline(rows[0], "fixed300")
    assert saved == model.predict_issue(query, context)
    heldout = baseline_record(100, "7X", 0., role="heldout")
    with pytest.raises(ValueError, match="Held-out"):
        model.accept([heldout], now=330.)
    with pytest.raises(ValueError, match="another training slot"):
        model.accept(rows[:1], now=330.)
    future = baseline_record(101, "7X", 0.)
    with pytest.raises(ValueError, match="released"):
        model.accept([future], now=329.)


def test_diagnostics_serialize_in_physical_units_without_nan():
    model, _ = fitted_model()
    summary = model.summary()
    diagnostic = summary["nuisance_diagnostics"]["fixed300"]
    assert diagnostic["residualized_action_design"]["columns"] == 9
    assert diagnostic["no_cost_rate_residualized_action_design"]["columns"] == 8
    assert diagnostic["global_cost_rate_action"]["original_norm"] > 0.
    assert diagnostic["oof_background_error_rmse"]["cost"] >= 0.
    assert diagnostic["oof_background_error_rmse"]["dt"] == 0.
    assert diagnostic["global_cost_rate_action"]["action_coefficient_physical"]["dt"] == 0.
    json.dumps(summary, allow_nan=False)
    json.dumps(model.predict_issue(issue(), observable_context()), allow_nan=False)
    json.dumps(model.contrast_diagnostic(issue(), observable_context()), allow_nan=False)
