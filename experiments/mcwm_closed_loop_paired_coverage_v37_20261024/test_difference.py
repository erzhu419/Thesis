"""Finite300 admission, independent contrasts and issue-only covariates."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("v37_difference", Path(__file__).with_name("difference.py"))
difference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(difference)


def pair(index, line, features, gain, common=1000.0):
    return dict(id=f"p{index}", role="train", selected_for_training=True,
        available_at=1030.0, admitted_at=1200.0, line_id=line, features=features,
        holds=[0.0, 1.875], outcomes={
            "low": dict(truth=dict(status="complete", dt=300.0,
                wait=common+gain[0], ride=common+gain[1])),
            "high": dict(truth=dict(status="complete", dt=300.0, wait=common, ride=common))})


def records():
    return [pair(0,"7X",[1,0,0,-.5],[-20,30]),
        pair(1,"7X",[1,0,0,.5],[40,-10]),
        pair(2,"102X",[0,1,0,.2],[10,20]),
        pair(3,"705X",[0,0,1,-.3],[-5,10])]


def test_contrasts_cancel_background_keep_physical_units_and_inputs_unchanged():
    original = records()
    untouched = copy.deepcopy(original)
    source_model = difference.DirectDifferenceReadout()
    source_model.accept(original, 1200.0)
    source_model.refit()
    models = []
    for common in (1000.0, 1e8):
        rows = records()
        for row in rows:
            for arm in ("low", "high"):
                for key in ("wait", "ride"):
                    row["outcomes"][arm]["truth"][key] += common
        model = difference.DirectDifferenceReadout()
        model.accept(rows, 1200.0)
        model.refit()
        models.append(model)
    np.testing.assert_array_equal(models[0].weights, models[1].weights)
    assert original == untouched
    gain = models[0].predict([1,0,0,.25])["gain"]
    assert gain["cost"] == 2*gain["wait"]+gain["ride"]
    summary = models[0].summary()
    assert summary["outcome_slots"] == 8 and summary["optimizer_updates"] == 0
    assert summary["controls_native_policy"] is False and summary["modifies_bellman_target"] is False


def test_line_constant_and_lowdim_are_distinct_and_admission_accumulates():
    full, base = difference.DirectDifferenceReadout(), difference.DirectDifferenceReadout(line_only=True)
    for model in (full, base):
        model.accept(records()[:2], 1200.0)
        model.refit()
        assert model.predict([0,1,0,.2])["gain"] is None
        model.accept(records()[2:], 1500.0)
        model.refit()
        assert model.summary()["pairs"] == 4 and model.summary()["refits"] == 2
    assert base.predict([1,0,0,-5])["gain"] == base.predict([1,0,0,5])["gain"]
    assert full.predict([1,0,0,-.5])["gain"] != full.predict([1,0,0,.5])["gain"]
    assert base.summary()["design"]["columns"] == 3 and full.summary()["design"]["columns"] == 4


@pytest.mark.parametrize("change", [
    {"role":"heldout"}, {"selected_for_training":False}, {"available_at":1201.0},
    {"admitted_at":1201.0}, {"admitted_at":1029.0}, {"holds":[0,7.5]},
])
def test_training_rights_native_release_and_actual_action_contract(change):
    row = records()[0]
    row.update(change)
    with pytest.raises(ValueError):
        difference.DirectDifferenceReadout().accept([row], 1200.0)


def test_truncated_missing_or_duplicate_pair_is_not_an_extra_or_zero_label():
    for change in ({"dt":299.0}, {"status":"missing"}):
        row = records()[0]
        row["outcomes"]["high"]["truth"].update(change)
        with pytest.raises(ValueError, match="exactly300"):
            difference.DirectDifferenceReadout().accept([row], 1200.0)
    model = difference.DirectDifferenceReadout()
    model.accept(records()[:1], 1200.0)
    with pytest.raises(ValueError, match="second"):
        model.accept(records()[:1], 1200.0)


def test_issue_prediction_uses_only_visible_line_headways_and_no_future_truth():
    from experiments.mcwm_closed_loop_v1.stream import EI

    event = np.zeros(max(EI.values())+1, np.float32)
    for key, value in {"forward_headway_seconds":660, "target_forward_headway_seconds":600,
            "backward_headway_seconds":540, "target_backward_headway_seconds":600}.items():
        event[EI[key]] = value
    issued = dict(line_id="7X", event=event, truth={"cost":1e99}, scene="forbidden_future")
    np.testing.assert_allclose(difference.issue_features(issued), [1,0,0,.2])
    model = difference.DirectDifferenceReadout()
    model.accept(records(), 1200.0)
    model.refit()
    before = model.predict_issue(issued)
    issued.update(truth={"cost":-1e99}, scene="changed", started_at=1e99)
    assert model.predict_issue(issued) == before
