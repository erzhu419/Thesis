"""Admission, physical action semantics and the unchanged low-dimensional fit."""
import copy

import numpy as np
import pytest

from experiments.mcwm_closed_loop_matched_response_v59_20261025 import difference as d


def pair(index=0, line="7X", feature=-0.5, gain=(20.0, -10.0)):
    return dict(id=f"train/{index}", index=index, role="train", selected_for_training=True,
        line_id=line, features=[float(line == value) for value in d.LINES] + [feature],
        holds=[0.0, 7.5], issued_at=500.0, available_at=830.0, admitted_at=900.0,
        outcomes={arm: dict(truth=dict(status="complete", dt=300.0, wait=1000.0 + dw,
            ride=2000.0 + dr, cost=4000.0 + 2 * dw + dr),
            execution=dict(native_applied_hold_seconds=hold))
            for arm, hold, dw, dr in (("low", 0.0, *gain), ("high", 7.5, 0.0, 0.0))})


def test_unmodified_ridge_contrast_units_and_background_cancellation():
    rows = [pair(), pair(1, "7X", 0.5, (-10, 30)), pair(2, "102X", 0.2), pair(3, "705X", 0.0)]
    untouched = copy.deepcopy(rows)
    models = []
    for background in (0, 1e7):
        values = copy.deepcopy(rows)
        for row in values:
            for arm in ("low", "high"):
                truth = row["outcomes"][arm]["truth"]
                truth["wait"] += background
                truth["ride"] += background
                truth["cost"] += 3 * background
        model = d.DirectDifferenceReadout()
        model.accept(values, 900.0)
        model.refit()
        models.append(model)
    np.testing.assert_array_equal(models[0].weights, models[1].weights)
    matrix, targets = models[0]._training_arrays()
    expected = np.linalg.solve(matrix.T @ matrix + np.eye(4), matrix.T @ (targets / 1000.0))
    np.testing.assert_array_equal(models[0].weights, expected)
    prediction = models[0].predict([1, 0, 0, 0.2])
    gain = prediction["gain"]
    assert gain["cost"] == 2 * gain["wait"] + gain["ride"]
    assert gain["scaled_cost"] == 1e-4 * gain["cost"]
    assert prediction["continuation"] == "common_frozen_mu14"
    assert models[0].summary()["optimizer_updates"] == 0
    assert rows == untouched


@pytest.mark.parametrize("change", [
    dict(role="heldout"), dict(selected_for_training=False), dict(available_at=901.0),
    dict(admitted_at=901.0), dict(admitted_at=829.0), dict(holds=[0.0, 1.875]),
])
def test_no_heldout_unadmitted_or_future_feedback_can_fit(change):
    row = pair()
    row.update(change)
    with pytest.raises(ValueError):
        d.DirectDifferenceReadout().accept([row], 900.0)


def test_actual_execution_and_complete_clocks_are_required():
    wrong = pair()
    wrong["outcomes"]["high"]["execution"]["native_applied_hold_seconds"] = 0.9375
    with pytest.raises(ValueError, match="executed"):
        d.DirectDifferenceReadout().accept([wrong], 900.0)
    short = pair()
    short["outcomes"]["high"]["truth"]["dt"] = 299.0
    with pytest.raises(ValueError, match="exactly300"):
        d.DirectDifferenceReadout().accept([short], 900.0)


def test_cumulative_admission_support_and_line_constant():
    model = d.DirectDifferenceReadout(line_only=True)
    model.accept([pair()], 900.0)
    model.refit()
    assert model.predict([0, 1, 0, 0])["status"] == "unsupported"
    assert model.predict([1, 0, 0, -10])["gain"] == model.predict([1, 0, 0, 10])["gain"]
    assert model.predict([1, 0, 0, 10])["headway_extrapolation"] is True
    with pytest.raises(ValueError, match="second"):
        model.accept([pair()], 900.0)
    model.accept([pair(1, "102X")], 1000.0)
    model.refit()
    assert model.summary()["refits"] == 2 and model.summary()["outcome_slots"] == 4


def test_issue_features_ignore_future_results():
    from experiments.mcwm_closed_loop_v1.stream import EI
    event = np.zeros(max(EI.values()) + 1, np.float32)
    for key, value in dict(forward_headway_seconds=660, target_forward_headway_seconds=600,
                           backward_headway_seconds=540, target_backward_headway_seconds=600).items():
        event[EI[key]] = value
    issued = dict(line_id="7X", event=event, truth=dict(cost=1e9))
    first = d.issue_features(issued)
    issued["truth"] = dict(cost=-1e9)
    assert d.issue_features(issued) == first
    np.testing.assert_allclose(first, [1, 0, 0, 0.2])
