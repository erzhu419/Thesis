"""Detect missing-state imputation, counted ties and unscored analytic errors."""
import copy
import json
from math import log1p
from unittest.mock import patch

import pytest

from experiments.mcwm_closed_loop_action_fork_v7_20261020.test_predictions import factual
from experiments.mcwm_closed_loop_action_local_v3_20261020.test_action_contract import state
from experiments.mcwm_closed_loop_action_response_v8_20261021 import analysis
from experiments.mcwm_closed_loop_action_response_v8_20261021.test_predictions import prediction, readouts
from experiments.mcwm_closed_loop_paired_world_v6_20261020.test_paired_world import paired_learner


def complete_record(*, zero_duration=False):
    item = paired_learner()
    states = state(time=19000., current=int(zero_duration))[1]
    if zero_duration:
        states[2] = 0.
    probe = prediction(item, states, readouts(item))
    dt, cost = (0., 0.) if zero_duration else (2., 6.)
    truth = {}
    for candidate in ("low", "high"):
        row = factual(probe.legacy, candidate, dt=dt, cost=cost)
        truth[candidate] = probe.legacy.score_transition(
            row, now=row.completed_at, candidate=candidate)
    return dict(status="complete", cell=dict(seed=6201, scenario="stable", target_line="7X"),
        selected=dict(time=19000., line="7X"), integrity=dict(all_pass=True),
        response=dict(labels=5), prediction=probe.summary(), truth=truth,
        comparison=probe.compare(truth["low"], truth["high"], low_cost_300=100., high_cost_300=100.),
        global_normalizer=dict(mean=item.world.norm.g_mean.cpu().tolist(),
                               std=item.world.norm.g_std.cpu().tolist()),
        budget=dict(prefix_sim_seconds=19000., new_suffix_sim_seconds=0,
                    reused_v7_branches=2, selected_prefix_labels=5, response_fit_labels=5,
                    truth_labels_enter_training=0, response_ridge_refits=1))


@pytest.mark.parametrize("zero_duration", [False, True])
def test_complete_prediction_rows_score_and_factual_ties_are_never_hits(zero_duration):
    record = complete_record(zero_duration=zero_duration)
    scored = analysis.score(record)
    assert scored["zero_duration"] is zero_duration
    aggregate = analysis.aggregate([scored])
    for version in ("legacy", "response"):
        for world in analysis.WORLDS:
            row = scored["versions"][version][world]
            assert row["directions"]["next_event_cost"] == "tie"
            assert row["direction_vs_300s"] == "tie"
            assert aggregate[version][world]["next_event_direction"] == dict(agree=0, disagree=0, tie=1)
            assert aggregate[version][world]["direction_vs_300s"] == dict(agree=0, disagree=0, tie=1)
            assert row["g4_mse"] >= 0
    if zero_duration:
        assert scored["versions"]["response"]["online"]["g4_mse"] == 0
        # An analytic flag must not erase an actual error in decoded outputs.
        corrupt = copy.deepcopy(record)
        corrupt["prediction"]["branches"]["online"]["candidates"]["high"]["global_decoded"][1] = 1.
        changed = analysis.score(corrupt)
        expected = (log1p(1.)/record["global_normalizer"]["std"][1])**2/8
        assert changed["versions"]["response"]["online"]["g4_mse"] == pytest.approx(expected)


def test_missing_failed_and_incomplete_states_remain_explicit(tmp_path):
    protocol = dict(training_seeds=[6201, 6202], scenarios=["stable", "local_speed_drop"],
                    target_lines=["7X", "102X", "705X"])
    (tmp_path/"protocol.json").write_text(json.dumps(protocol))
    root = tmp_path/"runs"
    good = complete_record()
    good_dir = root/"stable"/"7X"/"seed6201"
    good_dir.mkdir(parents=True)
    (good_dir/"summary.json").write_text(json.dumps(good))
    bad_dir = root/"stable"/"102X"/"seed6201"
    bad_dir.mkdir(parents=True)
    missing = dict(status="selection_missing", cell=dict(seed=6201, scenario="stable", target_line="102X"),
                   integrity=dict(all_pass=False), budget=dict(prefix_sim_seconds=13801., truth_labels_enter_training=0))
    (bad_dir/"summary.json").write_text(json.dumps(missing))
    failed_dir = root/"local_speed_drop"/"705X"/"seed6202"
    failed_dir.mkdir(parents=True)
    (failed_dir/"failure.json").write_text(json.dumps(dict(status="failed", message="native stop")))
    with patch.object(analysis, "HERE", tmp_path):
        result = analysis.analyze(root)
    assert result["status"] == "incomplete"
    assert result["integrity"]["complete_states"] == 1
    assert len(result["integrity"]["missing"]) == 10
    assert len(result["integrity"]["failures"]) == 1
    assert len(result["integrity"]["errors"]) == 1
    assert result["integrity"]["errors"][0]["status"] == "selection_missing"
    assert result["groups"]["overall"]["n"] == 1
    assert result["groups"]["local_speed_drop"]["n"] == 0
    assert "actual_mean_300s_gain" not in result["groups"]["local_speed_drop"]
    assert result["budget"]["prefix_sim_seconds"] == 19000.+13801.
    assert result["budget"]["truth_labels_enter_training"] == 0
