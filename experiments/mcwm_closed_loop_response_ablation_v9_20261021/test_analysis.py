"""Check real prediction scoring, missing-state accounting and compact size."""
from pathlib import Path
import copy
from itertools import product
import json
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent/"mcwm_closed_loop_diagnostic_v2_20261020"))

from experiments.mcwm_closed_loop_action_fork_v7_20261020.test_predictions import factual
from experiments.mcwm_closed_loop_action_local_v3_20261020.test_action_contract import state
from experiments.mcwm_closed_loop_paired_world_v6_20261020.test_paired_world import paired_learner
from experiments.mcwm_closed_loop_response_ablation_v9_20261021.ablation_world import (
    ARM_FEATURE_INDICES, FEATURE_GROUPS,
)
from experiments.mcwm_closed_loop_response_ablation_v9_20261021.analysis import (
    aggregate, analyze, metrics, score,
)
from experiments.mcwm_closed_loop_response_ablation_v9_20261021.predictions import (
    AblationPrediction, PREDICTION_ARMS,
)
from experiments.mcwm_closed_loop_response_ablation_v9_20261021.test_predictions import readouts


def fit_statistics(weights):
    """Representative production-shaped support statistics, without fit arrays."""
    coverage = dict(n=160, hold_seconds_min=0., hold_seconds_max=7.5,
                    positive_holding=30, zero_holding=130)
    return dict(labels=160, retained_labels=160, refits=32,
        designs={arm: dict(rows=160, columns=len(columns), rank=len(columns)-2,
                          active_columns=len(columns), ridge=1.,
                          ridge_precision_condition=1234.56789)
                 for arm, columns in ARM_FEATURE_INDICES.items()},
        fit_errors={arm: {family: dict(base_mse=.123456789, residual_mse=.023456789)
                         for family in branches["online"]}
                    for arm, branches in weights.items()},
        coefficient_group_l2={arm: {family: {group: float(weight[list(columns)].norm())
            for group, columns in FEATURE_GROUPS.items()}
            for family, weight in branches["online"].items()}
            for arm, branches in weights.items()},
        selected_holding_coverage=dict(overall=coverage,
            by_line={line: dict(all=coverage,
                at_diagnostic_station=dict(n=12, hold_seconds_min=0., hold_seconds_max=7.5,
                    positive_holding=4, zero_holding=8))
                for line in ("7X", "102X", "705X")}))


def make_record(item, *, zero_duration=False, gain300=10.):
    weights = readouts(item)
    canonical = state(time=19000., current=1 if zero_duration else 0)[1]
    if zero_duration:
        canonical[2] = 0.
    probe = AblationPrediction(item, canonical, np.linspace(.01, .18, 18, dtype=np.float32),
                              30., [0, 1, 2], weights, seed=70)
    facts = {}
    for candidate in ("low", "high"):
        row = factual(probe.legacy, candidate, dt=0 if zero_duration else 2,
                      cost=0 if zero_duration else 6)
        if zero_duration:
            row.ns[2] = 1/15
            row.ns[5+13] = probe.candidates[candidate]["executed_action"]
        facts[candidate] = probe.legacy.score_transition(
            row, now=row.completed_at, candidate=candidate)
    return dict(status="complete", cell=dict(seed=6201, scenario="stable", target_line="7X"),
        selected=dict(time=19000., sequence=17, line="7X", vehicle_id="current_bus",
                      trip_id="trip17", station="upstream", low_action=-1., high_action=-.75),
        global_normalizer=dict(mean=item.world.norm.g_mean.tolist(), std=item.world.norm.g_std.tolist()),
        prediction=probe.summary(), response=fit_statistics(weights),
        comparison=probe.compare(facts["low"], facts["high"],
                                 low_cost_300=1000., high_cost_300=1000.-gain300),
        truth={candidate: {key: facts[candidate][key]
                           for key in ("global_decoded", "bellman", "zero_duration")}
               for candidate in facts},
        integrity=dict(all_pass=True, checks=dict(training_prefix_exact=True,
            legacy_predictions_exact=True, full_v8_predictions_exact=True)),
        budget=dict(prefix_sim_seconds=19000., new_suffix_sim_seconds=0,
            reused_v7_branches=2, selected_prefix_labels=160, response_fit_labels=160,
            truth_labels_enter_training=0, response_ridge_refits=32, ridge_solutions=192))


@pytest.fixture(scope="module")
def records():
    item = paired_learner()
    return make_record(item), make_record(item, zero_duration=True)


def write_cell(root, record):
    cell = record["cell"]
    directory = root/cell["scenario"]/cell["target_line"]/f"seed{cell['seed']}"
    directory.mkdir(parents=True)
    (directory/"summary.json").write_text(json.dumps(record))
    return directory


def test_real_predictions_score_all_arms_and_aggregate_direction_and_ties(records):
    positive, zero = (score(record) for record in records)
    assert set(positive["arms"]) == set(PREDICTION_ARMS)
    assert positive["factual_gains"]["next_event_cost"] == 0
    assert positive["factual_gains"]["cost_300_seconds"] == 10
    for arm, values in positive["arms"].items():
        raw = records[0]["prediction"]["arms"][arm]["branches"]["online"]
        assert values["gains"] == raw["gains"]
        assert values["false_immediate_gain_error"] == abs(raw["gains"]["next_event_cost"])
        expected = "tie" if raw["gains"]["bellman_target"] == 0 else (
            "agree" if raw["gains"]["bellman_target"] > 0 else "disagree")
        assert values["direction_vs_300s"] == expected
    assert positive["frozen_base"] == metrics(
        records[0]["prediction"]["arms"]["base_analytic"]["branches"]["frozen_common"],
        records[0]["truth"], records[0]["global_normalizer"], positive["factual_gains"])
    combined = aggregate([positive, zero])
    assert combined["n"] == 2
    for arm in PREDICTION_ARMS:
        counts = combined["arms"][arm]["direction_vs_300s"]
        assert sum(counts.values()) == 2
        expected = {direction: sum(row["arms"][arm]["direction_vs_300s"] == direction
                                  for row in (positive, zero))
                    for direction in ("agree", "disagree", "tie")}
        assert counts == expected
    tied_record = copy.deepcopy(records[0])
    tied_record["comparison"]["factual"]["gains"]["cost_300_seconds"] = 0.
    tied = score(tied_record)
    assert all(value["direction_vs_300s"] == dict(agree=0, disagree=0, tie=1)
               for value in aggregate([tied])["arms"].values())


def test_zero_duration_flag_uses_decoded_errors_without_forcing_them_to_zero(records):
    record = copy.deepcopy(records[1])
    pristine = score(record)
    assert pristine["zero_duration"]
    assert all(value["g4_mse"] == 0 for value in pristine["arms"].values())
    # A broken analytic decoder must remain visible even if its flag is true.
    candidate = record["prediction"]["arms"]["base_analytic"]["branches"]["online"]["candidates"]["high"]
    assert candidate["analytic_within_batch"]
    candidate["global_decoded"][1] = 3.
    candidate["bellman"]["target"] += 1.
    scored = score(record)["arms"]["base_analytic"]
    expected = np.log1p(3.)**2/record["global_normalizer"]["std"][1]**2/8
    assert scored["g4_mse"] == pytest.approx(expected)
    assert scored["global_mae"][1] == 1.5
    assert scored["bellman_target_mse"] > 0


def test_missing_failed_and_invalid_states_are_retained_without_zero_imputation(tmp_path, records):
    valid = copy.deepcopy(records[0])
    write_cell(tmp_path, valid)
    failure_dir = tmp_path/"stable"/"102X"/"seed6201"
    failure_dir.mkdir(parents=True)
    (failure_dir/"failure.json").write_text(json.dumps(dict(status="failed", message="native exit")))
    invalid = copy.deepcopy(valid)
    invalid["cell"]["target_line"] = "705X"
    invalid["status"] = "integrity_failed"
    invalid["integrity"]["all_pass"] = False
    write_cell(tmp_path, invalid)
    result = analyze(tmp_path)
    assert result["status"] == "incomplete"
    assert result["integrity"]["complete_states"] == 1
    assert len(result["integrity"]["missing"]) == 10
    assert len(result["integrity"]["failures"]) == len(result["integrity"]["errors"]) == 1
    assert result["groups"]["overall"] == aggregate([score(valid)])
    assert result["groups"]["stable/102X"] == dict(n=0)
    assert result["groups"]["stable/705X"] == dict(n=0)
    assert result["budget"]["reused_v7_branches"] == 4  # Both recorded prefixes consumed budget.
    assert result["budget"]["truth_labels_enter_training"] == 0


def test_representative_twelve_state_summary_preserves_all_groups_and_stays_small(tmp_path, records):
    cells = list(product((6201, 6202), ("stable", "local_speed_drop"), ("7X", "102X", "705X")))
    for index, (seed, scenario, line) in enumerate(cells):
        # Real prediction/target records; only cell/support metadata is synthetic.
        record = copy.deepcopy(records[1 if index == 1 else 0])
        record["cell"] = dict(seed=seed, scenario=scenario, target_line=line)
        record["selected"]["line"] = line
        write_cell(tmp_path, record)
    result = analyze(tmp_path)
    assert result["status"] == "complete"
    assert result["integrity"]["complete_states"] == 12
    assert not any(result["integrity"][key] for key in ("missing", "failures", "errors"))
    assert len(result["groups"]) == 12
    assert result["groups"]["overall"]["n"] == 12
    assert result["groups"]["positive_dt"]["n"] == 11
    assert all(set(pair["arms"]) == set(PREDICTION_ARMS) for pair in result["pairs"])
    assert all(set(pair["fits"]["designs"]) == set(ARM_FEATURE_INDICES) for pair in result["pairs"])
    assert all(len(pair["fits"]["coefficient_group_l2"]) == 3 for pair in result["pairs"])
    size = len((json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n").encode())
    print(f"representative_compact_summary_bytes={size}")
    assert size <= 150000
