"""Verify paired truth, endpoint matching, missing states, and separate replay budgets."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v11_analysis_test", HERE/"analyze.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


@pytest.fixture
def protocol():
    return dict(training_seeds=[6201, 6202], scenarios=["stable", "local_speed_drop"],
        target_lines=["7X", "102X", "705X"], target_record_budget=285,
        preflight=dict(seed=6201, scenario="stable", target_record_budget=50))


def truth(action, *, missing_same_car=False):
    high = action == "high"
    values = dict(global_next=dict(status="complete", dt=0., wait=0., ride=0., cost=0.),
        same_vehicle_next=dict(status="complete", dt=100.+7.5*high, wait=100., ride=300.+50.*high, cost=500.+50.*high),
        fixed300=dict(status="complete", dt=300., wait=2000., ride=6000.-10.*high, cost=10000.-10.*high))
    if missing_same_car:
        values["same_vehicle_next"] = dict(status="missing", reason="beyond_native_suffix")
    return values


def prediction():
    result = dict(global_comparator=dict(arms={}))
    for model in ("base_analytic", "global58"):
        result["global_comparator"]["arms"][model] = dict(
            low=dict(dt=0., cost=0.), high=dict(dt=0., cost=0.),
            cost_gain_low_minus_high=0., raw_cost_gain_low_minus_high=2.)
    for model in analysis.MODELS:
        low = dict(global_next=dict(dt=0., wait=0., ride=0., cost=0.),
            same_vehicle_next=dict(dt=100., wait=100., ride=300., cost=500.),
            fixed300=dict(dt=250. if model == "legacy" else 300., wait=2000., ride=6000., cost=10000.))
        high = deepcopy(low)
        high["same_vehicle_next"].update(ride=350., cost=550.)
        high["fixed300"].update(ride=5970., cost=9970.)
        result[model] = dict(low=dict(endpoints=low), high=dict(endpoints=high),
            gains=dict(global_next=0., same_vehicle_next=-50., fixed300=30.), raw_global_next_gain=20.)
    return result


def anchor(line="7X"):
    return dict(selected=dict(sequence=4, line_id=line, started_at=18001., s=[1., 2.], raw_s=[9.],
                              issue_ledger=dict(time_seconds=18001., waiting_passengers=100.)),
        prefix=dict(external_seed=83201, counters=dict(policy_updates=456), receipts=[dict(deadline=18000)]),
        prediction=prediction())


def branch(action, line="7X", *, missing_same_car=False):
    return dict(status="complete", integrity=dict(all_pass=True, checks=dict(full_native_prefix_exact=True)), **anchor(line),
        truth=truth(action, missing_same_car=missing_same_car),
        execution=dict(requested_hold_seconds=7.5 if action == "high" else 0.,
                       executed_hold_seconds=7.5 if action == "high" else 0.),
        suffix=dict(seconds=300., training_delta=dict(policy_updates=0, world_updates=0, trust_updates=0)),
        budget=dict(prefix_sim_seconds=18001., suffix_sim_seconds=300., replay_target_labels=285,
                    truth_labels_enter_training=0), native=dict(other_batch_actions=[], prefix_matched_issues=5))


def day(p, *, preflight=False):
    rows = []
    for line in p["target_lines"]:
        for role in ("train", "heldout"):
            for action in ("low", "high"):
                rows.append(dict(line_id=line, role=role, hold_seconds=0. if action == "low" else 7.5,
                                 prediction=prediction(), truth=truth(action)))
    return dict(status="complete", integrity=dict(all_pass=True, checks=dict(prefix_trace_recorded=True)),
        budget=dict(target_labels=50 if preflight else 285), initializer=dict(updates=50000),
        coverage=dict(train={}, heldout={}), anchors={line: anchor(line) for line in p["target_lines"]},
        evaluations=rows, residual_model=dict(design=dict(rows=55, columns=13, rank=13)))


def populate(root, p, *, preflight=False):
    days, pairs = analysis.expected_paths(root, p, preflight=preflight)
    for _cell, path in days:
        path.mkdir(parents=True)
        (path/"summary.json").write_text(json.dumps(day(p, preflight=preflight)))
    for cell, path in pairs:
        path.mkdir(parents=True)
        (path/"summary.json").write_text(json.dumps(branch(cell["branch"], cell["target_line"])))


def test_global_comparator_never_gets_same_car_or_fixed300_errors():
    row = dict(hold_seconds=0., prediction=prediction(), truth=truth("low"))
    value = analysis.summarize_rows([row])
    assert set(value["endpoints"]["global_next"]["matched_errors"]) == set(analysis.GLOBAL_MODELS)
    assert set(value["endpoints"]["same_vehicle_next"]["matched_errors"]) == set(analysis.MODELS)
    assert set(value["endpoints"]["fixed300"]["matched_errors"]) == set(analysis.MODELS)
    assert value["endpoints"]["fixed300"]["matched_errors"]["residual"]["mae"] == [0., 0., 0., 0.]


def test_matched_cohort_drops_unsupported_without_zero_filling():
    rows = [dict(hold_seconds=0., prediction=prediction(), truth=truth("low")) for _ in range(2)]
    rows[1]["prediction"]["legacy"]["low"]["endpoints"]["same_vehicle_next"] = None
    value = analysis.summarize_rows(rows)["endpoints"]["same_vehicle_next"]
    assert value["available"]["residual"] == 2
    assert value["available"]["legacy"] == 1
    assert all(model["n"] == 1 for model in value["matched_errors"].values())
    rows[0]["truth"]["same_vehicle_next"] = dict(status="missing")
    value = analysis.summarize_rows(rows)["endpoints"]["same_vehicle_next"]
    assert value["missing_truth"] == 1
    assert all(model["n"] == 0 and model["mae"] is None for model in value["matched_errors"].values())


def test_gain_sign_and_tie_decision_regret():
    assert analysis.effect_metrics(10., -5.)["regret"] == 5.
    tie = analysis.effect_metrics(0., 12.)
    assert tie["selected_action"] == "low" and tie["regret"] == 12.
    assert tie["direction_correct"] is False and tie["predicted_tie"] is True
    assert analysis.effect_metrics(None, 3.) is None


def test_exact_anchor_prefix_prediction_and_zero_training_are_required():
    cell = dict(seed=6201, scenario="stable", line="7X")
    low, high = branch("low"), branch("high")
    valid = analysis.analyze_pair(cell, low, high, anchor())
    assert valid["integrity"]["all_pass"] is True
    assert valid["duration"] == "zero_dt"
    assert valid["raw_global_effects"]["residual"]["gain_signed_error"] == 20.
    assert valid["effects"]["fixed300"]["residual"]["gain_signed_error"] == 20.
    high["prediction"]["residual"]["gains"]["fixed300"] += .00001
    high["prefix"]["receipts"][0]["deadline"] += 1
    high["suffix"]["training_delta"]["world_updates"] = 1
    invalid = analysis.analyze_pair(cell, low, high, anchor())
    assert invalid["integrity"]["all_pass"] is False
    assert invalid["integrity"]["checks"]["anchor_prediction"] is False
    assert invalid["integrity"]["checks"]["common_prefix"] is False
    assert invalid["integrity"]["checks"]["suffix_zero_training"] is False
    assert analysis.summarize_effects([invalid])["fixed300"]["matched_n"] == 0


def test_same_car_missing_is_explicit_without_failing_native_preflight_integrity():
    low, high = branch("low"), branch("high", missing_same_car=True)
    pair = analysis.analyze_pair(dict(seed=6201, scenario="stable", line="7X"), low, high, anchor())
    assert pair["integrity"]["all_pass"] is True
    assert pair["truth"]["same_vehicle_next"]["status"] == "missing"
    assert pair["effects"]["same_vehicle_next"]["residual"] is None
    assert pair["truth"]["fixed300"]["true_gain"] == 10.


def test_wrong_actual_command_is_not_accepted_as_requested_action():
    high = branch("high")
    high["execution"]["executed_hold_seconds"] = 7.0
    pair = analysis.analyze_pair(dict(seed=6201, scenario="stable", line="7X"), branch("low"), high, anchor())
    assert pair["integrity"]["checks"]["actual_low_high_commands"] is False


def test_anchor_equality_without_complete_native_prefix_proof_blocks_gate():
    high = branch("high")
    del high["integrity"]["checks"]["full_native_prefix_exact"]
    pair = analysis.analyze_pair(dict(seed=6201, scenario="stable", line="7X"), branch("low"), high, anchor())
    assert pair["integrity"]["checks"]["anchor_state"] is True
    assert pair["integrity"]["checks"]["anchor_prefix"] is True
    assert pair["integrity"]["checks"]["full_native_prefix_exact"] is False
    assert pair["integrity"]["all_pass"] is False
    assert pair["prefix_matched_issues"] == dict(low=5, high=5)


def test_day_requires_resident_native_prefix_trace_record(tmp_path, protocol):
    populate(tmp_path, protocol, preflight=True)
    day_path = tmp_path/"day"/"summary.json"
    value = json.loads(day_path.read_text())
    del value["integrity"]["checks"]["prefix_trace_recorded"]
    day_path.write_text(json.dumps(value))
    result = analysis.analyze(tmp_path, protocol, preflight=True)
    assert result["integrity"]["checks"]["all_days_native_prefix_recorded"] is False
    assert result["integrity"]["all_pass"] is False


def test_complete_four_days_and_pairs_keep_unique_and_replay_budgets_separate(tmp_path, protocol):
    populate(tmp_path, protocol)
    value = analysis.analyze(tmp_path, protocol)
    assert value["status"] == "complete" and value["integrity"]["all_pass"] is True
    assert len(value["cells"]) == 4 and len(value["pairs"]) == 6
    assert value["budget"]["main_unique_training_slots"] == 1140
    extra = value["budget"]["extra_native_branch_budget"]
    assert extra["replay_target_labels"] == 3420 and extra["suffix_sim_seconds"] == 3600.
    assert extra["truth_labels_enter_training"] == 0
    assert value["independent_confirmation"]["status"] == "not_submitted"
    assert value["paired_overall"]["fixed300"]["matched_n"] == 6
    payload = json.dumps(value, allow_nan=False)
    assert len(payload.encode()) < 200000
    assert '"raw_s"' not in payload and '"receipts"' not in payload


def test_missing_selection_is_terminal_and_blocks_review_without_zero_truth(tmp_path, protocol):
    populate(tmp_path, protocol)
    _, pairs = analysis.expected_paths(tmp_path, protocol)
    (pairs[0][1]/"summary.json").write_text('{"status":"selection_missing","integrity":{"all_pass":false}}')
    value = analysis.analyze(tmp_path, protocol)
    assert value["status"] == "complete"
    assert value["integrity"]["all_pass"] is False
    assert value["independent_confirmation"]["review_ready"] is False
    assert value["missing"]["branch_statuses"]["selection_missing"] == 1
    missing = next(pair for pair in value["pairs"] if pair["branch_status"]["low"] == "selection_missing")
    assert missing["truth"]["global_next"]["true_gain"] is None


def test_preflight_gate_stays_small_and_does_not_pull_resident_states(tmp_path, protocol):
    populate(tmp_path, protocol, preflight=True)
    value = analysis.analyze(tmp_path, protocol, preflight=True)
    assert value["integrity"]["all_pass"] is True
    assert len(value["pairs"]) == 1
    assert "strata" not in value and "truth" not in value["pairs"][0]
    assert len(json.dumps(value, allow_nan=False).encode()) < 20000


def test_actual_fixed_clock_error_blocks_gate(tmp_path, protocol):
    populate(tmp_path, protocol, preflight=True)
    day_path = tmp_path/"day"/"summary.json"
    value = json.loads(day_path.read_text())
    value["evaluations"][0]["prediction"]["residual"]["low"]["endpoints"]["fixed300"]["dt"] = 299.
    day_path.write_text(json.dumps(value))
    result = analysis.analyze(tmp_path, protocol, preflight=True)
    assert result["integrity"]["checks"]["fixed300_residual_exact"] is False


def test_final_budget_keeps_failed_and_completed_preflight_attempts_without_estimates(tmp_path, protocol):
    populate(tmp_path, protocol)
    first, second = tmp_path/"preflight"/"day", tmp_path/"preflight2"/"day"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first/"failure.json").write_text('{"status":"failed","time":1234,"message":"native failure"}')
    (second/"summary.json").write_text(json.dumps(dict(status="complete", budget=dict(
        native_seconds=5003, target_labels=50, policy_updates=80, new_native_suffix_seconds=0))))
    pair = tmp_path/"preflight2"/"pair"/"low"
    pair.mkdir(parents=True)
    (pair/"summary.json").write_text(json.dumps(dict(status="complete", budget=dict(
        prefix_sim_seconds=4100, suffix_sim_seconds=300, replay_target_labels=50, replay_policy_updates=80))))
    other = tmp_path/"preflight2"/"pair"/"high"
    other.mkdir(parents=True)
    (other/"failure.json").write_text('{"status":"failed","message":"failed before native startup"}')
    result = analysis.analyze(tmp_path, protocol)["preflight_compute_budget"]
    assert [value["attempt"] for value in result] == ["preflight", "preflight2"]
    failed = result[0]["runs"][0]
    assert failed["native_seconds"] == 1234
    assert failed["target_labels"] is None and failed["policy_updates"] is None
    runs = {row["run"]: row for row in result[1]["runs"]}
    assert runs["day"]["target_labels"] == 50 and runs["day"]["suffix_seconds"] == 0
    assert runs["pair/low"]["native_seconds"] == 4400 and runs["pair/low"]["replay_target_labels"] == 50
    assert runs["pair/high"]["native_seconds"] is None and runs["pair/high"]["suffix_seconds"] is None
    assert len(json.dumps(result).encode()) < 2000
