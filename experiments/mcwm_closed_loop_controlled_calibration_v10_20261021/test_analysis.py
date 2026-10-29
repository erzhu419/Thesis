"""Verify matched comparisons, separate analytic zeros and compact collection."""

from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.analyze import (
    MAX_BYTES, analyze, summarize_group,
)


def evaluation(sequence=0, *, line="7X", phase="train_before", zero=False,
               supported=True, hold=0.):
    role, period = phase.split("_")
    truth = {endpoint: dict(status="complete", dt=0. if zero and endpoint == "global_next" else 100.,
        wait=20., ride=60., cost=100.+sequence)
        for endpoint in ("global_next", "same_vehicle_next", "fixed300")}
    errors = {endpoint: dict(signed_errors=dict(dt=2., wait=3., ride=4., cost=1.))
              for endpoint in truth} if supported else None
    prediction = dict(status="supported" if supported else "unsupported",
        endpoint_status={endpoint: ("analytic_zero_duration" if zero and endpoint == "global_next"
            else "supported") if supported else "unsupported" for endpoint in truth},
        gains=dict(global_next=0. if zero else 2., same_vehicle_next=4., fixed300=6.) if supported
            else {name: None for name in truth}, raw_global_next_gain=3. if supported else None)
    return dict(sequence=sequence, issued_at=3000. if period == "before" else 19000. if role == "heldout" else 12000.,
        line_id=line, role=role, hold_seconds=hold, status="complete", issue_phase=period,
        truth=truth, global_comparator={
            "base_analytic": dict(cost_signed_error=30., dt_signed_error=40.,
                cost_gain_low_minus_high=0. if zero else 7., raw_cost_gain_low_minus_high=8.),
            "global58": dict(cost_signed_error=50., dt_signed_error=60.,
                cost_gain_low_minus_high=0. if zero else 9., raw_cost_gain_low_minus_high=10.)},
        endpoint_prediction=prediction,
        endpoint_error=dict(status="complete" if supported else "unsupported", endpoints=errors))


def payload(rows):
    return dict(status="complete", integrity=dict(all_pass=True, checks=dict(
        heldout_never_fit=True, original_trust_update_rule=True, no_post_budget_training=True)),
        initializer=dict(initial_updates=50000, history_rows=32768, source_rows=4096),
        budget=dict(target_labels=285, packets=57, global58_labels=280, zero_global_admitted=5,
            policy_updates=456, native_seconds=24000, trust_updates=456),
        controlled_records=[dict(sequence=row["sequence"], line_id=row["line_id"], role=row["role"],
            hold_seconds=row["hold_seconds"], status=row["status"],
            selected_for_training=row["role"] == "train", endpoints=row["truth"]) for row in rows],
        evaluations=rows, endpoints=dict(complete=len(rows), pending=0, missing=0),
        endpoint_model=dict(design=dict(rows=20, columns=13, rank=10),
            endpoint_designs={name: dict(rows=20, columns=13, rank=10)
                for name in ("global_next", "same_vehicle_next", "fixed300")}),
        global58=dict(columns=58), cost=dict(full_day=10000))


def protocol():
    return dict(training_seeds=[6201, 6202], scenarios=["stable", "local_speed_drop"],
        target_lines=["7X", "102X", "705X"], target_record_budget=285)


def write_cell(root, seed, scenario, value):
    path = root/scenario/f"seed{seed}"
    path.mkdir(parents=True, exist_ok=True)
    (path/"summary.json").write_text(json.dumps(value))


def test_global_fairness_uses_identical_supported_records_and_preserves_unmatched():
    accepted = evaluation()
    unsupported = evaluation(1, supported=False)
    unsupported["global_comparator"]["base_analytic"]["cost_signed_error"] = 999999.
    local_only = evaluation(2)
    local_only["endpoint_prediction"]["endpoint_status"]["global_next"] = "unsupported"
    local_only["endpoint_prediction"]["gains"]["global_next"] = None
    local_only["endpoint_error"]["endpoints"]["global_next"] = dict(status="unsupported")
    result = summarize_group([accepted, unsupported, local_only])
    assert all(values["n"] == 1 for values in result["fair_global_errors"].values())
    assert result["fair_global_errors"]["base_analytic"]["mae"] == [40., 30.]
    assert result["unmatched"]["total"] == 2
    assert result["unmatched"]["candidate_positive_global_unsupported"] == 2
    assert result["nonfinite_values"] == 0  # unsupported None is never a bad float or zero.
    assert result["fair_global_gains"]["endpoint_candidate_raw"][0] == 1


def test_zero_duration_separate_missing_endpoint_never_imputed_and_no_false_gate(tmp_path):
    zero = evaluation(zero=True, phase="heldout_after")
    zero["endpoint_prediction"]["raw_global_next_gain"] = None
    missing = evaluation(1)
    missing["status"] = "missing"
    missing["truth"]["fixed300"] = dict(status="missing", reason="target_horizon")
    missing["endpoint_error"]["endpoints"]["fixed300"] = dict(status="missing")
    write_cell(tmp_path, 6201, "stable", payload([zero, missing]))
    result = analyze(tmp_path, protocol())
    zero_group = next(group for group in result["strata"] if group["duration"] == "zero_dt")
    positive = next(group for group in result["strata"] if group["duration"] == "positive_dt")
    assert zero_group["unmatched"]["raw_global_unsupported"] == 1
    assert "endpoint_candidate_raw" not in zero_group["fair_global_gains"]
    assert positive["endpoint_errors"]["fixed300"] == dict(n=0, mae=None, mse=None)
    assert positive["descriptive_unpaired_cost"]["fixed300"]["low"] == [0, None]
    assert result["integrity"]["missing_days"] == 3
    assert not result["independent_confirmation"]["checks"]["at_least_one_supported_heldout_comparison"]
    assert result["independent_confirmation"]["status"] == "not_submitted"


def test_failed_day_and_positive_global_coverage_survive_without_zero_filling(tmp_path):
    rows = [evaluation(i, zero=True, hold=hold) for i, hold in enumerate((0., 0., 7.5, 7.5))]
    write_cell(tmp_path, 6201, "stable", payload(rows))
    failed = tmp_path/"local_speed_drop/seed6201"
    failed.mkdir(parents=True)
    (failed/"failure.json").write_text(json.dumps(dict(status="failed", type="RuntimeError", message="native failure")))
    result = analyze(tmp_path, protocol())
    assert result["integrity"]["failed_days"] == 1
    cell = next(cell for cell in result["cells"] if cell.get("selected_train_coverage"))
    assert cell["selected_train_coverage"]["7X"] == dict(low=2, high=2)
    assert cell["selected_positive_global_coverage"]["7X"] == dict(low=0, high=0)
    assert cell["integrity"]["checks"]["original_trust_update_rule"] is True
    assert not result["independent_confirmation"]["checks"]["selected_positive_global_two_actions_per_line"]
    assert next(c for c in result["cells"] if c["status"] == "failed")["failure"]["message"] == "native failure"


def test_four_day_full_strata_fit_small_summary_budget_and_ready_is_only_review(tmp_path):
    rows = []
    for line, phase, zero, hold, replicate in product(
        ("7X", "102X", "705X"), ("train_before", "train_after", "heldout_after"),
        (False, True), (0., 7.5), range(2)):
        rows.append(evaluation(len(rows), line=line, phase=phase, zero=zero, hold=hold))
    for seed, scenario in product((6201, 6202), ("stable", "local_speed_drop")):
        write_cell(tmp_path, seed, scenario, payload(deepcopy(rows)))
    result = analyze(tmp_path, protocol())
    encoded = json.dumps(result, separators=(",", ":"), allow_nan=False).encode()
    assert len(encoded) <= MAX_BYTES
    assert len(result["strata"]) == 72
    assert result["status"] == "complete"
    assert result["independent_confirmation"]["review_ready"] is True
    assert result["independent_confirmation"]["status"] == "not_submitted"
    assert result["independent_confirmation"]["paired_native_truth_required"] is True
    assert all("trajectory" not in cell.get("cost", {}) for cell in result["cells"])
    assert "not paired same-state" in result["interpretation"]["randomized_groups"]
