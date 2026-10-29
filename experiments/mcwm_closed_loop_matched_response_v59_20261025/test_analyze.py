"""Heldout joins, causal release, regret and incomplete-output admission gates."""
import copy
import json
from pathlib import Path

import pytest

from experiments.mcwm_closed_loop_matched_response_v59_20261025 import analyze as a
from experiments.mcwm_closed_loop_matched_response_v59_20261025.test_difference import pair


def write(path, value, *, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in value) if lines else json.dumps(value))


def fixture(root):
    protocol = dict(scenarios=["stable", "local_speed_drop"], focus_lines=list(a.LINES),
        train_windows={"w1": [500, 600], "w2": [1000, 1100], "w3": [1500, 1600]},
        heldout_windows={"early": [18300, 19200], "late": [19200, 20700]})
    write(root / "protocol.json", protocol)
    for scenario in protocol["scenarios"]:
        collection = root / "runs/collection" / scenario
        train, heldout = [], []
        for index in range(9):
            row = pair(index, a.LINES[index % 3], (index // 3 - 1) * 0.25, (10, 0))
            row.update(window_name=f"w{index // 3 + 1}", source_sequence=index)
            train.append(row)
        packets = []
        for index in range(57):
            ids = [train[index]["id"]] if index < 9 else []
            packets.append(dict(index=index + 1, rows=[{} for _ in range(5)],
                receipt=dict(processed_at=900.0 + index * 300, pair_ids=ids,
                    paired_records=2 * len(ids), ordinary_records=5 - 2 * len(ids))))
            if ids:
                train[index]["admitted_at"] = packets[-1]["receipt"]["processed_at"]
        for index, (line, window) in enumerate((line, window) for line in a.LINES for window in ("early", "late")):
            row = pair(index, line, 0.1, (10, 0))
            row.update(id=f"heldout/{line}/{window}", role="heldout", selected_for_training=False,
                source_sequence=100 + index, window_name=window,
                issued_at=18500.0 if window == "early" else 19500.0,
                pair_receipt=dict(all_pass=True, overlap_seconds=100.0),
                comparison=dict(physical_separation=dict(status="separable", departure_separable=True)))
            for arm in ("low", "high"):
                row["outcomes"][arm]["next_event"] = dict(zero_duration=index % 3 == 0)
            heldout.append(row)
        write(collection / "admitted_pair_records.json", train)
        write(collection / "pair_truth.json", train + heldout)
        write(collection / "summary.json", dict(integrity=dict(all_pass=True)))
        write(collection / "dataset.json", dict(proof=dict(all_pass=True),
            metadata=dict(feedback_slots=285), packets=packets))
        write(root / "runs/learning" / scenario / "matched_audit.json", dict(all_pass=True,
            checks=dict(initial_state=True, rng=True, optimizer=True, actual_parallel=True),
            optimizer_overlap_seconds=12.0))
        for arm in ("fresh", "no_fresh"):
            learned = root / "runs/learning" / scenario / arm
            write(learned / "summary.json", dict(integrity=dict(all_pass=True),
                budget=dict(feedback_slots=285, optimizer_updates=456, post_stop_updates=0)))
            write(learned / "heldout.jsonl", [dict(slot=row["id"].removeprefix("heldout/"),
                line_id=row["line_id"], sequence=row["source_sequence"], issue=row["issued_at"],
                prediction=dict(delta_min=0.01 if arm == "fresh" else -0.01, holding_seconds=[0.0, 7.5]))
                for row in heldout], lines=True)
    return root


def test_complete_fit_selector_regret_and_bounded_summary(tmp_path):
    result = a.analyze(fixture(tmp_path))
    assert result["status"] == "complete" and result["integrity"]["all_pass"]
    assert result["next_control_gate"]["eligible_models"] == ["readout_line", "readout_lowdim"]
    assert len(result["states"]) == 12 and result["missing"] == []
    for case in result["cases"].values():
        assert case["finite_fit_packets"][-1]["cumulative_pairs"] == 9
        assert case["overall"]["models"]["fixed0"][a.AGGREGATE_COLUMNS.index("regret")] == 120.0
        assert case["overall"]["models"]["readout_line"][a.AGGREGATE_COLUMNS.index("regret")] == 0.0
        assert case["overall"]["models"]["no_fresh_Q"][a.AGGREGATE_COLUMNS.index("selected_cost")] == 24120.0
    assert len(json.dumps(result, separators=(",", ":")).encode()) <= 48000


def test_heldout_truth_changes_evaluation_never_fitted_weights(tmp_path):
    fixture(tmp_path)
    before = a.analyze(tmp_path)
    for scenario in ("stable", "local_speed_drop"):
        path = tmp_path / "runs/collection" / scenario / "pair_truth.json"
        rows = a.read(path)
        for row in rows:
            if row["role"] == "heldout":
                row["outcomes"]["low"]["truth"].update(wait=900.0, ride=2000.0, cost=3800.0)
        write(path, rows)
    after = a.analyze(tmp_path)
    assert before["cases"]["stable"]["finite_fit"] == after["cases"]["stable"]["finite_fit"]
    assert before["next_control_gate"]["passed"] and not after["next_control_gate"]["passed"]


def test_zero_truth_has_no_unique_direction(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "runs/collection/stable/pair_truth.json"
    rows = a.read(path)
    target = next(row for row in rows if row["role"] == "heldout")
    target["outcomes"]["low"]["truth"] = copy.deepcopy(target["outcomes"]["high"]["truth"])
    write(path, rows)
    result = a.analyze(tmp_path)
    zero = next(row for row in result["states"] if row["scenario"] == "stable" and row["slot"] == "7X/early")
    assert all(value[a.STATE_MODEL_COLUMNS.index("direction_correct")] is None and
        value[a.STATE_MODEL_COLUMNS.index("regret")] == 0.0 for value in zero["models"].values())
    assert result["cases"]["stable"]["overall"]["models"]["fixed0"][a.AGGREGATE_COLUMNS.index("direction_n")] == 5


def test_missing_and_truncated_outcomes_are_partial_not_zero(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "runs/collection/stable/pair_truth.json"
    rows = a.read(path)
    rows = [row for row in rows if row["id"] != "heldout/102X/late"]
    next(row for row in rows if row["id"] == "heldout/7X/early")["outcomes"]["low"]["truth"]["dt"] = 299.0
    write(path, rows)
    result = a.analyze(tmp_path)
    assert result["status"] == "partial_complete" and not result["next_control_gate"]["passed"]
    stable = result["cases"]["stable"]
    assert stable["heldout_coverage"]["actual"] == 5
    assert stable["overall"]["models"]["fixed0"][a.AGGREGATE_COLUMNS.index("n")] == 4
    assert len(result["missing"]) == 2


@pytest.mark.parametrize("failure", ["future", "heldout_fit", "two_slots", "join", "actual_action"])
def test_invalid_evidence_cannot_pass(failure, tmp_path):
    fixture(tmp_path)
    collection = tmp_path / "runs/collection/stable"
    if failure in ("future", "heldout_fit", "actual_action"):
        path = collection / "admitted_pair_records.json"
        rows = a.read(path)
        if failure == "future":
            rows[0]["available_at"] = 901.0
        elif failure == "heldout_fit":
            rows[0]["role"] = "heldout"
        else:
            rows[0]["outcomes"]["high"]["execution"]["native_applied_hold_seconds"] = 0.9375
    elif failure == "two_slots":
        path = collection / "dataset.json"
        rows = a.read(path)
        rows["packets"][0]["receipt"]["paired_records"] = 1
    else:
        path = tmp_path / "runs/learning/stable/fresh/heldout.jsonl"
        rows = a.jsonlines(path)
        rows[0]["sequence"] += 1
    write(path, rows, lines=failure == "join")
    with pytest.raises(ValueError):
        a.analyze(tmp_path)


def test_missing_matched_audit_blocks_advancement(tmp_path):
    fixture(tmp_path)
    (tmp_path / "runs/learning/stable/matched_audit.json").unlink()
    result = a.analyze(tmp_path)
    assert result["status"] == "partial_complete"
    assert not result["cases"]["stable"]["integrity"]["checks"]["matched_learning_audit"]
    assert not result["next_control_gate"]["passed"]
