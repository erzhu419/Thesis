"""Paired pilot effects retain missing runs and reject noncausal labels."""

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("action_local_pilot_analysis", HERE / "analysis.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def lines(path, values):
    path.write_text("".join(json.dumps(row) + "\n" for row in values))


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    p = json.loads((HERE / "protocol.json").read_text())
    root, reference = tmp_path / "runs", tmp_path / "reference"
    monkeypatch.setattr(analysis, "REFERENCE_ROOT", reference)
    model_stats = {model: dict(by_line={line: dict(by_period={phase: dict(n=1,
        mae_seconds=10. - 3 * index, bias_seconds=-10. + 3 * index)
        for phase in ("before", "early", "middle", "late")}) for line in analysis.LINES})
        for index, model in enumerate(analysis.PHYSICAL_MODELS)}
    for seed in p["training_seeds"]:
        for scenario in p["scenarios"]:
            shock = scenario == "local_speed_drop"
            for arm in p["arms"]:
                directory = root / "control" / scenario / arm / f"seed{seed}"
                offset = dict(old_model=0., action_real=100., action_model=-200., action_local=-300.)[arm]
                stable_cost = 30000. + offset + (seed - 6201) * 100
                cost = stable_cost + 1000 * shock
                physical = dict(coverage=dict(missing_saved_prediction=0, learned_labels=9),
                    initialization=dict(domain="source", used_by_line={line: 5 for line in analysis.LINES}),
                    models=model_stats)
                adapter_stats = dict(n=5, base_duration_mae_seconds=10., calibrated_duration_mae_seconds=8.,
                    base_cost_mae_passenger_seconds=20., calibrated_cost_mae_passenger_seconds=15.,
                    base_global_head_loss=1., calibrated_global_head_loss=.8)
                worlds = {arm: dict(local_adapter=dict(enabled=arm == "action_local", fit_labels=285,
                    paired_prequential=dict(by_phase={phase: adapter_stats for phase in
                        ("before", "early_after", "middle_after", "late_after")})))}
                summary = dict(status="complete", seed=seed, scenario=scenario, arm=arm,
                    target_time_seconds=24000., full_horizon_completed=True, target_online_training_events=285,
                    packets=57, policy_updates=456, sumo_version="SUMO 1.25.0",
                    generalized_passenger_seconds=cost, generalized_seconds_per_departed_passenger=cost / 10,
                    passengers=dict(departed_passengers=10, completed_passengers=9, unfinished_passengers=1,
                        accrued_wait_seconds_total=cost / 4, accrued_in_vehicle_seconds_total=cost / 2,
                        active_state_count_balance_error=0),
                    scenario_metadata=dict(actual_shock_at_seconds=9000 if shock else None),
                    feedback_receipts=[dict(deadline=deadline, processed_at=deadline + 1,
                        max_available_at=deadline - 1, records=5, sequence_ids=list(range(index * 5, index * 5 + 5)))
                        for index, deadline in enumerate(range(1200, 18001, 300))],
                    physical_intervals=physical, physical_learning_enters_policy=arm == "action_local",
                    worlds=worlds, execution_contract=dict(enabled=arm != "old_model"))
                write(directory / "summary.json", summary)
                write(directory / "protocol.json", dict(initial_updates=50000, history_rows=32768,
                    source_rows=4096, initializer=f"inputs/seed{seed}.pt"))
                if arm == "old_model":
                    write(reference / "control" / scenario / "model" / f"seed{seed}" / "summary.json", summary)
                lines(directory / "td_diagnostics.jsonl", [dict(deadline=deadline, processed_at=deadline + 1,
                    diagnostic=dict(real_td_loss=1.), local_counterfactual=dict(rows=10, applied_rows=4,
                        mean_absolute_target_q_delta=.25)) for deadline in range(1200, 18001, 300)])
                lines(directory / "metrics.jsonl", [dict(time_seconds=now,
                    accrued_wait_seconds=(stable_cost * now / 24000 + shock * 1000 * max(now - 9000, 0) / 15000) / 4,
                    accrued_in_vehicle_seconds=(stable_cost * now / 24000 + shock * 1000 * max(now - 9000, 0) / 15000) / 2)
                    for now in range(0, 24001, 300)])
                lines(directory / "prequential.jsonl", [dict(sequence=index, issued_at=10000., completed_at=10005.,
                    available_at=10035., scored_at=10040., selected_score=.5,
                    duration_absolute_error=10., cost_absolute_error=20.)
                    for index in range(0 if arm == "action_real" else 285)])
                lines(directory / "interval_predictions.jsonl", [dict(model=model, sequence=index,
                    line_id=line, completed_at=10005., available_at=10035., scored_at=10040.)
                    for model in analysis.PHYSICAL_MODELS for index, line in enumerate(analysis.LINES)])
    return root, p


def test_two_seed_screen_is_descriptive_and_keeps_missing_null():
    assert analysis.paired([-2., 4.]) == dict(values=[-2., 4.], n_pairs=2, mean=1.,
        minimum=-2., maximum=4., negative=1, positive=1)
    assert analysis.paired([None, 4.])["values"] == [None, 4.]
    assert analysis.paired([None, None])["mean"] is None


def test_complete_pilot_reports_chain_and_small_summary(pilot):
    root, p = pilot
    result = analysis.analyze(root, p)
    assert result["status"] == "complete", result["integrity"]["errors"]
    assert result["integrity"]["complete_controls"] == 16
    effect = next(row for row in result["full_day_paired_effects"] if row["candidate"] == "action_local"
                  and row["reference"] == "action_model" and row["scenario"] == "stable")
    assert effect["values"] == [-10., -10.]
    assert all(row["exact_cost"] for row in result["old_model_reproduction"])
    prediction = next(row for row in result["local_same_flow_effects"]["global_calibrated_minus_base"]
                      if row["arm"] == "action_local" and row["period"] == "all_after")
    assert prediction["cost_mae_passenger_seconds"]["values"] == [-5., -5.]
    assert result["td_diagnostics"][0]["periods"]["after"]["local_counterfactual"]["applied_rows"] == 4
    assert result["td_diagnostics"][0]["periods"]["after"]["local_counterfactual"]["applied_row_fraction"] == .4
    td = result["td_diagnostics"][0]["periods"]
    assert td["before"]["packets"] + td["after"]["packets"] == 57
    assert len(json.dumps(result, separators=(",", ":")).encode()) < 200000


def test_missing_and_early_labels_cannot_create_paired_gain(pilot):
    root, p = pilot
    (root / "control/stable/action_local/seed6202/summary.json").unlink()
    path = root / "control/local_speed_drop/action_local/seed6201/interval_predictions.jsonl"
    rows = list(analysis.jsonl(path))
    rows[0]["scored_at"] = 10000.
    lines(path, rows)
    result = analysis.analyze(root, p)
    assert result["status"] == "incomplete"
    assert result["integrity"]["missing"] == ["control/stable/action_local/seed6202"]
    assert any(row["check"] == "physical outcome availability" for row in result["integrity"]["errors"])
    effect = next(row for row in result["full_day_paired_effects"] if row["candidate"] == "action_local"
                  and row["reference"] == "action_model" and row["scenario"] == "local_speed_drop")
    assert effect["values"] == [None, -10.]
