"""Complete paired numbers, sparse states, and common-prediction integrity."""

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("diagnostic_v2_analysis", HERE / "analysis.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def lines(path, rows):
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))


def local_summary(shock):
    labels, models = {}, {name: {} for name in analysis.PHYSICAL_MODELS}
    for line in analysis.LINES:
        periods = {}
        for phase in ("warmup", "before", "crossing", "early", "middle", "late", "drain"):
            n = int(phase in ("before", "early", "late"))
            extra = shock and phase in ("early", "late")
            periods[phase] = dict(n=n, mean_duration_seconds=100.+20*extra if n else None,
                mean_holding_seconds=10.+5*extra if n else None, mean_nonholding_seconds=90.+15*extra if n else None)
        labels[line] = dict(by_period=periods)
        for index, model in enumerate(models):
            models[model][line] = dict(by_period={phase: dict(n=row["n"],
                mae_seconds=10.-3*index if row["n"] else None,
                bias_seconds=-10.+3*index if row["n"] else None) for phase, row in periods.items()})
    source = dict(overall=dict(n=15, mean_duration_seconds=100., mean_holding_seconds=10., mean_nonholding_seconds=90.))
    return dict(labels=dict(by_line=labels), models={name: dict(by_line=rows) for name, rows in models.items()},
        coverage=dict(direct_issued=9, direct_completed_labels=9, matched_labels=9,
            missing_saved_prediction=0, pending_predictions=0, learned_labels=9),
        initialization=dict(domain="source", available_by_line={line: 5 for line in analysis.LINES},
            used_by_line={line: 5 for line in analysis.LINES}, labels=source, features=["line_intercept"]))


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    root, reference = tmp_path / "runs", tmp_path / "reference"
    monkeypatch.setattr(analysis, "REFERENCE_ROOT", reference)
    write(root.parent / "inputs/source_summary.json", dict(status="complete", target_rows=0, policy_updates=0))
    for seed in analysis.PROTOCOL["training_seeds"]:
        for scenario in analysis.PROTOCOL["scenarios"]:
            shock = scenario == "local_speed_drop"
            for mode, arm in [("control", arm) for arm in analysis.PROTOCOL["arms"]] + [("prediction", "common")]:
                folder = root / mode / scenario / arm / f"seed{seed}"
                offset = {"real1": 0, "real075": -100, "model": -300, "context": -200, "common": 0}[arm]
                stable_cost = 30000.+100*(seed-6201)+offset
                cost = stable_cost+1000*shock
                receipts = [dict(deadline=deadline, processed_at=deadline+2, max_available_at=deadline-1,
                    records=5, sequence_ids=list(range(index*5, index*5+5))) for index, deadline in enumerate(range(1200, 18001, 300))]
                summary = dict(status="complete", seed=seed, scenario=scenario, arm=arm, mode=mode,
                    target_time_seconds=24000., full_horizon_completed=True, target_online_training_events=285,
                    packets=57, policy_updates=456 if mode == "control" else 0, sumo_version="SUMO 1.25.0",
                    generalized_passenger_seconds=cost, generalized_seconds_per_departed_passenger=cost/10,
                    passengers=dict(departed_passengers=10, completed_passengers=9, unfinished_passengers=1,
                        accrued_wait_seconds_total=cost/4, accrued_in_vehicle_seconds_total=cost/2,
                        active_state_count_balance_error=0), scenario_metadata=dict(actual_shock_at_seconds=9000 if shock else None),
                    feedback_receipts=receipts, physical_learning_enters_policy=False,
                    physical_intervals=local_summary(shock) if mode == "prediction" else None,
                    update_statistics={"before": {"train/actor_updated": dict(n=208, mean=.5)},
                                       "early_after": {"train/actor_updated": dict(n=248, mean=.5)}},
                    action_statistics={"before": {"proposed_hold_seconds": dict(n=100, mean=10.),
                        "executed_hold_seconds": dict(n=100, mean=15.), "execution_changed": dict(n=100, mean=.3)}})
                write(folder / "summary.json", summary)
                write(folder / "protocol.json", dict(initial_updates=50000, history_rows=32768,
                    source_rows=4096, initializer=f"inputs/seed{seed}.pt"))
                if mode == "control":
                    old_arm = {"real1": "replay", "model": "online", "context": "context"}.get(arm)
                    if old_arm:
                        write(reference / mode / scenario / old_arm / f"seed{seed}" / "summary.json", summary)
                    metrics = []
                    for now in range(0, 24001, 300):
                        accrued = stable_cost*now/24000+1000*shock*max(now-9000, 0)/15000
                        metrics.append(dict(time_seconds=now, accrued_wait_seconds=accrued/4,
                                            accrued_in_vehicle_seconds=accrued/2))
                    lines(folder / "metrics.jsonl", metrics)
                    lines(folder / "td_diagnostics.jsonl", [dict(deadline=deadline, processed_at=deadline+2,
                        proposed_action_change_seconds=.2, executed_action_change_seconds=.1,
                        diagnostic=dict(real_td_loss=1., real_td_contribution=.75,
                            effective_model_mean_weight_scale=.125, absolute_trust=dict(mean=.5, p05=.05, p95=.95),
                            actor_parameter_change=dict(l2=0.), critic_parameter_change=dict(l2=.01),
                            gradients=dict(model_to_real_scaled_norm_ratio=.1, cosine_real_model=.25)))
                        for deadline in range(1200, 18001, 300)])
                else:
                    global_logs = []
                    for index, model in enumerate(analysis.GLOBAL_MODELS):
                        for sequence in range(285):
                            issued = (7000., 10000., 16000.)[sequence//95]+sequence % 95
                            global_logs.append(dict(model_arm=model, sequence=sequence, issued_at=issued,
                                completed_at=issued+4, available_at=issued+34, scored_at=issued+35,
                                selected_score=3.-index, duration_absolute_error=6.-2*index,
                                cost_absolute_error=9.-3*index, selected_family_at_issue="direct",
                                selected_head_losses={head: 3.-index for head in analysis.HEADS},
                                family_head_losses={"direct": {head: 3.-index for head in analysis.HEADS}},
                                family_physical_errors={"direct": dict(cost_absolute_error=9.-3*index)}))
                    lines(folder / "prequential.jsonl", global_logs)
                    physical_logs = []
                    for model in analysis.PHYSICAL_MODELS:
                        for line_index, line in enumerate(analysis.LINES):
                            for index, phase in enumerate(("before", "early", "late")):
                                issued = (7000., 10000., 16000.)[index]
                                affected = shock and index > 0
                                actual, holding = 100.+20*affected, 10.+5*affected
                                physical_logs.append(dict(model=model, sequence=line_index*10+index, line_id=line,
                                    trip_id=f"trip{index}", vehicle_id=line, period=phase, issued_at=issued,
                                    completed_at=issued+actual, available_at=issued+actual+30, scored_at=issued+actual+31,
                                    actual_seconds=actual, holding_seconds=holding, nonholding_seconds=actual-holding))
                    lines(folder / "interval_predictions.jsonl", physical_logs)
    return root


def test_five_seed_t_interval_and_missing_values_are_not_zero():
    result = analysis.paired([1., 2., 3., 4., 5.])
    assert result["mean"] == 3
    assert result["ci95"] == pytest.approx([1.03675684, 4.96324316])
    assert analysis.paired([None]*5) == dict(values=[None]*5, n_pairs=0, mean=None,
                                          ci95=None, negative=0, positive=0)
    assert analysis.paired([2., None, None, None, None])["ci95"] is None


def test_complete_fifty_runs_keep_seed_effects_and_local_physical_mechanism(campaign):
    result = analysis.analyze(campaign)
    assert result["status"] == "complete", result["integrity"]["errors"]
    assert result["integrity"]["complete_controls"] == 40
    assert result["integrity"]["complete_predictions"] == 10
    effect = next(row for row in result["full_day_paired_effects"] if row["candidate"] == "model"
                  and row["reference"] == "real075" and row["scenario"] == "local_speed_drop")
    assert effect["values"] == [-20.]*5
    assert effect["ci95"] == [-20., -20.]
    matched = next(row for row in result["physical"]["matched_trip_shock_effects"]["paired_seed_effects"]
                   if row["line"] == "7X" and row["period"] == "early")
    assert matched["dt_seconds"]["mean"] == 20.
    assert matched["holding_seconds"]["mean"] == 5.
    assert matched["nonholding_seconds"]["mean"] == 15.
    assert all(row["exact_cost"] and row["exact_feedback"] for row in result["original_formal_reproduction"])
    assert all(row["recovery_status"] == "no_detectable_early_degradation" for row in result["window_cells"])
    assert result["td_diagnostics"]["rows"][0][-1] == .5
    assert result["td_diagnostics"]["rows"][0][-3:-1] == [None, None]
    model_td = next(row for row in result["td_diagnostics"]["rows"] if row[2] == "model")
    assert model_td[-3:-1] == [0., .1]
    assert len(json.dumps(result, separators=(",", ":")).encode()) <= 300000


def test_missing_run_and_unpaired_sequences_cannot_create_zero_or_paired_gain(campaign):
    missing = campaign / "control/local_speed_drop/real1/seed6205/summary.json"
    missing.unlink()
    path = campaign / "prediction/local_speed_drop/common/seed6204/interval_predictions.jsonl"
    rows = list(analysis.jsonl(path))
    rows[0]["sequence"] = 999
    lines(path, rows)
    result = analysis.analyze(campaign)
    assert result["status"] == "incomplete"
    assert len(result["integrity"]["missing"]) == 1
    effect = next(row for row in result["full_day_paired_effects"] if row["candidate"] == "model"
                  and row["reference"] == "real1" and row["scenario"] == "local_speed_drop")
    assert effect["n_pairs"] == 4 and effect["values"][-1] is None
    assert any("common sequences" in row["check"] for row in result["integrity"]["errors"])
    physical = next(row for row in result["physical"]["prediction_paired_effects"]
        if row["line"] == "7X" and row["period"] == "early_after" and row["scenario"] == "local_speed_drop")
    assert physical["mae_seconds"]["n_pairs"] == 4
    assert physical["mae_seconds"]["values"][3] is None


def test_matched_trip_delta_keeps_missing_counterparts_explicit():
    base = dict(line_id="7X", vehicle_id="bus", period="early", actual_seconds=100.,
                holding_seconds=10., nonholding_seconds=90.)
    stable = [dict(base, trip_id="matched"), dict(base, trip_id="only_stable")]
    shock = [dict(base, trip_id="matched", actual_seconds=120., holding_seconds=15., nonholding_seconds=105.),
             dict(base, trip_id="only_shock", actual_seconds=999.)]
    result = analysis._matched_trip_analysis({(6201, "stable"): stable, (6201, "local_speed_drop"): shock}, [6201])
    row = next(row for row in result["rows"] if row[1:3] == ["7X", "early"])
    assert row[3:] == [1, 1, 20., 5., 15.]
    missing = next(row for row in result["unmatched_by_seed_line"] if row["line"] == "7X")
    assert missing["stable_without_shock"] == missing["shock_without_stable"] == 1
    absent = next(row for row in result["rows"] if row[1:3] == ["102X", "early"])
    assert absent[3:] == [0, 0, None, None, None]


def test_context_fixed_family_decomposition_has_no_after_truth_oracle():
    rows = {}
    for seed in range(6201, 6206):
        online = dict(sequence=1, issued_at=10000., selected_family_at_issue="direct",
                      selected_score=2., cost_absolute_error=10.)
        context = dict(sequence=1, issued_at=10000., selected_family_at_issue="residual",
            selected_score=4., cost_absolute_error=20.,
            family_head_losses=dict(direct={head: 1. for head in analysis.HEADS}),
            family_physical_errors=dict(direct=dict(cost_absolute_error=5.)))
        rows[seed, "local_speed_drop"] = (dict(online=[online], context=[context]), True)
    result = analysis._family_decomposition(rows, list(range(6201, 6206)), ["local_speed_drop"])
    selection, calibration = result["paired_effects"]
    assert selection["score"]["mean"] == 3.
    assert selection["cost_mae"]["mean"] == 15.
    assert calibration["score"]["mean"] == -1.
    assert calibration["cost_mae"]["mean"] == -5.
    assert all(row[-1] == 1. for row in result["rows"])
    rows[6205, "local_speed_drop"][0]["context"][0]["family_physical_errors"] = {}
    missing = analysis._family_decomposition(rows, list(range(6201, 6206)), ["local_speed_drop"])
    assert missing["paired_effects"][1]["cost_mae"]["n_pairs"] == 4
    assert missing["paired_effects"][1]["cost_mae"]["values"][-1] is None


def test_global_feedback_metadata_distinguishes_real_zero_from_missing():
    assert analysis._feedback_coverage({}) == [[phase, *([None]*9)] for phase in ("before", "after")]
    before = dict(records=10, by_line={"7X": 2, "102X": 1, "other": 7},
                  directly_affected_upstream_records=0, directly_affected_upstream_by_line={})
    early = dict(records=5, by_line={"7X": 1, "other": 4},
                 directly_affected_upstream_records=1, directly_affected_upstream_by_line={"7X": 1})
    middle = dict(records=4, by_line={"705X": 2, "other": 2},
                  directly_affected_upstream_records=1, directly_affected_upstream_by_line={"705X": 1})
    result = analysis._feedback_coverage(dict(global_feedback_coverage=dict(
        before=before, early_after=early, middle_after=middle)))
    assert result[0] == ["before", 10, 3, 0, 2, 1, 0, 0, 0, 0]
    assert result[1] == ["after", 9, 3, 2, 1, 0, 2, 1, 0, 1]
    wired = analysis._feedback_coverage(dict(global_feedback_coverage=dict(before=before)))
    assert wired[1] == ["after", *([0]*9)]
