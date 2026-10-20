"""Cohort mixing, passive cost drift and missing frozen endpoints change decisions."""

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v4_analysis_test", HERE / "analysis.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def lines(path, values):
    path.write_text("".join(json.dumps(row) + "\n" for row in values))


def _run(directory, seed, scenario, arm, p, *, head=True):
    shock = scenario == "local_speed_drop"
    # Original two favour model TD; the fresh three do not. Combining them
    # without cohort labels would hide the development selection history.
    offset = {"action_real": 0, "action_model": -20 if seed <= 6202 else 40,
              "action_local": -10}[arm]
    cost = 30000. + offset + 1000 * shock
    physical = dict(coverage=dict(missing_saved_prediction=0),
        initialization=dict(domain="source", used_by_line={line: 5 for line in analysis.v3.LINES}),
        models={model: dict(by_line={line: dict(by_period={phase: dict(n=1, mae_seconds=5., bias_seconds=-3.)
            for phase in ("before", "early", "middle", "late")}) for line in analysis.v3.LINES})
            for model in analysis.v3.PHYSICAL_MODELS})
    adapter = dict(enabled=arm == "action_local", fit_labels=285,
        paired_prequential=dict(by_phase={phase: dict(n=1, base_duration_mae_seconds=10.,
            calibrated_duration_mae_seconds=8., base_cost_mae_passenger_seconds=20.,
            calibrated_cost_mae_passenger_seconds=15., base_global_head_loss=1., calibrated_global_head_loss=.8)
            for phase in ("before", "early_after", "middle_after", "late_after")}))
    heads = dict(head_order=["logdt", "logcost", "dWaiting", "dOnboard"],
        error_sign="prediction-minus-observed", overall=dict(n=285,
            base=dict(physical=dict(signed_mean=[-1.] * 4, mae=[2.] * 4, mse=[9.] * 4))),
        by_phase=dict(before=dict(n=135), early_after=dict(n=50), middle_after=dict(n=50), late_after=dict(n=50)),
        by_line={}, by_phase_and_line={})
    world = dict(local_adapter=adapter, **({"head_diagnostics": heads} if head else {}))
    summary = dict(status="complete", seed=seed, scenario=scenario, arm=arm,
        target_time_seconds=24000., full_horizon_completed=True, target_online_training_events=285,
        packets=57, policy_updates=456, sumo_version="SUMO 1.25.0",
        generalized_passenger_seconds=cost, generalized_seconds_per_departed_passenger=cost / 10,
        passengers=dict(departed_passengers=10, completed_passengers=9, unfinished_passengers=1,
            accrued_wait_seconds_total=cost / 4, accrued_in_vehicle_seconds_total=cost / 2,
            active_state_count_balance_error=0),
        scenario_metadata=dict(actual_shock_at_seconds=9000 if shock else None),
        feedback_receipts=[dict(deadline=deadline, processed_at=deadline + 1,
            max_available_at=deadline - 1, records=5, sequence_ids=list(range(i * 5, i * 5 + 5)))
            for i, deadline in enumerate(range(1200, 18001, 300))],
        physical_intervals=physical, physical_learning_enters_policy=arm == "action_local", worlds={arm: world})
    write(directory / "summary.json", summary)
    write(directory / "protocol.json", dict(initial_updates=50000, history_rows=32768,
        source_rows=4096, initializer=f"inputs/seed{seed}.pt"))
    lines(directory / "td_diagnostics.jsonl", [dict(deadline=deadline, processed_at=deadline + 1,
        diagnostic=dict(real_td_loss=1.), head_counterfactual=dict(rows=8,
            variants={name: dict(rows=8, applied_rows=4 if arm == "action_local" else 0,
                target_q=dict(actual_mean=1., counterfactual_mean=.75, mean_delta=.25 if arm == "action_local" else 0.,
                    mean_absolute_delta=.25 if arm == "action_local" else 0.,
                    max_absolute_delta=.5 if arm == "action_local" else 0.))
                for name in ("full_base", "omit_logdt", "omit_logcost", "omit_dWaiting", "omit_dOnboard")}))
        for deadline in range(1200, 18001, 300)])
    # Stable/shock ledger prefixes agree exactly through the intervention.
    lines(directory / "metrics.jsonl", [dict(time_seconds=now,
        accrued_wait_seconds=((30000. + offset) * now / 24000 + shock * 1000 * max(now - 9000, 0) / 15000) / 4,
        accrued_in_vehicle_seconds=((30000. + offset) * now / 24000 + shock * 1000 * max(now - 9000, 0) / 15000) / 2)
        for now in range(0, 24001, 300)])
    lines(directory / "prequential.jsonl", [dict(sequence=i, issued_at=10000., completed_at=10005.,
        available_at=10035., scored_at=10040., selected_score=.5, duration_absolute_error=10.,
        cost_absolute_error=20.) for i in range(0 if arm == "action_real" else 285)])
    lines(directory / "interval_predictions.jsonl", [dict(model=model, sequence=i,
        line_id=line, completed_at=10005., available_at=10035., scored_at=10040.)
        for model in analysis.v3.PHYSICAL_MODELS for i, line in enumerate(analysis.v3.LINES)])


@pytest.fixture
def campaign(tmp_path):
    p = json.loads((HERE.parent / "mcwm_closed_loop_action_local_v3_20261020/protocol.json").read_text())
    p["run_cells"] = [dict(seed=seed, scenario=scenario, arm=arm) for seed in range(6201, 6206)
        for scenario in p["scenarios"] for arm in (("action_model", "action_local") if seed <= 6202
                                                  else ("action_real", "action_model"))]
    root, prior, old = tmp_path / "runs", tmp_path / "v3", tmp_path / "v2"
    for row in p["run_cells"]:
        path = root / "control" / row["scenario"] / row["arm"] / f"seed{row['seed']}"
        _run(path, **row, p=p)
    for seed in (6201, 6202):
        for scenario in p["scenarios"]:
            for arm in ("action_real", "action_model", "action_local"):
                _run(prior / "runs/control" / scenario / arm / f"seed{seed}", seed, scenario, arm, p)
    for seed in range(6201, 6206):
        for scenario in p["scenarios"]:
            _run(old / "runs/control" / scenario / "model" / f"seed{seed}", seed, scenario, "action_real", p)
    return root, p, prior, old


def test_separate_cohorts_passive_reproduction_and_fixed_window(campaign):
    root, p, prior, old = campaign
    result = analysis.analyze(root, p, prior, old)
    assert result["status"] == "complete", result["integrity"]["errors"]
    effects = {row["cohort"]: row for row in result["stability_paired_effects"]
        if row["candidate"] == "action_model" and row["reference"] == "action_real"
        and row["scenario"] == "stable" and row["quantity"] == "full_day_seconds_per_departed_passenger"}
    assert effects["original_two_screened"]["values"] == [-2., -2.]
    assert effects["fresh_three"]["values"] == [4., 4., 4.]
    assert effects["all_five_development"]["values"] == [-2., -2., 4., 4., 4.]
    assert all(row["exact_cost_and_budget"] for row in result["passive_reproduction"])
    assert result["head_diagnostics"][0]["diagnostics"]["overall"]["base"]["physical"]["rmse"] == [3.] * 4
    assert all(row["windows"]["post_training_frozen"]["status"] == "complete"
               for row in result["new_control_cells"])
    active = next(row for row in result["head_counterfactuals"] if row["arm"] == "action_local")
    variants = active["periods"]["early_after"]["variants"]
    assert variants["omit_logdt"]["target_q"]["mean_absolute_delta"] == .25
    assert variants["omit_logdt"]["applied_rows"] == 40
    inactive = next(row for row in result["head_counterfactuals"] if row["arm"] == "action_model")
    assert inactive["periods"]["late_after"]["all_variant_deltas_exact_zero"] is True
    assert len(json.dumps(result, separators=(",", ":")).encode()) < 500000


def test_missing_endpoint_and_passive_cost_drift_fail_explicitly(campaign):
    root, p, prior, old = campaign
    directory = root / "control/stable/action_local/seed6201"
    summary = analysis.read(directory / "summary.json")
    summary["generalized_passenger_seconds"] += 4
    summary["generalized_seconds_per_departed_passenger"] += .4
    summary["passengers"]["accrued_wait_seconds_total"] += 1
    summary["passengers"]["accrued_in_vehicle_seconds_total"] += 2
    write(directory / "summary.json", summary)
    rows = [row for row in analysis.jsonl(directory / "metrics.jsonl") if row["time_seconds"] != 18300]
    lines(directory / "metrics.jsonl", rows)
    result = analysis.analyze(root, p, prior, old)
    assert result["status"] == "incomplete"
    assert any(row["scope"] == "passive_reproduction" for row in result["integrity"]["errors"])
    assert any(row["scope"] == "fixed_clock_cost" and row["status"] == "missing_endpoint"
               for row in result["integrity"]["errors"])
    effect = next(row for row in result["local_replay_paired_effects"] if row["scenario"] == "stable"
                  and row["quantity"] == "post_training_frozen_seconds_per_full_day_departed_passenger")
    assert effect["values"][0] is None


def test_missing_pair_is_retained_and_t_ci_uses_seed_units():
    result = analysis.paired([-1., None, 3.], [6203, 6204, 6205])
    assert result["values"] == [-1., None, 3.] and result["n_pairs"] == 2
    assert result["descriptive_95pct_t_ci"][0] < -20.
