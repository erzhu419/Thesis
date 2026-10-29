"""Reject meaningful attribution leakage and preserve unidentifiable recovery."""

import json

import pytest

from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.analyze import analyze
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.run_pair import parallel_timing


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root, *, stages=("stable", "local_speed_drop"), fresh_extra=0.):
    for scenario in stages:
        base = root/"runs"/scenario/"seed6201"
        summaries = {}
        for arm in ("fresh", "no_fresh"):
            directory = base/arm
            directory.mkdir(parents=True)
            metrics = [dict(time_seconds=now, accrued_wait_seconds=2.*now,
                accrued_in_vehicle_seconds=now+(fresh_extra*max(now-1200, 0) if arm == "fresh" else 0.),
                waiting_passengers=2, in_vehicle_passengers=1,
                departed_passengers=100) for now in range(0, 24001, 300)]
            quotas = dict(source=128, historical_target=64,
                fresh_target=64 if arm == "fresh" else 0,
                historical_replacement=64 if arm == "no_fresh" else 0)
            receipts = [dict(deadline=deadline, processed_at=deadline, max_available_at=deadline,
                records=5, sequence_ids=list(range(index*5, (index+1)*5)))
                for index, deadline in enumerate(range(1200, 18001, 300))]
            wait, ride = metrics[-1]["accrued_wait_seconds"], metrics[-1]["accrued_in_vehicle_seconds"]
            summary = dict(status="complete", integrity=dict(all_pass=True),
                initializer=dict(path="/resident/h2oplus_seed6201.pt", initial_updates=50000,
                    history_rows=32768, source_rows=4096),
                cell=dict(seed=6201, scenario=scenario, arm=arm),
                budget=dict(target_feedback_slots=285, training_fresh_outcomes=285 if arm == "fresh" else 0,
                    policy_updates=456, packets=57, native_seconds=24000,
                    world_updates=0, trust_updates=0, counterfactual_queries=0),
                receipts=receipts,
                replay=dict(feedback_slots_received=285, fresh_outcomes_stored=285 if arm == "fresh" else 0,
                    td_slots=quotas, fresh_current_s_a_use=["IQL value", "SAC actor", "temperature"],
                    fresh_outcome_use=["critic factual Bellman target"] if arm == "fresh" else []),
                runtime_contract=dict(action_transform="ExecutionAction", real_td_coefficient=1,
                    world_enabled=False, actor_value_covariates="fresh_current_s_a",
                    temperature_states="current", ratio_correction=False, td_slots=quotas),
                action_audit=dict(shared_rule_exact=True, raw_legacy_disagreements=3),
                clock_audit=dict(first_feedback_applied_at=1200., actor_real_clock=True),
                post_budget_counts=dict(before=dict(policy=456, world=0, trust=0),
                                        after=dict(policy=456, world=0, trust=0)),
                execution_wall=dict(native_started_wall=100. if arm == "fresh" else 110.,
                                    native_finished_wall=200. if arm == "fresh" else 210.),
                cost=dict(departed_passengers=100, accrued_wait_seconds=wait,
                    accrued_in_vehicle_seconds=ride, generalized_passenger_seconds=2*wait+ride,
                    generalized_seconds_per_departed_passenger=(2*wait+ride)/100),
                metrics_path="metrics.jsonl")
            write(directory/"summary.json", summary)
            (directory/"metrics.jsonl").write_text("".join(json.dumps(row)+"\n" for row in metrics))
            trace = [dict(started_at=now, sequence=index, s=[now/24000], a=[0.])
                     for index, now in enumerate((100, 500, 1199, 1300, 5000, 8999, 9100))]
            (directory/"prefix_trace.jsonl").write_text("".join(json.dumps(row)+"\n" for row in trace))
            summaries[arm] = summary
        write(base/"pair_summary.json", dict(status="complete", integrity=dict(all_pass=True),
              parallel=parallel_timing(summaries)))


def change_summary(root, arm, change):
    path = root/"runs"/"stable"/"seed6201"/arm/"summary.json"
    value = json.loads(path.read_text())
    change(value)
    write(path, value)


def test_complete_pilot_preserves_untriggered_recovery_and_has_no_automatic_confirmation(tmp_path):
    fixture(tmp_path)
    result = analyze(tmp_path, "pilot")
    assert result["integrity"]["all_pass"]
    for recovery in result["recovery"].values():
        assert recovery["recovery_status"] == "no_detectable_early_degradation"
        assert recovery["recovery_confirmed_seconds_after_shock"] is None
        assert recovery["recovery_onset_seconds_after_shock"] is None
    assert result["decision"]["stage_b"] == "closed"
    assert result["decision"]["independent_confirmation"] == "not_authorized_by_this_gate"
    assert len(json.dumps(result).encode()) < 30000


@pytest.mark.parametrize("change,failed", [
    (lambda row: row["replay"].update(fresh_outcomes_stored=1), "fresh_supervision_rights"),
    (lambda row: row["budget"].update(policy_updates=455), "exact_update_horizon"),
    (lambda row: row["post_budget_counts"]["after"].update(policy=457), "post_budget_zero_training"),
    (lambda row: row["action_audit"].update(shared_rule_exact=False), "actual_shared_f"),
    (lambda row: row["receipts"][0].update(sequence_ids=[5, 6, 7, 8, 9]), "unique_delayed_factual_feedback"),
])
def test_reject_leakage_budget_and_execution_changes(tmp_path, change, failed):
    fixture(tmp_path, stages=("stable",))
    change_summary(tmp_path, "no_fresh", change)
    result = analyze(tmp_path, "stable")
    assert result["status"] == "integrity_failed"
    assert result["integrity"]["checks"][f"stable/no_fresh/{failed}"] is False


def test_stable_cost_harm_is_reported_without_blocking_shock_dispatch(tmp_path):
    fixture(tmp_path, stages=("stable",), fresh_extra=3.)
    result = analyze(tmp_path, "stable")
    assert result["integrity"]["all_pass"]
    assert result["absolute_cost"]["stable"]["full_day"]["fresh_minus_no_fresh"][
        "generalized_passenger_seconds"] > 0
    assert result["decision"]["next_action"] == "run paired shock pilot"
    assert result["decision"]["cost_used_as_dispatch_gate"] is False
    # A canonical-vs-legacy gap is diagnostic, not a mismatch between the two arms' F.
    assert result["integrity"]["checks"]["stable/fresh/actual_shared_f"]


def test_pre_shock_future_denominator_is_not_used_as_past_state_evidence(tmp_path):
    fixture(tmp_path)
    path = tmp_path/"runs"/"local_speed_drop"/"seed6201"/"fresh"/"metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["departed_passengers"] = 999
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    assert analyze(tmp_path, "pilot")["integrity"]["checks"]["fresh/pre_shock_ledger_exact"]


def test_trace_drift_before_learning_fails_but_later_policy_divergence_is_allowed(tmp_path):
    fixture(tmp_path, stages=("stable",))
    path = tmp_path/"runs"/"stable"/"seed6201"/"no_fresh"/"prefix_trace.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["a"] = [1.]
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    assert analyze(tmp_path, "stable")["integrity"]["all_pass"]
    rows[0]["a"] = [1.]
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    assert analyze(tmp_path, "stable")["integrity"]["checks"]["stable/pre_feedback_prefix_exact"] is False


def test_missing_or_failed_native_runs_are_explicit(tmp_path):
    assert analyze(tmp_path, "stable")["status"] == "pending"
    directory = tmp_path/"runs"/"stable"/"seed6201"/"fresh"
    write(directory/"failure.json", dict(error="native exception"))
    result = analyze(tmp_path, "stable")
    assert result["status"] == "integrity_failed"
    assert result["failures"]


def test_parallel_measurement_requires_real_native_clock_overlap():
    summaries = {"fresh": dict(execution_wall=dict(native_started_wall=10., native_finished_wall=20.)),
                 "no_fresh": dict(execution_wall=dict(native_started_wall=15., native_finished_wall=25.))}
    result = parallel_timing(summaries)
    assert result["native_overlap_seconds"] == 5.
    assert result["arm_occupancy_factor"] == pytest.approx(20/15)
    summaries["no_fresh"]["execution_wall"]["native_started_wall"] = 21.
    assert parallel_timing(summaries)["positive_overlap"] is False
    assert parallel_timing({})["native_overlap_seconds"] is None
