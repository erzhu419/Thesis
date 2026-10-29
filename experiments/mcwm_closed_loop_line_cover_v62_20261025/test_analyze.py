"""Actual-cost decisions retain both comparators, all seeds and truthful gaps."""
import json

import pytest

from experiments.mcwm_closed_loop_line_cover_v62_20261025.analyze import (
    HERE, ROLES, analyze, run_directory)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))


def fixture_run(root, protocol, stage, seed, scenario, role, cost):
    directory = run_directory(root, protocol, stage, seed, scenario, role)
    arm = "fresh" if role == "fresh" else "no_fresh"
    reused = stage == "development" and role == "original_no_fresh"
    quota = dict(source=128, historical_target=64, fresh_target=64 if arm == "fresh" else 0,
                 historical_replacement=64 if arm == "no_fresh" else 0)
    receipts = []
    for index in range(57):
        deadline, ids = 1200+300*index, list(range(5*index, 5*index+5))
        receipt = dict(deadline=deadline, processed_at=deadline+2, records=5,
                       sequence_ids=ids, max_available_at=deadline-1)
        if role != "original_no_fresh":
            receipt["line_cover"] = dict(eligible_records=30,
                focus_latest_ids={line: value for line, value in zip(protocol["focus_lines"], ids)},
                selected_line_counts={"7X":1, "102X":1, "705X":1, "7S":2})
        receipts.append(receipt)
    waiting, riding = 2000., cost*100-4000.
    summary = dict(status="complete", cell=dict(seed=seed, scenario=scenario, arm=arm,
        external_seed=seed+100000, candidate="recent30" if role == "original_no_fresh" else "line_cover"),
        integrity=dict(all_pass=True, checks=dict(no_fresh_outcomes_retained=True,
            line_selection_complete=True, original_replay_and_sampler=True)),
        initializer=dict(path=f"/server/h2oplus_seed{seed}.pt", initial_updates=50000,
                         history_rows=32768, source_rows=4096),
        budget=dict(target_feedback_slots=285, packets=57, policy_updates=456, native_seconds=24000,
                    world_updates=0, trust_updates=0, counterfactual_queries=0,
                    training_fresh_outcomes=285 if arm == "fresh" else 0), receipts=receipts,
        replay=dict(feedback_slots_received=285, recent_covariates=285,
            fresh_outcomes_stored=285 if arm == "fresh" else 0, td_slots=quota,
            fresh_current_s_a_use=["IQL value", "SAC actor", "temperature"],
            fresh_outcome_use=["critic factual Bellman target"] if arm == "fresh" else []),
        runtime_contract=dict(action_transform="ExecutionAction", real_td_coefficient=1,
            world_enabled=False, ratio_correction=False, td_slots=quota,
            actor_value_covariates="fresh_current_s_a", temperature_states="current"),
        action_audit=dict(shared_rule_exact=True),
        clock_audit=dict(actor_real_clock=True, first_feedback_applied_at=1202.),
        post_budget_counts=dict(before=dict(policy=456, world=0, trust=0),
                                after=dict(policy=456, world=0, trust=0)),
        cost=dict(departed_passengers=100, accrued_wait_seconds=waiting,
            accrued_in_vehicle_seconds=riding, generalized_passenger_seconds=cost*100,
            generalized_seconds_per_departed_passenger=cost),
        execution_wall=dict(native_started_wall=-200. if reused else 0.,
                            native_finished_wall=-100. if reused else 100.))
    if role != "original_no_fresh":
        summary["feedback_selection"] = dict(eligible_record_counts_by_line={"7X":400, "102X":400, "705X":400},
            selected_record_counts_by_line={"7X":57, "102X":57, "705X":57, "7S":114},
            focus_available_packets={line:57 for line in protocol["focus_lines"]},
            focus_covered_packets={line:57 for line in protocol["focus_lines"]},
            label_age_seconds=dict(count=285, minimum=31., maximum=100., mean=43.),
            absent_lines_required=False, selection_reads_rewards_or_next_states=False)
    write(directory/"summary.json", summary)
    write(directory/"protocol.json", dict(policy_config=dict(batch_size=256, online_fraction=.5)))
    jsonl(directory/"metrics.jsonl", [dict(time_seconds=time, accrued_wait_seconds=waiting*fraction,
        accrued_in_vehicle_seconds=riding*fraction)
        for time, fraction in ((9000,.3), (12000,.45), (18300,.8), (24000,1.))])
    jsonl(directory/"prefix_trace.jsonl", [dict(started_at=1000, a=0), dict(started_at=1300, a=role)])
    return directory


def populate(tmp_path, stage="development", deltas=None, original_cost=100.):
    root = tmp_path/"v62"
    protocol = json.loads((HERE/"protocol.json").read_text())
    write(root/"protocol.json", protocol)
    paths = []
    for index, seed in enumerate(protocol[f"{stage}_seeds"]):
        for offset, scenario in enumerate(protocol["scenarios"]):
            delta = -2. if deltas is None else deltas[2*index+offset]
            for role in ROLES:
                cost = 100.+delta if role == "fresh" else original_cost if role == "original_no_fresh" else 100.
                paths.append(fixture_run(root, protocol, stage, seed, scenario, role, cost))
    return root, paths


def test_both_cost_gates_allow_development_with_reused_baseline(tmp_path):
    root, _ = populate(tmp_path)
    result = analyze(root, "development")
    assert result["integrity"]["all_pass"]
    assert result["decision"]["advance_to_confirmation"]
    assert result["reused_original_baseline_days"] == 4
    for cell in result["cells"]:
        assert cell["sources"]["original_no_fresh"]["reused"]
        assert cell["sources"]["original_no_fresh"]["recorded_arm"] == "no_fresh"
        assert cell["sources"]["original_no_fresh"]["recorded_candidate"] == "recent30"
        assert cell["actual_parallel"]["fresh_vs_no_fresh"]["positive_overlap"]
        assert cell["actual_parallel"]["fresh_vs_original_no_fresh"]["required"] is False
        assert cell["actual_parallel"]["fresh_vs_original_no_fresh"]["seconds"] is None


def test_beating_changed_no_fresh_cannot_replace_beating_original(tmp_path):
    root, _ = populate(tmp_path, original_cost=97.)
    result = analyze(root, "development")
    assert result["status"] == "complete" and result["integrity"]["all_pass"]
    assert result["aggregates"]["no_fresh"]["overall"]["mean_cost_difference"] == -2.
    assert result["aggregates"]["original_no_fresh"]["overall"]["mean_cost_difference"] == 1.
    assert not result["decision"]["advance_to_confirmation"]
    assert result["decision"]["reason"] == "candidate_closed_cost_gate_failed"


def test_all_seed_cells_retained_and_each_cell_win_not_required(tmp_path):
    root, _ = populate(tmp_path, deltas=[-2,-2,1,1])
    result = analyze(root, "development")
    assert result["decision"]["advance_to_confirmation"]
    summary = result["aggregates"]["original_no_fresh"]["overall"]
    assert summary["cells"] == 4 and summary["wins"] == summary["losses"] == 2
    assert summary["winning_fraction"] == summary["losing_fraction"] == .5
    assert summary["winning_mean_cost_difference"] == -2.
    assert summary["losing_mean_cost_difference"] == 1.


def test_shock_harm_closes_candidate_despite_overall_improvement(tmp_path):
    root, _ = populate(tmp_path, deltas=[-4,1,-4,1])
    result = analyze(root, "development")
    assert result["status"] == "complete"
    assert result["aggregates"]["original_no_fresh"]["overall"]["mean_cost_difference"] < 0
    assert not result["decision"]["scientific_gate_pass"]


def test_missing_reused_baseline_cannot_become_zero_or_dropped_success(tmp_path):
    root, paths = populate(tmp_path)
    (paths[2]/"summary.json").unlink()
    result = analyze(root, "development")
    assert result["status"] == "pending" and len(result["missing"]) == 1
    assert result["available_cells"] == 3
    assert not result["decision"]["advance_to_confirmation"]
    write(paths[2]/"failure.json", dict(type="NativeError", consumed_budget=dict(policy_updates=40)))
    result = analyze(root, "development")
    assert result["status"] == "integrity_failed"
    assert result["failed_run_work"][0]["consumed_budget"] == dict(policy_updates=40)
    assert result["decision"]["reason"] == "incomplete_or_integrity_failed"


@pytest.mark.parametrize("failure", ("budget", "leak", "coverage", "config", "prefix", "parallel"))
def test_integrity_failures_are_not_scientific_candidate_failures(tmp_path, failure):
    root, paths = populate(tmp_path)
    path = paths[1]
    summary = json.loads((path/"summary.json").read_text())
    if failure == "budget":
        summary["budget"]["policy_updates"] = 455
    elif failure == "leak":
        summary["replay"]["fresh_outcomes_stored"] = 1
    elif failure == "coverage":
        summary["receipts"][0]["line_cover"]["focus_latest_ids"]["7X"] = -100
    elif failure == "config":
        write(path/"protocol.json", dict(policy_config=dict(batch_size=512)))
    elif failure == "prefix":
        jsonl(path/"prefix_trace.jsonl", [dict(started_at=1000, a=7.5)])
    else:
        summary["execution_wall"] = dict(native_started_wall=100., native_finished_wall=200.)
    write(path/"summary.json", summary)
    result = analyze(root, "development")
    assert result["status"] == "integrity_failed"
    assert result["decision"]["reason"] == "incomplete_or_integrity_failed"
    assert not result["decision"]["advance_to_confirmation"]


def test_absent_focus_line_is_not_a_coverage_failure(tmp_path):
    root, paths = populate(tmp_path)
    path = paths[0]
    summary = json.loads((path/"summary.json").read_text())
    for receipt in summary["receipts"]:
        receipt["line_cover"].update(focus_latest_ids={}, selected_line_counts={"7S":5})
    report = summary["feedback_selection"]
    report.update(focus_available_packets={}, focus_covered_packets={}, selected_record_counts_by_line={"7S":285})
    write(path/"summary.json", summary)
    assert analyze(root, "development")["integrity"]["all_pass"]


def test_confirmation_requires_three_arm_overlap_and_is_only_ready_for_review(tmp_path):
    root, paths = populate(tmp_path, stage="confirmation")
    result = analyze(root, "confirmation")
    assert result["integrity"]["all_pass"]
    assert result["decision"]["confirmed_control_improvement"]
    assert result["decision"]["ordinary_dripping_goal_ready_for_root_review"]
    assert result["decision"]["ordinary_dripping_goal_achieved"] is False
    assert result["reused_original_baseline_days"] == 0
    assert len(result["aggregates"]["original_no_fresh"]["by_identity"]) == 3
    assert len(json.dumps(result, separators=(",", ":")).encode()) < 48000
    for index, path in enumerate(paths[:3]):
        summary = json.loads((path/"summary.json").read_text())
        start, end = ((0,100),(0,40),(60,100))[index]
        summary["execution_wall"] = dict(native_started_wall=start, native_finished_wall=end)
        write(path/"summary.json", summary)
    result = analyze(root, "confirmation")
    assert not result["integrity"]["all_pass"]
    assert any("no_fresh_vs_original_no_fresh/actual_parallel" in key
               for key in result["integrity"]["failed_checks"])


def test_native_sized_ledgers_float_ratios_and_twelve_line_coverage_stay_small(tmp_path):
    root, paths = populate(tmp_path, stage="confirmation")
    for index, path in enumerate(paths):
        summary = json.loads((path/"summary.json").read_text())
        waiting = 8438598+100*(index//3)
        riding = 16361600+347*(index//3)-(1000 if index % 3 == 0 else 0)
        total = 2*waiting+riding
        summary["cost"].update(departed_passengers=15079,
            accrued_wait_seconds=waiting, accrued_in_vehicle_seconds=riding,
            generalized_passenger_seconds=total,
            generalized_seconds_per_departed_passenger=total/15079)
        if "feedback_selection" in summary:
            summary["feedback_selection"]["eligible_record_counts_by_line"].update({line:123456
                for line in ("7S","122S","122X","311S","311X","406S","406X","102S","705S")})
        write(path/"summary.json", summary)
        jsonl(path/"metrics.jsonl", [dict(time_seconds=time,
            accrued_wait_seconds=waiting*fraction, accrued_in_vehicle_seconds=riding*fraction)
            for time, fraction in ((9000,.3), (12000,.45), (18300,.8), (24000,1.))])
    result = analyze(root, "confirmation")
    assert result["integrity"]["all_pass"]
    assert result["decision"]["confirmed_control_improvement"]
    assert len(json.dumps(result, separators=(",", ":")).encode()) < 48000
