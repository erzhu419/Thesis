"""Decision tests: realized costs, fixed cohorts, native timing and truthful gaps."""
import json
from pathlib import Path

import pytest

from experiments.mcwm_closed_loop_control_campaign_v60_20261025.control_analyze import (
    HERE, aggregate, analyze, compare_prefix, difference, window)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))


def fixture_run(root, candidate, stage, identity, scenario, arm, *, cost=100., overlap=True):
    name = f"seed{identity}" if candidate == "recent30" else f"external{identity}"
    directory = root/"runs"/candidate/stage/name/scenario/arm
    waiting, riding = 2000., cost*100-4000.
    ledger = dict(departed_passengers=100, accrued_wait_seconds=waiting,
        accrued_in_vehicle_seconds=riding, generalized_passenger_seconds=cost*100,
        generalized_seconds_per_departed_passenger=cost)
    budget = dict(native_seconds=24000, counterfactual_queries=0)
    cell = dict(scenario=scenario, arm=arm)
    summary = dict(status="complete", integrity=dict(all_pass=True,
        checks=dict(no_fresh_outcomes_retained=True)), cost=ledger, budget=budget, cell=cell,
        execution_wall=dict(native_started_wall=0. if overlap else 101.,
                            native_finished_wall=100. if overlap else 200.))
    if candidate == "recent30":
        cell["seed"] = identity
        budget.update(target_feedback_slots=285, policy_updates=456, world_updates=0, trust_updates=0)
        summary.update(initializer=dict(path=f"/server/seed{identity}.pt", initial_updates=50000,
            history_rows=32768, source_rows=4096),
            replay=dict(covariate_window=285, recent_covariates=285,
                critic_fresh_outcome_window=30 if arm == "fresh" else 0,
                fresh_outcomes_stored=285 if arm == "fresh" else 0,
                independent_outcome_sampler_calls=456),
            runtime_contract=dict(action_transform="ExecutionAction", world_enabled=False,
                td_slots=dict(fresh_target=64 if arm == "fresh" else 0)),
            clock_audit=dict(actor_real_clock=True, first_feedback_applied_at=1202.),
            action_audit=dict(shared_rule_exact=True),
            post_budget_counts=dict(before=dict(policy=456, world=0, trust=0),
                                    after=dict(policy=456, world=0, trust=0)))
        write(directory/"protocol.json", dict(policy_config=dict(batch_size=256, online_fraction=.5)))
    else:
        cell["external_seed"] = identity
        budget.update(inherited_feedback_slots=285, inherited_policy_updates=456,
                      new_feedback_slots=0, new_policy_updates=0)
        summary["intervention"] = dict(counts=dict(selected=57, selected_high=20, selected_zero=37),
            by_line={line: dict(selected=19, selected_high=6, selected_zero=13,
                training_station_unseen=15, headway_extrapolation=4)
                for line in ("7X", "102X", "705X")},
            used_bins=[dict(line_id=line, index=index)
                for line in ("7X", "102X", "705X") for index in range(19)])
    write(directory/"summary.json", summary)
    # Native hook has no zero-clock sample; primary cost must use the ledger.
    jsonl(directory/"metrics.jsonl", [dict(time_seconds=time,
        accrued_wait_seconds=waiting*fraction, accrued_in_vehicle_seconds=riding*fraction)
        for time, fraction in ((9000, .3), (12000, .45), (18300, .8), (24000, 1.))])
    jsonl(directory/"prefix_trace.jsonl", [dict(started_at=time, a=action)
        for time, action in ((1000, 0), (1200, 0), (19000, arm))])
    return directory


def populate(root, *, candidate="recent30", stage="development", deltas=None):
    protocol = json.loads((HERE/"protocol.json").read_text())
    write(root/"protocol.json", protocol)
    seeds = protocol[f"{stage}_seeds"] if candidate == "recent30" else protocol[candidate][f"{stage}_external_seeds"]
    arms = ("fresh", "no_fresh") if candidate == "recent30" else ("readout", "q_selector", "actor_mcwm")
    paths = []
    for index, identity in enumerate(seeds):
        for offset, scenario in enumerate(protocol["scenarios"]):
            delta = -2. if deltas is None else deltas[2*index+offset]
            for arm in arms:
                paths.append(fixture_run(root, candidate, stage, identity, scenario, arm,
                    cost=100.+delta if arm == arms[0] else 100.))
    return paths


def test_missing_zero_clock_does_not_invalidate_full_day_cost(tmp_path):
    paths = populate(tmp_path)
    summary = json.loads((paths[0]/"summary.json").read_text())
    full = window(summary, [], 0, 24000)
    assert full["status"] == "complete"
    assert full["generalized_seconds_per_departed_passenger"] == 98.
    report = analyze(tmp_path, "recent30", "development")
    assert report["integrity"]["all_pass"]
    assert report["decision"]["advance_to_confirmation"]
    assert report["aggregates"]["no_fresh"]["overall"]["mean_cost_difference"] == -2.
    assert report["aggregates"]["no_fresh"]["post_budget"]["mean_cost_difference"] == pytest.approx(-.4)


def test_gate_uses_all_fixed_cells_without_requiring_each_win(tmp_path):
    populate(tmp_path, deltas=[-2, -2, 1, 1])
    report = analyze(tmp_path, "recent30", "development")
    overall = report["aggregates"]["no_fresh"]["overall"]
    assert overall["cells"] == 4 and overall["wins"] == overall["losses"] == 2
    assert overall["mean_cost_difference"] == -.5
    assert report["decision"]["advance_to_confirmation"]


def test_stable_harm_or_shock_failure_closes_scientific_candidate(tmp_path):
    populate(tmp_path, deltas=[1, -4, 1, -4])
    report = analyze(tmp_path, "recent30", "development")
    assert report["status"] == "complete"
    assert report["integrity"]["all_pass"]
    assert not report["decision"]["advance_to_confirmation"]
    assert report["decision"]["reason"] == "candidate_closed_cost_gate_failed"


def test_missing_and_failed_cells_are_not_zero_or_dropped_successes(tmp_path):
    paths = populate(tmp_path)
    (paths[0]/"summary.json").unlink()
    report = analyze(tmp_path, "recent30", "development")
    assert report["status"] == "pending"
    assert len(report["missing"]) == 1
    assert report["available_cells"] == 3
    assert not report["decision"]["advance_to_confirmation"]
    write(paths[0]/"failure.json", dict(type="NativeError"))
    report = analyze(tmp_path, "recent30", "development")
    assert report["status"] == "integrity_failed"
    assert len(report["failures"]) == 1


@pytest.mark.parametrize("change", ("prefix", "overlap", "leak", "budget", "config"))
def test_native_control_admission_failures_block_cost_gate(tmp_path, change):
    paths = populate(tmp_path)
    path = paths[1]
    summary = json.loads((path/"summary.json").read_text())
    if change == "prefix":
        jsonl(path/"prefix_trace.jsonl", [dict(started_at=1000, a=7.5)])
    elif change == "overlap":
        summary["execution_wall"] = dict(native_started_wall=100., native_finished_wall=200.)
    elif change == "leak":
        summary["replay"]["fresh_outcomes_stored"] = 1
    elif change == "budget":
        summary["budget"]["policy_updates"] = 455
    else:
        write(path/"protocol.json", dict(policy_config=dict(batch_size=512)))
    write(path/"summary.json", summary)
    report = analyze(tmp_path, "recent30", "development")
    assert report["status"] == "integrity_failed"
    assert not report["decision"]["advance_to_confirmation"]


def test_prefix_stops_before_learning_and_never_compares_diverged_suffix(tmp_path):
    left, right = tmp_path/"left.jsonl", tmp_path/"right.jsonl"
    jsonl(left, [dict(started_at=1000, a=0), dict(started_at=1300, a=7.5)])
    jsonl(right, [dict(started_at=1000, a=0), dict(started_at=1300, a=0)])
    assert compare_prefix(left, right, 1202)["all_pass"]
    assert not compare_prefix(left, right, 1500)["all_pass"]


def test_missing_secondary_window_remains_explicit(tmp_path):
    paths = populate(tmp_path)
    path = paths[0]/"metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    jsonl(path, [row for row in rows if row["time_seconds"] != 18300])
    report = analyze(tmp_path, "recent30", "development")
    assert report["integrity"]["all_pass"]
    assert report["cells"][0]["comparisons"]["no_fresh"]["post_budget"]["status"] == "missing_or_invalid_endpoint"
    assert report["aggregates"]["no_fresh"]["post_budget"]["cells"] == 3


def test_confirmation_reports_three_independent_seeds_and_small_result(tmp_path):
    populate(tmp_path, stage="confirmation")
    report = analyze(tmp_path, "recent30", "confirmation")
    assert report["decision"]["confirmed_control_improvement"]
    assert report["decision"]["ordinary_dripping_goal_achieved"]
    assert not report["decision"]["advance_to_confirmation"]
    assert list(report["aggregates"]["no_fresh"]["by_identity"]) == ["6203", "6204", "6205"]
    assert len(json.dumps(report, separators=(",", ":")).encode()) < 48000


def test_paired_readout_requires_both_baselines_and_does_not_close_ordinary_goal(tmp_path):
    populate(tmp_path, candidate="paired_readout", stage="confirmation")
    report = analyze(tmp_path, "paired_readout", "confirmation")
    assert report["decision"]["confirmed_control_improvement"]
    assert not report["decision"]["ordinary_dripping_goal_achieved"]
    assert report["decision"]["counterfactual_assisted_only"]
    assert len(json.dumps(report, separators=(",", ":")).encode()) < 48000
    # A matched selector stronger than the readout blocks admission even if
    # the original actor baseline was beaten.
    path = tmp_path/"runs/paired_readout/confirmation/external106362/stable/q_selector"
    summary = json.loads((path/"summary.json").read_text())
    summary["cost"].update(accrued_in_vehicle_seconds=4000.,
        generalized_passenger_seconds=8000., generalized_seconds_per_departed_passenger=80.)
    write(path/"summary.json", summary)
    jsonl(path/"metrics.jsonl", [dict(time_seconds=24000,
        accrued_wait_seconds=2000., accrued_in_vehicle_seconds=4000.)])
    report = analyze(tmp_path, "paired_readout", "confirmation")
    assert report["integrity"]["all_pass"]
    assert not report["decision"]["confirmed_control_improvement"]


def test_three_arm_parallel_means_baselines_also_overlap(tmp_path):
    paths = populate(tmp_path, candidate="paired_readout")
    for index, path in enumerate(paths[:3]):
        summary = json.loads((path/"summary.json").read_text())
        start, end = ((0, 100), (0, 40), (60, 100))[index]
        summary["execution_wall"] = dict(native_started_wall=start, native_finished_wall=end)
        write(path/"summary.json", summary)
    report = analyze(tmp_path, "paired_readout", "development")
    assert not report["integrity"]["all_pass"]
    assert any("q_selector_vs_actor_mcwm" in name for name in report["integrity"]["failed_checks"])
