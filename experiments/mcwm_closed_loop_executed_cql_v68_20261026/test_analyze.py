"""Executed CQL scope, unchanged budgets and actual-cost acceptance gates."""
import copy
import json

import pytest

from experiments.mcwm_closed_loop_fresh_value_v64_20261025.test_analyze import jsonl, summary, write
from experiments.mcwm_closed_loop_executed_cql_v68_20261026.analyze import (
    HERE, SCHEDULE, EXPECTED_COUNTS, MAX_BYTES, analyze, encoded, export, split_summary)


def schedule():
    return dict(packets=57, critic_updates=456, value_updates=456,
        actor_updates=456, alpha_updates=456, packet_critic_steps=[8]*57,
        packet_value_steps=[8]*57, packet_actor_steps=[8]*57,
        packet_alpha_steps=[8]*57,
        optimizer_order_exact=True, schedule=dict(SCHEDULE), kl_penalty=False,
        reference_for_diagnostics_only=True, additional_random_draws=0)


def cql():
    return dict(coefficient=.2, temperature=1., penalty_calls=456, factual_rows=456*128,
        proposals_per_state=22, candidate_rows_before_dedup=456*128*22,
        candidate_rows_after_dedup=456*128*12,
        unique_candidates_per_state=dict(min=3, max=22, mean=12.),
        main_rng_preserved=True, numpy_draws=0, all_heads=True, exact_deduplication=True,
        isolated_uniform_proposals=10*456*128, isolated_positive_proposals=10*456*128,
        outcome_fields_read=False, factual_actions_transformed=False,
        candidate_actions_transformed=True, finite_penalty=True,
        penalty=dict(mean=.5,min=-.1,max=.8), weighted_penalty=dict(mean=.1,min=-.02,max=.16),
        by_head=[dict(head=i, factual_q_mean=1100.+i, candidate_q_mean=1100.2+i,
            logmeanexp_mean=1100.5+i, penalty_mean=.5, candidate_advantage_mean=.2,
            candidate_advantage_abs_mean=.4, deterministic_positive_minus_zero_mean=.3)
            for i in range(5)])


def drift():
    return dict(same_states=True, same_F=True, additional_random_draws=0, packets=57, states=285,
        groups=[dict(line="7X", window="early", count=285, gate_crossings=1,
            gate_kl_sum=.1, weighted_positive_kl_sum=.2, total_kl_sum=.3,
            holding_delta_sum_seconds=2., holding_abs_delta_sum_seconds=3.,
            holding_abs_delta_max_seconds=1.875)])


def populate(tmp_path, *, fresh=(98.,)*4, no_fresh=(99.,)*4, original_fresh=(103.,)*4):
    root = tmp_path/"v68"
    protocol = json.loads((HERE/"protocol.json").read_text()); write(root/"protocol.json", protocol)
    paths = []
    for i, seed in enumerate(protocol["development_seeds"]):
        ref = protocol["original_reference_by_seed"][str(seed)]
        for j, scene in enumerate(protocol["scenarios"]):
            index = 2*i+j
            for arm in ("fresh", "no_fresh", "original_fresh", "original_no_fresh"):
                old = arm.startswith("original_"); native_arm = arm.removeprefix("original_")
                directory = (root.parent/ref["directory"]/ref["relative"].format(seed=seed, scenario=scene)/native_arm
                    if old else root/"runs/executed_cql/development"/f"seed{seed}"/scene/arm)
                cost = 100. if arm == "original_no_fresh" else original_fresh[index] if arm == "original_fresh" else fresh[index] if arm == "fresh" else no_fresh[index]
                value = summary(seed, scene, native_arm, cost)
                if not old:
                    value["cell"]["candidate"] = "executed_cql"
                    value["actor_schedule_audit"] = schedule(); value["policy_drift"] = drift()
                    value["cql_audit"] = cql()
                    value["budget"].update(policy_updates=456, actor_updates=456, alpha_updates=456,
                                            critic_updates=456, value_updates=456)
                    value["post_budget_counts"] = dict(before=dict(EXPECTED_COUNTS), after=dict(EXPECTED_COUNTS))
                value["runtime_contract"]["optimizer_schedule"] = "native actor-before-critic"
                write(directory/"summary.json", value)
                config = dict(batch_size=256, online_fraction=.5,
                    online_actor_update_interval=1, online_actor_batch_mode="latest",
                    online_update_order="actor-before-critic", ensemble_size=5, online_cql_alpha=0.)
                if not old: config["online_cql_alpha"] = .2
                write(directory/"protocol.json", dict(policy_config=config))
                cost = value["cost"]
                jsonl(directory/"metrics.jsonl", [dict(time_seconds=t,
                    accrued_wait_seconds=cost["accrued_wait_seconds"]*f,
                    accrued_in_vehicle_seconds=cost["accrued_in_vehicle_seconds"]*f)
                    for t, f in ((9000,.3), (12000,.45), (18300,.8), (24000,1.))])
                jsonl(directory/"prefix_trace.jsonl", [dict(started_at=1000, a=0), dict(started_at=1300, a=arm)])
                if not old: paths.append(directory)
    return root, paths


def modify(path, key, value):
    data = json.loads(path.read_text()); target = data
    for k in key[:-1]: target = target[k]
    target[key[-1]] = value; write(path, data)


def test_four_arms_three_windows_and_fixed_actual_cost_gates(tmp_path):
    root, _ = populate(tmp_path); result = analyze(root)
    assert result["status"] == "complete" and result["decision"]["advance_to_confirmation"]
    assert result["available_cells"] == result["expected_cells"] == 4
    assert result["aggregates"]["matched"]["full_day"]["overall"]["mean_cost_difference"] == -1.
    assert result["aggregates"]["original_no_fresh"]["full_day"]["overall"]["mean_cost_difference"] == -2.
    assert result["aggregates"]["no_fresh_shift"]["full_day"]["overall"]["mean_cost_difference"] == -1.
    assert result["aggregates"]["original_fresh"]["early"]["overall"]["mean_cost_difference"] == pytest.approx(-.75)
    row = result["cells"][0]["periods"]["full_day"]
    assert set(row["arms"]) == {"fresh", "no_fresh", "original_fresh", "original_no_fresh"}
    assert row["arms"]["fresh"]["cost"] == 2*row["arms"]["fresh"]["wait"]+row["arms"]["fresh"]["ride"]
    crossing_column = result["policy_drift_columns"].index("gate_crossings")
    assert result["native_evidence"][0]["policy_drift"]["fresh"]["rows"][0][crossing_column] == 1
    assert result["decision"]["ordinary_dripping_goal_achieved"] is False


def test_degrading_new_control_cannot_manufacture_fresh_improvement(tmp_path):
    root, _ = populate(tmp_path, fresh=(101.,)*4, no_fresh=(110.,)*4)
    result = analyze(root)
    assert result["decision"]["matched_cost_gate_pass"]
    assert not result["decision"]["original_reference_gate_pass"]
    assert not result["decision"]["scientific_gate_pass"]
    assert result["aggregates"]["no_fresh_shift"]["full_day"]["overall"]["mean_cost_difference"] == 10.


def test_control_shift_is_explicit_but_not_an_extra_no_harm_gate(tmp_path):
    root, _ = populate(tmp_path, no_fresh=(104.,)*4)
    result = analyze(root)
    assert result["decision"]["scientific_gate_pass"]
    assert result["aggregates"]["no_fresh_shift"]["full_day"]["overall"]["mean_cost_difference"] == 4.


def test_collapsing_fresh_to_matched_control_is_not_success(tmp_path):
    root, _ = populate(tmp_path, fresh=(99.,)*4)
    result = analyze(root)
    assert not result["decision"]["matched_cost_gate_pass"]
    assert not result["decision"]["scientific_gate_pass"]


def test_shock_harm_is_not_hidden_by_stable_gains(tmp_path):
    root, _ = populate(tmp_path, fresh=(95.,101.,95.,101.), no_fresh=(100.,)*4)
    result = analyze(root)
    assert result["aggregates"]["matched"]["full_day"]["overall"]["mean_cost_difference"] < 0
    assert not result["decision"]["matched_cost_gate_pass"]


def test_original_fresh_full_day_and_worst_cell_must_not_worsen(tmp_path):
    root, _ = populate(tmp_path, original_fresh=(97.,)*4)
    result = analyze(root)
    conditions = result["optimization_assessment"]["conditions"]
    assert not conditions["full_day_not_worse_than_original_fresh"]
    assert not conditions["worst_cell_not_worse"]
    assert not result["decision"]["scientific_gate_pass"]


def test_full_day_win_cannot_hide_early_loss_or_stable_tail_harm(tmp_path):
    root, paths = populate(tmp_path)
    for path in paths:
        if path.name != "fresh": continue
        value = json.loads((path/"summary.json").read_text())["cost"]
        # Move unchanged full-day cost into early and tail windows.
        jsonl(path/"metrics.jsonl", [dict(time_seconds=t,
            accrued_wait_seconds=value["accrued_wait_seconds"]*f,
            accrued_in_vehicle_seconds=value["accrued_in_vehicle_seconds"]*f)
            for t, f in ((9000,.1), (12000,.4), (18300,.5), (24000,1.))])
    result = analyze(root); conditions = result["optimization_assessment"]["conditions"]
    assert result["decision"]["matched_cost_gate_pass"]
    assert not conditions["early_lower_than_original_fresh"]
    assert not conditions["stable_tail_not_higher_than_original_no_fresh"]
    assert not result["decision"]["scientific_gate_pass"]


@pytest.mark.parametrize("field,value", [
    (("actor_schedule_audit", "critic_updates"), 455),
    (("actor_schedule_audit", "actor_updates"), 57),
    (("actor_schedule_audit", "alpha_updates"), 56),
    (("actor_schedule_audit", "schedule"), dict(SCHEDULE, online_update_order="critic-before-actor")),
    (("actor_schedule_audit", "packet_actor_steps"), [1]*56),
    (("actor_schedule_audit", "optimizer_order_exact"), False),
    (("actor_schedule_audit", "kl_penalty"), True),
    (("cql_audit", "main_rng_preserved"), False),
    (("policy_drift", "same_F"), False),
    (("policy_drift", "states"), 284),
    (("budget", "policy_updates"), 57),
    (("budget", "critic_updates"), 455),
    (("post_budget_counts", "after", "actor"), 58),
    (("replay", "fresh_outcomes_stored"), 284),
])
def test_failed_execution_or_budget_audits_block_scientific_claim(tmp_path, field, value):
    root, paths = populate(tmp_path); modify(paths[0]/"summary.json", field, value)
    result = analyze(root)
    assert result["status"] == "integrity_failed"
    assert not result["decision"]["scientific_gate_pass"]
    assert result["decision"]["reason"] == "incomplete_or_integrity_failed"


def test_no_fresh_outcome_leak_and_nonparallel_native_days_are_rejected(tmp_path):
    root, paths = populate(tmp_path)
    modify(paths[1]/"summary.json", ("replay", "fresh_outcomes_stored"), 1)
    modify(paths[1]/"summary.json", ("execution_wall",), dict(native_started_wall=100.,native_finished_wall=200.))
    result = analyze(root)
    assert result["status"] == "integrity_failed"
    assert any("actual_parallel" in key for key in result["integrity"]["failed_checks"])


def test_missing_or_failed_native_work_is_not_zero_cost(tmp_path):
    root, paths = populate(tmp_path); path = paths[0]; (path/"summary.json").unlink()
    result = analyze(root)
    assert result["status"] == "pending" and result["available_cells"] == 3
    assert result["aggregates"]["matched"]["full_day"]["overall"]["missing_cells"] == 1
    assert not result["decision"]["scientific_gate_pass"]
    write(path/"failure.json", dict(type="NativeError", consumed_budget=dict(policy_updates=16)))
    result = analyze(root)
    assert result["status"] == "integrity_failed"
    assert result["failed_run_work"][0]["consumed_budget"]["policy_updates"] == 16


def native_sized_fixture(tmp_path):
    root, paths = populate(tmp_path)
    # v66's existing small summaries provide the real supported twelve-line
    # group shape, without reading checkpoints or native trajectories.
    source = HERE.parent/"mcwm_closed_loop_actor_constraint_v66_20261026"
    for path in paths:
        value = json.loads((path/"summary.json").read_text())
        cell = value["cell"]
        old = source/"runs/actor_constrained/development"/f"seed{cell['seed']}"/cell["scenario"]/cell["arm"]/"summary.json"
        value["policy_drift"] = json.loads(old.read_text())["policy_drift"]
        value["initializer"]["path"] = "/home/zhengliang01/scheduleurm_work/Thesis/outputs/frozen_initializer_seed"+str(cell["seed"])+".pt"
        protocol = json.loads((root/"protocol.json").read_text()); ref = protocol["original_reference_by_seed"][str(cell["seed"])]
        for arm in ("fresh", "no_fresh"):
            old_path = root.parent/ref["directory"]/ref["relative"].format(seed=cell["seed"],scenario=cell["scenario"])/arm/"summary.json"
            modify(old_path, ("initializer", "path"), value["initializer"]["path"])
        write(path/"summary.json", value)
    for path in root.parent.rglob("summary.json"):
        value = json.loads(path.read_text()); scalar = value["cost"]["generalized_seconds_per_departed_passenger"]
        primary = 2209.123456789+scalar-100.+.123456789*(value["cell"]["seed"]%10)
        wait = 8438598.256789+value["cell"]["seed"]+23.5136789123*scalar
        ride = 15079*primary-2*wait; total = 2*wait+ride
        value["cost"] = dict(departed_passengers=15079, accrued_wait_seconds=wait,
            accrued_in_vehicle_seconds=ride, generalized_passenger_seconds=total,
            generalized_seconds_per_departed_passenger=total/15079)
        write(path, value)
        jsonl(path.parent/"metrics.jsonl", [dict(time_seconds=t,
            accrued_wait_seconds=wait*f, accrued_in_vehicle_seconds=ride*f)
            for t, f in ((9000,.300123456), (12000,.450234567), (18300,.800345678), (24000,1.))])
    return root, analyze(root)


def test_twelve_line_split_roundtrip_cost_precision_and_both_pull_caps(tmp_path):
    root, original = native_sized_fixture(tmp_path)
    assert original["integrity"]["all_pass"]
    untouched = copy.deepcopy(original); cost, policy = split_summary(original)
    assert original == untouched
    restored = copy.deepcopy(cost); restored.pop("policy_drift_file")
    for target, source in zip(restored["native_evidence"], policy["native_evidence"]):
        assert (target["seed"], target["scenario"]) == (source["seed"], source["scenario"])
        target["policy_drift"] = source["policy_drift"]
    assert restored == original
    line_column = policy["policy_drift_columns"].index("line")
    assert all(len({row[line_column] for row in arm["rows"]}) == 12
               for cell in policy["native_evidence"] for arm in cell["policy_drift"].values())
    assert len(encoded(cost)) <= MAX_BYTES and len(encoded(policy)) <= MAX_BYTES
    out = root/"results/development.json"
    receipt = export(root, out)
    assert set(receipt["files"]) == {"development.json", "policy_drift.json"}
    assert all(size <= MAX_BYTES for size in receipt["files"].values())
    assert json.loads(out.read_text())["decision"] == original["decision"]
    assert json.loads(out.read_text())["aggregates"] == original["aggregates"]


def test_only_the_declared_cql_coefficient_is_changed(tmp_path):
    root, paths = populate(tmp_path)
    path = paths[0]/"protocol.json"
    modify(path, ("policy_config", "online_cql_alpha"), .3)
    result = analyze(root)
    assert not result["integrity"]["all_pass"]
    modify(path, ("policy_config", "online_cql_alpha"), .2)
    for directory in paths:
        modify(directory/"protocol.json", ("policy_config", "batch_size"), 512)
    result = analyze(root)
    assert not result["integrity"]["all_pass"]
    assert any("same_policy_config" in key for key in result["integrity"]["failed_checks"])


def test_export_size_failure_writes_neither_file(tmp_path, monkeypatch):
    from experiments.mcwm_closed_loop_executed_cql_v68_20261026 import analyze as module
    _, value = native_sized_fixture(tmp_path)
    value["extra"] = "x"*MAX_BYTES
    monkeypatch.setattr(module, "analyze", lambda _: value)
    out = tmp_path/"new/results/development.json"
    with pytest.raises(ValueError, match="development.json exceeds"):
        export(tmp_path, out)
    assert not out.exists() and not out.with_name("policy_drift.json").exists()


def test_original_optimizer_schedule_and_F_are_unchanged(tmp_path):
    root, paths = populate(tmp_path)
    assert analyze(root)["integrity"]["all_pass"]
    path = paths[0]/"summary.json"
    modify(path, ("runtime_contract", "optimizer_schedule"), "native critic-before-actor")
    value = analyze(root)
    assert any("unchanged_F_and_runtime" in key for key in value["integrity"]["failed_checks"])
    modify(path, ("runtime_contract", "optimizer_schedule"), "native actor-before-critic")
    modify(path, ("runtime_contract", "action_transform"), "changed_F")
    value = analyze(root)
    assert any("unchanged_F_and_runtime" in key for key in value["integrity"]["failed_checks"])
    assert not value["decision"]["scientific_gate_pass"]

@pytest.mark.parametrize("field,value", [
    (("coefficient",), .3),
    (("temperature",), .5),
    (("penalty_calls",), 455),
    (("factual_rows",), 456*256),
    (("candidate_rows_before_dedup",), 456*128*23),
    (("candidate_rows_after_dedup",), 456*128*23),
    (("unique_candidates_per_state", "mean"), 11.),
    (("factual_actions_transformed",), True),
    (("candidate_actions_transformed",), False),
    (("outcome_fields_read",), True),
    (("all_heads",), False),
    (("by_head",), cql()["by_head"][:-1]),
    (("weighted_penalty", "mean"), .5),
    (("main_rng_preserved",), False),
    (("numpy_draws",), 1),
])
def test_wrong_cql_scope_or_execution_blocks_claim(tmp_path, field, value):
    root, paths = populate(tmp_path)
    modify(paths[0]/"summary.json", ("cql_audit", *field), value)
    result = analyze(root)
    assert result["status"] == "integrity_failed"
    assert any("cql_" in key for key in result["integrity"]["failed_checks"])
    assert not result["decision"]["scientific_gate_pass"]


def test_cql_evidence_preserves_all_heads_and_separates_level_and_action_difference(tmp_path):
    root, _ = populate(tmp_path)
    result = analyze(root)
    columns = result["cql_head_columns"]
    value = result["native_evidence"][0]["executed_cql"]["fresh"]
    assert value["penalty_calls"] == 456 and value["factual_rows"] == 58368
    assert value["candidate_rows_before_dedup"] == 1284096
    assert [row[columns.index("head")] for row in value["head_rows"]] == list(range(5))
    assert value["head_rows"][0][columns.index("factual_q_mean")] == 1100.
    assert value["head_rows"][0][columns.index("deterministic_positive_minus_zero_mean")] == .3
    assert "exclude" in result["cql_diagnostics_scope"]
