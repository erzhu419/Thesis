"""Actual-cost attribution and packet-policy audit tests for fixed actor KL."""
import copy
import json

import pytest

from experiments.mcwm_closed_loop_fresh_value_v64_20261025.test_analyze import jsonl, summary, write
from experiments.mcwm_closed_loop_actor_constraint_v66_20261026.analyze import HERE, analyze


def constraint():
    return dict(coefficient=1., packets=57, reference_refreshes=57, updates=456,
        actor_updates=456, alpha_updates=456, packet_step_counts=[8]*57,
        packet_reference_fixed=True, full_hurdle_kl_scope=True,
        kl_orientation="new||packet_reference", additional_random_draws=0)


def drift():
    return dict(same_states=True, same_F=True, additional_random_draws=0, packets=57, states=285,
        groups=[dict(line="7X", window="early", count=285, gate_crossings=1,
            gate_kl_sum=.1, weighted_positive_kl_sum=.2, total_kl_sum=.3,
            holding_delta_sum_seconds=2., holding_abs_delta_sum_seconds=3.,
            holding_abs_delta_max_seconds=1.875)])


def populate(tmp_path, *, fresh=(98.,)*4, no_fresh=(99.,)*4, original_fresh=(103.,)*4):
    root = tmp_path/"v66"
    protocol = json.loads((HERE/"protocol.json").read_text()); write(root/"protocol.json", protocol)
    paths = []
    for i, seed in enumerate(protocol["development_seeds"]):
        ref = protocol["original_reference_by_seed"][str(seed)]
        for j, scene in enumerate(protocol["scenarios"]):
            index = 2*i+j
            for arm in ("fresh", "no_fresh", "original_fresh", "original_no_fresh"):
                old = arm.startswith("original_"); native_arm = arm.removeprefix("original_")
                directory = (root.parent/ref["directory"]/ref["relative"].format(seed=seed, scenario=scene)/native_arm
                    if old else root/"runs/actor_constrained/development"/f"seed{seed}"/scene/arm)
                cost = 100. if arm == "original_no_fresh" else original_fresh[index] if arm == "original_fresh" else fresh[index] if arm == "fresh" else no_fresh[index]
                value = summary(seed, scene, native_arm, cost)
                if not old:
                    value["cell"]["candidate"] = "actor_constrained"
                    value["actor_constraint_audit"] = constraint(); value["policy_drift"] = drift()
                write(directory/"summary.json", value)
                write(directory/"protocol.json", dict(policy_config=dict(batch_size=256, online_fraction=.5)))
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
    (("actor_constraint_audit", "coefficient"), 2.),
    (("actor_constraint_audit", "reference_refreshes"), 456),
    (("actor_constraint_audit", "packet_step_counts"), [8]*56),
    (("actor_constraint_audit", "additional_random_draws"), 1),
    (("policy_drift", "same_F"), False),
    (("policy_drift", "states"), 284),
    (("budget", "policy_updates"), 455),
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


def test_bounded_realistic_four_cell_summary(tmp_path):
    root, paths = populate(tmp_path)
    for path in paths:
        value = json.loads((path/"summary.json").read_text())
        value["policy_drift"]["groups"] = []
        for i, line in enumerate(("7X", "102X", "705X")):
            for window in ("pre_early", "early", "late_training"):
                group = copy.deepcopy(drift()["groups"][0]); group.update(line=line, window=window, count=31+i)
                value["policy_drift"]["groups"].append(group)
        value["policy_drift"]["groups"][-1]["count"] -= 3
        # 3*(31+32+33)-3=285 current paid states, with native-size strings.
        value["initializer"]["path"] = "/home/zhengliang01/scheduleurm_work/Thesis/outputs/frozen_initializer_seed"+str(value["cell"]["seed"])+".pt"
        # Match the same immutable initializer description in old reference summaries.
        protocol = json.loads((root/"protocol.json").read_text()); ref = protocol["original_reference_by_seed"][str(value["cell"]["seed"])]
        for arm in ("fresh", "no_fresh"):
            old = root.parent/ref["directory"]/ref["relative"].format(seed=value["cell"]["seed"],scenario=value["cell"]["scenario"])/arm/"summary.json"
            modify(old, ("initializer", "path"), value["initializer"]["path"])
        write(path/"summary.json", value)
    for path in root.parent.rglob("summary.json"):
        value = json.loads(path.read_text()); cost = value["cost"]
        scalar = cost["generalized_seconds_per_departed_passenger"]
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
    result = analyze(root)
    assert result["integrity"]["all_pass"]
    assert len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode()) < 48000
