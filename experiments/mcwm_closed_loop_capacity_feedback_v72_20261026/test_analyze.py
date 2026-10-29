"""Actual-cost acceptance, absolute attribution and missing-data boundaries."""
import json
import shutil

import pytest

from experiments.mcwm_closed_loop_capacity_feedback_v72_20261026 import analyze as module
from experiments.mcwm_closed_loop_capacity_feedback_v72_20261026.selection import (
    RISK_PREDICATE, summarize_selection)
from experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026 import test_analyze as previous
from experiments.mcwm_closed_loop_executed_cql_v68_20261026.test_analyze import write, jsonl, modify


def populate(tmp_path, *, fresh=(97.,)*4, no_fresh=(99.,)*4):
    previous_root, paths, _ = previous.populate(tmp_path, fresh=(98.,)*4)
    protocol = json.loads((module.HERE/"protocol.json").read_text())
    cached = tmp_path/protocol["previous_reference"]["directory"]
    previous_root.rename(cached)
    root = tmp_path/"v72"
    shutil.copytree(cached, root)
    write(root/"protocol.json", protocol)
    new_paths = []
    for index, path in enumerate(paths):
        path = root/path.relative_to(previous_root)
        row = module.old.read(path/"summary.json")
        arm = path.name
        row["cell"]["recipe"] = protocol["recipe"]
        cost = row["cost"]
        scalar = (fresh if arm == "fresh" else no_fresh)[index//2]
        cost.update(accrued_in_vehicle_seconds=100*scalar-4000.,
            generalized_passenger_seconds=100*scalar, generalized_seconds_per_departed_passenger=scalar)
        for receipt in row["receipts"]:
            ids = receipt["sequence_ids"]
            keys = [[receipt["deadline"]-5+j, identity] for j, identity in enumerate(ids)]
            receipt["max_available_at"] = keys[-1][0]
            receipt["capacity_selection"] = dict(eligible_records=10, eligible_risk_records=1,
                original_latest5_sequence_ids=ids.copy(), original_latest5_risk_records=1,
                selected_risk_records=1, replacement=None,
                candidate_line_counts={"102X": 10}, risk_candidate_line_counts={"102X": 1},
                selected_line_counts={"102X": 5}, risk_selected_line_counts={"102X": 1},
                candidate_available_span_seconds=[receipt["deadline"]-10, receipt["deadline"]-1],
                risk_candidate_available_span_seconds=[receipt["deadline"]-5]*2,
                selected_available_span_seconds=[keys[0][0], keys[-1][0]], selected_keys=keys,
                risk_predicate=RISK_PREDICATE, outcome_fields_used_for_ranking=False)
        row["capacity_selection"] = summarize_selection(row["receipts"])
        write(path/"summary.json", row)
        jsonl(path/"metrics.jsonl", [dict(time_seconds=t,
            accrued_wait_seconds=cost["accrued_wait_seconds"]*f,
            accrued_in_vehicle_seconds=cost["accrued_in_vehicle_seconds"]*f)
            for t, f in ((9000,.3), (12000,.45), (18300,.8), (24000,1.))])
        new_paths.append(path)
    # Legacy parent fixtures have100 departures; preserve their J with this trial's15079 cohort.
    for path in tmp_path.rglob("summary.json"):
        row = module.old.read(path)
        cost = row["cost"]
        factor = 15079/cost["departed_passengers"]
        for field in ("accrued_wait_seconds", "accrued_in_vehicle_seconds", "generalized_passenger_seconds"):
            cost[field] *= factor
        cost["departed_passengers"] = 15079
        write(path, row)
        points = module.old.read_metrics(path.parent, row)
        for point in points:
            for field in ("accrued_wait_seconds", "accrued_in_vehicle_seconds"):
                point[field] *= factor
        jsonl(path.parent/"metrics.jsonl", points)
    return root, new_paths, cached


def test_fixed_cost_gates_allow_changed_new_control_not_v70_exactness(tmp_path):
    root, _, _ = populate(tmp_path, no_fresh=(99.2,)*4)
    result, extra = module.analyze(root)
    assert result["integrity"]["all_pass"]
    assert result["decision"]["scientific_gate_pass"]
    assert result["aggregates"]["no_fresh_shift"]["full_day"]["overall"]["mean_cost_difference"] == pytest.approx(.2)
    assert not result["acceptance_protocol"]["no_fresh_full_trajectory_v70_exactness_required"]
    assert not result["decision"]["advance_to_confirmation"]
    assert not result["decision"]["ordinary_dripping_goal_achieved"]
    assert extra["cells"][0]["services"]["fresh"]["status"] == "service_summary_missing"


def test_degraded_new_baseline_cannot_manufacture_incremental_gain(tmp_path):
    root, _, _ = populate(tmp_path, fresh=(100.,)*4, no_fresh=(110.,)*4)
    result, _ = module.analyze(root)
    assert result["integrity"]["all_pass"]
    assert result["decision"]["matched_cost_gate_pass"]
    assert not result["decision"]["previous_reference_gate_pass"]
    assert not result["decision"]["scientific_gate_pass"]


def test_zero_fresh_or_shock_harm_cannot_pass(tmp_path):
    for index, fresh in enumerate(((99.,)*4, (94.,100.,94.,100.))):
        root, _, _ = populate(tmp_path/str(index), fresh=fresh)
        result, _ = module.analyze(root)
        assert result["integrity"]["all_pass"]
        assert not result["decision"]["matched_cost_gate_pass"]
        assert not result["decision"]["scientific_gate_pass"]


def test_stable_tail_is_per_cell_and_per_baseline_not_average(tmp_path):
    root, paths, _ = populate(tmp_path)
    path = paths[0]
    cost = module.old.read(path/"summary.json")["cost"]
    # Tail becomes19.9: better than matching new control's20.0, worse than old control's19.8.
    fraction = 1.-19.9/97.
    jsonl(path/"metrics.jsonl", [dict(time_seconds=t,
        accrued_wait_seconds=cost["accrued_wait_seconds"]*f,
        accrued_in_vehicle_seconds=cost["accrued_in_vehicle_seconds"]*f)
        for t, f in ((9000,.3), (12000,.45), (18300,fraction), (24000,1.))])
    # Matching new control has a slightly more costly tail, but full-day unchanged.
    control = paths[1]
    c = module.old.read(control/"summary.json")["cost"]
    jsonl(control/"metrics.jsonl", [dict(time_seconds=t,
        accrued_wait_seconds=c["accrued_wait_seconds"]*f,
        accrued_in_vehicle_seconds=c["accrued_in_vehicle_seconds"]*f)
        for t, f in ((9000,.3), (12000,.45), (18300,1.-20./99.), (24000,1.))])
    result, _ = module.analyze(root)
    gate = result["optimization_assessment"]["conditions"]
    assert gate["stable_tail_matched"]
    assert not gate["stable_tail_previous_no_fresh"]
    # The other stable cell's larger benefit must not cancel this cell's harm.
    assert result["aggregates"]["previous_no_fresh"]["post_budget"]["by_scenario"]["stable"]["mean_cost_difference"] < 0
    assert not result["decision"]["scientific_gate_pass"]


def test_exact_frozen_worst_cap_closes_average_winner(tmp_path):
    cap = 4.606804164732239
    root, _, _ = populate(tmp_path, fresh=(99.+cap+.001, 90.,90.,90.))
    result, _ = module.analyze(root)
    conditions = result["optimization_assessment"]["conditions"]
    assert conditions["matched_full_day"] and conditions["previous_no_fresh_full_day"]
    assert result["optimization_assessment"]["worst_full_day_cost_difference_cap"] == cap
    assert not conditions["matched_worst_cell"]
    assert not conditions["previous_no_fresh_worst_cell"]
    assert not result["decision"]["scientific_gate_pass"]


def test_missing_run_and_tail_never_become_zero(tmp_path):
    root, paths, _ = populate(tmp_path)
    missing = paths[0]/"summary.json"
    missing.unlink()
    result, _ = module.analyze(root)
    assert result["status"] == "pending"
    assert result["aggregates"]["matched"]["full_day"]["overall"]["missing_cells"] == 1
    assert not result["decision"]["scientific_gate_pass"]
    write(paths[0]/"failure.json", dict(type="NativeError", consumed_budget=dict(policy_updates=16)))
    result, _ = module.analyze(root)
    assert result["status"] == "integrity_failed"
    assert result["failed_run_work"][0]["consumed_budget"] == dict(policy_updates=16)


def test_missing_tail_endpoint_and_false_selection_audit_block_admission(tmp_path):
    root, paths, _ = populate(tmp_path)
    row = module.old.read(paths[0]/"summary.json")
    points = module.old.read_metrics(paths[0], row)
    jsonl(paths[0]/"metrics.jsonl", [row for row in points if row["time_seconds"] != 18300])
    result, _ = module.analyze(root)
    assert result["status"] == "integrity_failed"
    assert not result["optimization_assessment"]["conditions"]["stable_tail_matched"]
    # A self-reported aggregate cannot replace exact receipt recomputation.
    modify(paths[1]/"summary.json", ("capacity_selection", "selected_records"), 284)
    result, _ = module.analyze(root)
    assert any(name.endswith("selection/reported_aggregate_exact") for name in result["integrity"]["failed_checks"])


def test_final_waiting_membership_is_not_per_person_cost(tmp_path):
    left = dict(final=dict(waiting_inventory_complete=True,
        waiting_passengers=[dict(passenger_id="new", station_id="s", passable_lines=["102X"])], ledger_waiting=1))
    right = dict(final=dict(waiting_inventory_complete=True,
        waiting_passengers=[dict(passenger_id="old", station_id="s", passable_lines=["102X"])], ledger_waiting=1))
    result = module.passenger_comparison(left, right)
    assert result["left_minus_right"] == 0 and result["left_only"] == result["right_only"] == 1
    assert "new" not in json.dumps(result)
    assert module.passenger_comparison(left, None)["membership_difference"] is None


def test_service_missing_observations_and_transfer_metadata_are_explicit(tmp_path):
    point = dict(waiting_unique=2, ledger_waiting=2, waiting_inventory_complete=True,
        waiting_matches_ledger=True, waiting_passengers=[dict(metadata_observed=True,
            station_id="B", transfer_stop="B", completed_legs=1), dict(metadata_observed=False)])
    write(tmp_path/"service_summary.json", dict(final=point, final_waiting_groups=[
        dict(last_compatible_service=dict(departure=None))], service_event_counts=dict(departure=2),
        missing_observations=dict(service_onboard_ids=1)))
    jsonl(tmp_path/"service_trace.jsonl", [dict(record="service", kind="departure", onboard=50,
        capacity=50, eligible_waiting=3, native_stranded_count=3),
        dict(record="service", kind="departure", onboard=None, capacity=50,
            eligible_waiting=2, native_stranded_count=None)])
    result, _ = module.service_view(tmp_path)
    assert result["departures"] == dict(departure_observations=2,
        full_and_queue_departures=1, native_stranded_count_sum=3, departure_inventory_missing=1)
    assert result["final"]["at_recorded_transfer_stop"] == 1
    assert result["final"]["metadata_missing"] == 1
    assert result["final"]["compatible_last_departure_missing"] == 1


def test_three_bounded_outputs_keep_native_size_diagnostics_and_exact_costs(tmp_path):
    root, paths, _ = populate(tmp_path)
    lines=("102S","102X","122S","122X","311S","311X","406S","406X","705S","705X","7S","7X")
    for path in paths:
        row = module.old.read(path/"summary.json")
        group = row["policy_drift"]["groups"][0]
        row["policy_drift"]["groups"] = [dict(group, line=line, window=window,
            count=5 if i == 35 else 8) for i, (line, window) in enumerate(
                (line, window) for line in lines for window in ("early","late_training","pre_early"))]
        for i, receipt in enumerate(row["receipts"]):
            receipt["capacity_selection"].update(eligible_records=96, eligible_risk_records=12,
                candidate_line_counts={line: 8 for line in lines}, risk_candidate_line_counts={line: 1 for line in lines},
                selected_line_counts={lines[i%len(lines)]: 5}, risk_selected_line_counts={lines[i%len(lines)]: 1})
        row["capacity_selection"] = summarize_selection(row["receipts"])
        write(path/"summary.json", row)
    result, _ = module.analyze(root)
    receipt = module.export(root, root/"results/development.json")
    assert set(receipt["files"]) == {"development.json", "policy_drift.json", "recency.json"}
    assert all(size <= 48000 for size in receipt["files"].values())
    assert module.old.read(root/"results/development.json")["aggregates"] == result["aggregates"]
    assert len(module.old.read(root/"results/policy_drift.json")["native_evidence"][0]["policy_drift"]["fresh"]["rows"]) == 36
    assert len(module.old.read(root/"results/recency.json")["cells"][0]["capacity_coverage"]["fresh"]["rows"]) == 36
