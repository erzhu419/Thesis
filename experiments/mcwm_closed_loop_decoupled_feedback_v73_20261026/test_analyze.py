"""Frozen actual-cost gates, baseline regression and explicit information budget."""
import json
import shutil

import pytest

from experiments.mcwm_closed_loop_decoupled_feedback_v73_20261026 import analyze as module
from experiments.mcwm_closed_loop_decoupled_feedback_v73_20261026.selection import summarize_selection
from experiments.mcwm_closed_loop_capacity_feedback_v72_20261026 import test_analyze as previous
from experiments.mcwm_closed_loop_executed_cql_v68_20261026.test_analyze import write, jsonl, modify


def populate(tmp_path, *, fresh=(97.,)*4):
    capacity_root, _, cached = previous.populate(tmp_path)
    protocol = json.loads((module.HERE/"protocol.json").read_text())
    capacity_root.rename(tmp_path/protocol["capacity_reference"]["directory"])
    root = tmp_path/"v73"
    shutil.copytree(cached, root)
    write(root/"protocol.json", protocol)
    updates = [dict(packet=packet, step_in_packet=step, processed_at=1200.+300*(packet-1),
        critic_loss=.123456789, td_diagnostic=dict(dt_seconds=dict(mean=13.333333333)),
        cql=dict(regularizer_wall_seconds=.0012345)) for packet in range(1, 58) for step in (1, 8)]
    for source in (root, cached):
        jsonl(module.run_dir(source, 6201, "stable", "no_fresh")/"updates.jsonl", updates)
    paths = []
    for index, (seed, scene) in enumerate((seed, scene) for seed in (6201, 6203)
            for scene in ("stable", "local_speed_drop")):
        for arm in ("fresh", "no_fresh"):
            path = module.run_dir(root, seed, scene, arm)
            if arm == "no_fresh" and (seed, scene) != (6201, "stable"):
                shutil.rmtree(path)
                continue
            row = module.old.read(path/"summary.json")
            row["cell"]["recipe"] = protocol["recipe"]
            for number, receipt in enumerate(row["receipts"]):
                cov = receipt["sequence_ids"].copy()
                original_max = receipt["max_available_at"]
                keys = [[receipt["deadline"]-5+j, identity] for j, identity in enumerate(cov)]
                outcomes, replacement = cov.copy(), None
                # Every third packet adds one extra selected record; never changes the state path.
                if number % 3 == 0:
                    added = 1000000+receipt["deadline"]
                    outcomes[0] = added
                    keys[0] = [receipt["deadline"]-8, added]
                    replacement = dict(removed_sequence=cov[0], added_sequence=added,
                        removed_available_at=receipt["deadline"]-5, added_available_at=receipt["deadline"]-8,
                        added_risk=True, removed_risk=False)
                receipt.update(sequence_ids=outcomes, outcome_sequence_ids=outcomes.copy(),
                    covariate_sequence_ids=cov, covariate_records=5, outcome_records=5,
                    covariate_max_available_at=original_max, max_available_at=keys[-1][0],
                    overlap_records=4 if replacement else 5, union_records=6 if replacement else 5)
                receipt["capacity_selection"] = dict(eligible_records=10, eligible_risk_records=1,
                    original_latest5_sequence_ids=cov, original_latest5_risk_records=0 if replacement else 1,
                    selected_risk_records=1, replacement=replacement,
                    candidate_line_counts={"102X": 10}, risk_candidate_line_counts={"102X": 1},
                    selected_line_counts={"102X": 5}, risk_selected_line_counts={"102X": 1},
                    candidate_available_span_seconds=[receipt["deadline"]-10, receipt["deadline"]-1],
                    risk_candidate_available_span_seconds=[keys[0][0]]*2,
                    selected_available_span_seconds=[keys[0][0], keys[-1][0]], selected_keys=keys,
                    risk_predicate="issue_s:e[7]>0_and_e[8]>=1", outcome_fields_used_for_ranking=False)
            row["capacity_selection"] = summarize_selection(row["receipts"])
            row["replay"].update(covariate_records_received=285,
                outcome_labels_received=285 if arm == "fresh" else 0,
                covariate_unique_records=285, outcome_unique_records=285 if arm == "fresh" else 0,
                distinct_target_records_used=304 if arm == "fresh" else 285,
                selected_target_record_union=304, outcome_state_record_overlap=266 if arm == "fresh" else 0,
                fresh_current_s_a_use=["IQL value", "SAC actor", "temperature", "executed CQL"])
            if arm == "fresh":
                cost = row["cost"]
                cost.update(accrued_in_vehicle_seconds=15079*fresh[index]-2*cost["accrued_wait_seconds"],
                    generalized_passenger_seconds=15079*fresh[index],
                    generalized_seconds_per_departed_passenger=fresh[index])
                jsonl(path/"metrics.jsonl", [dict(time_seconds=t,
                    accrued_wait_seconds=cost["accrued_wait_seconds"]*f,
                    accrued_in_vehicle_seconds=cost["accrued_in_vehicle_seconds"]*f)
                    for t, f in ((9000,.3), (12000,.45), (18300,.8), (24000,1.))])
                paths.append(path)
            write(path/"summary.json", row)
    return root, paths, cached


def test_cached_control_and_five_day_information_contract(tmp_path):
    root, _, _ = populate(tmp_path)
    result, refs, policy, pools = module.analyze(root)
    assert result["integrity"]["all_pass"], result["integrity"]["failed_checks"]
    assert result["decision"]["scientific_gate_pass"]
    assert result["available_cells"] == 4
    assert result["budget"]["planned_new_native_days"] == 5
    assert result["budget"]["distinct_target_records_used"] == 4*304+285
    assert result["aggregates"]["matched"]["full_day"]["overall"]["mean_cost_difference"] == -2.
    assert refs["aggregates"]["previous_fresh"]["full_day"]["overall"]["mean_cost_difference"] == -1.
    assert pools["no_fresh_regression"]["all_pass"]
    assert pools["first_pair_actual_parallel"]["positive_overlap"]
    assert len(policy["cells"]) == 4
    assert not result["decision"]["advance_to_confirmation"]


@pytest.mark.parametrize("failure", ("late_trace", "ledger", "receipt", "CQL", "metrics"))
def test_exact_no_fresh_regression_blocks_expansion(tmp_path, failure):
    root, _, _ = populate(tmp_path)
    path = module.run_dir(root, 6201, "stable", "no_fresh")
    if failure == "late_trace":
        jsonl(path/"prefix_trace.jsonl", [dict(started_at=1000,a=0), dict(started_at=23000,a=1.875)])
    elif failure == "ledger":
        modify(path/"summary.json", ("cost", "accrued_wait_seconds"), 2001.)
    elif failure == "receipt":
        row = module.old.read(path/"summary.json")
        row["receipts"][30]["covariate_sequence_ids"][0] += 10000
        write(path/"summary.json", row)
    elif failure == "CQL":
        modify(path/"summary.json", ("cql_audit", "main_rng_preserved"), False)
    else:
        points = module.old.read_metrics(path, module.old.read(path/"summary.json"))
        points[0]["accrued_wait_seconds"] += 1
        jsonl(path/"metrics.jsonl", points)
    gate = module.no_fresh_regression(root)
    assert not gate["all_pass"]
    result, *_ = module.analyze(root)
    assert not result["decision"]["scientific_gate_pass"]


def test_stable_tail_is_individual_and_v72_is_only_descriptive(tmp_path):
    root, paths, _ = populate(tmp_path)
    cost = module.old.read(paths[0]/"summary.json")["cost"]
    fraction = 1.-20.1/97.
    jsonl(paths[0]/"metrics.jsonl", [dict(time_seconds=t,
        accrued_wait_seconds=cost["accrued_wait_seconds"]*f,
        accrued_in_vehicle_seconds=cost["accrued_in_vehicle_seconds"]*f)
        for t, f in ((9000,.3), (12000,.45), (18300,fraction), (24000,1.))])
    result, refs, *_ = module.analyze(root)
    assert result["integrity"]["all_pass"]
    assert result["aggregates"]["matched"]["post_budget"]["by_scenario"]["stable"]["mean_cost_difference"] < 0
    assert not result["optimization_assessment"]["conditions"]["both_stable_tails"]
    assert not result["decision"]["scientific_gate_pass"]
    assert refs["aggregates"]["capacity_fresh"]["full_day"]["overall"]["mean_cost_difference"] == 0.


@pytest.mark.parametrize("fresh", ((99.,)*4, (94.,100.,94.,100.), (103.607,90.,90.,90.)))
def test_zeroed_fresh_shock_harm_and_worst_cell_cannot_pass(tmp_path, fresh):
    root, _, _ = populate(tmp_path, fresh=fresh)
    result, *_ = module.analyze(root)
    assert result["integrity"]["all_pass"]
    assert not result["decision"]["scientific_gate_pass"]


def test_pool_accounting_cannot_report_285_total_records(tmp_path):
    root, paths, _ = populate(tmp_path)
    modify(paths[0]/"summary.json", ("replay", "distinct_target_records_used"), 285)
    result, *_ = module.analyze(root)
    assert not result["integrity"]["all_pass"]
    assert any(name.endswith("pool_actual_union_exact") for name in result["integrity"]["failed_checks"])


def test_update_log_allows_profiler_time_but_keeps_physical_dt(tmp_path):
    root, _, _ = populate(tmp_path)
    path = module.run_dir(root, 6201, "stable", "no_fresh")/"updates.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["cql"]["regularizer_wall_seconds"] = 6.987654321
    jsonl(path, rows)
    assert module.no_fresh_regression(root)["all_pass"]
    rows[-1]["td_diagnostic"]["dt_seconds"]["mean"] += .1
    jsonl(path, rows)
    gate = module.no_fresh_regression(root)
    assert not gate["all_pass"]
    assert gate["update_log"]["compared_rows"] == 114
    assert gate["update_log"]["first_differences"] == [dict(row=113,
        fields=["td_diagnostic.dt_seconds.mean"])]


def test_missing_native_day_never_becomes_zero_cost(tmp_path):
    root, paths, _ = populate(tmp_path)
    (paths[-1]/"summary.json").unlink()
    result, *_ = module.analyze(root)
    assert result["status"] == "pending"
    assert result["aggregates"]["matched"]["full_day"]["overall"]["missing_cells"] == 1
    assert not result["decision"]["scientific_gate_pass"]


def test_high_precision_costs_and_full_line_diagnostics_fit_four_pulls(tmp_path):
    root, paths, _ = populate(tmp_path)
    lines = ("102S","102X","122S","122X","311S","311X","406S","406X","705S","705X","7S","7X")
    for path in paths:
        row = module.old.read(path/"summary.json")
        group = row["policy_drift"]["groups"][0]
        row["policy_drift"]["groups"] = [dict(group, line=line, window=window,
            count=5 if i == 35 else 8) for i, (line, window) in enumerate(
                (line, window) for line in lines for window in ("early","late_training","pre_early"))]
        for i, receipt in enumerate(row["receipts"]):
            receipt["capacity_selection"].update(candidate_line_counts={line: 8 for line in lines},
                risk_candidate_line_counts={line: 1 for line in lines},
                selected_line_counts={lines[i%12]: 5}, risk_selected_line_counts={lines[i%12]: 1})
        row["capacity_selection"] = summarize_selection(row["receipts"])
        write(path/"summary.json", row)
    for path in tmp_path.rglob("summary.json"):
        row = module.old.read(path)
        scalar = row["cost"]["generalized_seconds_per_departed_passenger"]
        wait = 8438598.256789+23.5136789123*scalar
        total = 15079*(2209.123456789+scalar-100.)
        row["cost"] = dict(departed_passengers=15079, accrued_wait_seconds=wait,
            accrued_in_vehicle_seconds=total-2*wait, generalized_passenger_seconds=total,
            generalized_seconds_per_departed_passenger=total/15079)
        write(path, row)
        jsonl(path.parent/"metrics.jsonl", [dict(time_seconds=t,
            accrued_wait_seconds=wait*f, accrued_in_vehicle_seconds=(total-2*wait)*f)
            for t, f in ((9000,.300123456), (12000,.450234567), (18300,.800345678), (24000,1.))])
    main, *_ = module.analyze(root)
    receipt = module.export(root, root/"results/development.json")
    assert len(receipt["files"]) == 4 and all(size <= 48000 for size in receipt["files"].values())
    assert module.old.read(root/"results/development.json")["aggregates"] == main["aggregates"]
    policy = module.old.read(root/"results/policy_drift.json")
    assert len(policy["cells"][0]["policy_drift"]["rows"]) == 36
    regression = module.export(root, root/"regression_gate.json", regression_gate=True)
    assert regression["all_pass"]
