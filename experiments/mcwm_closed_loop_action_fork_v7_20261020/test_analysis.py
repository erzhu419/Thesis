"""Block mismatched prefixes, training on branch truth, and missing-state claims."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v7_analysis_test", HERE / "analysis.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def protocol():
    return dict(name="v7-test", preflight_directory="preflight", policy_updates_per_packet=8,
                initializer_updates=50000, history_rows=32768, source_rows=4096,
                model_init_updates=400, model_refit_updates=10, feedback_records_per_packet=5,
                target_seed_offset=77000,
                selection_window_seconds=[10200, 13800], preflight=dict(seed=6201,
                scenario="local_speed_drop", target_line="7X", selection_window_seconds=[1500, 3900],
                model_init_updates=2, model_refit_updates=1))


def branch(cell, *, preflight=False):
    time = 1600 if preflight else 10300
    reward = -2. if cell["branch"] == "low" else -1.
    factual = dict(candidate=cell["branch"], global_decoded=[1., -reward, 1., 1.],
        bellman=dict(scaled_reward=reward, bootstrap=3., target=reward+3.),
        started_at=time, completed_at=time+1, zero_duration=False, terminal=False,
        next_vehicle_id="next", branches={})
    factual.update(anchor_update=8, world_version=3, common_family="direct", target_seed=42)
    prediction = dict(valid_pair=True, anchor_update=8, world_version=3, common_family="direct", target_seed=42, branches={key: dict(candidates={role: dict(
        global_raw=[1., 2. if role == "low" else 1., 1., 1.],
        global_decoded=[1., 2. if role == "low" else 1., 1., 1.],
        standardized_global=[1., 2. if role == "low" else 1., 1., 1.],
        bellman=dict(scaled_reward=-2. if role == "low" else -1., bootstrap=3., target=1. if role == "low" else 2.),
        trust=dict(weight=.2, log_q=-1., tau=1.)) for role in a.BRANCHES})
        for key in ("online", "frozen_common")})
    for key in prediction["branches"]:
        factual["branches"][key] = dict(global_signed_error=[0.]*4, standardized_global_signed_error=[0.]*4,
            bellman_signed_error=dict(scaled_reward=0., bootstrap=0., target=0.),
            six_losses=[0.]*6, trust_outcome=None)
    return dict(status="complete", cell=cell, preflight=preflight,
        selected=dict(time=time, sequence=123, line=cell["target_line"], vehicle_id="bus", trip_id="trip",
            station=5, state=[1., 2.], raw_state=[1., 2.], low_action=-1., high_action=1.,
            remaining_actions=[], prefix_cost=100., prefix_departed=10),
        prefix=dict(initializer_receipt=dict(path="resident.pt", initial_updates=50000, history_rows=32768, source_rows=4096,
                model_init_updates=2 if preflight else 400, model_refit_updates=1 if preflight else 10),
            feedback_receipts=[dict(sequence=1, records=5, deadline=1200, processed_at=1201)], policy_updates=8, world_updates=2 if preflight else 20,
            world_refits=1, trust_updates=8, physical_labels=3, external_seed=cell["seed"]+77000,
            target_labels=5, reference_feedback_prefix_exact=True),
        prediction=prediction, next_event=factual,
        suffix=dict(start_time=time, end_time=time+300, seconds=300, cost=100. if cell["branch"] == "low" else 80.,
            wait=30., ride=70., departed=10, ledger_balance={}, training_updates=0, world_updates=0, trust_updates=0, physical_updates=0, native_steps=300),
        native=dict(sumo_version="1.25", passenger_ledger_balanced=True,
            first_action_applied=-1. if cell["branch"] == "low" else 1., other_batch_actions_unchanged=True, cutoff_not_terminal=True),
        budget=dict(prefix_sim_seconds=time, suffix_sim_seconds=300, truth_next_global_records=1,
            truth_labels_enter_training=0, prefix_target_labels=5, prefix_policy_updates=8))


def write(root, cell, value):
    path = root / a.cell_path(cell) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_fixed_12_states_and_24_branches():
    p = protocol()
    assert len(a.state_cells(p)) == 12
    assert len(a.branch_cells(p)) == 24
    assert len(a.branch_cells(p, preflight=True)) == 2
    assert len({(row["seed"], row["scenario"], row["target_line"], row["branch"])
                for row in a.branch_cells(p)}) == 24


def test_native_gate_rejects_prefix_mismatch_and_truth_training(tmp_path):
    p = protocol()
    cells = a.branch_cells(p, preflight=True)
    for cell in cells:
        write(tmp_path, cell, branch(cell, preflight=True))
    assert a.native_preflight_gate(tmp_path, p)["all_gates_pass"]
    high = branch(cells[1], preflight=True)
    high["prefix"]["feedback_receipts"][0]["sequence"] = 2
    high["suffix"]["training_updates"] = 1
    high["budget"]["truth_labels_enter_training"] = 1
    write(tmp_path, cells[1], high)
    gate = a.native_preflight_gate(tmp_path, p)
    assert not gate["all_gates_pass"]
    assert not gate["checks"]["initializer_training_and_exogenous_seed_identical"]
    assert not gate["checks"]["high/frozen_suffix_policy"]
    assert not gate["checks"]["high/truth_never_used_for_training"]


def test_direction_tie_and_missing_are_not_wins():
    assert a.direction_agreement(1., 2.) == "agree"
    assert a.direction_agreement(1., -2.) == "disagree"
    assert a.direction_agreement(0., 0.) == "tie"
    assert a.direction_agreement(None, 1.) == "missing"


def test_complete_analysis_uses_cost_gain_sign_and_separates_budget(tmp_path):
    p = protocol()
    for cell in a.branch_cells(p):
        write(tmp_path, cell, branch(cell))
    result = a.analyze(tmp_path, p)
    assert result["status"] == "complete", result["integrity"]["errors"]
    assert result["integrity"]["complete_pairs"] == 12
    assert result["budget"]["suffix_sim_seconds"] == 7200
    assert result["budget"]["prefix_sim_seconds_replayed"] == 24*10300
    assert result["budget"]["truth_labels_enter_training"] == 0
    pair = result["pairs"][0]
    assert pair["evidence"]["factual_effect"]["gains"]["cost_300_seconds"] == 20.
    assert pair["evidence"]["worlds"]["online"]["scores"]["bellman_vs_cost_300_direction"] == "agree"
    assert result["aggregates"]["stable"]["online"]["n_pairs"] == 6


def test_missing_selection_preserves_missing_without_zero_cost(tmp_path):
    p = protocol()
    cells = a.branch_cells(p)
    for cell in cells:
        write(tmp_path, cell, branch(cell))
    missing = dict(status="selection_missing", cell=cells[0], preflight=False,
                   budget=dict(prefix_sim_seconds=13800, suffix_sim_seconds=0,
                               truth_next_global_records=0, truth_labels_enter_training=0, prefix_target_labels=5, prefix_policy_updates=8))
    write(tmp_path, cells[0], missing)
    result = a.analyze(tmp_path, p)
    assert result["status"] == "incomplete"
    assert result["integrity"]["complete_pairs"] == 11
    assert len(result["pairs"]) == 11
    assert result["branch_statuses"][0]["status"] == "selection_missing"
    assert result["budget"]["suffix_sim_seconds"] == 23*300
    assert result["aggregates"]["stable"]["online"]["n_pairs"] == 5


def test_native_gate_rejects_changed_model_budget_and_nonreference_prefix(tmp_path):
    p = protocol()
    for cell in a.branch_cells(p, preflight=True):
        row = branch(cell, preflight=True)
        row["prefix"]["initializer_receipt"]["model_init_updates"] = 400
        row["prefix"]["initializer_receipt"]["model_refit_updates"] = 10
        row["prefix"]["reference_feedback_prefix_exact"] = False
        write(tmp_path, cell, row)
    gate = a.native_preflight_gate(tmp_path, p)
    assert not gate["all_gates_pass"]
    assert not gate["checks"]["low/same_model_init_updates"]
    assert not gate["checks"]["high/same_model_refit_updates"]
    assert not gate["checks"]["low/same_v6_reference_feedback_prefix"]
