"""Reject passive drift, incomplete shadow matching and missing held-future costs."""

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v5_analysis_test", HERE / "analysis.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
fixture_spec = importlib.util.spec_from_file_location("v4_readonly_fixture_v5", analysis.V4_ROOT / "test_analysis.py")
fixtures = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixtures)
write, lines = fixtures.write, fixtures.lines


def diagnostics(anchors):
    context = dict(branches=["base", "saved", "current"],
        historical=[dict(deadline=deadline, n=20, overall=dict(base=dict(loss=1.), saved=dict(loss=.9), current=dict(loss=.8)),
                         identifiability=dict(rank=2)) for deadline in anchors],
        next_packet_frozen_anchor=[dict(deadline=deadline, n=5, current_context_fixed_at_anchor=True,
            evaluator_fixed_at_anchor=True, scored_before_next_packet_training=True) for deadline in anchors],
        pending_anchor_deadline=None)
    shadow = dict(anchors=[dict(deadline=deadline, steps=8, active_model_batches=8,
        local_exact_active=True, trust_weights_fixed_to_active=True,
        branches={name: dict(actor_parameter_change=.01) for name in analysis.BRANCHES}) for deadline in anchors],
        next_packet=[dict(anchor_deadline=deadline, rows=5, common_frozen_anchor_target=True,
            branches={name: dict(td_error=.1) for name in analysis.BRANCHES}) for deadline in anchors], pending_anchor_count=0)
    return context, shadow


@pytest.fixture
def campaign(tmp_path):
    p = json.loads((HERE / "protocol.json").read_text())
    root, prior = tmp_path / "runs", tmp_path / "v4"
    for cell in p["run_cells"]:
        relative = Path("control") / cell["scenario"] / cell["arm"] / f"seed{cell['seed']}"
        for directory in (root / relative, prior / "runs" / relative):
            fixtures._run(directory, **cell, p=p)
            recorded = analysis.read(directory / "protocol.json")
            recorded["protocol"] = p
            write(directory / "protocol.json", recorded)
            # The production trace starts at300. No time0 record is required
            # for either adaptation or the strict post-update interval.
            values = [row for row in analysis.jsonl(directory / "metrics.jsonl") if row["time_seconds"] > 0]
            lines(directory / "metrics.jsonl", values)
        directory = root / relative
        summary = analysis.read(directory / "summary.json")
        context, shadow = diagnostics(p["context_probe_deadlines"])
        summary["worlds"]["action_local"]["context_diagnostics"] = context
        summary["shadow_diagnostics"] = shadow
        write(directory / "summary.json", summary)
    return root, p, prior


def test_four_passive_days_need_no_time_zero_endpoint(campaign):
    root, p, prior = campaign
    result = analysis.analyze(root, p, v4_root=prior)
    assert result["status"] == "complete", result["integrity"]["errors"]
    assert len(result["control_cells"]) == len(result["context_diagnostics"]) == len(result["shadow_diagnostics"]) == 4
    assert all(row["exact_cost_and_budget"] and row["same_initializer"] for row in result["passive_reproduction"])
    assert all(row["windows"]["post_training_frozen"]["status"] == "complete" for row in result["control_cells"])


def test_shadow_mismatch_future_gap_and_passive_cost_drift_are_explicit(campaign):
    root, p, prior = campaign
    directory = root / "control/stable/action_local/seed6201"
    summary = analysis.read(directory / "summary.json")
    summary["shadow_diagnostics"]["anchors"][0]["local_exact_active"] = False
    summary["shadow_diagnostics"]["next_packet"].pop()
    summary["generalized_passenger_seconds"] += 4
    summary["generalized_seconds_per_departed_passenger"] += .4
    summary["passengers"]["accrued_wait_seconds_total"] += 1
    summary["passengers"]["accrued_in_vehicle_seconds_total"] += 2
    write(directory / "summary.json", summary)
    result = analysis.analyze(root, p, v4_root=prior)
    assert result["status"] == "incomplete"
    checks = {row.get("check") for row in result["integrity"]["errors"]}
    assert {"shadow_local_exact_active", "shadow_four_future_packets", "passive_v4_reproduction"} <= checks


def test_missing_holdout_endpoint_preserves_missing_value(campaign):
    root, p, prior = campaign
    directory = root / "control/stable/action_local/seed6201"
    values = [row for row in analysis.jsonl(directory / "metrics.jsonl") if row["time_seconds"] != 18300]
    lines(directory / "metrics.jsonl", values)
    result = analysis.analyze(root, p, v4_root=prior)
    assert result["status"] == "incomplete"
    window = result["control_cells"][0]["windows"]["post_training_frozen"]
    assert window["status"] == "missing_endpoint"
    assert window["generalized_passenger_seconds"] is window["seconds_per_full_day_departed_passenger"] is None


def test_native_preflight_requires_eighty_live_updates(campaign):
    root, p, prior = campaign
    directory = root / "control/local_speed_drop/action_local/seed6201"
    summary = analysis.read(directory / "summary.json")
    summary.update(target_time_seconds=4201, target_online_training_events=50, packets=10,
                   policy_updates=80, full_horizon_completed=False, preflight=True)
    summary["scenario_metadata"]["actual_shock_at_seconds"] = 1350
    summary["feedback_receipts"] = summary["feedback_receipts"][:10]
    world = summary["worlds"]["action_local"]
    world["local_adapter"]["fit_labels"] = world["head_diagnostics"]["overall"]["n"] = 50
    world["context_diagnostics"], summary["shadow_diagnostics"] = diagnostics(p["preflight"]["context_probe_deadlines"])
    write(directory / "summary.json", summary)
    recorded = analysis.read(directory / "protocol.json")
    recorded["protocol"] = dict(p, **p["preflight"])
    write(directory / "protocol.json", recorded)
    lines(directory / "prequential.jsonl", list(analysis.jsonl(directory / "prequential.jsonl"))[:50])
    lines(directory / "td_diagnostics.jsonl", list(analysis.jsonl(directory / "td_diagnostics.jsonl"))[:10])
    gate = analysis.native_preflight_gate(root, p)
    assert gate["all_gates_pass"], gate
    summary["policy_updates"] = 10
    write(directory / "summary.json", summary)
    gate = analysis.native_preflight_gate(root, p)
    assert not gate["all_gates_pass"] and not gate["checks"]["policy_updates"]
    summary.update(policy_updates=80, full_horizon_completed=True)
    write(directory / "summary.json", summary)
    gate = analysis.native_preflight_gate(root, p)
    assert not gate["all_gates_pass"] and not gate["checks"]["full_horizon_completed"]


def test_frozen_preflight_rejects_parameter_drift_and_missing_trust(campaign):
    root, p, _ = campaign
    cell = dict(seed=6201, scenario="local_speed_drop", arm="frozen_world")
    source = root / "control/local_speed_drop/action_local/seed6201"
    directory = root / "control/local_speed_drop/frozen_world/seed6201"
    summary = analysis.read(source / "summary.json")
    summary.update(arm="frozen_world", target_time_seconds=4201, target_online_training_events=50,
        packets=10, policy_updates=80, preflight=True, full_horizon_completed=False,
        physical_learning_enters_policy=False)
    summary["scenario_metadata"]["actual_shock_at_seconds"] = 1350
    summary["feedback_receipts"] = summary["feedback_receipts"][:10]
    frozen = dict(enabled=True, initial_fit_updates=2,
        models={name: dict(changed_tensor_count=0, max_absolute_change=0.)
                for name in ("source", "direct", "residual")}, online_neural_gradient_updates=0,
        skipped_refit_packets=10, family_selection_adapts=True, trust_update_calls=80,
        local_readout_enabled=False, nominal_refit_rng_draws_preserved=True)
    summary["worlds"] = dict(frozen_world=dict(frozen_parameters=frozen,
        gradient_updates=0, target_refits=0, prequential_rows=50))
    write(directory / "summary.json", summary)
    recorded = analysis.read(source / "protocol.json")
    recorded["protocol"] = dict(p, **p["preflight"])
    write(directory / "protocol.json", recorded)
    lines(directory / "prequential.jsonl", list(analysis.jsonl(source / "prequential.jsonl"))[:50])
    lines(directory / "td_diagnostics.jsonl", list(analysis.jsonl(source / "td_diagnostics.jsonl"))[:10])
    gate = analysis.native_frozen_preflight_gate(root, p)
    assert gate["all_gates_pass"], gate
    frozen["models"]["direct"]["changed_tensor_count"] = 1
    frozen["trust_update_calls"] = 0
    write(directory / "summary.json", summary)
    gate = analysis.native_frozen_preflight_gate(root, p)
    assert not gate["all_gates_pass"]
    assert not gate["checks"]["frozen_neural_parameters"]
    assert not gate["checks"]["frozen_trust_learning_active"]
