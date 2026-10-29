"""Reject passive drift, missing held-future comparisons and unmatched updates."""

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v6_analysis_test", HERE / "analysis.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
spec = importlib.util.spec_from_file_location("v4_readonly_fixture_v6", analysis.V4_ROOT / "test_analysis.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
write, lines = fixtures.write, fixtures.lines


def diagnostics(anchors, rows=285, updates=456):
    branches = {name: dict(g4_mse=[1.] * 4) for name in analysis.WORLD_BRANCHES}
    paired = dict(enabled=True, local_adapter_enabled=False, selected_rows=rows,
        issue_time=dict(overall=dict(n=rows, branches=branches)), frozen_parameters_unchanged=True,
        trust_updates=updates, pending_anchor=None,
        next_packet_frozen_anchor=[dict(deadline=deadline, n=5,
            evaluator_fixed_at_anchor=True, scored_before_next_packet_training=True,
            overall=dict(n=5, branches=branches)) for deadline in anchors],
        trust_anchor_steps=[dict(deadline=deadline, steps=[dict(
            identical_training_indices_and_next_actions=True, online={}, frozen={}) for _ in range(8)])
            for deadline in anchors])
    shadow = dict(anchors=[dict(deadline=deadline, steps=8, active_model_batches=8,
        online_exact_active=True, trust_weights_fixed_to_active=True,
        common_state_rows_matched=True, common_rng_streams_matched=True,
        branches={name: dict(actor_parameter_change=.01) for name in analysis.SHADOW_BRANCHES}) for deadline in anchors],
        next_packet=[dict(anchor_deadline=deadline, rows=5, common_frozen_anchor_target=True,
            branches={name: dict(td_error=.1) for name in analysis.SHADOW_BRANCHES}) for deadline in anchors],
        pending_anchor_count=0, target_environment_forks=0)
    return paired, shadow


@pytest.fixture
def campaign(tmp_path):
    p = json.loads((HERE / "protocol.json").read_text())
    root, prior = tmp_path / "runs", tmp_path / "v4"
    for cell in analysis.run_cells(p):
        relative = Path("control") / cell["scenario"] / cell["arm"] / f"seed{cell['seed']}"
        for directory in (root / relative, prior / "runs" / relative):
            fixtures._run(directory, **cell, p=p)
            recorded = analysis.read(directory / "protocol.json")
            recorded["protocol"] = p
            write(directory / "protocol.json", recorded)
            values = [row for row in analysis.jsonl(directory / "metrics.jsonl") if row["time_seconds"] > 0]
            lines(directory / "metrics.jsonl", values)
        directory = root / relative
        summary = analysis.read(directory / "summary.json")
        paired, shadow = diagnostics(p["paired_probe_deadlines"])
        summary["worlds"]["action_model"]["paired_world_diagnostics"] = paired
        summary["shadow_diagnostics"] = shadow
        write(directory / "summary.json", summary)
    return root, p, prior


def test_four_passive_days_preserve_action_model(campaign):
    root, p, prior = campaign
    result = analysis.analyze(root, p, v4_root=prior)
    assert result["status"] == "complete", result["integrity"]["errors"]
    assert len(result["control_cells"]) == len(result["paired_world_diagnostics"]) == len(result["shadow_diagnostics"]) == 4
    assert all(row["exact_cost_and_budget"] and row["same_initializer"] for row in result["passive_reproduction"])
    assert all(row["windows"]["post_training_frozen"]["status"] == "complete" for row in result["control_cells"])


def test_shadow_future_gap_and_passive_cost_drift_block_conclusion(campaign):
    root, p, prior = campaign
    directory = root / "control/stable/action_model/seed6201"
    summary = analysis.read(directory / "summary.json")
    summary["shadow_diagnostics"]["anchors"][0]["online_exact_active"] = False
    summary["worlds"]["action_model"]["paired_world_diagnostics"]["next_packet_frozen_anchor"].pop()
    summary["generalized_passenger_seconds"] += 4
    summary["generalized_seconds_per_departed_passenger"] += .4
    summary["passengers"]["accrued_wait_seconds_total"] += 1
    summary["passengers"]["accrued_in_vehicle_seconds_total"] += 2
    write(directory / "summary.json", summary)
    result = analysis.analyze(root, p, v4_root=prior)
    assert result["status"] == "incomplete"
    checks = {row.get("check") for row in result["integrity"]["errors"]}
    assert {"shadow_online_exact_active", "paired_four_future_packets", "passive_v4_action_model_reproduction"} <= checks


def test_missing_holdout_endpoint_preserves_missing_cost(campaign):
    root, p, prior = campaign
    directory = root / "control/stable/action_model/seed6201"
    values = [row for row in analysis.jsonl(directory / "metrics.jsonl") if row["time_seconds"] != 18300]
    lines(directory / "metrics.jsonl", values)
    result = analysis.analyze(root, p, v4_root=prior)
    assert result["status"] == "incomplete"
    window = result["control_cells"][0]["windows"]["post_training_frozen"]
    assert window["status"] == "missing_endpoint"
    assert window["generalized_passenger_seconds"] is window["seconds_per_full_day_departed_passenger"] is None


def test_native_preflight_requires_live_own_trust_and_unchanged_frozen_parameters(campaign):
    root, p, _ = campaign
    directory = root / "control/local_speed_drop/action_model/seed6201"
    summary = analysis.read(directory / "summary.json")
    summary.update(target_time_seconds=4201, target_online_training_events=50, packets=10,
                   policy_updates=80, full_horizon_completed=False, preflight=True)
    summary["scenario_metadata"]["actual_shock_at_seconds"] = 1350
    summary["feedback_receipts"] = summary["feedback_receipts"][:10]
    paired, summary["shadow_diagnostics"] = diagnostics(p["preflight"]["paired_probe_deadlines"], 50, 80)
    summary["worlds"]["action_model"]["paired_world_diagnostics"] = paired
    write(directory / "summary.json", summary)
    recorded = analysis.read(directory / "protocol.json")
    recorded["protocol"] = dict(p, **p["preflight"])
    write(directory / "protocol.json", recorded)
    lines(directory / "prequential.jsonl", list(analysis.jsonl(directory / "prequential.jsonl"))[:50])
    lines(directory / "td_diagnostics.jsonl", list(analysis.jsonl(directory / "td_diagnostics.jsonl"))[:10])
    gate = analysis.native_preflight_gate(root, p)
    assert gate["all_gates_pass"], gate
    paired.update(trust_updates=0, frozen_parameters_unchanged=False)
    summary["policy_updates"] = 10
    write(directory / "summary.json", summary)
    gate = analysis.native_preflight_gate(root, p)
    assert not gate["all_gates_pass"]
    assert not gate["checks"]["policy_updates"]
    assert not gate["checks"]["paired_own_trust_updates"]
    assert not gate["checks"]["paired_frozen_parameters"]
