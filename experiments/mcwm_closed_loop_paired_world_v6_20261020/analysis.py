"""Resident same-row world/trust scores and passive optimizer contrasts."""

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V5_ROOT = HERE.parent / "mcwm_closed_loop_context_shadow_v5_20261020"
V4_ROOT = HERE.parent / "mcwm_closed_loop_head_diagnostic_v4_20261020"
spec = importlib.util.spec_from_file_location("mcwm_v5_readonly_analysis_v6", V5_ROOT / "analysis.py")
v5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v5)
read, jsonl, at, difference = v5.read, v5.jsonl, v5.at, v5.difference
WORLD_BRANCHES = {"online", "frozen_own", "frozen_common"}
SHADOW_BRANCHES = {"Online", "Frozen"}


def run_cells(p):
    cells = p["run_cells"]
    expected = {(seed, scenario, "action_model") for seed in (6201, 6202)
                for scenario in ("stable", "local_speed_drop")}
    if len(cells) != 4 or {(row["seed"], row["scenario"], row["arm"]) for row in cells} != expected:
        raise ValueError("Retain the four fixed action_model development days")
    return cells


def result_name():
    return "paired_world_summary.json"


def diagnostic_checks(world, shadow, anchors, records, updates):
    """Reject missing causal pairing or optimizer mismatch before interpreting it."""
    future = world.get("next_packet_frozen_anchor", [])
    trust_anchors = world.get("trust_anchor_steps", [])
    forks, next_shadow = shadow.get("anchors", []), shadow.get("next_packet", [])
    return dict(
        paired_diagnosis_enabled=world.get("enabled") is True,
        local_readout_disabled=world.get("local_adapter_enabled") is False,
        paired_selected_rows=world.get("selected_rows") == records,
        paired_issue_rows=at(world, "issue_time", "overall", "n") == records,
        paired_issue_branches=set(at(world, "issue_time", "overall", "branches") or {}) == WORLD_BRANCHES,
        paired_frozen_parameters=world.get("frozen_parameters_unchanged") is True,
        paired_own_trust_updates=world.get("trust_updates") == updates,
        paired_four_future_packets=[row.get("deadline") for row in future] == anchors,
        paired_future_common_evaluation=bool(future) and all(row.get("n") == 5
            and row.get("evaluator_fixed_at_anchor") is True
            and row.get("scored_before_next_packet_training") is True
            and set(at(row, "overall", "branches") or {}) == WORLD_BRANCHES for row in future),
        paired_no_pending_packet=world.get("pending_anchor") is None,
        paired_eight_trust_steps=[row.get("deadline") for row in trust_anchors] == anchors
            and all(len(row.get("steps", [])) == 8 for row in trust_anchors),
        paired_trust_common_training_rows=bool(trust_anchors) and all(
            step.get("identical_training_indices_and_next_actions") is True
            and isinstance(step.get("online"), dict) and isinstance(step.get("frozen"), dict)
            for row in trust_anchors for step in row.get("steps", [])),
        shadow_four_anchors=[row.get("deadline") for row in forks] == anchors,
        shadow_eight_updates=bool(forks) and all(row.get("steps") == 8
            and row.get("active_model_batches") == 8 for row in forks),
        shadow_online_exact_active=bool(forks) and all(row.get("online_exact_active") is True for row in forks),
        shadow_shared_actual_trust=bool(forks) and all(row.get("trust_weights_fixed_to_active") is True for row in forks),
        shadow_common_state_and_rng=bool(forks) and all(row.get("common_state_rows_matched") is True
            and row.get("common_rng_streams_matched") is True for row in forks),
        shadow_two_branches=bool(forks) and all(set(row.get("branches", {})) == SHADOW_BRANCHES for row in forks),
        shadow_four_future_packets=[row.get("anchor_deadline") for row in next_shadow] == anchors,
        shadow_common_future_targets=bool(next_shadow) and all(row.get("rows") == 5
            and row.get("common_frozen_anchor_target") is True
            and set(row.get("branches", {})) == SHADOW_BRANCHES for row in next_shadow),
        shadow_no_pending_packets=shadow.get("pending_anchor_count") == 0,
        shadow_no_target_rollout=shadow.get("target_environment_forks") == 0)


def _run_checks(directory, summary, recorded, p, cell, *, preflight=False):
    checks = v5._run_checks(directory, summary, recorded, p, cell, preflight=preflight)
    config = dict(p, **p["preflight"]) if preflight else p
    world = at(summary, "worlds", "action_model", "paired_world_diagnostics") or {}
    checks["physical_readout_not_deployed"] = summary.get("physical_learning_enters_policy") is False
    checks.update(diagnostic_checks(world, summary.get("shadow_diagnostics") or {},
        config["paired_probe_deadlines"], config["target_record_budget"], config["policy_update_budget"]))
    return checks


def native_preflight_gate(root=None, protocol=None):
    p = protocol or read(HERE / "protocol.json")
    root = Path(root) if root is not None else HERE / p["preflight_directory"]
    cell = dict(seed=p["preflight"]["seed"], scenario=p["preflight"]["scenario"], arm="action_model")
    directory = root / "control" / cell["scenario"] / cell["arm"] / f"seed{cell['seed']}"
    summary, recorded, failure = [read(directory / name) for name in ("summary.json", "protocol.json", "failure.json")]
    checks = _run_checks(directory, summary, recorded, p, cell, preflight=True) if summary and recorded else {}
    checks.update(summary_exists=summary is not None, protocol_exists=recorded is not None, no_failure=failure is None)
    return dict(cell=cell, checks=checks, all_gates_pass=all(checks.values()))


def analyze(root=HERE / "runs", protocol=None, *, v4_root=V4_ROOT):
    root, v4_root = Path(root), Path(v4_root)
    p = protocol or read(HERE / "protocol.json")
    errors, missing, failures, cells, reproduction, worlds, shadows, td = [], [], [], [], [], [], [], []
    initializers = {}
    for cell in run_cells(p):
        label = f"control/{cell['scenario']}/{cell['arm']}/seed{cell['seed']}"
        directory = root / label
        summary, recorded, failure = [read(directory / name) for name in ("summary.json", "protocol.json", "failure.json")]
        if failure:
            failures.append(dict(cell=label, **failure))
        if summary is None or recorded is None:
            missing.append(label)
            continue
        checks = _run_checks(directory, summary, recorded, p, cell)
        errors.extend(dict(cell=label, check=key) for key, valid in checks.items() if not valid)
        initializers.setdefault(str(cell["seed"]), set()).add(recorded.get("initializer"))
        cost = v5.cost_cell(directory, summary)
        cells.append(dict(**cell, **cost))
        errors.extend(dict(cell=label, check="fixed_clock_cost", window=name, **value)
                      for name, value in cost["windows"].items() if value["status"] != "complete")
        prior = read(v4_root / "runs" / label / "summary.json")
        prior_protocol = read(v4_root / "runs" / label / "protocol.json")
        fields = ("generalized_passenger_seconds", "generalized_seconds_per_departed_passenger", "passengers",
                  "policy_updates", "target_online_training_events", "target_time_seconds", "packets", "feedback_receipts")
        same = prior is not None and all(summary.get(key) == prior.get(key) for key in fields)
        same_initializer = prior_protocol is not None and recorded.get("initializer") == prior_protocol.get("initializer")
        reproduction.append(dict(**cell, exact_cost_and_budget=same, same_initializer=same_initializer,
            cost_delta=difference(summary.get("generalized_passenger_seconds"),
                                  prior.get("generalized_passenger_seconds") if prior else None)))
        if not same or not same_initializer:
            errors.append(dict(cell=label, check="passive_v4_action_model_reproduction"))
        worlds.append(dict(**cell, diagnostics=at(summary, "worlds", "action_model", "paired_world_diagnostics")))
        shadows.append(dict(**cell, diagnostics=summary.get("shadow_diagnostics")))
        td.append(dict(**cell, periods=v5.v4.v3._td_summary(list(jsonl(directory / "td_diagnostics.jsonl")), summary, "action_model")))
    for seed, paths in initializers.items():
        if len(paths) != 1:
            errors.append(dict(seed=seed, check="different_initializers"))
    return dict(status="complete" if len(cells) == 4 and not errors and not missing and not failures else "incomplete",
        integrity=dict(expected_runs=4, complete_runs=len(cells), errors=errors, missing=missing, failures=failures,
                       initializer_paths={seed: sorted(paths) for seed, paths in initializers.items()}),
        passive_reproduction=reproduction, control_cells=cells, paired_world_diagnostics=worlds,
        shadow_diagnostics=shadows, active_td_diagnostics=td,
        interpretation="Four original development days reproduce action_model exactly. Online and initialized frozen twins score identical issue-time states/actions and selected outcomes. Own-family scores include family selection; common-family scores isolate parameter differences. Next-packet Bellman scores use fixed anchor evaluators before fitting. Each world trains its own trust metric; shadows fix actual online trust, matched states and random streams for eight optimizer steps. Only Online is deployed. Shadow changes cannot establish passenger benefit. Cost intervals use full-day departed cohorts; there are no additional target labels after the original update budget.")


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, separators=(",", ":"), allow_nan=False))
