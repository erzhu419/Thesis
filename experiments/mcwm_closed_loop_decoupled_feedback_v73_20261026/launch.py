"""Reuse the first two full days, require exact full no_fresh regression, then run three fresh days."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V60 = HERE.parent / "mcwm_closed_loop_control_campaign_v60_20261025"
TERMINAL = {"done", "failed", "cancelled"}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def cells(protocol):
    """Four new fresh days and exactly one complete old-path regression day."""
    result = [dict(seed=seed, scenario=scenario, arm="fresh",
        relative=f"runs/executed_cql/development/seed{seed}/{scenario}/fresh")
        for seed in protocol["development_seeds"] for scenario in protocol["scenarios"]]
    preflight = protocol["native_preflight"]
    baseline = dict(seed=preflight["seed"], scenario=preflight["scenario"], arm="no_fresh",
        relative=f"runs/executed_cql/development/seed{preflight['seed']}/{preflight['scenario']}/no_fresh")
    result.insert(1, baseline)
    if len(result) != 5 or len({row["relative"] for row in result}) != 5:
        raise ValueError("v73 is frozen to four fresh days plus one baseline regression")
    return result

def native_jobs(expected, infra):
    return [infra.task(cell["relative"], HERE / "control.py",
                ["--seed", cell["seed"], "--scenario", cell["scenario"], "--arm", cell["arm"],
                 "--out", infra.remote(HERE / cell["relative"])]) for cell in expected]


def receipt(infra, batch, expected):
    path = HERE / (batch + "_submission.json")
    return (json.loads(path.read_text()) if path.exists()
            else infra.submit(native_jobs(expected, infra), batch))


def preflight(rows, protocol):
    """Only compare pre-update common provenance, never arm-dependent learned targets."""
    declared = protocol["native_preflight"]
    checks = {}
    for arm, row in zip(("fresh", "no_fresh"), rows):
        checks[arm + "/technical"] = row.get("technical_gate_pass") is True
        checks[arm + "/cell"] = (row.get("seed") == declared["seed"] and
            row.get("scenario") == declared["scenario"] and row.get("arm") == arm)
        for key in ("feedback_slots", "policy_updates", "critic_updates", "value_updates",
                    "actor_updates", "alpha_updates"):
            checks[arm + "/" + key] = row.get(key) == declared[key]
        checks[arm + "/schedule"] = row.get("packet_schedule") == [8, 8, 8, 8]
    for key in ("initializer", "first_feedback", "execution_contract"):
        checks[key + "/present"] = all(isinstance(row.get(key), dict) and bool(row[key]) for row in rows)
        checks[key + "/same"] = checks[key + "/present"] and rows[0][key] == rows[1][key]
    checks["prefix_before_first_update/present"] = all(
        isinstance(row.get("prefix_before_first_update"), dict) and
        {"trace_path", "rows", "clock", "initial_updates"} <= row["prefix_before_first_update"].keys()
        for row in rows)
    return dict(all_pass=all(checks.values()), failed_checks=[key for key, value in checks.items() if not value])


def watch(receipt, expected, infra, read_only):
    """Observe the native gates; full receipts and service records stay resident."""
    paths = [cell["relative"] + "/" + name for cell in expected for name in ("gate.json", "failure.json")]
    found = read_only(infra.presence, paths)
    indexed = {row["signature"].split("/runs/", 1)[1]: row["id"] for row in receipt["submitted"]}
    complete, pending, failures = [], [], []

    def resident(cell, identifier):
        relative = cell["relative"]
        if found.get(relative + "/failure.json"):
            failures.append(dict(cell=cell, id=identifier,
                failure=read_only(infra.collect, relative + "/failure.json")))
            return True
        if found.get(relative + "/gate.json"):
            gate = read_only(infra.collect, relative + "/gate.json")
            row = dict(cell=cell, id=identifier)
            if gate.get("status") == "complete" and gate.get("integrity", {}).get("all_pass") is True:
                complete.append(row)
            else:
                failures.append(dict(row, gate=gate))
            return True
        return False

    for cell in expected:
        identifier = indexed.get(cell["relative"].removeprefix("runs/"))
        if identifier is None:
            raise ValueError("Existing receipt omits declared cell: " + cell["relative"])
        if not resident(cell, identifier):
            pending.append((cell, identifier))
    still = []
    if pending:
        statuses = {row["id"]: row for row in read_only(infra.status, [key for _, key in pending])}
        if set(statuses) != {key for _, key in pending}:
            raise ValueError("Scheduler omitted an existing task; no replacement permitted")
        terminal = [(cell, key) for cell, key in pending if statuses[key]["status"] in TERMINAL]
        if terminal:
            found.update(read_only(infra.presence, [cell["relative"] + "/" + name
                for cell, _ in terminal for name in ("gate.json", "failure.json")]))
        for cell, key in pending:
            status = statuses[key]
            if status["status"] in TERMINAL:
                if not resident(cell, key):
                    failures.append(dict(cell=cell, id=key, scheduler_status=status["status"],
                        reason="Terminal task without native gate; incurred work retained"))
            else:
                still.append(dict(cell=cell, id=key, scheduler_status=status["status"], node=status.get("node")))
    return dict(completed=len(complete), failures=failures, pending=still, all_terminal=not still)


def summary_task(infra, batch, *, argv, outputs, read_only):
    """Submit one resident comparison and wait for its bounded outputs."""
    submission = HERE / (batch + "_submission.json")
    if not submission.exists():
        infra.submit([infra.task("analysis/" + batch, HERE / "analyze.py", argv)], batch)
    identifier = json.loads(submission.read_text())["submitted"][0]["id"]
    statuses = read_only(infra.status, [identifier])
    if len(statuses) != 1 or statuses[0]["id"] != identifier:
        raise ValueError("Scheduler omitted existing summary; no replacement permitted")
    status = statuses[0]
    found = read_only(infra.presence, outputs)
    missing = [path for path in outputs if not found.get(path)]
    failed = status["status"] in ("failed", "cancelled") or (missing and status["status"] in TERMINAL)
    return dict(id=identifier, missing=missing, failed=failed, status=status,
                ready=not missing and not failed)


def _advance(read_only):
    ready = HERE / "execution_ready.json"
    if not ready.exists() or json.loads(ready.read_text()).get("ready") is not True:
        return dict(stage="preparing_decoupled_feedback_development", done=False, goal_achieved=False,
                    completed=0, pending=[], failures=[], all_terminal=False)
    infra = module("v73_launch_infra", HERE / "infra.py")
    protocol = json.loads((HERE / "protocol.json").read_text())
    expected = cells(protocol)
    declared = protocol["native_preflight"]
    first = [cell for cell in expected if cell["seed"] == declared["seed"] and
             cell["scenario"] == declared["scenario"]]
    if len(first) != 2 or declared.get("reuse_full_day") is not True or first[1]["arm"] != "no_fresh":
        raise ValueError("Native preflight must reuse one full fresh day and one full no_fresh regression")
    batch = "decoupled_feedback_development"
    first_receipt = receipt(infra, batch + "_preflight", first)
    paths = [cell["relative"] + "/preflight.json" for cell in first]
    found = read_only(infra.presence, paths)
    if not all(found.get(path) for path in paths):
        current = watch(first_receipt, first, infra, read_only)
        stopped = current["all_terminal"] or bool(current["failures"])
        return dict(current, stage="decoupled_feedback_preflight_attention" if stopped else
            "decoupled_feedback_native_preflight", done=stopped, goal_achieved=False,
            requires_agent_review=stopped, native_tasks_may_still_run=bool(current["pending"]),
            missing_preflight=[path for path in paths if not found.get(path)])
    rows = [read_only(infra.collect, path) for path in paths]
    gate = preflight(rows, protocol)
    if gate["all_pass"]:
        prefix = read_only(infra.common_prefix, rows)
        gate["common_prefix"] = prefix
        if not prefix["all_pass"]:
            gate["all_pass"] = False
            gate["failed_checks"].append("full_pre_update_prefix_exact")
    current = watch(first_receipt, first, infra, read_only)
    if not gate["all_pass"] or current["failures"]:
        return dict(current, stage="decoupled_feedback_preflight_attention", done=True,
            goal_achieved=False, preflight=gate, requires_agent_review=True,
            native_tasks_may_still_run=bool(current["pending"]),
            performance_verdict="not_evaluated_incomplete")

    # The first fresh day continues normally. Remaining days require the FULL old-path
    # no_fresh regression, not merely its first eight updates or first packet.
    regression_cell = first[1]
    if any(row["cell"] == regression_cell for row in current["pending"]):
        return dict(current, stage="decoupled_feedback_full_nofresh_regression", done=False,
            goal_achieved=False, preflight=gate, requires_agent_review=False)
    regression_path = "regression_gate.json"
    regression_task = summary_task(infra, batch + "_regression",
        argv=["--regression-gate", "--out", infra.remote(HERE / regression_path)],
        outputs=[regression_path], read_only=read_only)
    if regression_task["failed"]:
        return dict(current, stage="decoupled_feedback_regression_attention", done=True,
            goal_achieved=False, preflight=gate, requires_agent_review=True,
            regression_task=regression_task, native_tasks_may_still_run=bool(current["pending"]),
            performance_verdict="not_evaluated_incomplete")
    if not regression_task["ready"]:
        return dict(current, stage="decoupled_feedback_regression_comparison", done=False,
            goal_achieved=False, preflight=gate, regression_task=regression_task)
    regression = read_only(infra.collect, regression_path)
    if not (regression.get("status") == "complete" and regression.get("all_pass") is True and
            regression.get("integrity", {}).get("all_pass") is True):
        return dict(current, stage="decoupled_feedback_regression_attention", done=True,
            goal_achieved=False, preflight=gate, regression=regression, requires_agent_review=True,
            native_tasks_may_still_run=bool(current["pending"]),
            performance_verdict="not_evaluated_incomplete")

    remaining = [cell for cell in expected if cell not in first]
    if len(remaining) != 3 or any(cell["arm"] != "fresh" for cell in remaining):
        raise ValueError("Only three remaining fresh days are authorized")
    other_receipt = receipt(infra, batch + "_remaining", remaining)
    combined = dict(submitted=first_receipt["submitted"] + other_receipt["submitted"])
    current = watch(combined, expected, infra, read_only)
    if current["failures"]:
        return dict(current, stage="decoupled_feedback_native_attention", done=True,
            goal_achieved=False, preflight=gate, regression=regression, requires_agent_review=True,
            native_tasks_may_still_run=bool(current["pending"]),
            performance_verdict="not_evaluated_incomplete")
    if not current["all_terminal"]:
        return dict(current, stage="decoupled_feedback_development_native", done=False,
            goal_achieved=False, preflight=gate, regression=regression, attention=None)
    outputs = protocol["summary_outputs"]
    main = outputs[0]
    summary = summary_task(infra, batch + "_analysis",
        argv=["--out", infra.remote(HERE / main)], outputs=outputs, read_only=read_only)
    if summary["failed"]:
        return dict(current, stage="decoupled_feedback_summary_attention", done=True,
            goal_achieved=False, preflight=gate, regression=regression, attention=summary["status"],
            missing_summary=summary["missing"], requires_agent_review=True,
            performance_verdict="not_evaluated_incomplete")
    if not summary["ready"]:
        return dict(current, stage="decoupled_feedback_development_summary", done=False,
            goal_achieved=False, preflight=gate, regression=regression, attention=None,
            missing_summary=summary["missing"])
    result, *_ = [read_only(infra.collect, path) for path in outputs]
    if result.get("status") != "complete" or result.get("integrity", {}).get("all_pass") is not True:
        return dict(current, stage="decoupled_feedback_technical_attention", done=True,
            goal_achieved=False, preflight=gate, regression=regression, summary_outputs=outputs,
            failed_checks=result.get("integrity", {}).get("failed_checks", []),
            result_failures=result.get("failures", []), missing=result.get("missing", []),
            performance_verdict="not_evaluated_incomplete", attention=None, requires_agent_review=True)
    decision = result.get("decision", {})
    passed = decision.get("scientific_gate_pass")
    if not isinstance(passed, bool):
        raise ValueError("Development summary has no boolean scientific_gate_pass")
    return dict(current, stage="development_gain_requires_root_review" if passed else "development_candidate_closed",
        done=True, goal_achieved=False, candidate_passed=passed, summary_outputs=outputs,
        preflight=gate, regression=regression, decision=decision, attention=None, requires_agent_review=True)


def advance():
    helper = module("v73_v60_readonly", V60 / "supervisor.py")
    try:
        return _advance(helper.read_only)
    except helper.ReadUnavailable as error:
        return dict(stage="waiting_transient_development_read", done=False, goal_achieved=False,
                    message=str(error))
