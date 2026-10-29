"""Fixed line-cover development and independent full-day confirmation."""
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


def cells(protocol, stage):
    arms = ("fresh", "no_fresh") if stage == "development" else ("fresh", "no_fresh", "original_no_fresh")
    return [dict(seed=seed, scenario=scenario, arm=arm,
                 relative=f"runs/line_cover/{stage}/seed{seed}/{scenario}/{arm}")
            for seed in protocol[stage + "_seeds"] for scenario in protocol["scenarios"]
            for arm in arms]


def native_jobs(protocol, stage, infra):
    result = []
    for cell in cells(protocol, stage):
        original = cell["arm"] == "original_no_fresh"
        runner = V60 / "control.py" if original else HERE / "control.py"
        argv = ["--seed", cell["seed"], "--scenario", cell["scenario"],
                "--arm", "no_fresh" if original else cell["arm"],
                "--out", infra.remote(HERE / cell["relative"])]
        if original:
            argv += ["--candidate", "recent30"]
        result.append(infra.task(cell["relative"], runner, argv))
    return result


def watch(receipt, expected, infra, read_only):
    paths = [cell["relative"] + "/" + name for cell in expected for name in ("summary.json", "failure.json")]
    found = read_only(infra.presence, paths)
    indexed = {row["signature"].split("/runs/", 1)[1]: row["id"] for row in receipt["submitted"]}
    pending, complete, failures = [], [], []

    def resident(cell, identifier):
        relative = cell["relative"]
        if found.get(relative + "/failure.json"):
            failure = read_only(infra.collect, relative + "/failure.json")
            failures.append(dict(cell=cell, id=identifier, failure=failure))
            return True
        if found.get(relative + "/summary.json"):
            summary = read_only(infra.collect, relative + "/summary.json")
            row = dict(cell=cell, id=identifier)
            (complete if summary.get("status") == "complete" else failures).append(row)
            return True
        return False

    for cell in expected:
        relative = cell["relative"]
        identifier = indexed.get(relative.removeprefix("runs/"))
        if identifier is None:
            raise ValueError("Existing receipt omits declared cell: " + relative)
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
                for cell, _ in terminal for name in ("summary.json", "failure.json")]))
        for cell, key in pending:
            status = statuses[key]
            if status["status"] in TERMINAL:
                if not resident(cell, key):
                    failures.append(dict(cell=cell, id=key, scheduler_status=status["status"],
                        reason="Terminal task without declared output; incurred work retained"))
            else:
                still.append(dict(cell=cell, id=key, scheduler_status=status["status"], node=status.get("node")))
    return dict(completed=len(complete), failures=failures, pending=still, all_terminal=not still)


def _advance(read_only):
    ready = HERE / "execution_ready.json"
    if not ready.exists() or json.loads(ready.read_text()).get("ready") is not True:
        return dict(stage="preparing_line_cover_control", done=False, goal_achieved=False)
    infra = module("v62_launch_infra", HERE / "infra.py")
    protocol = json.loads((HERE / "protocol.json").read_text())
    for stage in ("development", "confirmation"):
        batch = "line_cover_" + stage
        receipt_path = HERE / (batch + "_submission.json")
        receipt = (json.loads(receipt_path.read_text()) if receipt_path.exists()
                   else infra.submit(native_jobs(protocol, stage, infra), batch))
        current = watch(receipt, cells(protocol, stage), infra, read_only)
        if not current["all_terminal"]:
            return dict(stage="line_cover_control_" + stage, done=False, goal_achieved=False, **current)
        relative = "results/line_cover/" + stage + ".json"
        analysis_batch = batch + "_analysis"
        analysis_receipt = HERE / (analysis_batch + "_submission.json")
        if not analysis_receipt.exists():
            infra.submit([infra.task("cost/" + stage, HERE / "analyze.py",
                ["--stage", stage, "--out", infra.remote(HERE / relative)])], analysis_batch)
        if not read_only(infra.presence, [relative]).get(relative):
            identifier = json.loads(analysis_receipt.read_text())["submitted"][0]["id"]
            rows = read_only(infra.status, [identifier])
            if len(rows) != 1 or rows[0]["id"] != identifier:
                raise ValueError("Scheduler omitted existing summary task; no replacement permitted")
            status = rows[0]
            return dict(stage="line_cover_summary_" + stage, done=False, goal_achieved=False,
                attention=status if status["status"] in TERMINAL else None)
        result = read_only(infra.collect, relative)
        if result.get("status") != "complete" or result.get("integrity", {}).get("all_pass") is not True:
            return dict(stage="line_cover_technical_attention", done=False, goal_achieved=False,
                result=relative, failed_checks=result.get("integrity", {}).get("failed_checks", []),
                failures=result.get("failures", []), missing=result.get("missing", []),
                performance_verdict="not_evaluated_incomplete", requires_agent_review=True)
        decision = result.get("decision", {})
        key = "advance_to_confirmation" if stage == "development" else "confirmed_control_improvement"
        if not isinstance(decision.get(key), bool):
            raise ValueError("Completed line-cover result has no declared boolean decision: " + key)
        if stage == "development" and not decision[key]:
            return dict(stage="line_cover_candidate_closed_requires_next_control", done=False,
                goal_achieved=False, result=relative, decision=decision, requires_agent_review=True)
        if stage == "confirmation":
            return dict(stage="line_cover_gain_requires_root_review" if decision[key]
                else "line_cover_candidate_closed_after_confirmation", done=False,
                goal_achieved=False, confirmed_candidate=decision[key], result=relative,
                decision=decision, requires_agent_review=True)
    raise RuntimeError("Fixed line-cover campaign exhausted without a decision")


def advance():
    helper = module("v62_v60_readonly", V60 / "supervisor.py")
    try:
        return _advance(helper.read_only)
    except helper.ReadUnavailable as error:
        return dict(stage="waiting_transient_line_cover_read", done=False, goal_achieved=False,
                    message=str(error))
