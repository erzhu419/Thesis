"""One fixed independent recent30 replication; retain the failed v61 gate."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V60 = HERE.parent / "mcwm_closed_loop_control_campaign_v60_20261025"
V62 = HERE.parent / "mcwm_closed_loop_line_cover_v62_20261025"
TERMINAL = {"done", "failed", "cancelled"}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def cells(protocol):
    return [dict(seed=seed, scenario=scenario, arm=arm,
                 relative=f"runs/recent30/confirmation/seed{seed}/{scenario}/{arm}")
            for seed in protocol["confirmation_seeds"] for scenario in protocol["scenarios"]
            for arm in ("fresh", "no_fresh")]


def native_jobs(protocol, infra):
    return [infra.task(cell["relative"], V60 / "control.py",
                ["--candidate", "recent30", "--seed", cell["seed"], "--scenario", cell["scenario"],
                 "--arm", cell["arm"], "--out", infra.remote(HERE / cell["relative"])])
            for cell in cells(protocol)]


def _advance(read_only):
    ready = HERE / "execution_ready.json"
    if not ready.exists() or json.loads(ready.read_text()).get("ready") is not True:
        return dict(stage="preparing_recent30_replication", done=False, goal_achieved=False)
    infra = module("v63_launch_infra", HERE / "infra.py")
    watcher = module("v63_existing_watch_v62", V62 / "launch.py")
    protocol = json.loads((HERE / "protocol.json").read_text())
    batch = "recent30_confirmation"
    receipt_path = HERE / (batch + "_submission.json")
    receipt = (json.loads(receipt_path.read_text()) if receipt_path.exists()
               else infra.submit(native_jobs(protocol, infra), batch))
    current = watcher.watch(receipt, cells(protocol), infra, read_only)
    if not current["all_terminal"]:
        return dict(stage="recent30_replication_native", done=False, goal_achieved=False, **current)
    relative = "results/replication.json"
    analysis_batch = batch + "_analysis"
    analysis_receipt = HERE / (analysis_batch + "_submission.json")
    if not analysis_receipt.exists():
        infra.submit([infra.task("cost/replication", HERE / "analyze.py",
            ["--out", infra.remote(HERE / relative)])], analysis_batch)
    if not read_only(infra.presence, [relative]).get(relative):
        identifier = json.loads(analysis_receipt.read_text())["submitted"][0]["id"]
        rows = read_only(infra.status, [identifier])
        if len(rows) != 1 or rows[0]["id"] != identifier:
            raise ValueError("Scheduler omitted existing replication summary; no replacement permitted")
        status = rows[0]
        return dict(stage="recent30_replication_summary", done=False, goal_achieved=False,
                    attention=status if status["status"] in TERMINAL else None)
    result = read_only(infra.collect, relative)
    if result.get("status") != "complete" or result.get("integrity", {}).get("all_pass") is not True:
        return dict(stage="replication_technical_attention", done=False, goal_achieved=False,
            result=relative, failed_checks=result.get("integrity", {}).get("failed_checks", []),
            failures=result.get("failures", []), missing=result.get("missing", []),
            performance_verdict="not_evaluated_incomplete", requires_agent_review=True)
    decision = result.get("decision", {})
    passed = decision.get("confirmed_control_improvement")
    if not isinstance(passed, bool):
        raise ValueError("Replication summary has no boolean confirmed_control_improvement")
    return dict(stage="replication_gain_requires_root_review" if passed else "replication_candidate_closed",
        done=False, goal_achieved=False, confirmed_candidate=passed, result=relative,
        decision=decision, requires_agent_review=True)


def advance():
    helper = module("v63_v60_readonly", V60 / "supervisor.py")
    try:
        return _advance(helper.read_only)
    except helper.ReadUnavailable as error:
        return dict(stage="waiting_transient_replication_read", done=False, goal_achieved=False,
                    message=str(error))
