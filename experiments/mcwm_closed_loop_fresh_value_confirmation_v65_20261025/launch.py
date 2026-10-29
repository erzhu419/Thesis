"""One fixed three-seed confirmation of the frozen v64 algorithm."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V60 = HERE.parent / "mcwm_closed_loop_control_campaign_v60_20261025"
V62 = HERE.parent / "mcwm_closed_loop_line_cover_v62_20261025"
V64 = HERE.parent / "mcwm_closed_loop_fresh_value_v64_20261025"
TERMINAL = {"done", "failed", "cancelled"}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def cells(protocol):
    return [dict(seed=seed, scenario=scenario, arm=arm,
                 relative=f"runs/fresh_value/confirmation/seed{seed}/{scenario}/{arm}")
            for seed in protocol["confirmation_seeds"] for scenario in protocol["scenarios"]
            for arm in ("fresh", "no_fresh")]


def native_jobs(protocol, infra):
    return [infra.task(cell["relative"], V64 / "control.py",
                ["--seed", cell["seed"], "--scenario", cell["scenario"], "--arm", cell["arm"],
                 "--out", infra.remote(HERE / cell["relative"])]) for cell in cells(protocol)]


def _advance(read_only):
    ready = HERE / "execution_ready.json"
    if not ready.exists() or json.loads(ready.read_text()).get("ready") is not True:
        return dict(stage="preparing_fresh_value_confirmation", done=False, goal_achieved=False,
                    completed=0, pending=[], failures=[], all_terminal=False)
    infra = module("v65_launch_infra", HERE / "infra.py")
    watcher = module("v65_existing_watch_v62", V62 / "launch.py")
    protocol = json.loads((HERE / "protocol.json").read_text())
    batch = "fresh_value_confirmation"
    receipt_path = HERE / (batch + "_submission.json")
    receipt = (json.loads(receipt_path.read_text()) if receipt_path.exists()
               else infra.submit(native_jobs(protocol, infra), batch))
    current = watcher.watch(receipt, cells(protocol), infra, read_only)
    if not current["all_terminal"]:
        return dict(current, stage="fresh_value_confirmation_native", done=False,
                    goal_achieved=False, attention=None)
    relative = "results/confirmation.json"
    analysis_batch = batch + "_analysis"
    analysis_receipt = HERE / (analysis_batch + "_submission.json")
    if not analysis_receipt.exists():
        infra.submit([infra.task("cost/confirmation", HERE / "analyze.py",
            ["--out", infra.remote(HERE / relative)])], analysis_batch)
    if not read_only(infra.presence, [relative]).get(relative):
        identifier = json.loads(analysis_receipt.read_text())["submitted"][0]["id"]
        rows = read_only(infra.status, [identifier])
        if len(rows) != 1 or rows[0]["id"] != identifier:
            raise ValueError("Scheduler omitted existing confirmation summary; no replacement permitted")
        status = rows[0]
        return dict(current, stage="fresh_value_confirmation_summary", done=False, goal_achieved=False,
                    attention=status if status["status"] in TERMINAL else None)
    result = read_only(infra.collect, relative)
    if result.get("status") != "complete" or result.get("integrity", {}).get("all_pass") is not True:
        return dict(current, stage="confirmation_technical_attention", done=True, goal_achieved=False,
            result=relative, failed_checks=result.get("integrity", {}).get("failed_checks", []),
            result_failures=result.get("failures", []), missing=result.get("missing", []),
            performance_verdict="not_evaluated_incomplete", attention=None, requires_agent_review=True)
    decision = result.get("decision", {})
    passed = decision.get("confirmed_control_improvement")
    if not isinstance(passed, bool):
        raise ValueError("Confirmation summary has no boolean confirmed_control_improvement")
    return dict(current, stage="confirmation_gain_requires_root_review" if passed else "confirmation_candidate_closed",
        done=True, goal_achieved=False, confirmed_candidate=passed, result=relative,
        decision=decision, attention=None, requires_agent_review=True)


def advance():
    helper = module("v65_v60_readonly", V60 / "supervisor.py")
    try:
        return _advance(helper.read_only)
    except helper.ReadUnavailable as error:
        return dict(stage="waiting_transient_confirmation_read", done=False, goal_achieved=False,
                    message=str(error))
