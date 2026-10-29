"""One preflight seed job, one remaining seed job, then resident small readouts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V60 = HERE.parent/"mcwm_closed_loop_control_campaign_v60_20261025"
V70 = HERE.parent/"mcwm_closed_loop_recent30_mixed_cql_v70_20261026"
TERMINAL = {"done", "failed", "cancelled"}
OUTPUTS = ("results/diagnostic.json", "results/service.json")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def receipt(infra, batch, *, seed=None):
    path = HERE/(batch+"_submission.json")
    if path.is_file():
        return json.loads(path.read_text())
    if seed is None:
        runner, argv, label = HERE/"analyze.py", ["--out", infra.remote(HERE/OUTPUTS[0])], "tail_exchange_analysis"
    else:
        runner, argv, label = HERE/"pair.py", ["--seed", seed, "--out", infra.remote(HERE/"runs"/f"seed{seed}")], f"seed{seed}"
    return infra.submit([infra.task(label, runner, argv)], batch)


def statuses(infra, receipts, read_only):
    ids = [row["id"] for receipt in receipts for row in receipt["submitted"]]
    values = read_only(infra.status, ids)
    if len(values) != len(ids) or {row["id"] for row in values} != set(ids):
        raise ValueError("Scheduler omitted existing receipt IDs; no replacement permitted")
    return values


def watch(infra, receipts, seeds, read_only):
    values = statuses(infra, receipts, read_only)
    by_id = {row["id"]: row for row in values}
    paths = [f"runs/seed{seed}/seed_summary.json" for seed in seeds]
    found = read_only(infra.presence, paths)
    completed, failures = [], []
    for seed, receipt, path in zip(seeds, receipts, paths):
        if len(receipt["submitted"]) != 1:
            raise ValueError("Each seed reserves one task for both native prefixes")
        row = by_id[receipt["submitted"][0]["id"]]
        if row["status"] in ("failed", "cancelled"):
            failures.append(dict(seed=seed, task=row, reason="native_seed_task_failed"))
        elif found.get(path):
            summary = read_only(infra.collect, path)
            audit = summary.get("integrity", {})
            valid = summary.get("status") == "complete" and audit.get("all_pass") is True \
                and bool(audit.get("checks")) and all(value is True for value in audit["checks"].values())
            if not valid:
                failures.append(dict(seed=seed, reason="native_seed_integrity_failed", path=path))
            elif row["status"] == "done":
                completed.append(seed)
        elif row["status"] in TERMINAL:
            failures.append(dict(seed=seed, task=row, reason="terminal_seed_summary_missing", path=path))
    return dict(completed=len(completed), completed_seeds=completed, failures=failures,
        pending=[row["id"] for row in values if row["status"] not in TERMINAL],
        all_terminal=all(row["status"] in TERMINAL for row in values), native_jobs=values)


def attention(current, stage, **fields):
    return dict(current, stage=stage, done=True, goal_achieved=False, requires_agent_review=True,
        native_tasks_may_still_run=bool(current.get("pending")), **fields)


def _advance(read_only):
    ready = HERE/"execution_ready.json"
    if not ready.is_file() or json.loads(ready.read_text()).get("ready") is not True:
        return dict(stage="preparing_tail_exchange", done=False, goal_achieved=False)
    infra = module("v71_launch_infra", HERE/"infra.py")
    protocol = json.loads((HERE/"protocol.json").read_text())
    first_seed = protocol["native_preflight"]["seed"]
    remaining = [seed for seed in protocol["seeds"] if seed != first_seed]
    if protocol["seeds"] != [6201, 6203] or first_seed != 6201 or len(remaining) != 1 \
            or protocol["native_preflight"].get("reuse_full_day") is not True:
        raise ValueError("Fixed v71 reuses6201 preflight and only adds6203")
    first = receipt(infra, "tail_exchange_preflight", seed=first_seed)
    current = watch(infra, [first], [first_seed], read_only)
    if current["failures"]:
        return attention(current, "tail_exchange_preflight_attention")
    paths = [f"runs/seed{first_seed}/{arm}/preflight.json" for arm in ("fresh", "no_fresh")]
    found = read_only(infra.presence, paths)
    missing = [path for path in paths if not found.get(path)]
    if missing:
        if current["all_terminal"]:
            return attention(current, "tail_exchange_preflight_attention", missing_preflight=missing)
        return dict(current, stage="tail_exchange_native_preflight", done=False, goal_achieved=False,
            missing_preflight=missing)
    old = module("v71_v70_preflight", V70/"launch.py")
    gate = old.preflight([read_only(infra.collect, path) for path in paths], protocol)
    if not gate["all_pass"]:
        return attention(current, "tail_exchange_preflight_attention", preflight=gate)
    other = receipt(infra, "tail_exchange_remaining", seed=remaining[0])
    current = watch(infra, [first, other], [first_seed, remaining[0]], read_only)
    if current["failures"]:
        return attention(current, "tail_exchange_native_attention", preflight=gate)
    if not current["all_terminal"]:
        return dict(current, stage="tail_exchange_native", done=False, goal_achieved=False, preflight=gate)
    summary = receipt(infra, "tail_exchange_analysis")
    state = statuses(infra, [summary], read_only)[0]
    found = read_only(infra.presence, OUTPUTS)
    missing = [path for path in OUTPUTS if not found.get(path)]
    if state["status"] in ("failed", "cancelled") or missing and state["status"] in TERMINAL:
        return attention(current, "tail_exchange_summary_attention", preflight=gate,
            attention=state, missing_summary=missing)
    if missing or state["status"] != "done":
        return dict(current, stage="tail_exchange_summary", done=False, goal_achieved=False,
            preflight=gate, missing_summary=missing)
    result, service = [read_only(infra.collect, path) for path in OUTPUTS]
    if result.get("status") != "complete" or result.get("integrity", {}).get("all_pass") is not True:
        return attention(current, "tail_exchange_technical_attention", preflight=gate,
            result=OUTPUTS[0], failed_checks=result.get("integrity", {}).get("failed_checks", []),
            missing=result.get("missing", []), result_failures=result.get("failures", []))
    decision = result.get("decision", {})
    if not isinstance(decision.get("feedback_redistribution_may_be_considered"), bool):
        raise ValueError("Diagnostic decision must explicitly say whether redistribution may be considered")
    return attention(current, "tail_exchange_complete_requires_root_review", preflight=gate,
        result=OUTPUTS[0], service=OUTPUTS[1], decision=decision,
        service_status=service.get("status"), new_recipe_started=False)


def advance():
    helper = module("v71_v60_readonly", V60/"supervisor.py")
    try:
        return _advance(helper.read_only)
    except helper.ReadUnavailable as error:
        return dict(stage="waiting_transient_tail_exchange_read", done=False, goal_achieved=False,
            message=str(error))
