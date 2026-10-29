"""Re-export completed v72 summaries without changing its analysis or cost gates."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments.mcwm_closed_loop_capacity_feedback_v72_20261026 import analyze as v72
from experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026 import metrics as old

MAX_BYTES = v72.MAX_BYTES


def split_export(main, auxiliary, cost_name="development.json"):
    """Move descriptive old references, preserving every numeric value and gate."""
    cost, policy = old.split_summary(main)
    auxiliary = deepcopy(auxiliary)
    original = dict(
        scope="Descriptive v64/v65 original no_fresh reference; no scientific gates changed.",
        aggregates=cost["aggregates"].pop("original_no_fresh"),
        cells=[],
    )
    for cell in cost["cells"]:
        periods = {}
        for name, period in cell["periods"].items():
            periods[name] = dict(
                arm=period["arms"].pop("original_no_fresh"),
                comparison=period["comparisons"].pop("original_no_fresh"),
            )
        original["cells"].append(dict(identity=cell["identity"], scenario=cell["scenario"], periods=periods))
    reference_name = "reference_summary.json"
    payloads = {cost_name: cost, "policy_drift.json": policy,
        "recency.json": auxiliary, reference_name: original}
    cost["reference_summary_file"] = reference_name
    return payloads


def actual_budget(root, protocol):
    """Export actual completed-run budgets, rather than only the planned totals."""
    rows = []
    for seed in protocol["development_seeds"]:
        for scene in protocol["scenarios"]:
            for arm in protocol["arms"]:
                directory = root/"runs/executed_cql/development"/f"seed{seed}"/scene/arm
                summary = old.read(directory/"summary.json")
                rows.append(dict(seed=seed, scenario=scene, arm=arm,
                    status=summary.get("status") if summary else "missing",
                    budget=summary.get("budget") if summary else None,
                    post_budget_counts=summary.get("post_budget_counts") if summary else None,
                    factual_trace_rows=summary.get("factual_trace_rows") if summary else None))
    complete = [row for row in rows if row["status"] == "complete"]
    fields = ("target_feedback_slots", "policy_updates", "critic_updates", "value_updates",
              "actor_updates", "alpha_updates", "packets", "native_seconds",
              "world_updates", "trust_updates", "counterfactual_queries")
    totals = {name: sum(row["budget"][name] for row in complete) for name in fields}
    return dict(expected_native_days=len(rows), completed_native_days=len(complete), totals=totals, runs=rows)


def export(root, out):
    root, out = Path(root), Path(out)
    main, auxiliary = v72.analyze(root)
    actual = actual_budget(root, old.read(root/"protocol.json"))
    native_runs = actual.pop("runs")
    main["budget"]["actual"] = actual
    payloads = split_export(main, auxiliary, out.name)
    payloads["reference_summary.json"]["native_budget_runs"] = native_runs
    encoded = {out.with_name(name): old.encoded(value) for name, value in payloads.items()}
    for path, raw in encoded.items():
        if len(raw) > MAX_BYTES:
            raise ValueError(f"{path.name} exceeds {MAX_BYTES} bytes: {len(raw)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    for path, raw in encoded.items():
        with path.open("xb") as handle:
            handle.write(raw)
    cost = payloads[out.name]
    return dict(status=cost["status"], decision=cost["decision"],
        integrity=cost["integrity"], files={path.name: len(raw) for path, raw in encoded.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.root, args.out), allow_nan=False))
