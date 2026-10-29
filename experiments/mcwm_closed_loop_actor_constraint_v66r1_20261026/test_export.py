"""The export split preserves every cost/gate and all twelve native lines."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.mcwm_closed_loop_actor_constraint_v66_20261026.analyze import analyze, drift_view
from experiments.mcwm_closed_loop_actor_constraint_v66_20261026.test_analyze import populate
from experiments.mcwm_closed_loop_actor_constraint_v66r1_20261026 import export as module


def native_sized_fixture(tmp_path):
    root, _ = populate(tmp_path)
    value = analyze(root)
    # Use the already collected small summaries' full supported twelve lines,
    # rather than the earlier insufficient three-diagnostic-line fixture.
    for row in value["native_evidence"]:
        for arm in ("fresh", "no_fresh"):
            path = module.SOURCE/"runs/actor_constrained/development"/f"seed{row['seed']}"/row["scenario"]/arm/"summary.json"
            row["policy_drift"][arm] = drift_view(json.loads(path.read_text())["policy_drift"])
    return value


def test_split_roundtrip_retains_costs_gates_and_full_twelve_line_rows(tmp_path):
    original = native_sized_fixture(tmp_path); untouched = deepcopy(original)
    main, drift = module.split_summary(original)
    assert original == untouched
    restored = deepcopy(main); restored.pop("policy_drift_file")
    for target, source in zip(restored["native_evidence"], drift["native_evidence"]):
        assert (target["seed"], target["scenario"]) == (source["seed"], source["scenario"])
        target["policy_drift"] = source["policy_drift"]
    assert restored == original
    assert main["aggregates"] == original["aggregates"]
    assert main["optimization_assessment"] == original["optimization_assessment"]
    assert main["decision"] == original["decision"]
    column = drift["policy_drift_columns"].index("line")
    assert len({r[column] for cell in drift["native_evidence"] for arm in cell["policy_drift"].values() for r in arm["rows"]}) == 12
    assert len(module.encoded(main)) <= module.MAX_BYTES
    assert len(module.encoded(drift)) <= module.MAX_BYTES


def test_export_writes_two_new_small_files_without_retraining(tmp_path, monkeypatch):
    value = native_sized_fixture(tmp_path); calls = []
    def existing_analysis(path):
        calls.append(path)
        return value
    monkeypatch.setattr(module, "analyze", existing_analysis)
    out = tmp_path/"new/results/development.json"
    receipt = module.export(module.SOURCE, out)
    assert calls == [module.SOURCE]
    assert receipt["new_native_days"] == receipt["new_feedback_slots"] == receipt["new_policy_updates"] == 0
    assert set(receipt["files"]) == {"development.json", "policy_drift.json"}
    assert all(size <= module.MAX_BYTES for size in receipt["files"].values())
    assert json.loads(out.read_text())["decision"] == value["decision"]
    assert json.loads(out.read_text())["policy_drift_file"] == "policy_drift.json"


def test_size_failure_writes_neither_partial_file(tmp_path, monkeypatch):
    value = native_sized_fixture(tmp_path)
    value["unchanged_extra_evidence"] = "x"*module.MAX_BYTES
    monkeypatch.setattr(module, "analyze", lambda _: value)
    out = tmp_path/"new/results/development.json"
    with pytest.raises(ValueError, match="development.json exceeds"):
        module.export(module.SOURCE, out)
    assert not out.exists() and not out.with_name("policy_drift.json").exists()
