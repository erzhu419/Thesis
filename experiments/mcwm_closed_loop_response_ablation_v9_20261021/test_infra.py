"""Check high-priority submissions and native gate behavior without remote calls."""

import importlib.util
from pathlib import Path
import shlex
import sys
from unittest.mock import patch

import pytest

HERE = Path(__file__).resolve().parent


@pytest.fixture
def launch_module(monkeypatch):
    # Other numbered experiments also use the top-level name infra.
    spec = importlib.util.spec_from_file_location("v9_test_infra", HERE / "infra.py")
    infra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infra)
    monkeypatch.setitem(sys.modules, "infra", infra)
    spec = importlib.util.spec_from_file_location("v9_test_launch", HERE / "launch.py")
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    return launch


@pytest.fixture
def protocol():
    return dict(name="v9-test", training_seeds=[6201, 6202],
                scenarios=["stable", "local_speed_drop"], target_lines=["7X", "102X", "705X"],
                preflight=dict(seed=6201, scenario="local_speed_drop", target_line="7X"))


def test_task_specs_use_high_priority_for_runner_and_finalizer(launch_module, protocol):
    for runner in ("run.py", "finalize.py"):
        task = launch_module.infra.task(protocol, "local-only", runner=runner)
        assert task["priority"] == "high"
        assert task["cwd"] == str(HERE)
        assert launch_module.infra.remote(HERE / runner) in shlex.split(task["cmd"])


def test_twelve_unique_prefix_paths_and_single_preflight(launch_module, protocol):
    with patch.object(launch_module, "protocol", return_value=protocol):
        cells = launch_module.cells()
        tasks = [launch_module.make_job(**cell) for cell in cells]
        assert len(tasks) == 12
        assert len({task["signature"] for task in tasks}) == 12
        for cell, task in zip(cells, tasks):
            args = shlex.split(task["cmd"])
            expected = HERE / "runs" / cell["scenario"] / cell["target_line"] / f"seed{cell['seed']}"
            assert args[args.index("--out") + 1] == launch_module.infra.remote(expected)
            assert "--branch" not in args
            assert task["priority"] == "high"
        task = launch_module.make_job(**protocol["preflight"], preflight=True)
        args = shlex.split(task["cmd"])
        assert args[args.index("--out") + 1] == launch_module.infra.remote(HERE / "preflight")
        assert "--preflight" in args and task["priority"] == "high"


@pytest.mark.parametrize("gate", [
    {"status": "failed", "integrity": {"all_pass": True}},
    {"status": "complete", "integrity": {"all_pass": False}},
])
def test_failed_native_gate_prevents_submit(launch_module, tmp_path, gate, monkeypatch):
    monkeypatch.setattr(launch_module, "HERE", tmp_path)
    with patch.object(launch_module.infra, "collect_json", return_value=gate) as collect, \
            patch.object(launch_module.infra, "submit") as submit:
        with pytest.raises(RuntimeError, match="preflight"):
            launch_module.submit_campaign()
        collect.assert_called_once_with("preflight/gate.json", max_bytes=12000)
        submit.assert_not_called()


def test_passed_small_gate_submits_all_high_prefix_tasks(launch_module, protocol):
    with patch.object(launch_module, "protocol", return_value=protocol), \
            patch.object(Path, "is_file", return_value=False), \
            patch.object(launch_module.infra, "collect_json", return_value={"status": "complete", "integrity": {"all_pass": True}}) as collect, \
            patch.object(launch_module.infra, "submit", return_value={"submitted": []}) as submit:
        launch_module.submit_campaign()
        collect.assert_called_once_with("preflight/gate.json", max_bytes=12000)
        tasks, batch = submit.call_args.args
        assert batch == "diagnostic" and len(tasks) == 12
        assert all(task["priority"] == "high" for task in tasks)


def test_source_readiness_requires_completed_v8_predictions(launch_module, tmp_path):
    import json

    source, inputs, v7, v8 = (tmp_path / name for name in ("source", "inputs", "v7", "v8"))
    for path in (source, inputs, v7, v8):
        path.mkdir()
    (source / "source_summary.json").write_text(json.dumps({"status": "complete"}))
    (source / "source_intervals.json").write_text(json.dumps([{}] * 102))
    for name in ("h2oplus_seed6201.pt", "h2oplus_seed6202.pt", "history.npz", "source_9101.npz"):
        (inputs / name).touch()
    (v7 / "action_fork_summary.json").write_text("{}")
    result_path = v8 / "response_summary.json"

    def local_readiness(_node, command, **_kwargs):
        namespace = {}
        exec(shlex.split(command)[-1], namespace)
        return 0, '{"v8_predictions_available":true}', ""

    with patch.multiple(launch_module.infra, RESIDENT_SOURCE=source, RESIDENT_INPUTS=inputs,
                        RESIDENT_V7=v7, RESIDENT_V8=v8), \
            patch.object(launch_module.infra.scheduler, "run_on", side_effect=local_readiness):
        with pytest.raises(FileNotFoundError):
            launch_module.source_ready()
        result_path.write_text(json.dumps({"status": "partial", "integrity": {"complete_states": 11}}))
        with pytest.raises(AssertionError):
            launch_module.source_ready()
        result_path.write_text(json.dumps({"status": "complete", "integrity": {"complete_states": 12}}))
        assert launch_module.source_ready()["v8_predictions_available"]


def test_oversized_resident_summary_is_not_downloaded(launch_module):
    with patch.object(launch_module.infra.scheduler, "run_on", return_value=(0, "150001\n", "")), \
            patch.object(launch_module.infra.subprocess, "run") as copy:
        with pytest.raises(RuntimeError, match="small-result budget"):
            launch_module.infra.collect_json("ablation_summary.json")
        copy.assert_not_called()


def test_staging_rejects_previous_version_paths(launch_module):
    previous = HERE.parent / "mcwm_closed_loop_action_response_v8_20261021" / "run.py"
    with patch.object(launch_module.infra.scheduler, "run_on") as remote:
        with pytest.raises(ValueError):
            launch_module.infra.stage([previous])
        remote.assert_not_called()
