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
    spec = importlib.util.spec_from_file_location("v8_test_infra", HERE / "infra.py")
    infra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infra)
    monkeypatch.setitem(sys.modules, "infra", infra)
    spec = importlib.util.spec_from_file_location("v8_test_launch", HERE / "launch.py")
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    return launch


@pytest.fixture
def protocol():
    return dict(name="v8-test", training_seeds=[6201, 6202],
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
