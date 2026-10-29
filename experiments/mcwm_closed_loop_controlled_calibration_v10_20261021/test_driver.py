"""Verify the four-day submission, failed gate stop, and small-pull boundary offline."""

import importlib.util
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shlex
import sys
from unittest.mock import patch

import pytest

HERE = Path(__file__).resolve().parent


@pytest.fixture
def launch_module(monkeypatch):
    # Other numbered experiments use these same short module names.
    spec = importlib.util.spec_from_file_location("v10_test_infra", HERE / "infra.py")
    infra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infra)
    monkeypatch.setitem(sys.modules, "infra", infra)
    spec = importlib.util.spec_from_file_location("v10_test_launch", HERE / "launch.py")
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    return launch


@pytest.fixture
def protocol():
    return dict(name="v10-test", training_seeds=[6201, 6202], scenarios=["stable", "local_speed_drop"],
                preflight=dict(seed=6201, scenario="stable"))


def test_four_unique_day_paths_and_single_preflight(launch_module, protocol):
    with patch.object(launch_module, "protocol", return_value=protocol):
        cells = launch_module.cells()
        tasks = [launch_module.make_job(**cell) for cell in cells]
        assert len(tasks) == 4
        assert {(cell["seed"], cell["scenario"]) for cell in cells} == {
            (6201, "stable"), (6201, "local_speed_drop"),
            (6202, "stable"), (6202, "local_speed_drop"),
        }
        assert len({task["signature"] for task in tasks}) == 4
        for cell, task in zip(cells, tasks):
            args = shlex.split(task["cmd"])
            expected = HERE / "runs" / cell["scenario"] / f"seed{cell['seed']}"
            assert args[args.index("--out") + 1] == launch_module.infra.remote(expected)
            assert "--target-line" not in args and "--preflight" not in args
            assert task["priority"] == "high"
        args = shlex.split(launch_module.make_job(**protocol["preflight"], preflight=True)["cmd"])
        assert args[args.index("--out") + 1] == launch_module.infra.remote(HERE / "preflight")
        assert "--preflight" in args


@pytest.mark.parametrize("gate", [
    {"status": "failed", "integrity": {"all_pass": True}},
    {"status": "complete", "integrity": {"all_pass": False}},
])
def test_failed_native_gate_prevents_all_day_submissions(launch_module, protocol, tmp_path, gate, monkeypatch):
    monkeypatch.setattr(launch_module, "HERE", tmp_path)
    with patch.object(launch_module, "protocol", return_value=protocol), \
            patch.object(launch_module.infra, "collect_json", return_value=gate) as collect, \
            patch.object(launch_module.infra, "submit") as submit:
        with pytest.raises(RuntimeError, match="preflight"):
            launch_module.submit_campaign()
        collect.assert_called_once_with("preflight/gate.json", max_bytes=20000)
        submit.assert_not_called()


def test_passed_small_gate_submits_all_high_days(launch_module, protocol):
    gate = {"status": "complete", "integrity": {"all_pass": True}}
    with patch.object(launch_module, "protocol", return_value=protocol), \
            patch.object(Path, "is_file", return_value=False), \
            patch.object(launch_module.infra, "collect_json", return_value=gate), \
            patch.object(launch_module.infra, "submit", return_value={"submitted": []}) as submit:
        launch_module.submit_campaign()
        tasks, batch = submit.call_args.args
        assert batch == "diagnostic" and len(tasks) == 4
        assert all(task["priority"] == "high" for task in tasks)


def test_finalizer_waits_on_new_runs_and_uses_high_priority(launch_module, protocol, monkeypatch):
    with patch.object(launch_module, "protocol", return_value=protocol), \
            patch.object(Path, "read_text", return_value=json.dumps({"submitted": []})), \
            patch.object(launch_module.infra, "submit", return_value={"submitted": []}) as submit:
        launch_module.submit_finalizer()
    tasks, batch = submit.call_args.args
    assert batch == "diagnostic_finalize" and len(tasks) == 1
    task = tasks[0]
    args = shlex.split(task["cmd"])
    assert task["priority"] == "high"
    assert args[2] == launch_module.infra.remote(HERE / "analyze.py")
    assert args[args.index("--root") + 1] == launch_module.infra.remote(HERE / "runs")
    assert args[args.index("--out") + 1] == launch_module.infra.remote(HERE / "calibration_summary.json")
    assert "--wait" in args


def test_oversized_resident_summary_is_not_copied(launch_module):
    with patch.object(launch_module.infra.scheduler, "run_on", return_value=(0, "200001\n", "")), \
            patch.object(launch_module.infra.subprocess, "run") as copy:
        with pytest.raises(RuntimeError, match="small-result budget"):
            launch_module.infra.collect_json("calibration_summary.json")
        copy.assert_not_called()


def test_staging_rejects_earlier_version_paths_before_remote_write(launch_module):
    previous = HERE.parent / "mcwm_closed_loop_action_response_v8_20261021" / "run.py"
    with patch.object(launch_module.infra.scheduler, "run_on") as remote:
        with pytest.raises(ValueError):
            launch_module.infra.stage([previous])
        remote.assert_not_called()


def test_second_preflight_has_new_output_and_submission_name(launch_module, protocol):
    protocol["preflight_directory"] = "preflight2"
    with patch.object(launch_module, "protocol", return_value=protocol), \
            patch.object(launch_module, "source_ready"), \
            patch.object(launch_module.infra, "submit", return_value={"submitted": []}) as submit:
        launch_module.submit_preflight()
    tasks, batch = submit.call_args.args
    args = shlex.split(tasks[0]["cmd"])
    assert batch == "preflight2"
    assert args[args.index("--out") + 1] == launch_module.infra.remote(HERE / "preflight2")
    assert tasks[0]["signature"].endswith("/preflight2")


def test_second_preflight_gate_leaves_old_failed_evidence_untouched(launch_module, protocol, tmp_path, monkeypatch):
    protocol["preflight_directory"] = "preflight2"
    old = tmp_path / "preflight"
    old.mkdir()
    evidence = old / "failure.json"
    evidence.write_text('{"status":"failed","reason":"original native endpoint failure"}')
    original = evidence.read_text()
    monkeypatch.setattr(launch_module, "HERE", tmp_path)
    with patch.object(launch_module, "protocol", return_value=protocol), \
            patch.object(launch_module.infra, "remote", side_effect=str), \
            patch.object(launch_module.infra, "collect_json", return_value={"status": "complete", "integrity": {"all_pass": True}}) as collect, \
            patch.object(launch_module.infra, "submit", return_value={"submitted": []}):
        launch_module.submit_campaign()
    collect.assert_called_once_with("preflight2/gate.json", max_bytes=20000)
    assert evidence.read_text() == original


def test_driver_metadata_observes_current_preflight_directory(launch_module, protocol, tmp_path, monkeypatch):
    protocol["preflight_directory"] = "preflight2"
    for name in ("preflight", "preflight2"):
        (tmp_path / name).mkdir()
    (tmp_path / "preflight/failure.json").write_text('{"status":"failed"}')
    (tmp_path / "preflight2/gate.json").write_text('{"status":"complete"}')
    monkeypatch.setitem(sys.modules, "launch", launch_module)
    spec = importlib.util.spec_from_file_location("v10_test_continue", HERE / "continue_batch.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    monkeypatch.setattr(driver, "HERE", tmp_path)

    def local_metadata(_node, command, **_kwargs):
        output = StringIO()
        with redirect_stdout(output):
            exec(shlex.split(command)[-1], {})
        return 0, output.getvalue(), ""

    with patch.object(launch_module, "protocol", return_value=protocol), \
            patch.object(launch_module.infra, "remote", side_effect=str), \
            patch.object(launch_module.infra.scheduler, "run_on", side_effect=local_metadata):
        value = driver.metadata()
    assert value["preflight_complete"] is True
    assert value["preflight_failed"] is False
