"""Check native gating, paired allocation, priority, and the small-pull boundary offline."""

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
    spec = importlib.util.spec_from_file_location("v11_test_infra", HERE / "infra.py")
    infra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infra)
    monkeypatch.setitem(sys.modules, "infra", infra)
    spec = importlib.util.spec_from_file_location("v11_test_launch", HERE / "launch.py")
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    return launch


@pytest.fixture
def protocol():
    return dict(name="v11-test", training_seeds=[6201, 6202], scenarios=["stable", "local_speed_drop"],
                target_lines=["7X", "102X", "705X"], preflight=dict(seed=6201, scenario="stable"))


def test_four_days_and_twelve_branches_have_distinct_high_tasks(launch_module, protocol):
    with patch.object(launch_module, "protocol", return_value=protocol):
        days = launch_module.cells()
        branches = launch_module.pair_cells()
        tasks = [launch_module.make_job(**cell) for cell in days]
        tasks += [launch_module.make_job(mode="pair", **cell) for cell in branches]
    assert len(days) == 4 and len(branches) == 12
    assert {row["seed"] for row in branches} == {6201}
    assert {(row["scenario"], row["target_line"], row["branch"]) for row in branches} == {
        (scenario, line, branch) for scenario in protocol["scenarios"]
        for line in protocol["target_lines"] for branch in ("low", "high")}
    assert len({task["signature"] for task in tasks}) == 16
    assert all(task["priority"] == "high" for task in tasks)
    assert all(task["skip_launch_staging"] is True for task in tasks)
    for row, task in zip(branches, tasks[4:]):
        argv = shlex.split(task["cmd"])
        expected = HERE / "pairs" / row["scenario"] / row["target_line"] / "seed6201" / row["branch"]
        assert argv[argv.index("--out") + 1] == launch_module.infra.remote(expected)
        assert argv[argv.index("--mode") + 1] == "pair"
        assert argv[argv.index("--target-line") + 1] == row["target_line"]
        assert argv[argv.index("--branch") + 1] == row["branch"]


@pytest.mark.parametrize("gate", [
    {"status": "failed", "integrity": {"all_pass": True}},
    {"status": "complete", "integrity": {"all_pass": False}},
])
def test_failed_aggregate_preflight_blocks_four_days(launch_module, protocol, tmp_path, monkeypatch, gate):
    monkeypatch.setattr(launch_module, "HERE", tmp_path)
    with patch.object(launch_module, "protocol", return_value=protocol), \
            patch.object(launch_module.infra, "collect_json", return_value=gate), \
            patch.object(launch_module.infra, "submit") as submit:
        with pytest.raises(RuntimeError, match="gate"):
            launch_module.submit_campaign()
        submit.assert_not_called()


def test_preflight_stages_keep_separate_paths_and_failures(launch_module, protocol, tmp_path, monkeypatch):
    protocol["preflight_directory"] = "preflight2"
    (tmp_path / "preflight").mkdir()
    old = tmp_path / "preflight" / "failure.json"
    old.write_text('{"status":"failed","reason":"retained first attempt"}')
    monkeypatch.setattr(launch_module, "HERE", tmp_path)
    gate = dict(status="complete", integrity=dict(all_pass=True))
    with patch.object(launch_module, "protocol", return_value=protocol), \
            patch.object(launch_module.infra, "remote", side_effect=str), \
            patch.object(launch_module.infra, "collect_json", return_value=gate) as collect, \
            patch.object(launch_module.infra, "submit", return_value={"submitted": []}) as submit:
        launch_module.submit_preflight_pairs()
        collect.assert_called_once_with("preflight2/day/gate.json", max_bytes=20000)
        tasks, batch = submit.call_args.args
    assert batch == "preflight2_pairs" and len(tasks) == 2
    for branch, task in zip(("low", "high"), tasks):
        args = shlex.split(task["cmd"])
        assert args[args.index("--out") + 1] == str(tmp_path / "preflight2" / "pair" / branch)
        assert "--preflight" in args and task["priority"] == "high"
    assert json.loads(old.read_text())["reason"] == "retained first attempt"


def test_missing_or_failed_main_day_blocks_pair_budget(launch_module, protocol):
    incomplete = [dict(status="complete", all_pass=True, failed=False) for _ in range(3)]
    failed = incomplete + [dict(status="complete", all_pass=False, failed=False)]
    with patch.object(launch_module, "protocol", return_value=protocol), \
            patch.object(launch_module.infra, "submit") as submit:
        for rows in (incomplete, failed):
            with patch.object(launch_module, "days_status", return_value=rows):
                with pytest.raises(RuntimeError, match="four"):
                    launch_module.submit_pairs()
        submit.assert_not_called()


def test_passed_days_allocate_only_twelve_branches(launch_module, protocol):
    rows = [dict(status="complete", all_pass=True, failed=False) for _ in range(4)]
    with patch.object(launch_module, "protocol", return_value=protocol), \
            patch.object(launch_module, "days_status", return_value=rows), \
            patch.object(launch_module.infra, "submit", return_value={"submitted": []}) as submit:
        launch_module.submit_pairs()
    tasks, batch = submit.call_args.args
    assert batch == "paired" and len(tasks) == 12
    assert all(task["priority"] == "high" for task in tasks)


def test_finalizers_use_high_and_do_not_collect_runs(launch_module, protocol):
    with patch.object(launch_module, "protocol", return_value=protocol), \
            patch.object(Path, "read_text", return_value=json.dumps({"submitted": []})), \
            patch.object(launch_module.infra, "submit", return_value={"submitted": []}) as submit:
        launch_module.submit_preflight_finalizer()
        tasks, name = submit.call_args.args
        argv = shlex.split(tasks[0]["cmd"])
        assert "--preflight" in argv and "--wait" in argv
        assert name == "preflight_finalize"
        assert tasks[0]["priority"] == "high"
        launch_module.submit_finalizer()
        tasks, name = submit.call_args.args
        argv = shlex.split(tasks[0]["cmd"])
        assert name == "diagnostic_finalize" and tasks[0]["priority"] == "high"
        assert argv[argv.index("--root") + 1] == launch_module.infra.remote(HERE)
        assert argv[argv.index("--out") + 1] == launch_module.infra.remote(HERE / "residual_summary.json")


def test_oversized_summary_is_rejected_before_copy(launch_module):
    with patch.object(launch_module.infra.scheduler, "run_on", return_value=(0, "200001\n", "")), \
            patch.object(launch_module.infra.subprocess, "run") as copy:
        with pytest.raises(RuntimeError, match="small-result budget"):
            launch_module.infra.collect_json("residual_summary.json")
        copy.assert_not_called()


def test_staging_rejects_old_experiment_before_remote_write(launch_module):
    previous = HERE.parent / "mcwm_closed_loop_controlled_calibration_v10_20261021" / "run.py"
    with patch.object(launch_module.infra.scheduler, "run_on") as remote:
        with pytest.raises(ValueError):
            launch_module.infra.stage([previous])
        remote.assert_not_called()


def test_metadata_preserves_missing_branch_and_uses_current_preflight(launch_module, protocol, tmp_path, monkeypatch):
    protocol["preflight_directory"] = "preflight2"
    for relative in ("preflight", "preflight2/day", "preflight2/pair/low", "preflight2/pair/high",
                     "runs/stable/seed6201", "pairs/stable/7X/seed6201/low"):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    (tmp_path / "preflight/failure.json").write_text('{"status":"failed"}')
    (tmp_path / "preflight2/day/gate.json").write_text('{"status":"complete"}')
    passed = '{"status":"complete","integrity":{"all_pass":true}}'
    (tmp_path / "runs/stable/seed6201/summary.json").write_text(passed)
    (tmp_path / "pairs/stable/7X/seed6201/low/summary.json").write_text(passed)
    monkeypatch.setitem(sys.modules, "launch", launch_module)
    spec = importlib.util.spec_from_file_location("v11_test_continue", HERE / "continue_batch.py")
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
    assert value["preflight_day_complete"] is True and value["preflight_day_failed"] is False
    assert value["days_complete"] == value["days_passed"] == 1
    assert value["pairs_complete"] == 1
    assert value["failures"] == []
