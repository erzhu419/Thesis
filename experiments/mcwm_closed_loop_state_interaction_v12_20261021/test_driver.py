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
    spec = importlib.util.spec_from_file_location("v12_test_infra", HERE / "infra.py")
    infra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infra)
    monkeypatch.setitem(sys.modules, "infra", infra)
    spec = importlib.util.spec_from_file_location("v12_test_launch", HERE / "launch.py")
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    return launch


@pytest.fixture
def protocol():
    return dict(name="v12-test", training_seeds=[6201, 6202], scenarios=["stable", "local_speed_drop"],
                target_lines=["7X", "102X", "705X"], preflight=dict(seed=6201, scenario="stable"))


def test_four_days_and_twenty_four_branches_have_distinct_high_tasks(launch_module, protocol):
    with patch.object(launch_module, "protocol", return_value=protocol):
        days = launch_module.cells()
        branches = launch_module.pair_cells()
        tasks = [launch_module.make_job(**cell) for cell in days]
        tasks += [launch_module.make_job(mode="pair", **cell) for cell in branches]
    assert len(days) == 4 and len(branches) == 24
    assert {row["seed"] for row in branches} == {6201}
    assert {(row["scenario"], row["target_line"], row["anchor_index"], row["branch"])
            for row in branches} == {
        (scenario, line, anchor_index, branch) for scenario in protocol["scenarios"]
        for line in protocol["target_lines"] for anchor_index in (0, 1)
        for branch in ("low", "high")}
    assert len({task["signature"] for task in tasks}) == 28
    assert all(task["priority"] == "high" for task in tasks)
    assert all(task["skip_launch_staging"] is True for task in tasks)
    for row, task in zip(branches, tasks[4:]):
        argv = shlex.split(task["cmd"])
        expected = (HERE / "pairs" / row["scenario"] / row["target_line"] / "seed6201" /
                    f"state{row['anchor_index']}" / row["branch"])
        assert argv[argv.index("--out") + 1] == launch_module.infra.remote(expected)
        assert argv[argv.index("--mode") + 1] == "pair"
        assert argv[argv.index("--target-line") + 1] == row["target_line"]
        assert argv[argv.index("--branch") + 1] == row["branch"]
        assert argv[argv.index("--anchor-index") + 1] == str(row["anchor_index"])
        assert task["allowed_nodes"] == [f"node{i:03d}" for i in range(1, 7)]


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


def test_passed_days_allocate_only_twenty_four_branches(launch_module, protocol):
    rows = [dict(status="complete", all_pass=True, failed=False) for _ in range(4)]
    with patch.object(launch_module, "protocol", return_value=protocol), \
            patch.object(launch_module, "days_status", return_value=rows), \
            patch.object(launch_module.infra, "submit", return_value={"submitted": []}) as submit:
        launch_module.submit_pairs()
    tasks, batch = submit.call_args.args
    assert batch == "paired" and len(tasks) == 24
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
        assert argv[argv.index("--out") + 1] == launch_module.infra.remote(HERE / "interaction_summary.json")


def test_oversized_summary_is_rejected_before_copy(launch_module):
    with patch.object(launch_module.infra.scheduler, "run_on", return_value=(0, "200001\n", "")), \
            patch.object(launch_module.infra.subprocess, "run") as copy:
        with pytest.raises(RuntimeError, match="small-result budget"):
            launch_module.infra.collect_json("interaction_summary.json")
        copy.assert_not_called()


def test_staging_rejects_old_experiment_before_remote_write(launch_module):
    previous = HERE.parent / "mcwm_closed_loop_controlled_calibration_v10_20261021" / "run.py"
    with patch.object(launch_module.infra.scheduler, "run_on") as remote:
        with pytest.raises(ValueError):
            launch_module.infra.stage([previous])
        remote.assert_not_called()


def test_resident_readiness_requires_each_old_summary_and_full_prefix(launch_module, protocol,
                                                                    tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "source_summary.json").write_text('{"status":"complete"}')
    (source / "source_intervals.json").write_text(json.dumps([{}] * 102))
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in ("h2oplus_seed6201.pt", "h2oplus_seed6202.pt", "history.npz",
                 "history_metadata.json", "source_9101.npz"):
        (inputs / name).touch()
    modules = {
        "RESIDENT_V7": ["bootstrap.py"], "RESIDENT_V8": ["response_world.py"],
        "RESIDENT_V9": ["ablation_world.py"], "RESIDENT_V10": ["endpoints.py"],
        "RESIDENT_V11": ["residual.py", "paired_stream.py", "coverage.py", "prefix_trace.py", "run.py"],
    }
    monkeypatch.setattr(launch_module.infra, "RESIDENT_SOURCE", source)
    monkeypatch.setattr(launch_module.infra, "RESIDENT_INPUTS", inputs)
    for name, filenames in modules.items():
        directory = tmp_path / name
        directory.mkdir()
        for filename in filenames:
            (directory / filename).touch()
        monkeypatch.setattr(launch_module.infra, name, directory)
    reference = tmp_path / "reference"
    protocol["reference_v11_root"] = str(reference)
    for relative in [*(Path("runs") / launch_module.cell_path(row)
                       for row in launch_module.cells(protocol)), Path("preflight2/day")]:
        directory = reference / relative
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text('{}')
        (directory / "prefix_trace.jsonl").write_text('{}\n')
    truth_files = [reference / "pairs" / scenario / line / "seed6201" / branch / "summary.json"
                   for scenario in protocol["scenarios"] for line in protocol["target_lines"]
                   for branch in ("low", "high")]
    truth_files.extend(reference / "preflight2/pair" / branch / "summary.json"
                       for branch in ("low", "high"))
    for path in truth_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')

    def local_check(_node, command, **_kwargs):
        output = StringIO()
        with redirect_stdout(output):
            exec(shlex.split(command)[-1], {})
        return 0, output.getvalue(), ""

    with patch.object(launch_module, "protocol", return_value=protocol), \
            patch.object(launch_module.infra, "remote", side_effect=str), \
            patch.object(launch_module.infra.scheduler, "run_on", side_effect=local_check):
        ready = launch_module.source_ready()
        assert ready["reference_snapshot_files"] == 10
        assert ready["reference_paired_truth_files"] == 14
        truth_files[-1].unlink()
        with pytest.raises(AssertionError, match="preflight2/pair/high/summary"):
            launch_module.source_ready()
        truth_files[-1].write_text('{}')
        missing = reference / "runs/local_speed_drop/seed6202/prefix_trace.jsonl"
        missing.unlink()
        with pytest.raises(AssertionError, match="prefix_trace"):
            launch_module.source_ready()


def test_metadata_preserves_missing_branch_and_uses_current_preflight(launch_module, protocol, tmp_path, monkeypatch):
    protocol["preflight_directory"] = "preflight2"
    for relative in ("preflight", "preflight2/day", "preflight2/pair/low", "preflight2/pair/high",
                     "runs/stable/seed6201", "pairs/stable/7X/seed6201/state0/low"):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    (tmp_path / "preflight/failure.json").write_text('{"status":"failed"}')
    (tmp_path / "preflight2/day/gate.json").write_text('{"status":"complete"}')
    passed = '{"status":"complete","integrity":{"all_pass":true}}'
    (tmp_path / "runs/stable/seed6201/summary.json").write_text(passed)
    (tmp_path / "pairs/stable/7X/seed6201/state0/low/summary.json").write_text(passed)
    monkeypatch.setitem(sys.modules, "launch", launch_module)
    spec = importlib.util.spec_from_file_location("v12_test_continue", HERE / "continue_batch.py")
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


def test_pair_failure_is_retained_for_final_summary_without_relaxing_native_gates(launch_module,
                                                                                monkeypatch):
    monkeypatch.setitem(sys.modules, "launch", launch_module)
    spec = importlib.util.spec_from_file_location("v12_test_failure_continue", HERE / "continue_batch.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    value = dict(preflight_day_failed=False, preflight_pairs_failed=[],
                 failures=["pairs/stable/705X/seed6201/state1/low"], result_exists=True)
    with patch.object(driver, "metadata", return_value=value), patch.object(driver, "log"):
        with pytest.raises(RuntimeError, match="retain"):
            driver.wait_until("waiting_four_common_prefix_days", lambda v: True)
        assert driver.wait_until("waiting_native_pairs_and_summary", lambda v: v["result_exists"],
                                 allow_recorded_run_failures=True) == value
        value["preflight_pairs_failed"] = ["low"]
        with pytest.raises(RuntimeError, match="retain"):
            driver.wait_until("waiting_native_pairs_and_summary", lambda v: True,
                              allow_recorded_run_failures=True)
