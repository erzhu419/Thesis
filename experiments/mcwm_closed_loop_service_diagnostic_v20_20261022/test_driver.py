"""The first observational suffix consumes one of six and cannot be duplicated."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('v20_service_launch_test', HERE/'launch.py')
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


def test_preflight_and_remaining_are_six_unique_fixed_branches(monkeypatch):
    calls = []
    monkeypatch.setattr(launch.infra, 'collect', lambda *args, **kwargs:
        dict(status='complete', integrity=dict(all_pass=True)))
    monkeypatch.setattr(launch.infra, 'submit', lambda rows, name: calls.append((name, rows)))
    launch.submit_preflight()
    launch.submit_remaining()
    assert [len(rows) for _, rows in calls] == [1, 5]
    jobs = [row for _, rows in calls for row in rows]
    assert len({row['signature'] for row in jobs}) == 6
    assert sum('branches/native/7X/low' in row['signature'] for row in jobs) == 1
    assert all(row['priority'] == 'high' and row['no_auto_requeue'] is True for row in jobs)
    assert all(row['allowed_nodes'] == [f'node{x:03d}' for x in range(1, 7)] for row in jobs)
    assert all('--mapping tail' not in row['cmd'] and '--source' in row['cmd'] for row in jobs)
    p = launch.protocol()
    assert p['diagnostic_compute_budget']['new_branch_queries_maximum'] == len(jobs)
    assert p['suffix_seconds']*len(jobs) == 1800
    assert p['invariants']['training_labels_per_day'] == 285


def test_failed_preflight_stops_remaining_five(monkeypatch):
    monkeypatch.setattr(launch.infra, 'collect', lambda *args, **kwargs:
        dict(status='complete', integrity=dict(all_pass=False)))
    submitted = []
    monkeypatch.setattr(launch.infra, 'submit', lambda *args: submitted.append(args))
    with pytest.raises(RuntimeError, match='preflight must pass'):
        launch.submit_remaining()
    assert submitted == []


def test_receipt_and_result_writes_preserve_existing_evidence(tmp_path):
    path = tmp_path/'result.json'
    launch.infra.write(path, dict(first=True))
    with pytest.raises(FileExistsError):
        launch.infra.write(path, dict(second=True))
    assert json.loads(path.read_text()) == dict(first=True)
