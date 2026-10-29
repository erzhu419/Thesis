"""Budgeted stages must stay disjoint and scientific failure must stop expansion."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def driver(monkeypatch):
    infra = module('v21_driver_infra_test', HERE/'infra.py')
    monkeypatch.setitem(sys.modules, 'infra', infra)
    launch = module('v21_driver_launch_test', HERE/'launch.py')
    monkeypatch.setitem(sys.modules, 'launch', launch)
    continuation = module('v21_driver_continuation_test', HERE/'continue_batch.py')
    p = dict(name='v21-driver-test', preflight=dict(seed=6201, scenario='stable', target_seed_offset=79000),
        target_seed_offset=80000, training_seeds=[6201, 6202], scenarios=['stable', 'local_speed_drop'],
        target_lines=['7X', '102X', '705X'], heldout_windows=[dict(name='heldout1'), dict(name='heldout2')])
    monkeypatch.setattr(launch, 'protocol', lambda:p)
    return launch, continuation


def test_four_day_plan_and_48_heldout_branches_are_disjoint_and_high(driver, monkeypatch):
    launch, _ = driver
    calls = []
    monkeypatch.setattr(launch, 'source_ready', lambda:None)
    monkeypatch.setattr(launch, 'require_gate', lambda *a, **k:None)
    monkeypatch.setattr(launch, 'days_status', lambda:[dict(status='complete', all_pass=True, coverage=True)]*3)
    monkeypatch.setattr(launch.infra, 'submit', lambda rows, batch:calls.append((batch, rows)))
    launch.submit_preflight()
    launch.submit_service_pairs()
    launch.submit_response_days()
    launch.submit_response_pairs()
    assert [(batch, len(rows)) for batch, rows in calls] == [
        ('preflight', 1), ('service_pairs', 12), ('response_days', 3), ('response_pairs', 36)]
    jobs = [row for _, rows in calls for row in rows]
    assert len({row['signature'] for row in jobs}) == 52
    assert all(row['priority'] == 'high' and row['no_auto_requeue'] is True for row in jobs)
    assert all(row['allowed_nodes'] == [f'node{i:03d}' for i in range(1, 7)] for row in jobs)
    assert all('--preflight' in row['cmd'] for _, rows in calls[:2] for row in rows)
    assert all('--preflight' not in row['cmd'] for _, rows in calls[2:] for row in rows)
    assert set(tuple(sorted(c.items())) for c in launch.cells(preflight=True)).isdisjoint(
        tuple(sorted(c.items())) for c in launch.cells())
    # All siblings refer to a main flow from their own stage; A is never replayed into B.
    assert all('stage_a/runs/' in row['cmd'] and 'stage_a/pairs/' in row['cmd'] for row in calls[1][1])
    assert all('stage_b/runs/' in row['cmd'] and 'stage_b/pairs/' in row['cmd'] for row in calls[3][1])


@pytest.mark.parametrize('scientific', [None, dict(passed=False)])
def test_integrity_pass_alone_never_admits_response_training(driver, monkeypatch, scientific):
    launch, _ = driver
    value = dict(status='complete', integrity=dict(all_pass=True))
    if scientific is not None:
        value['scientific_gate'] = scientific
    monkeypatch.setattr(launch.infra, 'collect', lambda *a, **k:value)
    submitted = []
    monkeypatch.setattr(launch.infra, 'submit', lambda *a:submitted.append(a))
    with pytest.raises(RuntimeError, match='scientific gate'):
        launch.submit_response_days()
    assert submitted == []


def test_missing_a_coverage_blocks_service_pairs(driver, monkeypatch):
    launch, _ = driver
    monkeypatch.setattr(launch.infra, 'collect', lambda *a, **k:dict(status='complete',
        integrity=dict(all_pass=True), design_coverage=dict(all_pass=False)))
    with pytest.raises(RuntimeError, match='coverage'):
        launch.submit_service_pairs()


def test_incomplete_b_days_block_response_pairs(driver, monkeypatch):
    launch, _ = driver
    monkeypatch.setattr(launch, 'require_gate', lambda *a, **k:None)
    monkeypatch.setattr(launch, 'days_status', lambda:[dict(status='complete', all_pass=True, coverage=True),
        dict(status='pending'), dict(status='complete', all_pass=True, coverage=True)])
    with pytest.raises(RuntimeError, match='response days'):
        launch.submit_response_pairs()


def test_coordinator_stops_normally_after_scientific_rejection(driver, monkeypatch):
    launch, continuation = driver
    monkeypatch.setattr(continuation, 'exists', lambda name:name in ('preflight', 'service_pairs', 'service_finalize'))
    monkeypatch.setattr(continuation, 'wait_until', lambda *a, **k:{})
    monkeypatch.setattr(continuation.infra, 'collect', lambda *a, **k:dict(status='complete',
        integrity=dict(all_pass=True), scientific_gate=dict(passed=False, reason='history not predictive')))
    called = []
    monkeypatch.setattr(launch, 'submit_response_days', lambda:called.append('response_days'))
    logs = []
    monkeypatch.setattr(continuation, 'log', lambda stage, **kw:logs.append((stage, kw)))
    continuation.main()
    assert called == []
    assert logs[-1][0] == 'scientific_gate_stopped'
    assert logs[-1][1]['response_days_submitted'] is False


def test_resident_metadata_retains_failure_and_stage_paths(driver, monkeypatch, tmp_path):
    launch, continuation = driver
    monkeypatch.setattr(continuation.infra, 'remote', lambda _:str(tmp_path))
    monkeypatch.setattr(continuation.infra, 'PYTHON', sys.executable)
    first = tmp_path/launch.day_path(launch.cells(preflight=True)[0], preflight=True)
    first.mkdir(parents=True)
    (first/'summary.json').write_text(json.dumps(dict(status='complete', integrity=dict(all_pass=True),
        design_coverage=dict(all_pass=True))))
    (first/'gate.json').write_text('{}')
    failed = tmp_path/launch.evaluation_path(launch.evaluation_cells(preflight=True)[0], 'low', preflight=True)
    failed.mkdir(parents=True)
    (failed/'failure.json').write_text(json.dumps(dict(type='RuntimeError', message='prefix differs', time=19000)))
    def run_on(node, command, **kwargs):
        return 0, subprocess.check_output(command, shell=True, text=True), ''
    monkeypatch.setattr(continuation.infra.scheduler, 'run_on', run_on)
    value = continuation.metadata()
    assert value['first']['coverage'] is True and value['first_gate'] is True
    assert value['service_pairs_terminal'] == 1
    assert value['service_pairs_bad'][0]['failure']['message'] == 'prefix differs'
    assert value['response_days_terminal'] == value['response_pairs_terminal'] == 0


def test_existing_evidence_is_not_replaced(driver, tmp_path):
    launch, _ = driver
    path = tmp_path/'result.json'
    launch.infra.write(path, dict(first=True))
    with pytest.raises(FileExistsError):
        launch.infra.write(path, dict(second=True))
    assert json.loads(path.read_text()) == dict(first=True)
