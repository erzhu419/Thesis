"""Training admission and recovery must not create unbudgeted native tasks."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def driver(monkeypatch):
    infra = module('v23_driver_infra', HERE/'infra.py')
    monkeypatch.setitem(sys.modules, 'infra', infra)
    launch = module('v23_driver_launch', HERE/'launch.py')
    monkeypatch.setitem(sys.modules, 'launch', launch)
    continuation = module('v23_driver_continuation', HERE/'continue_batch.py')
    monkeypatch.setattr(launch, 'protocol', lambda:dict(name='v23-test',
        preflight=dict(seed=6201, scenario='stable'), target_lines=['7X','102X','705X'],
        heldout_windows=[dict(name='heldout1'), dict(name='heldout2')]))
    return launch, continuation


def native_gate(*, training=True):
    return dict(status='complete', integrity=dict(all_pass=True),
        design_coverage=dict(all_pass=True), training_gate=dict(passed=training),
        service_validation_eligible=training)


def test_tasks_reserve_parallel_cpus_and_six_two_arm_service_tasks(driver):
    launch, _ = driver
    day = launch.day_job()
    assert day['cpu'] == 2 and day['ram_mb'] == 24000
    assert '--preflight' in day['cmd']
    branches = [launch.service_pair_job(c) for c in launch.evaluation_cells()]
    assert len(branches) == len({r['signature'] for r in branches}) == 6
    assert all(r['cpu'] == 2 and r['ram_mb'] == 24000 and '--preflight' in r['cmd']
        and 'run_pair.py' in r['cmd'] for r in branches)
    assert all(r['priority'] == 'high' and r['no_auto_requeue'] is True
        for r in [day, *branches])
    assert not hasattr(launch, 'submit_response_days')


def test_failed_training_gate_cannot_submit_heldouts(driver, monkeypatch):
    launch, _ = driver
    monkeypatch.setattr(launch.infra, 'collect', lambda *a, **k:native_gate(training=False))
    monkeypatch.setattr(launch.infra, 'submit', lambda *a, **k:pytest.fail('No heldouts after training rejection'))
    with pytest.raises(RuntimeError, match='scoreable reversal'):
        launch.submit_service_pairs()


def test_incomplete_coverage_still_gets_training_analysis_but_no_heldouts(driver, monkeypatch):
    launch, _ = driver
    gate = native_gate()
    gate['design_coverage']['all_pass'] = False
    monkeypatch.setattr(launch.infra, 'collect', lambda *a, **k:gate)
    submitted = []
    monkeypatch.setattr(launch.infra, 'submit', lambda rows, batch:submitted.append((batch, rows)))
    launch.submit_finalizer(stage='train')
    assert submitted[0][0] == 'training_finalize'
    assert '--stage train' in submitted[0][1][0]['cmd']
    with pytest.raises(RuntimeError, match='coverage is incomplete'):
        launch.submit_service_pairs()
    assert len(submitted) == 1


def test_training_rejection_is_terminal_and_reuses_result(driver, monkeypatch, tmp_path):
    launch, c = driver
    monkeypatch.setattr(c, 'HERE', tmp_path)
    monkeypatch.setattr(c, 'exists', lambda _:True)
    monkeypatch.setattr(c, 'wait_until', lambda *a, **k:{})
    monkeypatch.setattr(c, 'collect_small', lambda *a, **k:native_gate(training=False))
    monkeypatch.setattr(launch, 'submit_service_pairs', lambda:pytest.fail('Rejected training must stop'))
    c.main()
    value = json.loads((tmp_path/'continuation_result.json').read_text())
    assert value['status'] == 'stopped_training_gate'
    assert value['service_pairs_submitted'] is False and value['response_days_submitted'] is False
    monkeypatch.setattr(c, 'wait_until', lambda *a, **k:pytest.fail('Terminal coordinator must not poll or submit'))
    c.main()


def test_existing_receipts_reuse_twelve_heldouts_and_never_submit_b(driver, monkeypatch, tmp_path):
    _, c = driver
    monkeypatch.setattr(c, 'HERE', tmp_path)
    monkeypatch.setattr(c, 'exists', lambda _:True)
    monkeypatch.setattr(c, 'wait_until', lambda *a, **k:{})
    monkeypatch.setattr(c, 'collect_small', lambda *a, **k:native_gate())
    monkeypatch.setattr(c.infra, 'submit', lambda *a, **k:pytest.fail('Receipts already exist'))
    c.main()
    value = json.loads((tmp_path/'continuation_result.json').read_text())
    assert value['status'] == 'complete_service_validation'
    assert value['response_days_submitted'] is False


def test_metadata_retains_real_failure_without_simulation_retry(driver, monkeypatch, tmp_path):
    launch, c = driver
    monkeypatch.setattr(c, 'HERE', tmp_path)
    monkeypatch.setattr(c.infra, 'remote', str)
    monkeypatch.setattr(c, 'readonly_remote', lambda code:subprocess.check_output(
        [sys.executable, '-c', code], text=True))
    day = tmp_path/launch.day_path()
    day.mkdir(parents=True)
    (day/'summary.json').write_text(json.dumps(native_gate()))
    (day/'gate.json').write_text(json.dumps(native_gate()))
    (tmp_path/'training_gate.json').write_text(json.dumps(native_gate()))
    bad = tmp_path/launch.evaluation_path(launch.evaluation_cells()[0], 'low')
    bad.mkdir(parents=True)
    (bad/'failure.json').write_text(json.dumps(dict(type='RuntimeError', message='prefix differs')))
    value = c.metadata()
    assert value['training_gate']['training_passed'] is True
    assert value['service_pairs']['terminal'] == 1
    assert value['service_pairs']['bad'][0]['status'] == 'failed'
    monkeypatch.setattr(c, 'metadata', lambda:value)
    with pytest.raises(RuntimeError, match='no automatic simulation retry'):
        c.wait_until('test', lambda _:True)
