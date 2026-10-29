"""Read retries may recover SSH; they must never re-dispatch native work."""
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
    infra = module('parallel_continuation_infra', HERE/'infra.py')
    monkeypatch.setitem(sys.modules, 'infra', infra)
    launch = module('parallel_continuation_launch', HERE/'launch_parallel.py')
    monkeypatch.setitem(sys.modules, 'launch_parallel', launch)
    continuation = module('parallel_continuation', HERE/'continue_parallel.py')
    monkeypatch.setattr(launch, 'protocol', lambda:dict(name='parallel-test', target_lines=['7X','102X','705X'],
        heldout_windows=[dict(name='heldout1'),dict(name='heldout2')]))
    return launch, continuation


def native_gate(*, scientific=True):
    return dict(status='complete', integrity=dict(all_pass=True), scientific_gate=dict(passed=scientific))


def test_only_rc255_reads_retry_then_return(driver, monkeypatch):
    _, c = driver
    calls, sleeps, logs = [], [], []
    def run_on(*args, **kwargs):
        calls.append(args)
        if len(calls)<3:
            raise RuntimeError('[node001] cmd failed (rc=255): kex_exchange_identification: Connection closed')
        return 0, '{}', ''
    monkeypatch.setattr(c.infra.scheduler, 'run_on', run_on)
    monkeypatch.setattr(c.time, 'sleep', sleeps.append)
    monkeypatch.setattr(c, 'log', lambda stage, **kw:logs.append((stage,kw)))
    assert c.readonly_remote('print("{}")')=='{}'
    assert len(calls)==3 and sleeps==[45,45]
    assert [x[1]['retry'] for x in logs]==[1,2]


def test_ssh_read_retry_is_bounded_and_never_submits(driver, monkeypatch):
    _, c = driver
    calls, sleeps = [], []
    def run_on(*args, **kwargs):
        calls.append(args)
        raise RuntimeError('[node001] cmd failed (rc=255): Connection closed')
    monkeypatch.setattr(c.infra.scheduler, 'run_on', run_on)
    monkeypatch.setattr(c.time, 'sleep', sleeps.append)
    monkeypatch.setattr(c.infra, 'submit', lambda *a:pytest.fail('A failed read must not dispatch work'))
    with pytest.raises(RuntimeError, match='rc=255'):
        c.readonly_remote('print("{}")')
    assert len(calls)==4 and sleeps==[45,45,45]


def test_remote_python_error_and_bad_json_do_not_retry(driver, monkeypatch):
    _, c = driver
    calls = []
    def failure(*args, **kwargs):
        calls.append(1)
        raise RuntimeError('[node001] cmd failed (rc=1): invalid native summary')
    monkeypatch.setattr(c.infra.scheduler, 'run_on', failure)
    monkeypatch.setattr(c.time, 'sleep', lambda _:pytest.fail('Only SSH transport may retry'))
    with pytest.raises(RuntimeError, match='rc=1'):
        c.readonly_remote('raise ValueError()')
    assert len(calls)==1
    calls.clear()
    def malformed(*args, **kwargs):
        calls.append(1); return 0, '{bad json', ''
    monkeypatch.setattr(c.infra.scheduler, 'run_on', malformed)
    with pytest.raises(json.JSONDecodeError):
        c.metadata()
    assert len(calls)==1


def test_scientific_rejection_is_terminal_without_b_days(driver, monkeypatch, tmp_path):
    launch, c = driver
    monkeypatch.setattr(c, 'HERE', tmp_path)
    monkeypatch.setattr(c, 'wait_until', lambda *a, **k:{})
    monkeypatch.setattr(c, 'collect_small', lambda source, name, limit:
        native_gate(scientific=False) if name=='service_gate.json' else native_gate())
    monkeypatch.setattr(launch, 'submit_response_days', lambda:pytest.fail('Scientific rejection must stop B'))
    c.main()
    result = json.loads((tmp_path/'continuation_result.json').read_text())
    assert result['status']=='stopped_scientific_gate'
    assert result['response_days_submitted'] is False


def test_existing_receipts_reuse_all_submissions(driver, monkeypatch, tmp_path):
    launch, c = driver
    monkeypatch.setattr(c, 'HERE', tmp_path)
    monkeypatch.setattr(c, 'exists', lambda _:True)
    monkeypatch.setattr(c, 'wait_until', lambda *a, **k:{})
    monkeypatch.setattr(c, 'collect_small', lambda *a, **k:native_gate())
    monkeypatch.setattr(c.infra, 'submit', lambda *a:pytest.fail('Existing receipts must not be resubmitted'))
    c.main()
    assert json.loads((tmp_path/'continuation_result.json').read_text())['status']=='complete'
    # A finished coordinator is also safe to invoke again.
    c.main()


def test_response_submission_is_36_unique_one_cpu_native_branches(driver, monkeypatch):
    _, c = driver
    calls = []
    monkeypatch.setattr(c.infra, 'submit', lambda rows, batch:calls.append((batch,rows)))
    c.submit_response_pairs()
    batch, rows = calls[0]
    assert batch=='response_pairs' and len(rows)==36
    assert len({r['signature'] for r in rows})==36
    assert all(r['priority']=='high' and r['no_auto_requeue'] is True and r['cpu']==1 for r in rows)
    assert all('stage_b/runs/' in r['cmd'] and 'stage_b/pairs/' in r['cmd'] for r in rows)


def test_small_service_gate_is_copied_only_into_review(driver, monkeypatch, tmp_path):
    _, c = driver
    old = tmp_path/'old'
    old.mkdir()
    source = old/'service_gate.json'
    source.write_text(json.dumps(native_gate()))
    root = tmp_path/'review'
    root.mkdir()
    monkeypatch.setattr(c, 'HERE', root)
    monkeypatch.setattr(c.infra, 'remote', lambda path:str(path))
    monkeypatch.setattr(c, 'readonly_remote', lambda code:subprocess.check_output([sys.executable,'-c',code],text=True))
    before = source.read_text()
    result = c.collect_small(source, 'service_gate.json', 20000)
    assert result['scientific_gate']['passed'] is True
    assert source.read_text()==before
    assert json.loads((root/'service_gate.json').read_text())==result


def test_metadata_executes_read_only_and_retains_real_failures(driver, monkeypatch, tmp_path):
    _, c = driver
    old, root = tmp_path/'old', tmp_path/'review'
    monkeypatch.setattr(c, 'HERE', root)
    monkeypatch.setattr(c, 'ORIGINAL', old)
    monkeypatch.setattr(c.infra, 'remote', lambda path:str(path))
    monkeypatch.setattr(c, 'readonly_remote', lambda code:subprocess.check_output([sys.executable,'-c',code],text=True))
    a = old/'stage_a/runs/stable/seed6201'
    a.mkdir(parents=True)
    (a/'summary.json').write_text(json.dumps(native_gate()))
    old.joinpath('service_gate.json').write_text(json.dumps(native_gate(scientific=False)))
    parity = root/'parity/low'
    parity.mkdir(parents=True)
    (parity/'failure.json').write_text(json.dumps(dict(type='RuntimeError', message='prefix differs')))
    value = c.metadata()
    assert value['service_day']['all_pass'] is True
    assert value['service_gate']['scientific_passed'] is False
    assert value['parity']['terminal']==1 and value['parity']['bad'][0]['status']=='failed'
    assert value['response_days']['terminal']==value['response_pairs']['terminal']==0
    monkeypatch.setattr(c, 'metadata', lambda:value)
    with pytest.raises(RuntimeError, match='no automatic simulation retry'):
        c.wait_until('test', lambda _:True)
