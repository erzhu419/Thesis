"""Exercise admission gates and task contracts without contacting the scheduler."""
import importlib.util
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.setattr(sys, 'path', list(sys.path))
    infra = load('v16_driver_test_infra', 'infra.py')
    monkeypatch.setitem(sys.modules, 'infra', infra)
    launch = load('v16_driver_test_launch', 'launch.py')
    monkeypatch.setitem(sys.modules, 'launch', launch)
    continuation = load('v16_driver_test_continuation', 'continue_batch.py')
    calls = []

    def submit(jobs, name):
        calls.append((name, jobs))
        return dict(submitted=[dict(id=f'mock-{name}-{i}') for i in range(len(jobs))])

    def network_forbidden(*args, **kwargs):
        raise AssertionError('Driver tests must not contact the scheduler or collect resident files')

    monkeypatch.setattr(infra, 'submit', submit)
    monkeypatch.setattr(infra.scheduler, 'run_on', network_forbidden)
    monkeypatch.setattr(infra, 'collect_json', network_forbidden)
    return SimpleNamespace(infra=infra, launch=launch, continuation=continuation, calls=calls)


@pytest.mark.parametrize('gate', [
    dict(status='pending', integrity=dict(all_pass=True)),
    dict(status='complete', integrity=dict(all_pass=False)),
    dict(status='complete', integrity={}),
])
def test_four_days_require_complete_passing_native_preflight(driver, monkeypatch, tmp_path, gate):
    monkeypatch.setattr(driver.launch, 'HERE', tmp_path)
    path = tmp_path / 'preflight/gate.json'
    path.parent.mkdir()
    path.write_text(json.dumps(gate))
    with pytest.raises(RuntimeError, match='gate must pass'):
        driver.launch.submit_days()
    assert driver.calls == []


def test_missing_preflight_gate_cannot_submit_four_days(driver, monkeypatch, tmp_path):
    monkeypatch.setattr(driver.launch, 'HERE', tmp_path)

    def missing(relative, *, max_bytes):
        assert relative == 'preflight/gate.json' and max_bytes == 20000
        raise FileNotFoundError(relative)

    monkeypatch.setattr(driver.infra, 'collect_json', missing)
    with pytest.raises(FileNotFoundError):
        driver.launch.submit_days()
    assert driver.calls == []


@pytest.mark.parametrize('bad_day', [
    dict(status='pending', all_pass=None),
    dict(status='complete', all_pass=False),
    dict(status='failed', all_pass=False),
])
def test_heldout_requires_all_four_completed_passing_days(driver, monkeypatch, bad_day):
    rows = [dict(status='complete', all_pass=True) for _ in range(4)]
    rows[2] = bad_day
    monkeypatch.setattr(driver.launch, 'days_status', lambda: rows)
    with pytest.raises(RuntimeError, match='All new main days must pass'):
        driver.launch.submit_evaluations()
    assert driver.calls == []


def test_new_day_and_heldout_scope_and_label_budget(driver):
    launch = driver.launch
    p = launch.protocol()
    assert launch.cells() == [dict(seed=seed, scenario=scene)
        for seed in (6201, 6202) for scene in ('stable', 'local_speed_drop')]
    assert len(launch.evaluation_cells()) == 24
    assert len({str(launch.evaluation_path(cell, arm))
        for cell in launch.evaluation_cells() for arm in ('low', 'high')}) == 48
    assert len(launch.evaluation_cells(preflight=True)) == 1
    assert p['horizon_seconds'] == 24000 and p['initializer_updates'] == 50000
    assert (p['history_rows'], p['source_rows']) == (32768, 4096)
    packets = len(range(p['feedback_first_deadline'], p['feedback_last_deadline'] + 1,
                        p['feedback_interval_seconds']))
    assert packets == 57
    assert packets * p['feedback_records_per_packet'] == p['target_record_budget'] == 285
    assert packets * p['policy_updates_per_packet'] == p['policy_update_budget'] == 456
    assert p['training_pair_slots_per_day_maximum'] == 9 * 2 == 18
    assert p['ordinary_slots_when_complete'] + 18 == 285
    assert p['compute_budget']['main_native_seconds'] == 4 * 24000
    assert p['compute_budget']['training_branches_maximum'] == 4 * 9 * 2
    assert p['compute_budget']['heldout_native_suffix_seconds'] == 24 * 2 * 300


def assert_task(job, infra, *, ram_mb, runner='run.py'):
    assert job['priority'] == 'high'
    assert (job['cpu'], job['ram_mb'], job['vram']) == (1, ram_mb, 0)
    assert job['allowed_nodes'] == [f'node{i:03d}' for i in range(1, 7)]
    assert job['cwd'] == str(HERE)
    assert job['allow_cpu_training'] is True and job['skip_launch_staging'] is True
    assert all(job['env'][key] == '1'
               for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))
    command = shlex.split(job['cmd'])
    assert command[:3] == [infra.PYTHON, '-u', infra.remote(HERE / runner)]
    return command[3:]


def test_all_native_jobs_keep_high_resources_modes_sources_and_exact_output_paths(driver):
    launch, infra = driver.launch, driver.infra
    for preflight in (False, True):
        for cell in launch.cells(preflight=preflight):
            args = assert_task(launch.day_job(cell, preflight=preflight), infra, ram_mb=16000)
            expected = ['--mode', 'day', '--seed', str(cell['seed']), '--scenario', cell['scenario'],
                        '--out', infra.remote(HERE / launch.day_path(cell, preflight=preflight))]
            assert args == expected + (['--preflight'] if preflight else [])
        for cell in launch.evaluation_cells(preflight=preflight):
            for arm in ('low', 'high'):
                args = assert_task(launch.evaluation_job(cell, arm, preflight=preflight), infra, ram_mb=8000)
                expected = ['--mode', 'branch', '--seed', str(cell['seed']), '--scenario', cell['scenario'],
                    '--source', infra.remote(HERE / launch.day_path(cell, preflight=preflight)),
                    '--target-id', f"heldout/{cell['line']}/{cell['window']}", '--branch', arm,
                    '--out', infra.remote(HERE / launch.evaluation_path(cell, arm, preflight=preflight))]
                assert args == expected + (['--preflight'] if preflight else [])
    assert driver.calls == []


def test_successful_gates_submit_exactly_four_days_and_48_heldout_branches(driver, monkeypatch):
    launch = driver.launch
    gates = []
    monkeypatch.setattr(launch, 'require_gate', lambda relative: gates.append(str(relative)))
    monkeypatch.setattr(launch, 'days_status', lambda: [dict(status='complete', all_pass=True)] * 4)
    launch.submit_days()
    launch.submit_evaluations()
    assert gates == ['preflight/gate.json']
    assert [(name, len(jobs)) for name, jobs in driver.calls] == [('diagnostic', 4), ('paired', 48)]


def test_finalizers_use_high_small_cpu_jobs_and_bounded_resident_summary_paths(driver, monkeypatch):
    original_read = Path.read_text

    def read(path, *args, **kwargs):
        if path in (HERE / 'preflight_pairs_submission.json', HERE / 'paired_submission.json'):
            return '{"submitted": []}'
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read)
    for preflight in (True, False):
        driver.launch.submit_finalizer(preflight=preflight)
        name, jobs = driver.calls[-1]
        assert name == ('preflight_finalize' if preflight else 'diagnostic_finalize')
        assert len(jobs) == 1
        args = assert_task(jobs[0], driver.infra, ram_mb=1024, runner='analyze.py')
        root = HERE / 'preflight' if preflight else HERE
        out = root / 'gate.json' if preflight else root / 'learning_summary.json'
        assert args == ['--root', driver.infra.remote(root), '--out', driver.infra.remote(out), '--wait'] \
            + (['--preflight'] if preflight else [])
    assert driver.launch.SUMMARY_MAX_BYTES == 150000


def snapshot(**updates):
    value = dict(preflight_day=dict(status='complete', all_pass=True), preflight_day_gate=True,
        preflight_pairs_terminal=0, preflight_pairs_bad=[], preflight_gate=False,
        days_complete=0, days_passed=0, days_bad=[], pairs_terminal=0, pairs_bad=[],
        failures=[], result_exists=False)
    value.update(updates)
    return value


def test_continuation_waits_before_submitting_incomplete_days_heldout(driver, monkeypatch, tmp_path):
    run = driver.continuation
    monkeypatch.setattr(run, 'HERE', tmp_path)
    monkeypatch.setattr(run, 'exists', lambda batch: batch in
        {'preflight', 'preflight_pairs', 'preflight_finalize', 'diagnostic'})
    monkeypatch.setattr(run, 'metadata', lambda: snapshot(preflight_gate=True, days_complete=3, days_passed=3))

    class Waiting(Exception):
        pass

    def pause(seconds):
        assert seconds == 60
        raise Waiting

    monkeypatch.setattr(run.time, 'sleep', pause)
    with pytest.raises(Waiting):
        run.main()
    assert driver.calls == []


def test_continuation_requires_preflight_and_obeys_full_submission_order(driver, monkeypatch, tmp_path):
    run, launch = driver.continuation, driver.launch
    monkeypatch.setattr(run, 'HERE', tmp_path)
    committed, events = set(), []
    monkeypatch.setattr(run, 'exists', lambda batch: batch in committed)
    with pytest.raises(RuntimeError, match='Submit native preflight day first'):
        run.main()
    assert events == [] and driver.calls == []
    committed.add('preflight')

    def submitted(batch):
        events.append(batch)
        committed.add(batch)
        return dict(submitted=[dict(id='mock-' + batch)])

    monkeypatch.setattr(launch, 'submit_evaluations', lambda *, preflight=False:
                        submitted('preflight_pairs' if preflight else 'paired'))
    monkeypatch.setattr(launch, 'submit_finalizer', lambda *, preflight=False:
                        submitted('preflight_finalize' if preflight else 'diagnostic_finalize'))
    monkeypatch.setattr(launch, 'submit_days', lambda: submitted('diagnostic'))
    result = dict(status='complete', integrity=dict(all_pass=True), bytes=100)
    states = iter([snapshot(), snapshot(preflight_gate=True),
        snapshot(preflight_gate=True, days_complete=4, days_passed=4),
        snapshot(preflight_gate=True, days_complete=4, days_passed=4, pairs_terminal=48,
                 result_exists=True, result=result)])
    monkeypatch.setattr(run, 'metadata', lambda: next(states))
    monkeypatch.setattr(run.time, 'sleep', lambda _: pytest.fail('Every supplied stage is ready'))
    (tmp_path / launch.RESULT_NAME).write_text(json.dumps(result))
    run.main()
    assert events == ['preflight_pairs', 'preflight_finalize', 'diagnostic', 'paired', 'diagnostic_finalize']
    assert driver.calls == []
