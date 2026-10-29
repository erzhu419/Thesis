"""Pair execution order may change; scientific inputs and result order may not."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def runner(monkeypatch):
    for name in ('bootstrap', 'history', 'shift_model', 'candidate'):
        monkeypatch.setitem(sys.modules, name, load('v21_parallel_'+name, HERE/(name+'.py')))
    return load('v21_parallel_runner', HERE/'parallel_run.py')


def transition(action):
    return dict(s=[0.]*229, a=[action], ns=[1.]*229, r=-1., dt=2., terminal=False,
        episode_id=0, domain='target', started_at=1250., completed_at=1252., available_at=1282.,
        vehicle_id='bus', line_id='7X', station_id='stop', next_vehicle_id='other', sequence=20,
        raw_s=[0.]*229, raw_ns=[1.]*229, trip_id='trip')


def row():
    return dict(id='train/7X/early', index=0, role='train', line_id='7X', features=[1.,0.,0.,.2],
        selected=dict(sequence=20, started_at=1250.),
        service_history=dict(status='supported', mean_residual_scaled=.1))


def write_summary(cmd):
    arm = cmd[cmd.index('--branch')+1]
    out = Path(cmd[cmd.index('--out')+1])
    out.mkdir(parents=True)
    value = dict(status='complete', integrity=dict(all_pass=True), selected=dict(sequence=20),
        prefix=dict(receipts=[]), prediction=dict(shared='unsupported'),
        native=dict(other_batch_actions=[]),
        truth=dict(fixed300=dict(status='complete', dt=300., wait=100. if arm=='low' else 103.,
            ride=200. if arm=='low' else 190., cost=400. if arm=='low' else 396.)),
        global_transition=transition(-1. if arm=='low' else -.75),
        service_shift=dict(stations=['a','b','c'], arrival=[1.,2.,3.] if arm=='low' else [8.,9.,10.],
            departure=[2.,3.,4.] if arm=='low' else [9.,10.,11.]))
    (out/'summary.json').write_text(json.dumps(value))
    return arm


def test_launches_both_before_wait_and_returns_low_high_even_when_high_finishes_first(runner, monkeypatch, tmp_path):
    children, completed, waited = [], [], []
    class Child:
        def __init__(self, cmd, stdout, stderr):
            assert not waited
            self.args, self.pid, self.state, self.calls = cmd, 100+len(children), None, 0
            self.arm = write_summary(cmd)
            stdout.write(self.arm+' launched\n')
            children.append(self)
        def wait(self, timeout=None):
            assert len(children)==2
            waited.append(self.arm); self.calls += 1
            if self.arm=='low' and self.calls==1:
                raise subprocess.TimeoutExpired(self.args, timeout)
            self.state=0; completed.append(self.arm); return 0
        def poll(self): return self.state
        def terminate(self): raise AssertionError('A successful pair must not be terminated')
    monkeypatch.setattr(runner.subprocess, 'Popen', Child)
    selected = row(); original = deepcopy(selected)
    (tmp_path/'prefix_trace.jsonl').write_text('existing native prefix\n')
    pair = runner.acquire_native_pair(SimpleNamespace(seed=6201, scenario='stable', preflight=True), tmp_path, selected)
    assert completed==['high','low']
    assert list(pair['outcomes'])==['low','high']
    assert pair['outcomes']['low']['truth']['wait']==100.
    assert pair['outcomes']['high']['truth']['wait']==103.
    assert pair['service_targets']['arrival']==[-.5,-.5,-.5]
    assert selected==original  # Waiting wall time cannot advance the fixed issue clock.
    assert (tmp_path/'prefix_trace.jsonl').read_text()=='existing native prefix\n'
    assert all(Path(c.args[2]).name=='parallel_run.py' and '--preflight' in c.args for c in children)
    assert all(c.state==0 for c in children)
    assert (tmp_path/'training_branches/0/low.log').read_text()=='low launched\n'
    assert (tmp_path/'training_branches/0/high.log').read_text()=='high launched\n'


@pytest.mark.parametrize('needs_kill', [False, True])
def test_failed_sibling_reaps_live_child_and_preserves_logs(runner, monkeypatch, tmp_path, needs_kill):
    children = []
    class Child:
        def __init__(self, cmd, stdout, stderr):
            self.args, self.pid, self.state = cmd, 100+len(children), None
            self.arm = cmd[cmd.index('--branch')+1]
            self.terminated, self.killed, self.reaped = False, False, False
            stdout.write(self.arm+' started\n')
            children.append(self)
        def wait(self, timeout=None):
            assert len(children)==2
            if self.arm=='high':
                self.state=3; self.reaped=True; return 3
            if self.killed:
                self.state=-9; self.reaped=True; return -9
            if self.terminated and not needs_kill:
                self.state=-15; self.reaped=True; return -15
            raise subprocess.TimeoutExpired(self.args, timeout)
        def poll(self): return self.state
        def terminate(self): self.terminated=True
        def kill(self): self.killed=True
    monkeypatch.setattr(runner.subprocess, 'Popen', Child)
    with pytest.raises(subprocess.CalledProcessError) as failure:
        runner.acquire_native_pair(SimpleNamespace(seed=6201, scenario='stable', preflight=True), tmp_path, row())
    assert failure.value.returncode==3
    low, high = children
    assert low.terminated and low.reaped and low.killed==needs_kill
    assert high.reaped and not high.terminated
    assert all(c.poll() is not None for c in children)
    assert (tmp_path/'training_branches/0/low.log').read_text()=='low started\n'
    assert (tmp_path/'training_branches/0/high.log').read_text()=='high started\n'


def test_second_launch_failure_reaps_first_child(runner, monkeypatch, tmp_path):
    children = []
    class Child:
        def __init__(self, cmd, stdout, stderr):
            arm = cmd[cmd.index('--branch')+1]
            if arm=='high': raise OSError('Unable to launch second simulator')
            self.args, self.pid, self.state, self.terminated = cmd, 100, None, False
            children.append(self); stdout.write('low started\n')
        def poll(self): return self.state
        def terminate(self): self.terminated=True
        def wait(self, timeout=None): self.state=-15; return self.state
    monkeypatch.setattr(runner.subprocess, 'Popen', Child)
    with pytest.raises(OSError, match='second simulator'):
        runner.acquire_native_pair(SimpleNamespace(seed=6201, scenario='stable', preflight=True), tmp_path, row())
    assert children[0].terminated and children[0].poll()==-15
    assert (tmp_path/'training_branches/0/low.log').read_text()=='low started\n'
    assert (tmp_path/'training_branches/0/high.log').is_file()
