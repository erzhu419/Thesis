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
    for name in ('bootstrap', 'history', 'shift_model', 'candidate', 'sampling'):
        monkeypatch.setitem(sys.modules, name, load('v21_parallel_'+name, HERE/(name+'.py')))
    return load('v21_parallel_runner', HERE/'run.py')


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
        native=dict(other_batch_actions=[]), execution_wall=dict(pid=101 if arm=='low' else 102, native_started_wall=10. if arm=='low' else 12., native_finished_wall=20. if arm=='low' else 22.),
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
    pair = runner.acquire_native_pair(SimpleNamespace(seed=6201, scenario='stable', preflight=True), tmp_path, selected, lambda:1250.)
    assert completed==['high','low']
    assert list(pair['outcomes'])==['low','high']
    assert pair['outcomes']['low']['truth']['wait']==100.
    assert pair['outcomes']['high']['truth']['wait']==103.
    assert pair['service_targets']['arrival']==[-.5,-.5,-.5]
    assert selected==original  # Waiting wall time cannot advance the fixed issue clock.
    assert (tmp_path/'prefix_trace.jsonl').read_text()=='existing native prefix\n'
    assert all(Path(c.args[2]).name=='run.py' and '--preflight' in c.args for c in children)
    assert all(c.state==0 for c in children)
    assert (tmp_path/'training_branches/0/low.log').read_text()=='low launched\n'
    assert (tmp_path/'training_branches/0/high.log').read_text()=='high launched\n'

    receipt = json.loads((tmp_path/'training_branches/0/parallel_execution.json').read_text())
    assert receipt['all_pass'] and receipt['overlap_seconds'] == 8.
    assert receipt['parent_native_before'] == receipt['parent_native_after'] == 1250.


def test_native_contract_and_line_reachability_are_kept_in_the_new_plan(runner):
    p = runner.load_protocol()
    plans = runner.planned_cells(p, 6201)
    train = [r for r in plans if r['role']=='train']
    assert len(train)==9 and len({r['index'] for r in train})==9
    assert all(sum(r['line_id']==line for r in train)==3 for line in p['target_lines'])
    assert p['target_record_budget']==285 and p['initializer_updates']==50000 and p['horizon_seconds']==24000
    assert p['target_seed_offset']==p['preflight']['target_seed_offset']==90000
    assert list(runner.make_readouts(p,service_only=True))==['shared','route']
    for line in ('102X','705X'):
        first = next(r for r in train if r['line_id']==line and r['window_name']=='bootstrap')
        assert first['window']['start']==3600 and first['window']['end']==6600
