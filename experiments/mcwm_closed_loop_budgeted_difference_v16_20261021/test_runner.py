"""Check sparse controls and the resident, sequential paired acquisition path."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('v16_native_runner_test', HERE/'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def protocol(preflight=False):
    value = json.loads((HERE/'protocol.json').read_text())
    if preflight:
        value.update(value['preflight'])
    return value


def test_sparse_plan_has_nine_training_pairs_and_six_distinct_heldouts():
    np.random.seed(321)
    before = np.random.get_state()[1].copy()
    plan = runner.planned_cells(protocol(), 6201)
    assert plan == runner.planned_cells(protocol(), 6201)
    assert np.array_equal(before, np.random.get_state()[1])
    train = [row for row in plan if row['role'] == 'train']
    heldout = [row for row in plan if row['role'] == 'heldout']
    assert len(train) == 9 and len(heldout) == 6
    assert len({row['id'] for row in plan}) == 15
    assert [row['index'] for row in train] == list(range(9))
    for line in protocol()['target_lines']:
        arms = [row['nominal_arm'] for row in train if row['line_id'] == line]
        assert set(arms[:2]) == {'low','high'}
        assert abs(arms.count('low')-arms.count('high')) == 1
    assert all('nominal_arm' not in row for row in heldout)
    assert min(row['window']['start'] for row in heldout) > protocol()['feedback_last_deadline']


def test_preflight_is_one_new_train_pair_and_one_new_heldout_anchor():
    plan = runner.planned_cells(protocol(True),6201)
    assert [row['id'] for row in plan] == [
        'train/7X/preflight_train','heldout/7X/preflight_holdout']
    assert plan[0]['window'] == dict(name='preflight_train',start=1200,end=2700)
    assert plan[1]['window']['start'] > protocol(True)['feedback_last_deadline']


def test_sparse_choices_are_actual_existing_execution_actions_and_preserve_floor():
    transform = runner.ExecutionAction(dict(max_hold_seconds=60.,scheduled_headway_scale_seconds=360.))
    state = np.zeros(229,np.float32)
    state[0] = .4
    event = state[5:19]
    event[3:7] = [1.,1.2,1.,1.]
    event[8],event[9],event[11],event[12] = .4,.1,1.,1.
    choices = runner.action_choices(transform,state)
    assert 30.*(1.+float(choices['low'][0])) == 0.
    assert 30.*(1.+float(choices['high'][0])) == 7.5
    event[8],event[12] = .1,0.
    assert runner.action_choices(transform,state) is None


def transition_payload(action):
    return dict(s=[0.]*229,a=[action],ns=[1.]*229,r=-1.,dt=2.,terminal=False,
        episode_id=0,domain='target',started_at=1250.,completed_at=1252.,available_at=1282.,
        vehicle_id='bus',line_id='7X',station_id='stop',next_vehicle_id='other',
        sequence=20,raw_s=[0.]*229,raw_ns=[1.]*229,trip_id='trip')


def test_acquisition_replays_two_sequential_children_and_keeps_same_source(monkeypatch,tmp_path,capsys):
    launched = []
    waiting = []
    class Child:
        def __init__(self,cmd,stdout,stderr):
            self.args,self.pid,self.calls = cmd,100+len(launched),0
            assert stderr == subprocess.STDOUT
            assert not waiting  # A second libsumo process cannot overlap this child.
            waiting.append(self)
            launched.append(cmd)
            arm = cmd[cmd.index('--branch')+1]
            target = Path(cmd[cmd.index('--out')+1])
            target.mkdir(parents=True)
            value = dict(status='complete',integrity=dict(all_pass=True),
                selected=dict(sequence=20),prefix=dict(receipts=[]),prediction=dict(shared='unsupported'),
                native=dict(other_batch_actions=[dict(sequence=21,a=[-.5])]),
                truth=dict(fixed300=dict(status='complete',dt=300.,wait=100.,ride=200.,cost=400.)),
                global_transition=transition_payload(-1. if arm == 'low' else -.75))
            (target/'summary.json').write_text(json.dumps(value))
        def wait(self,timeout):
            assert timeout == 45
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(self.args,timeout)
            assert waiting.pop() is self
            return 0
    monkeypatch.setattr(runner.subprocess,'Popen',Child)
    args = SimpleNamespace(seed=6201,scenario='stable',preflight=True)
    record = dict(id='train/7X/preflight_train',index=0,line_id='7X',features=[1.,0.,0.,.5],
        selected=dict(sequence=20,started_at=1250.))
    pair = runner.acquire_native_pair(args,tmp_path,record)
    assert len(launched) == 2 and not waiting
    assert [cmd[cmd.index('--branch')+1] for cmd in launched] == ['low','high']
    assert all('--preflight' in cmd and cmd[cmd.index('--source')+1] == str(tmp_path) for cmd in launched)
    assert all(cmd[cmd.index('--mode')+1] == 'branch' for cmd in launched)
    assert pair['source_sequence'] == 20 and pair['role'] == 'train'
    assert pair['outcomes']['high']['transition'].a.dtype == np.float32
    assert pair['outcomes']['low']['transition'].sequence == 20  # Budget owns label-ID normalization.
    assert '"child_pid": 100' in capsys.readouterr().out


def test_normalized_cache_roundtrip_retains_delayed_pair_identity(tmp_path):
    from budget import PairPacketBudget
    budget = PairPacketBudget(protocol(True))
    budget.reserve(20,'train/7X/preflight_train')
    pair = dict(id='train/7X/preflight_train',index=0,role='train',source_sequence=20,
        issued_at=1250.,line_id='7X',features=[1.,0.,0.,.2],outcomes={
            arm:dict(truth=dict(status='complete',dt=300.,wait=100.,ride=200.,cost=400.),
                transition=runner.transition_from_dict(transition_payload(action)))
            for arm,action in (('low',-1.),('high',-.75))})
    normalized = budget.add_pair(pair)
    (tmp_path/'pair_cache').mkdir()
    (tmp_path/'pair_cache/0.json').write_text(json.dumps(runner.encode_pair(normalized)))
    loaded = runner.cached_pair(tmp_path,0)
    assert loaded['available_at'] == 1580.
    assert loaded['selected_for_training'] is False
    for arm,sequence in (('low',-1),('high',-2)):
        assert loaded['outcomes'][arm]['transition'].sequence == sequence
        assert loaded['outcomes'][arm]['transition'].available_at == 1580.

