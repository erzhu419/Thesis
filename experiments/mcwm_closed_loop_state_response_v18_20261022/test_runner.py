"""Check full preflight, native pair controls, coverage and label-age reporting."""
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
spec = importlib.util.spec_from_file_location('v18_native_runner_test', HERE/'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def protocol():
    return runner.load_protocol()


def test_three_fitted_readouts_share_admitted_pairs_and_preserve_old_predictions():
    from difference import DirectDifferenceReadout
    records=[]
    for index,line in enumerate(('7X','102X','705X')):
        records.append(dict(id=f'pair/{line}',line_id=line,role='train',selected_for_training=True,
            available_at=330.,admitted_at=600.,features=[float(index==j) for j in range(3)]+[.1*(index+1),.2*(index+1)],
            outcomes={arm:dict(truth=dict(status='complete',dt=300.,wait=wait,ride=ride,cost=2*wait+ride))
                for arm,wait,ride in (('low',1000.+index,2000.-index),('high',1003.,1992.))}))
    models=runner.make_readouts(protocol())
    assert list(models)==['shared','route','candidate']
    for model in models.values():
        model.accept(records,600.)
        model.refit()
        assert [row['id'] for row in model.records]==[row['id'] for row in records]
        assert model.summary()['outcome_labels']==6
    for name,line_only in (('shared',False),('route',True)):
        old=DirectDifferenceReadout(line_only=line_only,ridge=protocol()['difference_ridge'])
        old.accept(records,600.)
        old.refit()
        for record in records:
            assert models[name].predict(record['features'])==old.predict(record['features'][:4])
    for record in records:
        shared=models['shared'].predict(record['features'])
        candidate=models['candidate'].predict(record['features'])
        assert candidate['support_pairs']==shared['support_pairs']==1
        assert candidate['gain']['wait']==shared['gain']['wait']
        assert len(candidate['features'])==5


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


def test_preflight_never_shrinks_the_pilot_initialization_budget_or_clock(monkeypatch,tmp_path):
    value = protocol()
    value['preflight'] = dict(target_record_budget=50,model_init_updates=2,
        model_refit_updates=1,preflight_stop_seconds=5000,target_lines=['7X'])
    (tmp_path/'protocol.json').write_text(json.dumps(value))
    monkeypatch.setattr(runner,'HERE',tmp_path)
    full = runner.load_protocol()
    assert (full['initializer_updates'],full['model_init_updates'],full['model_refit_updates']) == (50000,400,10)
    assert (full['target_record_budget'],full['policy_update_budget'],full['horizon_seconds']) == (285,456,24000)
    plan = runner.planned_cells(full,6201)
    assert sum(row['role']=='train' for row in plan) == 9
    assert sum(row['role']=='heldout' for row in plan) == 6


def design_fixture():
    p = protocol()
    plans = runner.planned_cells(p,6201)
    records,admitted = [],[]
    for plan in plans:
        at = plan['window']['start']+10.
        records.append(dict(id=plan['id'],role=plan['role'],line_id=plan['line_id'],
            window_name=plan['window_name'],selected=dict(started_at=at)))
        if plan['role'] == 'train':
            admitted_at = at+400.
            if plan['window_name'] == p['training_windows'][-1]['name']:
                admitted_at = 17400.+300.*p['target_lines'].index(plan['line_id'])
            admitted.append(dict(id=plan['id'],line_id=plan['line_id'],issued_at=at,
                admitted_at=admitted_at))
    return p,plans,records,admitted


def test_coverage_requires_nine_admitted_pairs_and_all_line_windows():
    p,plans,records,admitted = design_fixture()
    coverage,freshness = runner.design_report(p,plans,records,admitted)
    assert coverage['all_pass'] is True
    assert (coverage['training_expected'],coverage['training_selected'],coverage['training_admitted']) == (9,9,9)
    assert (coverage['heldout_expected'],coverage['heldout_selected']) == (6,6)
    assert freshness['last_pair_admitted_at'] == 18000.
    for index,line in enumerate(p['target_lines']):
        assert coverage['by_line'][line]['training_admitted'] == 3
        assert coverage['by_line'][line]['heldout_selected'] == 2
        assert coverage['by_line'][line]['missing_heldout_windows'] == []
        last = 17400.+300.*index
        assert freshness['by_line'][line]['last_pair_admitted_at'] == last
        for window in p['heldout_windows']:
            row = freshness['heldout'][f"heldout/{line}/{window['name']}"]
            assert row['age_since_last_line_pair_admitted'] == window['start']+10.-last


def test_missing_windows_are_coverage_loss_with_no_invented_label_age():
    p,plans,records,admitted = design_fixture()
    missing_id = f"heldout/7X/{p['heldout_windows'][1]['name']}"
    records = [row for row in records if row['id'] != missing_id]
    coverage,freshness = runner.design_report(p,plans,records,admitted)
    assert coverage['all_pass'] is False
    assert coverage['training_admitted'] == 9 and coverage['heldout_selected'] == 5
    assert coverage['by_line']['7X']['missing_heldout_windows'] == [p['heldout_windows'][1]['name']]
    assert freshness['heldout'][missing_id]['status'] == 'selection_missing'
    assert freshness['heldout'][missing_id]['age_since_last_line_pair_admitted'] is None


def test_unadmitted_late_pair_cannot_make_reported_labels_fresher():
    p,plans,records,admitted = design_fixture()
    missing_id = f"train/7X/{p['training_windows'][-1]['name']}"
    admitted = [row for row in admitted if row['id'] != missing_id]
    coverage,freshness = runner.design_report(p,plans,records,admitted)
    assert coverage['all_pass'] is False
    assert coverage['by_line']['7X']['training_selected'] == 3
    assert coverage['by_line']['7X']['training_admitted'] == 2
    assert coverage['by_line']['7X']['unadmitted_training_windows'] == [p['training_windows'][-1]['name']]
    actual_last = max(row['admitted_at'] for row in admitted if row['line_id']=='7X')
    assert freshness['by_line']['7X']['last_pair_admitted_at'] == actual_last


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
    record = dict(id='train/7X/preflight_train',index=0,line_id='7X',features=[1.,0.,0.,.5,.3],
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
    budget = PairPacketBudget(protocol())
    budget.reserve(20,'train/7X/preflight_train')
    pair = dict(id='train/7X/preflight_train',index=0,role='train',source_sequence=20,
        issued_at=1250.,line_id='7X',features=[1.,0.,0.,.2,.3],outcomes={
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

