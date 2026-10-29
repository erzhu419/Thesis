"""The shadow gate and first-branch reuse must bound all new native queries."""
import importlib.util
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest

HERE=Path(__file__).resolve().parent


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,HERE/file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.setattr(sys,'path',list(sys.path))
    infra=load('v19_test_infra','infra.py');monkeypatch.setitem(sys.modules,'infra',infra)
    launch=load('v19_test_launch','launch.py');monkeypatch.setitem(sys.modules,'launch',launch)
    run=load('v19_test_continue','continue_batch.py')
    calls=[]
    monkeypatch.setattr(infra,'submit',lambda jobs,name:calls.append((name,jobs)) or dict(submitted=[dict(id=name)]))
    def forbidden(*args,**kwargs):raise AssertionError('Driver test cannot contact resident data or scheduler')
    monkeypatch.setattr(infra,'collect_json',forbidden)
    monkeypatch.setattr(infra.scheduler,'run_on',forbidden)
    return SimpleNamespace(infra=infra,launch=launch,run=run,calls=calls)


@pytest.mark.parametrize('decision',[dict(status='complete',integrity=dict(all_pass=False),material_difference=True),
    dict(status='complete',integrity=dict(all_pass=True),material_difference=False)])
def test_native_queries_require_valid_material_shadow_evidence(driver,monkeypatch,tmp_path,decision):
    monkeypatch.setattr(driver.launch,'HERE',tmp_path)
    (tmp_path/'shadow_decision.json').write_text(json.dumps(decision))
    with pytest.raises(RuntimeError):driver.launch.submit_preflight()
    assert not driver.calls


def test_remaining_queries_require_passing_first_native_branch(driver,monkeypatch,tmp_path):
    monkeypatch.setattr(driver.launch,'HERE',tmp_path)
    (tmp_path/'preflight_gate.json').write_text(json.dumps(dict(status='complete',integrity=dict(all_pass=False))))
    with pytest.raises(RuntimeError):driver.launch.submit_remaining()
    assert not driver.calls


def test_one_plus_five_queries_reuse_first_path_and_source(driver,monkeypatch):
    launch=driver.launch
    monkeypatch.setattr(launch,'gate',lambda _:dict(material_difference=True))
    launch.submit_preflight();launch.submit_remaining()
    assert [(name,len(jobs)) for name,jobs in driver.calls]==[('preflight_retest',1),('intervention',5)]
    jobs=[job for _,batch in driver.calls for job in batch]
    assert len({job['signature'] for job in jobs})==6
    paths=[]
    for job in jobs:
        argv=shlex.split(job['cmd'])
        paths.append(argv[argv.index('--out')+1])
        assert argv[argv.index('--mode')+1]=='branch'
        assert argv[argv.index('--seed')+1]=='6201'
        assert argv[argv.index('--scenario')+1]=='stable'
        assert argv[argv.index('--source')+1].endswith('mcwm_closed_loop_state_response_v18_20261022/runs/stable/seed6201')
        assert job['priority']=='high' and (job['cpu'],job['ram_mb'],job['vram'])==(1,8000,0)
        assert job['no_auto_requeue'] is True
        assert job['allowed_nodes']==[f'node{i:03d}' for i in range(1,7)]
    assert len(set(paths))==6 and paths[0].endswith('/branches_retest/7X/low')
    p=launch.protocol()
    assert p['diagnostic_compute_budget']['new_branch_queries_maximum']==6
    assert p['diagnostic_compute_budget']['new_suffix_seconds_maximum']==1800
    assert p['invariants']['training_labels_per_day']==285
    assert p['diagnostic_compute_budget']['new_training_labels']==0


def test_shadow_is_one_high_read_only_task(driver):
    driver.launch.submit_shadow()
    name,jobs=driver.calls[0]
    assert name=='shadow' and len(jobs)==1
    job=jobs[0]
    assert job['priority']=='high' and job['ram_mb']==4096
    assert job['no_auto_requeue'] is True
    assert 'shadow.py' in job['cmd'] and '--source' in job['cmd'] and '--out' in job['cmd']
    assert '--mode day' not in job['cmd']


def test_continuation_does_not_expand_an_incomplete_native_preflight(driver,monkeypatch):
    run=driver.run
    monkeypatch.setattr(run,'exists',lambda name:name=='preflight_retest')
    monkeypatch.setattr(run,'metadata',lambda:dict(first=dict(status='pending'),bad=[],failures=[]))
    class Waiting(Exception):pass
    def sleep(seconds):
        assert seconds==60
        raise Waiting
    monkeypatch.setattr(run.time,'sleep',sleep)
    with pytest.raises(Waiting):run.main()
    assert not driver.calls


def test_continuation_orders_preflight_gate_remaining_five_and_finalizer(driver,monkeypatch,tmp_path):
    run=driver.run;monkeypatch.setattr(run,'HERE',tmp_path)
    monkeypatch.setattr(run,'exists',lambda name:name=='preflight_retest')
    result=dict(status='complete',integrity=dict(all_pass=True),bytes=2)
    states=iter([dict(first=dict(status='complete'),bad=[],failures=[]),
        dict(preflight_gate=True,bad=[],failures=[]),dict(result_exists=True,result=result,bad=[],failures=[])])
    monkeypatch.setattr(run,'metadata',lambda:next(states))
    monkeypatch.setattr(driver.launch,'gate',lambda _:dict(material_difference=True))
    (tmp_path/'intervention_summary.json').write_text('{}')
    def finalize(*,preflight=False):
        driver.calls.append(('preflight_finalize' if preflight else 'intervention_finalize',[]))
        return dict(submitted=[])
    monkeypatch.setattr(driver.launch,'submit_finalizer',finalize)
    run.main()
    assert [name for name,_ in driver.calls]==['preflight_finalize','intervention','intervention_finalize']
