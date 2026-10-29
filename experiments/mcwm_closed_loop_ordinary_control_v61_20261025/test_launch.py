import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v61_launch_test',HERE/'launch.py')
launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)


def test_fixed_all_cells_and_disjoint_confirmation():
    p=json.loads((HERE/'protocol.json').read_text())
    dev=launch.cells(p,'development');confirmation=launch.cells(p,'confirmation')
    assert len(dev)==8 and len(confirmation)==12
    assert {x['seed'] for x in dev}=={6201,6202}
    assert {x['seed'] for x in confirmation}=={6203,6204,6205}
    assert all(sum(x['arm']==a for x in confirmation)==6 for a in ('fresh','no_fresh'))


def test_jobs_reuse_frozen_runner_with_new_output_directories():
    class Infra:
        @staticmethod
        def remote(path):return str(path)
        @staticmethod
        def task(label,runner,argv):return dict(label=label,runner=runner,argv=argv)
    p=json.loads((HERE/'protocol.json').read_text())
    jobs=launch.native_jobs(p,'development',Infra)
    assert len(jobs)==8
    assert all(job['runner']==launch.SOURCE/'control.py' for job in jobs)
    assert all(str(HERE) in job['argv'][-1] for job in jobs)
    assert all(job['argv'][:2]==['--candidate','recent30'] for job in jobs)


def test_unready_source_cannot_dispatch_paid_control(monkeypatch,tmp_path):
    class Infra:pass
    monkeypatch.setattr(launch,'HERE',tmp_path)
    monkeypatch.setattr(launch,'module',lambda *args: Infra())
    result=launch._advance(lambda function,*args,**kwargs:function(*args,**kwargs))
    assert result['stage']=='preparing_ordinary_control'
    assert result['done'] is False and result['goal_achieved'] is False
