"""The paid GPU prefix and final-gate publication have distinct destinations."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.mcwm_closed_loop_frozen_validation_v75_20261027 import infra


def test_three_gpu_jobs_use_original_cache_resources_and_new_outputs():
    jobs = infra.initializer_jobs()
    assert len(jobs) == 3
    for seed, job in zip((6206, 6207, 6208), jobs):
        assert job['priority'] == 'high' and job['allowed_nodes'] == ['jtl110gpu2']
        assert job['cpu'] == 6 and job['vram'] == 10000 and job['ram_mb'] == 24000
        assert '--seed '+str(seed) in job['cmd']
        assert 'frozen_validation_v75_20261027/inputs' in job['cmd']
        assert job['env']['THESIS_H2O_ROOT'].endswith('v14_terminal_taper_pilot_v1/workspace')
        assert 'formal_v2_full' in job['env']['THESIS_DATASET']


def setup_publish(tmp_path, monkeypatch, seed=6206):
    p=json.loads((infra.HERE/'protocol.json').read_text())
    (tmp_path/'protocol.json').write_text(json.dumps(p))
    monkeypatch.setattr(infra,'HERE',tmp_path)
    monkeypatch.setattr(infra,'remote',lambda path:str(Path('/hpc/v75')/Path(path).relative_to(tmp_path)))
    monkeypatch.setattr(infra,'presence',lambda paths:{path:False for path in paths})
    calls=[]
    gate=dict(status='complete',all_pass=True,seed=seed,offline_updates=50000,
        online_updates=0,initializer_path='/gpu/v75/inputs/h2oplus_seed6206.pt',
        recipe_path='/gpu/v75/inputs/seed6206/recipe.json')
    def run_on(node, command, **kwargs):
        calls.append((node,command))
        return 0,json.dumps({'6206':gate}) if 'result={}' in command else '', ''
    monkeypatch.setattr(infra,'scheduler',SimpleNamespace(run_on=run_on,
        _ssh_target_for_node=lambda node:'operator@hpc'))
    return calls


def test_gate_is_published_after_server_side_weights_and_recipe(tmp_path, monkeypatch):
    calls=setup_publish(tmp_path,monkeypatch)
    result=infra.promote_initializers([6206])
    transfers=[i for i,(_,cmd) in enumerate(calls) if cmd.startswith('rsync ')]
    gate=[i for i,(_,cmd) in enumerate(calls) if 'write_text' in cmd]
    assert len(transfers)==2 and len(gate)==1 and max(transfers)<gate[0]
    assert '/hpc/v75/inputs/h2oplus_seed6206.pt' in calls[gate[0]][1]
    assert result['published']==[6206]
    assert json.loads((tmp_path/'initializer_seed6206_publication.json').read_text())['local_checkpoint_bytes']==0


def test_wrong_seed_gate_cannot_publish_a_checkpoint(tmp_path, monkeypatch):
    calls=setup_publish(tmp_path,monkeypatch,seed=6207)
    with pytest.raises(ValueError,match='fixed seed/update budget'):
        infra.promote_initializers([6206])
    assert not any(cmd.startswith('rsync ') for _,cmd in calls)
