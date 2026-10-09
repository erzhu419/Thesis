"""Check update accounting, actual critic mixing, split gradients and task placement."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alignment_common import P, milestones
import torch
from run import ModelTrainer
from model import BusModel
from launch import placements, job
from collect_results import summarize


def test_update_schedule_matches_original_budget_and_taper():
    previous, before, after = P['warmup_events'], 0, 0
    for point in milestones(P):
        for event in range(previous, point):
            if event < P['actor_taper_event']:
                before += 1
            else:
                after += 1
        previous=point
    assert (before,after)==(10000,10000)
    assert before+after+P['offline_updates_in_initializer']==70000
    assert set(P['model_refresh_events']) <= set(milestones(P))
    assert P['deployment_hold_scale']==1.0


def test_model_loss_is_one_quarter_not_double_counted_across_partitions():
    trainer=ModelTrainer.__new__(ModelTrainer)
    trainer.config=SimpleNamespace(critic_loss='mse',bootstrap_probability=1.)
    trainer.model_fraction=.25
    model=torch.tensor(3.,requires_grad=True)
    trainer.model_loss=model.square()
    offline=torch.tensor([[1.,2.]],requires_grad=True)
    online=torch.tensor([[2.,3.]],requires_grad=True)
    target=torch.zeros(2)
    actual=.5*trainer._ensemble_td_loss(offline,target)+.5*trainer._ensemble_td_loss(online,target)
    expected=.75*(.5*offline.square().mean()+.5*online.square().mean())+.25*model.square()
    assert torch.allclose(actual,expected)
    actual.backward()
    assert torch.allclose(model.grad,torch.tensor(1.5))
    assert offline.grad is not None and online.grad is not None


def test_no_model_preserves_real_loss():
    trainer=ModelTrainer.__new__(ModelTrainer)
    trainer.config=SimpleNamespace(critic_loss='mse',bootstrap_probability=1.)
    trainer.model_fraction=.25
    trainer.model_loss=None
    q=torch.tensor([[1.,3.]])
    assert trainer._ensemble_td_loss(q,torch.zeros(2))==5.


def test_target_refit_does_not_unfreeze_source_model():
    base=BusModel()
    residual=BusModel(base)
    residual.encoder.requires_grad_(True)
    residual.head.requires_grad_(True)
    assert all(not p.requires_grad for p in base.parameters())
    assert all(p.requires_grad for p in residual.head.parameters())


def test_jobs_cover_all_servers_and_pair_devices():
    items=placements()
    assert len(items)==18
    assert {i['node'] for i in items}=={'node007','jtl110gpu','jtl110gpu2','jtl311linux'}
    for a,b in zip(items[::2],items[1::2]):
        assert (a['node'],a['seed'],a['scenario'])==(b['node'],b['seed'],b['scenario'])
    tests=[job(i,case=c) for i in items for c in range(10)]
    assert len(tests)==180
    assert {j['require_node'] for j in tests}=={f'node{i:03}' for i in range(1,7)}
    assert all(j['vram']==0 and j['cpu']==1 for j in tests)
    assert all('/gates/' in j['wait_for_files'][0] for j in tests)


def test_partial_results_do_not_become_complete(tmp_path):
    result=summarize(tmp_path)
    assert not result['complete']
    assert result['training_complete']==result['evaluation_complete']==0
    assert len(result['missing'])==198
