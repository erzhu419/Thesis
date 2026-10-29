"""Worker evidence gates, frozen executed-action scoring and real triplet launch."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

os.environ.setdefault('THESIS_H2O_ROOT',str(Path(__file__).resolve().parents[1]/'sumo_eval/runtime/H2Oplus'))

from experiments.mcwm_closed_loop_shared_level_v36_20261024 import run, learning
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.test_learning import initial, rows
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng, _state


def test_old_metrics_and_every_arm_sampling_rng_gates_are_independent():
    reference=dict(packet=1,receipt=dict(sequence_ids=[1,2,3,4,5]),sampling=[dict(recent=[i]) for i in range(8)],
        rng=dict(global_state=[1],sampler=[2],diagnostic=[3]),updates=8,
        optimizer_steps=dict(critic_optimizer=dict(states=12,min=1,max=50008)),metrics=dict(loss=1.))
    assert all(run._check_packet(reference,reference,exact_reference=reference).values())
    shared=deepcopy(reference);shared['metrics']['loss']=2.
    assert all(run._check_packet(shared,reference).values())
    with pytest.raises(RuntimeError,match='old_metrics_exact'):
        run._check_packet(shared,reference,exact_reference=reference)
    for field, message in [('sampling','all_eight_sampling_draws'),('rng','global_and_private_rng'),
                           ('optimizer_steps','optimizer_steps')]:
        changed=deepcopy(shared);changed[field]={}
        with pytest.raises(RuntimeError,match=message):run._check_packet(changed,reference)


def test_shared_frozen_issue_uses_saved_actual_actions_and_keeps_heads_and_rng():
    learner=learning.attach_shared_level(initial())
    with torch.no_grad():
        learner.critic.level_weight.fill_(.03)
        learner.critic.action_weight[0].fill_(.7)
        learner.critic.action_weight[1].fill_(-.2)
    run.freeze(learner)
    row=rows(.8)[0]
    selected=dict(s=row.s.tolist(),memory_s=row.memory_s)
    actions=dict(low=[-1.],high=[-.9375])
    before=deepcopy(_state(learner));rng=_rng()
    private={name:deepcopy(getattr(learner,name).bit_generator.state) for name in ('rng','diagnostic_rng')}
    result=run.heldout_prediction(learner,selected,actions)
    prediction=result['prediction']
    assert prediction['executed_actions']==[-1.,-.9375]
    assert prediction['holding_seconds']==[0.,1.875]
    # F would shrink the saved high action to .234375 s at this tail state.
    repeat=learner.action_transform(torch.tensor(row.s)[None].expand(2,-1),torch.tensor([[-1.],[-.9375]]))
    assert learner.action_transform.hold_seconds(repeat)[1].item()!=1.875
    assert len(prediction['q_heads'])==2 and len(prediction['min_heads'])==2
    assert result['variants']['full']=={key:prediction[key] for key in result['variants']['full']}
    assert len(result['components']['level_heads'])==1
    assert _equal(before,_state(learner)) and _equal(rng,_rng())
    assert all(_equal(state,getattr(learner,name).bit_generator.state) for name,state in private.items())


def test_optimizer_interval_intersection_does_not_count_overlapping_probe_envelopes():
    def records(values):
        return [dict(update_started_wall=start,update_finished_wall=finish) for start,finish in values]

    disjoint=dict(zero_memory=records([(0.,1.),(9.,10.)]),
        per_head_level=records([(2.,3.),(7.,8.)]),shared_level=records([(4.,5.),(6.,7.)]))
    assert run.optimizer_overlap(disjoint)==0.
    actual=dict(zero_memory=records([(0.,2.),(4.,6.)]),
        per_head_level=records([(1.,3.),(4.5,8.)]),shared_level=records([(1.5,2.5),(5.,7.)]))
    assert run.optimizer_overlap(actual)==1.5


def test_triplet_starts_three_distinct_real_workers_with_common_training_overlap(tmp_path,monkeypatch):
    # Small subprocesses exercise the orchestration itself. Native optimizer
    # correctness is covered by the full learning tests and saved-reference gates.
    runner=tmp_path/'run.py'
    runner.write_text('''import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--seed');p.add_argument('--scenario');p.add_argument('--out');p.add_argument('--child-arm')
a=p.parse_args();r=Path(a.out);r.mkdir();start=time.time();time.sleep(.2)
finish=time.time()
(r/'packets.jsonl').write_text(json.dumps(dict(update_started_wall=start,update_finished_wall=finish))+'\\n')
value=dict(integrity=dict(all_pass=True),budget=dict(feedback_slots=285,shadow_updates=456),
 wall=dict(pid=os.getpid(),started=start,training_started=start,training_finished=time.time(),finished=time.time()))
(r/'summary.json').write_text(json.dumps(value))
''')
    monkeypatch.setattr(run,'HERE',tmp_path)
    out=tmp_path/'triplet'
    run.triplet(SimpleNamespace(out=out,seed=6202,scenario='stable'),dict(arms=list(learning.ARMS)))
    result=json.loads((out/'summary.json').read_text())
    assert result['integrity']['all_pass']
    assert len({value['pid'] for value in result['workers'].values()})==3
    assert result['training_overlap_wall_seconds']>0.
    assert result['optimizer_overlap_wall_seconds']>0.
    assert result['budget']['shadow_updates']==1368
    assert result['budget']['unique_original_labels']==285
