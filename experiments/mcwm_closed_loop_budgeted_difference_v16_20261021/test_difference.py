"""Direct contrasts cancel shared levels and use admitted paired outcomes only."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v16_difference_test',HERE/'difference.py')
difference=importlib.util.module_from_spec(spec);spec.loader.exec_module(difference)


def pair(index,line,features,delta,*,common=1000.):
    wait,ride=delta
    return dict(id=f'p{index}',role='train',selected_for_training=True,admitted_at=600.,available_at=430.,
        line_id=line,features=features,outcomes={
            'low':dict(truth=dict(status='complete',dt=300.,wait=common+wait,ride=common+ride)),
            'high':dict(truth=dict(status='complete',dt=300.,wait=common,ride=common))})


def records():
    return [pair(0,'7X',[1,0,0,-.5],[-20,30]),pair(1,'7X',[1,0,0,.5],[40,-10]),
            pair(2,'102X',[0,1,0,.2],[10,20]),pair(3,'705X',[0,0,1,-.3],[-5,10])]


def test_direct_targets_ignore_common_level_and_preserve_cost_identity_and_clock():
    rows=records();changed=copy.deepcopy(rows)
    for row in changed:
        for arm in ('low','high'):
            for key in ('wait','ride'):row['outcomes'][arm]['truth'][key]+=1e9
    models=[]
    for data in (rows,changed):
        m=difference.DirectDifferenceReadout();m.accept(data,600.);m.refit();models.append(m)
    assert np.allclose(models[0].weights,models[1].weights)
    pred=models[0].predict([1,0,0,.25])
    assert pred['gain']['cost']==2*pred['gain']['wait']+pred['gain']['ride']
    assert pred['gain']['dt']==0. and pred['fixed300_clock']==dict(low=300.,high=300.)
    assert models[0].summary()['outcome_labels']==8


def test_line_only_is_constant_with_state_and_primary_can_vary():
    full=difference.DirectDifferenceReadout();line=difference.DirectDifferenceReadout(line_only=True)
    for model in (full,line):model.accept(records(),600.);model.refit()
    assert line.predict([1,0,0,-5.])['gain']==line.predict([1,0,0,5.])['gain']
    assert full.predict([1,0,0,-.5])['gain']!=full.predict([1,0,0,.5])['gain']
    assert full.summary()['design']['columns']==4 and line.summary()['design']['columns']==3


def test_missing_line_returns_unsupported_without_zero_gain():
    m=difference.DirectDifferenceReadout();m.accept(records()[:2],600.);m.refit()
    pred=m.predict([0,1,0,.2])
    assert pred['status']=='unsupported' and pred['gain'] is None and pred['support_pairs']==0


def test_future_unadmitted_heldout_or_duplicate_pair_cannot_fit():
    for mutation in (dict(role='heldout'),dict(selected_for_training=False),dict(available_at=601.)):
        row=records()[0];row.update(mutation);m=difference.DirectDifferenceReadout()
        with pytest.raises(ValueError):m.accept([row],600.)
    m=difference.DirectDifferenceReadout();m.accept(records()[:1],600.)
    with pytest.raises(ValueError,match='second'):
        m.accept(records()[:1],600.)


def test_issue_features_are_issue_headways_and_never_read_scene_or_truth():
    from experiments.mcwm_closed_loop_v1.stream import EI
    event=np.zeros(max(EI.values())+1,np.float32)
    event[EI['forward_headway_seconds']]=660.
    event[EI['target_forward_headway_seconds']]=600.
    event[EI['backward_headway_seconds']]=540.
    event[EI['target_backward_headway_seconds']]=600.
    event[EI['bus_capacity']]=50.
    issued=dict(line_id='7X',event=event,started_at=3000.,scenario='forbidden_future_flag')
    features=difference.issue_features(issued)
    assert np.allclose(features,[1,0,0,.2])
