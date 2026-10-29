import math
import pytest
from experiments.mcwm_closed_loop_trace_support_v39_20261024.test_support import case
from experiments.mcwm_closed_loop_trace_design_v40_20261024.behavior import actor_parameters
from experiments.mcwm_closed_loop_overlap_v41_20261024.overlap import geometry_at
from experiments.mcwm_closed_loop_trace_support_v39_20261024.support import atom_geometry
from experiments.mcwm_closed_loop_sync_v42_20261024.controls import oracle_gate_behavior,score


def test_gate_control_exchanges_normalized_probability_pair_without_modifying_source():
    behavior=dict(positive_probability=.2,zero_probability=.8,mean=-1.,std=.6)
    target=dict(positive_probability=.9,zero_probability=.1,mean=3.,std=2.)
    hybrid=oracle_gate_behavior(behavior,target)
    assert behavior['positive_probability']==.2
    assert hybrid['positive_probability']==.9 and hybrid['zero_probability']==.1
    assert hybrid['mean']==-1. and hybrid['std']==.6


def test_gate_alignment_does_not_force_unit_ratio_when_gaussian_still_differs():
    _,reference,row=case(time=.8,proposal=0.,gate=0.)
    behavior=actor_parameters(reference,row['s'])
    target=dict(behavior,positive_probability=.8,zero_probability=.2,mean=1.)
    control=score(reference.action_transform,row['s'],row['a'],oracle_gate_behavior(behavior,target),target,
        atom_geometry(reference.action_transform,row['s'],row['a']),geometry_at(reference.action_transform,row['s']))
    assert control['log_c']==pytest.approx(-.5)
    assert control['c']==pytest.approx(math.exp(-.5))
    assert control['overlap']<1. and not control['training_authorized']


def test_all_expectations_are_recomputed_under_the_hypothetical_behavior():
    _,reference,row=case(time=.8,proposal=-1.)
    behavior=actor_parameters(reference,row['s'])
    target=dict(behavior,positive_probability=.8,zero_probability=.2)
    geometry=geometry_at(reference.action_transform,row['s'])
    point=atom_geometry(reference.action_transform,row['s'],row['a'])
    old=score(reference.action_transform,row['s'],row['a'],behavior,target,point,geometry)
    aligned=score(reference.action_transform,row['s'],row['a'],oracle_gate_behavior(behavior,target),target,point,geometry)
    assert old['expected_log_c']<0.
    assert aligned['expected_log_c']==0. and aligned['overlap']==1.
