import importlib.util
import json
from pathlib import Path

import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v60_test_launch',HERE/'control_launch.py')
launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)


def complete(eligible):
    return dict(status='complete',integrity=dict(all_pass=True),
        next_control_gate=dict(passed=bool(eligible),eligible_models=eligible))


def test_both_passing_readouts_do_not_choose_largest_heldout_gain():
    result=launch.choose(complete(['readout_lowdim','readout_line']))
    assert result['candidate']=='paired_readout' and result['model']=='readout_line'
    assert result['ordinary_single_trajectory_dripping'] is False


def test_failed_cost_readout_is_closed_before_recency_control():
    result=launch.choose(complete([]))
    assert result['candidate']=='recent30'
    assert result['v59_disposition']=='closed_failed_gate'


def test_incomplete_admission_is_not_a_performance_failure():
    result=launch.choose(dict(status='partial_complete',integrity=dict(all_pass=False)))
    assert result['candidate']=='recent30'
    assert result['v59_disposition']=='closed_incomplete_admission'


def test_pass_without_predeclared_model_is_an_error():
    payload=complete([]);payload['next_control_gate']['passed']=True
    with pytest.raises(ValueError,match='no eligible'):
        launch.choose(payload)


def test_confirmation_is_fixed_and_disjoint_from_development():
    p=json.loads((HERE/'protocol.json').read_text());selection=dict(candidate='recent30')
    dev=launch.cells(p,selection,'development');confirm=launch.cells(p,selection,'confirmation')
    assert len(dev)==8 and len(confirm)==12
    assert {r['seed'] for r in dev}=={6201,6202}
    assert {r['seed'] for r in confirm}=={6203,6204,6205}
    assert all(sum(c['arm']==arm for c in confirm)==6 for arm in ('fresh','no_fresh'))


def test_readout_keeps_real_actor_baseline_and_matched_selector():
    p=json.loads((HERE/'protocol.json').read_text())
    result=launch.cells(p,dict(candidate='paired_readout',model='readout_line'),'development')
    assert len(result)==12
    assert {c['arm'] for c in result}=={'readout','q_selector','actor_mcwm'}
