"""Check same-prefix effect accounting without native queries or training."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v19_analysis_test_module', HERE/'analyze.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path):
    source, root = tmp_path/'v18', tmp_path/'v19'
    day = source/'runs/stable/seed6201'
    write(root/'protocol.json',dict(intervention=dict(first_branch=dict(
        line='7X',arm='low',output_relative='branches_retest/7X/low'))))
    write(day/'summary.json', dict(status='complete',
        integrity=dict(all_pass=True, checks=dict(physical_source_fixed=True, original_trust_rule=True)),
        initializer=dict(initial_updates=50000, history_rows=32768, source_rows=4096),
        budget=dict(target_labels=285, ordinary_labels=267, paired_action_labels=18, training_pairs=9,
            paired_outcomes_per_pair=2, packets=57, native_seconds=24000.,
            policy_updates=456, trust_updates=456, world_updates=1140)))
    records = []
    for i, line in enumerate(analysis.LINES):
        at = 16700.+i*100
        predictions = {name: dict(status='supported', support_pairs=2,
            gain=dict(dt=0., wait=2., ride=1., cost=5.), fixed300_clock=dict(low=300., high=300.))
            for name in ('route', 'shared', 'candidate')}
        anchor = dict(id=f'train/{line}/late', role='train', index=i*3+2, line_id=line,
            selected=dict(sequence=100+i, started_at=at, line_id=line, vehicle_id=line+'_3',
                station_id='diagnostic', s=[at/24000], event=[1.]),
            prediction=predictions, features=[float(i == j) for j in range(3)]+[.1,.5],
            prefix=dict(external_seed=84201, counters=dict(policy=50360, world=900, trust=360,
                target_labels=225), receipts=[dict(deadline=16500)], issued_count=101+i))
        records.append(anchor)
        for arm, seconds in (('low',0.), ('high',7.5)):
            for policy in ('native','tail'):
                wait = 100.+(5. if policy == 'native' else -10.) if arm == 'low' else 100.
                def endpoint(dt):
                    return dict(status='complete', dt=dt, wait=wait, ride=200., cost=2*wait+200.)
                value = dict(status='complete', selected=deepcopy(anchor['selected']), prefix=deepcopy(anchor['prefix']),
                    prediction=deepcopy(predictions),
                    integrity=dict(all_pass=True, checks={name:True for name in analysis.PREFIX_CHECKS}),
                    execution=dict(requested_hold_seconds=seconds, executed_hold_seconds=seconds),
                    native=dict(other_batch_actions=[], prefix_matched_issues=101+i),
                    suffix=dict(seconds=300, training_delta=dict(policy=0, world=0, trust=0, readout_refits=0)),
                    truth=dict(global_next=endpoint(2.), same_vehicle_next=endpoint(30.), fixed300=endpoint(300.)),
                    budget=dict(prefix_sim_seconds=at, suffix_sim_seconds=300,
                        replay_target_labels=225, replay_policy_updates=360, truth_labels_enter_training=0))
                if policy == 'tail':
                    value['phase_intervention'] = dict(mapping='tail_after_fork_batch',
                        actor_real_clock=True, actor_clock_max_abs_error=.0003,
                        fork_batch_native=True, tail_formula_exact=True, tail_policy_issues=200)
                    path = root/('branches_retest' if line == '7X' and arm == 'low' else 'branches')/line/arm
                else:
                    path = day/'training_branches'/str(anchor['index'])/arm
                write(path/'summary.json', value)
    write(day/'acquisitions.json',dict(records=records))
    return source, root


def mutate(root, line='7X', arm='low', transform=None):
    path=root/('branches_retest' if line == '7X' and arm == 'low' else 'branches')/line/arm/'summary.json'
    value=json.loads(path.read_text())
    transform(value)
    write(path,value)


def test_complete_fixed_six_suffixes_count_no_new_training_and_compare_true_gain(tmp_path):
    source, root=fixture(tmp_path)
    value=analysis.analyze(source,root)
    assert value['status']=='complete' and value['integrity']['all_pass']
    assert value['budget']['new_main_days']==value['budget']['new_training_labels']==0
    assert value['budget']['diagnostic_action_outcomes']==6
    assert value['budget']['completed_tail_suffix_seconds']==1800
    assert value['budget']['reused_native_action_outcomes']==6
    assert value['budget']['replayed_target_labels']==1350
    assert value['model_metrics']['shared']['native']['regret_total']==0.
    assert value['model_metrics']['shared']['tail']['regret_total']==60.
    assert value['model_metrics']['zero']['regret_change']==-30.
    assert value['missing']['selection']==0 and value['unsupported']['candidate']==0
    for row in value['cases']:
        assert row['delta_F_gain']==dict(dt=0.,wait=-15.,ride=0.,cost=-30.)
        assert row['true_sign_changed']
        assert row['model_effects']['shared']['native'][-1]==0.
        assert row['model_effects']['shared']['tail'][-1]==20.
        assert row['model_effects']['shared']['regret_change']==20.
        assert row['model_effects']['zero']['native'][-1]==10.
        assert row['model_effects']['zero']['tail'][-1]==0.
    assert len(json.dumps(value,separators=(',',':')).encode()) < analysis.MAX_BYTES


def test_first_low_preflight_is_one_of_six_and_does_not_require_future_arms(tmp_path):
    source,root=fixture(tmp_path)
    for path in (root/'branches').glob('*/*/summary.json'):
        if path.parent.parent.name!='7X' or path.parent.name!='low':path.unlink()
    value=analysis.analyze(source,root,preflight=True)
    assert value['status']=='complete' and value['integrity']['all_pass']
    assert value['budget']['diagnostic_action_outcomes']==1
    assert value['budget']['completed_tail_suffix_seconds']==300
    assert len(value['cases'])==1 and value['cases'][0]['line_id']=='7X'
    assert value['budget']['preflight_is_first_of_six']


@pytest.mark.parametrize('field,change',[
    ('same_batch', lambda v:v['native'].update(other_batch_actions=[dict(a=[.1])])),
    ('clock', lambda v:v['phase_intervention'].update(actor_real_clock=False)),
    ('prediction', lambda v:v['prediction']['candidate']['gain'].update(cost=6.)),
    ('suffix', lambda v:v['suffix']['training_delta'].update(policy=1)),
    ('training', lambda v:v['budget'].update(truth_labels_enter_training=1)),
])
def test_material_intervention_contract_errors_block_gate(tmp_path,field,change):
    source,root=fixture(tmp_path)
    mutate(root,transform=change)
    value=analysis.analyze(source,root)
    assert value['status']=='complete' and not value['integrity']['all_pass'],field
    assert value['cases'][0]['integrity']['failed_checks'],field


def test_missing_and_zero_global_duration_are_preserved_separately(tmp_path):
    source,root=fixture(tmp_path)
    for arm in analysis.ARMS:
        mutate(root,transform=lambda v:v['truth']['global_next'].update(dt=0.),arm=arm)
    value=analysis.analyze(source,root)
    assert value['cases'][0]['duration']['tail']=='zero_dt'
    assert value['duration_counts']['tail']==dict(zero_dt=1,positive_dt=2)
    assert value['cases'][0]['delta_F_gain']['cost']==-30.
    (root/'branches/7X/high/summary.json').unlink()
    value=analysis.analyze(source,root)
    assert value['status']=='incomplete' and not value['integrity']['all_pass']
    assert value['cases'][0]['truth']['tail']['fixed300']['gain'] is None
    assert value['cases'][0]['delta_F_gain'] is None
    assert value['missing']['endpoints']['tail']['fixed300']==1


def test_unsupported_prediction_remains_none_instead_of_zero(tmp_path):
    source,root=fixture(tmp_path)
    day=source/'runs/stable/seed6201'
    acquired=json.loads((day/'acquisitions.json').read_text())
    acquired['records'][0]['prediction']['candidate'].update(status='unsupported',gain=None)
    write(day/'acquisitions.json',acquired)
    for arm in analysis.ARMS:
        for path in (day/'training_branches/2'/arm/'summary.json',
                root/('branches_retest' if arm == 'low' else 'branches')/'7X'/arm/'summary.json'):
            v=json.loads(path.read_text())
            v['prediction']['candidate'].update(status='unsupported',gain=None)
            write(path,v)
    value=analysis.analyze(source,root)
    assert value['integrity']['all_pass']
    assert value['cases'][0]['model_effects']['candidate']['prediction'] is None
    assert value['cases'][0]['model_effects']['candidate']['regret_change'] is None
    assert value['unsupported']['candidate']==1
    assert value['model_metrics']['candidate']['native']['n']==2


def test_source_budget_changes_are_not_hidden_as_diagnostic_replay(tmp_path):
    source,root=fixture(tmp_path)
    path=source/'runs/stable/seed6201/summary.json'
    value=json.loads(path.read_text());value['budget']['target_labels']=286;write(path,value)
    result=analysis.analyze(source,root)
    assert not result['integrity']['all_pass']
    assert 'training_285' in result['source']['integrity']['failed_checks']


def test_failed_original_first_arm_is_preserved_but_only_retest_is_counted(tmp_path):
    source,root=fixture(tmp_path)
    failed=root/'branches/7X/low/failure.json'
    original=dict(status='failed',type='KeyError',message='nominal_arm',time=2737.)
    write(failed,original)
    protocol=json.loads((root/'protocol.json').read_text())
    protocol['diagnostic_compute_budget']=dict(failed_preflight_prefix=dict(
        native_seconds_replayed=2737,replayed_target_labels=30,replayed_policy_updates=48))
    write(root/'protocol.json',protocol)
    value=analysis.analyze(source,root)
    assert value['status']=='complete' and value['integrity']['all_pass']
    assert value['budget']['completed_tail_outcomes']==6
    assert value['budget']['completed_tail_suffix_seconds']==1800
    assert value['budget']['prefix_replay_seconds']==6*16800.
    assert value['budget']['total_prefix_replay_seconds']==6*16800.+2737
    assert value['budget']['total_replayed_target_labels']==value['budget']['replayed_target_labels']+30
    assert value['budget']['total_replayed_policy_updates']==value['budget']['replayed_policy_updates']+48
    assert json.loads(failed.read_text())==original
    (root/'branches_retest/7X/low/summary.json').unlink()
    value=analysis.analyze(source,root,preflight=True)
    assert value['status']=='incomplete' and not value['integrity']['all_pass']
    assert value['cases'][0]['branch_status']['low']['tail']=='missing'
