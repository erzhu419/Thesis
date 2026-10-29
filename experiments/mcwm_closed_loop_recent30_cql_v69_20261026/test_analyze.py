"""V69 recency composition keeps the stronger CQL control and actual-cost gates."""
import copy
import json
import shutil

import pytest

from experiments.mcwm_closed_loop_executed_cql_v68_20261026 import test_analyze as prior
from experiments.mcwm_closed_loop_recent30_cql_v69_20261026 import analyze as module


def populate(tmp_path, *, fresh=(98.,)*4):
    protocol=json.loads((module.HERE/'protocol.json').read_text())
    cached,_=prior.populate(tmp_path,fresh=(101.,)*4,no_fresh=(99.,)*4)
    cached_root=tmp_path/protocol['original_cql_reference']['directory']
    cached.rename(cached_root)
    root=tmp_path/'v69'
    prior.write(root/'protocol.json',protocol)
    shutil.copytree(cached_root/'runs',root/'runs')
    paths=[]
    for i,seed in enumerate(protocol['development_seeds']):
        for j,scene in enumerate(protocol['scenarios']):
            for arm in ('fresh','no_fresh'):
                relative=f'runs/executed_cql/development/seed{seed}/{scene}/{arm}'
                path=root/relative; old_path=cached_root/relative
                # Native trace has two rows, including one after first feedback.
                old=module.old.read(old_path/'summary.json'); old['factual_trace_rows']=2
                prior.write(old_path/'summary.json',old)
                value=copy.deepcopy(old)
                value['cell']['recipe']=protocol['recipe']
                value['replay'].update(covariate_window=285,
                    critic_fresh_outcome_window=30 if arm=='fresh' else 0,
                    independent_outcome_sampler_seed=seed+600000,
                    independent_outcome_sampler_calls=456,
                    maximum_fresh_outcome_age_in_admitted_labels=29 if arm=='fresh' else None,
                    no_fresh_original_batch_unchanged=arm=='no_fresh')
                if arm=='fresh':
                    cost=value['cost']; scalar=fresh[2*i+j]
                    cost.update(accrued_in_vehicle_seconds=100*scalar-4000.,
                        generalized_passenger_seconds=100*scalar,
                        generalized_seconds_per_departed_passenger=scalar)
                    prior.jsonl(path/'metrics.jsonl',[dict(time_seconds=t,
                        accrued_wait_seconds=cost['accrued_wait_seconds']*fraction,
                        accrued_in_vehicle_seconds=cost['accrued_in_vehicle_seconds']*fraction)
                        for t,fraction in ((9000,.3),(12000,.45),(18300,.8),(24000,1.))])
                prior.write(path/'summary.json',value)
                paths.append(path)
    return root,paths,cached_root


def test_parent_analysis_accepts_new_replay_recipe_and_all_three_comparators(tmp_path):
    root,_,_=populate(tmp_path)
    main,recency=module.analyze(root)
    assert main['status']=='complete' and main['integrity']['all_pass']
    assert main['decision']['scientific_gate_pass']
    assert main['recipe']=='recent30_fresh_value_cql'
    assert main['aggregates']['matched']['full_day']['overall']['mean_cost_difference']==-1.
    assert main['aggregates']['original_no_fresh']['full_day']['overall']['mean_cost_difference']==-2.
    assert recency['aggregates']['v68_no_fresh']['full_day']['overall']['mean_cost_difference']==-1.
    assert recency['aggregates']['v68_fresh']['full_day']['overall']['mean_cost_difference']==-3.
    assert all(all(cell['no_fresh_exact'].values()) for cell in recency['native_evidence'])
    assert all(cell['full_trace']['compared_rows']==2 for cell in recency['native_evidence'])
    assert main['acceptance_protocol']['early_window']=='descriptive_only'
    assert main['acceptance_protocol']['historical_v68_verdict_unchanged']
    assert main['decision']['ordinary_dripping_goal_achieved'] is False


@pytest.mark.parametrize('failure',('late_trace','trace_count','metrics','ledger','receipts','initializer','config','runtime'))
def test_native_no_fresh_must_reproduce_complete_cached_path(tmp_path,failure):
    root,paths,_=populate(tmp_path); path=paths[1]
    value=module.old.read(path/'summary.json')
    if failure=='late_trace':
        prior.jsonl(path/'prefix_trace.jsonl',[dict(started_at=1000,a=0),dict(started_at=1300,a=1.875)])
    elif failure=='metrics':
        rows=module.old.read_metrics(path,value); rows[0]['accrued_wait_seconds']+=1
        prior.jsonl(path/'metrics.jsonl',rows)
    elif failure=='ledger':value['cost']['accrued_wait_seconds']+=1
    elif failure=='receipts':value['receipts'][0]['sequence_ids'][0]=10000
    elif failure=='trace_count':value['factual_trace_rows']=3
    elif failure=='initializer':value['initializer']['path']='/server/other.pt'
    elif failure=='runtime':value['runtime_contract']['optimizer_schedule']='other'
    else:prior.modify(path/'protocol.json',('policy_config','ensemble_size'),4)
    prior.write(path/'summary.json',value)
    main,recency=module.analyze(root)
    assert main['status']=='integrity_failed'
    assert not main['decision']['scientific_gate_pass']
    assert any('/matched_v68_no_fresh/' in key for key in recency['integrity']['failed_checks'])


@pytest.mark.parametrize('field,value',(
    ('critic_fresh_outcome_window',285),
    ('independent_outcome_sampler_seed',6201),
    ('independent_outcome_sampler_calls',455),
    ('maximum_fresh_outcome_age_in_admitted_labels',30),
    ('covariate_window',30)))
def test_wrong_recency_or_current_state_window_blocks_admission(tmp_path,field,value):
    root,paths,_=populate(tmp_path)
    prior.modify(paths[0]/'summary.json',('replay',field),value)
    main,_=module.analyze(root)
    assert main['status']=='integrity_failed' and not main['decision']['scientific_gate_pass']
    assert any('/recency/' in key for key in main['integrity']['failed_checks'])


def test_missing_and_failed_native_work_is_never_zero_cost(tmp_path):
    root,paths,_=populate(tmp_path)
    (paths[0]/'summary.json').unlink()
    main,extra=module.analyze(root)
    assert main['status']=='pending' and not main['decision']['scientific_gate_pass']
    assert extra['aggregates']['v68_no_fresh']['full_day']['overall']['missing_cells']==1
    prior.write(paths[0]/'failure.json',dict(type='NativeError',consumed_budget=dict(policy_updates=16)))
    main,_=module.analyze(root)
    assert main['status']=='integrity_failed'
    assert len(main['failed_run_work'])==1
    assert main['failed_run_work'][0]['consumed_budget']==dict(policy_updates=16)


def test_missing_tail_endpoint_does_not_become_zero_or_pass(tmp_path):
    root,paths,_=populate(tmp_path)
    value=module.old.read(paths[0]/'summary.json')
    rows=module.old.read_metrics(paths[0],value)
    prior.jsonl(paths[0]/'metrics.jsonl',[row for row in rows if row['time_seconds']!=18300])
    main,_=module.analyze(root)
    tail=main['aggregates']['matched']['post_budget']['overall']
    assert tail['missing_cells']==1
    assert not main['decision']['post_budget_gate_pass']
    assert not main['decision']['scientific_gate_pass']


def test_full_trace_includes_a_final_clock_event_without_nonfinite_json(tmp_path):
    root,paths,cached=populate(tmp_path)
    current=paths[1]
    reference=cached/current.relative_to(root)
    for path in (current,reference):
        rows=[json.loads(line) for line in (path/'prefix_trace.jsonl').read_text().splitlines()]
        rows.append(dict(started_at=24000,a=0))
        prior.jsonl(path/'prefix_trace.jsonl',rows)
        prior.modify(path/'summary.json',('factual_trace_rows',),3)
    main,extra=module.analyze(root)
    assert main['integrity']['all_pass']
    assert extra['native_evidence'][0]['full_trace']['compared_rows']==3
    module.old.encoded(extra)


def test_early_harm_is_descriptive_but_tail_harm_closes_candidate(tmp_path):
    root,paths,_=populate(tmp_path)
    for path in paths[::2]:
        cost=module.old.read(path/'summary.json')['cost']
        prior.jsonl(path/'metrics.jsonl',[dict(time_seconds=t,
            accrued_wait_seconds=cost['accrued_wait_seconds']*fraction,
            accrued_in_vehicle_seconds=cost['accrued_in_vehicle_seconds']*fraction)
            for t,fraction in ((9000,.1),(12000,.45),(18300,.8),(24000,1.))])
    main,_=module.analyze(root)
    assert main['aggregates']['matched']['early']['overall']['mean_cost_difference']>0
    assert main['decision']['scientific_gate_pass']
    for path in paths[::2]:
        cost=module.old.read(path/'summary.json')['cost']
        prior.jsonl(path/'metrics.jsonl',[dict(time_seconds=t,
            accrued_wait_seconds=cost['accrued_wait_seconds']*fraction,
            accrued_in_vehicle_seconds=cost['accrued_in_vehicle_seconds']*fraction)
            for t,fraction in ((9000,.1),(12000,.45),(18300,.5),(24000,1.))])
    main,_=module.analyze(root)
    assert main['decision']['matched_cost_gate_pass']
    assert not main['decision']['post_budget_gate_pass']
    assert not main['decision']['scientific_gate_pass']


def test_fresh_collapsed_to_control_and_shock_harm_do_not_pass(tmp_path):
    root,_,_=populate(tmp_path,fresh=(99.,)*4)
    main,_=module.analyze(root)
    assert not main['decision']['matched_cost_gate_pass']
    assert not main['decision']['scientific_gate_pass']


def test_export_keeps_twelve_lines_all_heads_precision_and_three_small_files(tmp_path):
    root,paths,_=populate(tmp_path)
    lines=('102S','102X','122S','122X','311S','311X','406S','406X','705S','705X','7S','7X')
    windows=('early','late_training','pre_early')
    for path in paths:
        value=module.old.read(path/'summary.json')
        group=value['policy_drift']['groups'][0]
        value['policy_drift']['groups']=[dict(group,line=line,window=window,
            count=5 if i==35 else 8) for i,(line,window) in
            enumerate((line,window) for line in lines for window in windows)]
        prior.write(path/'summary.json',value)
    # Native-size ledgers exercise byte limits without rounding costs or
    # dropping line/head evidence to fit a tiny synthetic fixture.
    for file in tmp_path.rglob('summary.json'):
        value=module.old.read(file); scalar=value['cost']['generalized_seconds_per_departed_passenger']
        primary=2209.123456789+scalar-100.
        wait=8438598.256789+23.5136789123*scalar
        ride=15079*primary-2*wait
        value['cost']=dict(departed_passengers=15079,accrued_wait_seconds=wait,
            accrued_in_vehicle_seconds=ride,generalized_passenger_seconds=2*wait+ride,
            generalized_seconds_per_departed_passenger=(2*wait+ride)/15079)
        if 'cql_audit' in value:value['cql_audit']['regularizer_wall_seconds']=36.123456789
        prior.write(file,value)
        prior.jsonl(file.parent/'metrics.jsonl',[dict(time_seconds=t,
            accrued_wait_seconds=wait*fraction,accrued_in_vehicle_seconds=ride*fraction)
            for t,fraction in ((9000,.300123456),(12000,.450234567),(18300,.800345678),(24000,1.))])
    main,extra=module.analyze(root)
    assert main['integrity']['all_pass']
    receipt=module.export(root,root/'results/development.json')
    assert set(receipt['files'])=={'development.json','policy_drift.json','recency.json'}
    assert all(size<=module.MAX_BYTES for size in receipt['files'].values())
    cost=module.old.read(root/'results/development.json')
    policy=module.old.read(root/'results/policy_drift.json')
    assert cost['aggregates']==main['aggregates']
    assert module.old.read(root/'results/recency.json')==extra
    for cell in policy['native_evidence']:
        for arm in cell['policy_drift'].values():
            assert len({row[0] for row in arm['rows']})==12
            assert len(arm['rows'])==36
    for cell in cost['native_evidence']:
        for arm in cell['executed_cql'].values():
            assert len(arm['head_rows'])==5
            assert 'regularizer_wall_seconds' in arm


def test_oversized_export_writes_no_artifacts(tmp_path,monkeypatch):
    root,_,_=populate(tmp_path)
    main,extra=module.analyze(root); extra['overflow']='x'*module.MAX_BYTES
    monkeypatch.setattr(module,'analyze',lambda _: (main,extra))
    out=root/'results/development.json'
    with pytest.raises(ValueError,match='recency.json exceeds'):
        module.export(root,out)
    assert not out.parent.exists()
