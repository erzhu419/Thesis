"""Check actual-cost attribution, reused update budget and incomplete evidence."""
import json
from pathlib import Path
import pytest

from experiments.mcwm_closed_loop_support_td_v83_20261027 import analyze as A

PROTOCOL = dict(development_seeds=[6207,6208],scenarios=['stable','local_speed_drop'],
    external_seed_offset=100000,recipe='latest5_covariates_capacity_outcomes',
    full_day_departed_passengers=15079,v75_reference='v75',v81_reference='v81')


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def jsonl(path, values):
    path.write_text(''.join(json.dumps(value)+'\n' for value in values))


def populate(tmp_path, monkeypatch, *, deltas=(-2.,-2.,-2.,-2.)):
    root = tmp_path/'v83'
    write(root/'protocol.json',PROTOCOL)
    for index,(seed,scene) in enumerate((seed,scene) for seed in PROTOCOL['development_seeds']
            for scene in PROTOCOL['scenarios']):
        for label in ('fresh','no_fresh','v75_fresh','v75_no_fresh','v81_fresh','v81_no_fresh'):
            arm = A.native_arm(label)
            current = label in A.ARMS
            directory = A.run_dir(root,seed,scene,arm) if current else A.reference_dir(root,PROTOCOL,seed,scene,arm,label[:3])
            per = (100.+deltas[index] if arm == 'fresh' else 100.) if current else (101. if arm == 'fresh' else 100. if label.startswith('v81_') else 102.)
            n = PROTOCOL['full_day_departed_passengers']
            initializer = dict(path=f'resident/seed{seed}.pt',initial_updates=50000,history_rows=32768,source_rows=4096)
            counts = dict(critic=456,value=456,actor=456,alpha=456,policy=456,world=0,trust=0)
            budget = dict(native_seconds=24000,packets=57,target_feedback_slots=285,
                **{key+'_updates':value for key,value in counts.items()})
            summary = dict(status='complete',cell=dict(seed=seed,scenario=scene,arm=arm,
                candidate='executed_cql',recipe=PROTOCOL['recipe'],external_seed=seed+100000),
                runtime_contract=dict(F='fixed'),initializer=initializer,factual_trace_rows=2,receipts=[dict(same_native=True)],
                clock_audit=dict(first_feedback_applied_at=1205.),
                execution_wall=dict(native_started_wall=100.,native_finished_wall=200.),
                budget=budget,post_budget_counts=dict(before=counts,after=counts),
                td_gradient_gate=dict(all_pass=True,checks=dict(singleton_gradient_blocked=True)),
                replay=dict(arm=arm,covariate_records_received=285,
                    outcome_labels_received=285 if arm == 'fresh' else 0,
                    distinct_target_records_used=306 if arm == 'fresh' else 285),
                cost=dict(departed_passengers=n,accrued_wait_seconds=10*n,
                    accrued_in_vehicle_seconds=(per-20)*n,generalized_passenger_seconds=per*n,
                    generalized_seconds_per_departed_passenger=per))
            write(directory/'summary.json',summary)
            write(directory/'protocol.json',dict(policy_config=dict(ensemble_size=2,alpha=.2)))
            jsonl(directory/'prefix_trace.jsonl',[dict(started_at=1100.,original='same'),
                dict(started_at=1205.,arm=arm)])
            jsonl(directory/'context_issues.jsonl',[dict(sequence=0,ctx='same')])
            jsonl(directory/'context_selected.jsonl',[dict(packet=1,ctx='same')])
            jsonl(directory/'updates.jsonl',[dict(packet=index//2+1,
                step_in_packet=1 if index%2==0 else 8,loss=float(index),
                regularizer_wall_seconds=1. if current else 2.) for index in range(114)])
            import torch
            torch.save(dict(state=dict(weight=torch.tensor([1.,2.])),global_rng=dict(torch=torch.tensor([1])),
                outcome_rng=dict(state=123),config=dict(fixed=True),counts=counts,frozen_at=18000.),
                directory/'final_learner.pt')
            jsonl(directory/'metrics.jsonl',[dict(time_seconds=t,
                accrued_wait_seconds=summary['cost']['accrued_wait_seconds']*fraction,
                accrued_in_vehicle_seconds=summary['cost']['accrued_in_vehicle_seconds']*fraction)
                for t,fraction in ((9000,.3),(12000,.45),(18300,.8),(24000,1.))])
            if current:
                write(directory/'preflight.json',dict(technical_gate_pass=True,initializer=initializer,
                    first_feedback=dict(ids=[1,2,3,4,5]),execution_contract=dict(F='fixed')))
                write(directory/'context_summary.json',dict(integrity=dict(all_pass=True,
                    checks=dict(causal_issue=True,old_truth_excluded=True)),
                    first_packet=dict(actual_updates=8,reused_main_day=True,no_extra_steps=True,
                        checks=dict(Adam_continuous=True,RNG_preserved=True)),
                    issues=dict(counts=dict(observed=100)),selected=dict(counts=dict(covariates=285))))
    monkeypatch.setattr(A,'native_checks',lambda *args,**kwargs:dict(native_contract=True))
    return root


def modify(path, update):
    value = json.loads(path.read_text())
    update(value)
    write(path,value)


def test_eight_days_and_six_comparators_count_actual_budget(tmp_path,monkeypatch):
    root = populate(tmp_path,monkeypatch)
    main,evidence,pools,training = A.analyze(root)
    assert main['integrity']['all_pass'],main['integrity']['failed_checks']
    assert main['decision']['scientific_gate_pass']
    assert set(main['aggregates']) == set(A.COMPARISONS)
    assert main['aggregates']['matched']['full_day']['overall']['mean_cost_difference'] == -2.
    assert main['aggregates']['no_fresh_v75_shift']['full_day']['overall']['mean_cost_difference'] == -2.
    assert len(evidence['pairs']) == 4 and len(pools['cells']) == len(training['cells']) == 8
    budget = main['budget']
    assert budget['actual_native_seconds'] == 192000
    assert budget['actual_covariate_slots'] == 2280
    assert budget['actual_complete_fresh_results'] == 1140
    assert budget['actual_updates_per_optimizer'] == dict(critic=3648,value=3648,actor=3648,alpha=3648)
    assert budget['new_initialization_updates'] == budget['extra_first8_steps'] == 0


def test_positive_stable_tail_is_visible_without_new_hard_gate(tmp_path,monkeypatch):
    root = populate(tmp_path,monkeypatch)
    for seed in PROTOCOL['development_seeds']:
        directory = A.run_dir(root,seed,'stable','fresh')
        points = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
        for point in points:
            if point['time_seconds'] == 18300:
                point['accrued_in_vehicle_seconds'] -= 10*15079
        jsonl(directory/'metrics.jsonl',points)
    main = A.analyze(root)[0]
    assert main['decision']['scientific_gate_pass']
    tails = main['optimization_assessment']['stable_tail_non_degradation']['matched']
    assert tails['complete'] and not tails['every_cell_non_degrading']
    assert tails['worst_cost_difference'] > 0
    assert main['decision']['old_gate_reopened'] is False


def test_matching_win_does_not_hide_bad_old_absolute_reference(tmp_path,monkeypatch):
    root = populate(tmp_path,monkeypatch,deltas=(3.,3.,3.,3.))
    for seed in PROTOCOL['development_seeds']:
        for scene in PROTOCOL['scenarios']:
            paths=[A.run_dir(root,seed,scene,'no_fresh')/'summary.json',
                A.reference_dir(root,PROTOCOL,seed,scene,'no_fresh','v81')/'summary.json']
            def bad_baseline(row):
                cost = row['cost']
                n = cost['departed_passengers']
                cost.update(accrued_in_vehicle_seconds=90*n,generalized_passenger_seconds=110*n,
                    generalized_seconds_per_departed_passenger=110.)
            for path in paths:
                modify(path,bad_baseline)
                directory = path.parent
                summary = json.loads(path.read_text())
                points = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
                for point in points:
                    fraction = {9000:.3,12000:.45,18300:.8,24000:1.}[point['time_seconds']]
                    point['accrued_in_vehicle_seconds'] = summary['cost']['accrued_in_vehicle_seconds']*fraction
                jsonl(directory/'metrics.jsonl',points)
    main = A.analyze(root)[0]
    assert main['integrity']['all_pass']
    assert main['optimization_assessment']['conditions']['matched_full_day']
    assert not main['optimization_assessment']['conditions']['v75_absolute_full_day']
    assert not main['decision']['scientific_gate_pass']


def test_worst_cell_cannot_be_hidden_by_mean(tmp_path,monkeypatch):
    root = populate(tmp_path,monkeypatch,deltas=(-6.,-6.,-6.,1.))
    main = A.analyze(root)[0]
    assert main['integrity']['all_pass']
    assert main['optimization_assessment']['conditions']['matched_full_day']
    assert main['optimization_assessment']['matched_worst'] == 1.
    assert main['optimization_assessment']['v75_matched_worst'] == -1.
    assert not main['decision']['scientific_gate_pass']


@pytest.mark.parametrize('damage',['missing_arm','no_overlap','extra_preflight_updates','future_context','missing_tail'])
def test_incomplete_or_invalid_evidence_cannot_pass(tmp_path,monkeypatch,damage):
    root = populate(tmp_path,monkeypatch)
    directory = A.run_dir(root,6208,'stable','no_fresh')
    if damage == 'missing_arm':
        (directory/'summary.json').unlink()
    elif damage == 'no_overlap':
        modify(directory/'summary.json',lambda row:row.update(execution_wall=dict(native_started_wall=210.,native_finished_wall=310.)))
    elif damage == 'extra_preflight_updates':
        modify(directory/'context_summary.json',lambda row:row['first_packet'].update(actual_updates=16,no_extra_steps=False))
    elif damage == 'future_context':
        modify(directory/'context_summary.json',lambda row:row['integrity']['checks'].update(causal_issue=False))
    else:
        points = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
        jsonl(directory/'metrics.jsonl',[row for row in points if row['time_seconds'] != 18300])
    main = A.analyze(root)[0]
    assert not main['integrity']['all_pass']
    assert not main['decision']['scientific_gate_pass']
    if damage == 'missing_arm':
        assert main['budget']['actual_covariate_slots'] is None
        group=main['aggregates']['matched']['full_day']['overall']
        assert group['expected_cells']-group['complete_cells'] == 1


def test_four_bounded_exports(tmp_path,monkeypatch):
    root = populate(tmp_path,monkeypatch)
    result = A.export(root,root/'results/development.json')
    assert set(result['files']) == {'development.json','evidence.json','pools.json','training.json'}
    assert all(value <= 48000 for value in result['files'].values())


def test_full_precision_costs_and_all_references_still_export_small(tmp_path,monkeypatch):
    root=populate(tmp_path,monkeypatch)
    for index,(seed,scene) in enumerate((seed,scene) for seed in PROTOCOL['development_seeds']
            for scene in PROTOCOL['scenarios']):
        offsets=dict(fresh=(-31,13,7,-3)[index],no_fresh=0,
            v75_fresh=(-20,3,-4,5)[index],v75_no_fresh=(10,-4,8,2)[index],
            v81_fresh=(-25,7,1,4)[index],v81_no_fresh=0)
        for label,offset in offsets.items():
            arm=A.native_arm(label)
            directory=A.run_dir(root,seed,scene,arm) if label in A.ARMS else A.reference_dir(
                root,PROTOCOL,seed,scene,arm,label[:3])
            wait=8400000+seed*7+index*13+offset*70
            ride=16300000+seed*11+index*27+offset*39
            modify(directory/'summary.json',lambda row:row['cost'].update(accrued_wait_seconds=wait,
                accrued_in_vehicle_seconds=ride,generalized_passenger_seconds=2*wait+ride,
                generalized_seconds_per_departed_passenger=(2*wait+ride)/15079))
            jsonl(directory/'metrics.jsonl',[dict(time_seconds=t,accrued_wait_seconds=wait*f,
                accrued_in_vehicle_seconds=ride*f) for t,f in ((9000,.3),(12000,.45),(18300,.8),(24000,1.))])
    result=A.export(root,root/'results/development.json')
    assert all(value<=48000 for value in result['files'].values()),result['files']


@pytest.mark.parametrize('damage',['update','late_prefix','context','snapshot','gate'])
def test_no_fresh_drift_or_missing_td_gate_cannot_be_hidden(tmp_path,monkeypatch,damage):
    root=populate(tmp_path,monkeypatch)
    directory=A.run_dir(root,6208,'stable','no_fresh')
    if damage=='update':
        rows=[json.loads(line) for line in (directory/'updates.jsonl').read_text().splitlines()]
        rows[20]['loss']+=1e-10
        jsonl(directory/'updates.jsonl',rows)
    elif damage=='late_prefix':
        rows=[json.loads(line) for line in (directory/'prefix_trace.jsonl').read_text().splitlines()]
        rows[-1]['arm']='changed_late'
        jsonl(directory/'prefix_trace.jsonl',rows)
    elif damage=='context':
        jsonl(directory/'context_selected.jsonl',[dict(packet=1,ctx='different')])
    elif damage=='snapshot':
        import torch
        value=torch.load(directory/'final_learner.pt',weights_only=False)
        value['state']['weight'][0]+=1e-5
        torch.save(value,directory/'final_learner.pt')
    else:
        modify(directory/'summary.json',lambda row:row['td_gradient_gate']['checks'].update(singleton_gradient_blocked=False))
    main,evidence,*_=A.analyze(root)
    assert not main['integrity']['all_pass']
    assert not main['decision']['scientific_gate_pass']
    if damage!='gate':
        pair=next(row for row in evidence['pairs'] if row['seed']==6208 and row['scenario']=='stable')
        assert not pair['no_fresh_v81_regression']['all_pass']


def test_v81_fresh_benefit_cannot_be_lost_under_a_good_v75_comparison(tmp_path,monkeypatch):
    root=populate(tmp_path,monkeypatch)
    for seed in PROTOCOL['development_seeds']:
        for scene in PROTOCOL['scenarios']:
            directory=A.reference_dir(root,PROTOCOL,seed,scene,'fresh','v81')
            def better_reference(row):
                n=row['cost']['departed_passengers']
                row['cost'].update(accrued_in_vehicle_seconds=77*n,
                    generalized_passenger_seconds=97*n,generalized_seconds_per_departed_passenger=97.)
            modify(directory/'summary.json',better_reference)
            rows=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
            for row in rows:
                row['accrued_in_vehicle_seconds']=77*15079*{9000:.3,12000:.45,18300:.8,24000:1.}[row['time_seconds']]
            jsonl(directory/'metrics.jsonl',rows)
    main=A.analyze(root)[0]
    assert main['integrity']['all_pass']
    assert main['optimization_assessment']['conditions']['fresh_mean_not_worse_than_v75']
    assert not main['optimization_assessment']['conditions']['fresh_mean_not_worse_than_v81']
    assert not main['decision']['scientific_gate_pass']
