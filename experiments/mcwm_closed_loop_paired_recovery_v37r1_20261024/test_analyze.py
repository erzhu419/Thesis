"""Recovery rights: no invented heldouts, exact joins, and separated MC noise."""
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v37r1_analysis_test', HERE/'analyze.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
LINES = ('7X','102X','705X')


def write(path, value, *, lines=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in value) if lines else json.dumps(value))


def record(index, line, window, benefit, *, role='train'):
    issue = 9600.+3000*(index//3) if role=='train' else (18400. if window=='early' else 21200.)
    high = dict(status='complete',dt=300.,wait=10000.,ride=20000.,cost=40000.)
    low = dict(high,wait=10000.+benefit,ride=20000.-benefit,cost=40000.+benefit)
    return dict(id=f'train/{index}' if role=='train' else f'heldout/{line}/{window}',
        role=role,index=index,line_id=line,window_name=window,issued_at=issue,
        source_sequence=100+index,features=[float(line==value) for value in LINES]+[(index//3-1)*.25],
        holds=[0.,1.875],selected_for_training=role=='train',available_at=issue+330.,admitted_at=issue+600.,
        outcomes={arm:dict(truth=truth,next_event=dict(status='complete',zero_duration=index%3==0,
            wait=0.,ride=0.,cost=0.)) for arm,truth in (('low',low),('high',high))},
        comparison=dict(physical_separation=dict(status='separable',departure_separable=True,
            next_arrival_separable=True)),pair_receipt=dict(overlap_seconds=12.3456789))


def target(index):
    value=(index+1)*.0123456789
    primary=dict(reward_scaled=0.,value_contribution=value,sac_contribution=-value/3,target=value*2/3)
    noise=.01987654321
    native=dict(primary,sac_contribution=primary['sac_contribution']+noise,target=primary['target']+noise)
    return dict(bellman_high_minus_low=primary,native_mc_bellman_high_minus_low=native,
        crn_sampling_noise={key:native[key]-primary[key] for key in primary},
        probe_contract=dict(primary='common_random_numbers_single_row_same_seed'))


def fixture(tmp_path):
    source,root=tmp_path/'source',tmp_path/'recovery/runs'
    write(source/'protocol.json',dict(scenarios=['stable','local_speed_drop'],focus_lines=list(LINES)))
    for scenario in ('stable','local_speed_drop'):
        native,shadow=source/'runs/collection'/scenario,root/scenario
        budget=dict(target_labels=285,packets=57,ordinary_labels=267,paired_action_labels=18,
            training_pairs=9,prefix_native_seconds=24000.,suffixes=26,suffix_native_seconds=7800,
            native_model_optimizer_updates=0)
        write(native/'summary.json',dict(integrity=dict(all_pass=False,checks=dict(total285=True,
            train9=True,all_declared_slots_selected=False,heldout6=False)),budget=budget))
        write(shadow/'summary.json',dict(integrity=dict(all_pass=True),optimizer_overlap_wall_seconds=12.3456789))
        train=[record(i,LINES[i%3],f'w{i//3+1}',20.*(i%3+1)+40.*(i//3-1)) for i in range(9)]
        benefits=[10.,20.,30.,0.] if scenario=='stable' else [-70.,-80.,-90.,-100.]
        heldout=[record(i,line,window,benefits[i],role='heldout') for i,(line,window) in enumerate(
            [('7X','early'),('102X','early'),('705X','early'),('705X','tail')])]
        write(native/'pair_truth.json',train+list(reversed(heldout)))
        write(native/'dataset.json',dict(packets=[dict(index=i+1,
            receipt=dict(pair_ids=[row['id']],processed_at=row['admitted_at'])) for i,row in enumerate(train)]))
        for arm in analysis.ARMS:
            write(shadow/arm/'summary.json',dict(integrity=dict(all_pass=True),
                budget=dict(feedback_slots=285,shadow_updates=456,ordinary_slots=267,
                    paired_slots=18,paired_zero_duration_rows=1),
                target_contract=dict(reward_scale=.0001,mixed_backup_lambda=.1,discount=1.)))
            scores=[]
            for i,row in enumerate(heldout):
                delta=.1 if arm=='zero_memory' else -.1
                if arm=='shared_level':delta=.1 if benefits[i]>0 else -.1
                scores.append(dict(slot=row['id'].removeprefix('heldout/'),sequence=row['source_sequence'],
                    issue=row['issued_at'],prediction=dict(delta_min=delta,min_heads=[0,1]),**target(i)))
            write(shadow/arm/'heldout.jsonl',scores,lines=True)
            write(shadow/arm/'pair_targets.jsonl',[dict(pair_index=i,phase=phase,
                prediction=dict(delta_min=.123456789),**target(i)) for i in range(9)
                for phase in analysis.STAGES],lines=True)
    return root,source


def test_partial_training_completion_has_exact_eight_real_heldouts_and_bounded_summary(tmp_path):
    root,source=fixture(tmp_path)
    result=analysis.analyze(root,source)
    assert result['status']=='partial_complete' and result['training_integrity']['all_pass']
    assert result['scientific_complete'] is False and result['formal_independent_gate'] is False
    assert len(result['states'])==8 and len(result['missing'])==4
    for case in result['cases'].values():
        assert case['heldout_coverage']==dict(expected=6,actual=4,missing=['102X/tail','7X/tail'])
        assert len(case['training_pairs'])==9 and all(value['pairs']==9 for value in case['finite_fit'].values())
        assert all(len(values)==9 and all(len(stages)==3 and all(len(stage)==7 for stage in stages)
            for stages in values) for values in case['pair_target_evolution'].values())
        assert case['pair_target_evolution']['zero_memory'][0][0][-1]==pytest.approx(.019877)
    stable=result['cases']['stable']['overall']['models']
    assert stable['fixed0']==[4,3,0,0,60.]
    assert stable['fixedhigh']==[4,3,3,4,0.]
    assert stable['shared_level']==[4,3,3,3,0.]
    fixedhigh_column=result['models'].index('fixedhigh')
    assert all(row[6][fixedhigh_column][0] is None for row in result['states'])
    assert set(result['cases']['stable']['finite300_heldout_mae'])=={'finite300_line','finite300_lowdim'}
    payload=json.dumps(result,separators=(',',':'),allow_nan=False)
    assert len(payload.encode())<=24000,len(payload.encode())


def test_holdout_truth_changes_regret_but_never_fixed_readout_fit(tmp_path):
    root,source=fixture(tmp_path)
    before=analysis.analyze(root,source)
    path=source/'runs/collection/stable/pair_truth.json'
    rows=json.loads(path.read_text())
    for row in rows:
        if row['role']=='heldout':
            for key in ('wait','ride','cost'):row['outcomes']['low']['truth'][key]+=10**6
    write(path,rows)
    after=analysis.analyze(root,source)
    assert before['cases']['stable']['finite_fit']==after['cases']['stable']['finite_fit']
    assert before['cases']['stable']['overall']!=after['cases']['stable']['overall']


@pytest.mark.parametrize('failure',['source_training','worker_budget','missing_stage','wrong_join','wrong_randomness'])
def test_training_or_join_failure_cannot_be_renamed_partial_success(tmp_path,failure):
    root,source=fixture(tmp_path)
    if failure=='source_training':
        path=source/'runs/collection/stable/summary.json';row=json.loads(path.read_text())
        row['integrity']['checks']['total285']=False;write(path,row)
    elif failure=='worker_budget':
        path=root/'stable/zero_memory/summary.json';row=json.loads(path.read_text())
        row['budget']['shadow_updates']=448;write(path,row)
    else:
        path=root/'stable/zero_memory'/('pair_targets.jsonl' if failure in ('missing_stage','wrong_randomness') else 'heldout.jsonl')
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if failure=='missing_stage':rows.pop()
        elif failure=='wrong_join':rows[0]['sequence']+=1
        else:rows[0]['probe_contract']['primary']='independent_draws'
        write(path,rows,lines=True)
    with pytest.raises(RuntimeError):analysis.analyze(root,source)
