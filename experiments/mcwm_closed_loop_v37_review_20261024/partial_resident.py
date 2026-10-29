"""Exploratory readout on existing admitted pairs; no SGD or model checkpoint."""
import json
from pathlib import Path
import sys

REPO=Path('/home/zhengliang01/scheduleurm_work/Thesis')
sys.path.insert(0,str(REPO))
from experiments.mcwm_closed_loop_paired_coverage_v37_20261024.difference import DirectDifferenceReadout

ROOT=REPO/'experiments/mcwm_closed_loop_paired_coverage_v37_20261024'


def read(path):return json.loads(path.read_text())


def model_choice(predicted,benefit):
    high=predicted>0
    return [predicted,int(high),None if benefit==0 else (predicted>0)-(predicted<0)==(benefit>0)-(benefit<0),
        max(-benefit,0.) if high else max(benefit,0.)]


def run():
    result=dict(status='partial_exploratory_only',long_Q_updates_executed=0,new_SGD_updates=0,
        missing_heldout=4,cases={},columns=dict(prediction=['benefit','high','direction_correct','regret'],
            costs=['wait','ride','cost'],features=['7X','102X','705X','headway_imbalance']))
    for scenario in ('stable','local_speed_drop'):
        out=ROOT/'runs'/'collection'/scenario
        dataset=read(out/'dataset.json');records=read(out/'pair_truth.json')
        train=[r for r in records if r['role']=='train' and r['selected_for_training']]
        assert len(train)==9
        readers=dict(line=DirectDifferenceReadout(line_only=True),lowdim=DirectDifferenceReadout(line_only=False))
        fitted=[]
        for packet in dataset['packets']:
            receipt=packet['receipt']
            new=[r for r in train if r['id'] in receipt['pair_ids']]
            if new:
                for reader in readers.values():reader.accept(new,now=receipt['processed_at']);reader.refit()
                fitted.extend(r['id'] for r in new)
        assert len(fitted)==len(set(fitted))==9
        case=dict(fits={name:reader.summary() for name,reader in readers.items()},heldout=[],
            pair_inputs=[],all_285_slots_unique=len({r['sequence'] for p in dataset['packets'] for r in p['rows']})==285,
            actual_ordinary_paired=[sum(r['sequence']>=0 for p in dataset['packets'] for r in p['rows']),
                sum(r['sequence']<0 for p in dataset['packets'] for r in p['rows'])],
            pair_source_factual_excluded=not {r['source_sequence'] for r in train}.intersection(
                r['sequence'] for p in dataset['packets'] for r in p['rows']),
            all_feedback_available_before_admit=all(r['available_at']<=p['receipt']['processed_at']
                for p in dataset['packets'] for r in p['rows']))
        for record in records:
            arms=record['outcomes'];td={a:arms[a]['transition'] for a in ('low','high')}
            a,b=td['low'],td['high']
            ns_delta=[abs(x-y) for x,y in zip(a['ns'],b['ns'])]
            m_delta=[abs(x-y) for x,y in zip(a['memory_ns']['vector'],b['memory_ns']['vector'])]
            case['pair_inputs'].append(dict(id=record['id'],role=record['role'],
                reward_high_minus_low=b['r']-a['r'],dt_high_minus_low=b['dt']-a['dt'],
                next_state_changed_columns=[i for i,v in enumerate(ns_delta) if v],
                next_state_delta_max=max(ns_delta),next_memory_delta_max=max(m_delta),
                next_memory_changed_columns=[i for i,v in enumerate(m_delta) if v],
                next_memory_feature_names=a['memory_ns']['feature_names'],
                next_memory_line_ids=a['memory_ns']['line_ids'],
                memory_current_identical=a['memory_s']['vector']==b['memory_s']['vector'],
                actual_actions_match=a['a']==record['actions']['low'] and b['a']==record['actions']['high'],
                feedback_pairs_causal=record['role']!='train' or record['admitted_at']>=record['available_at']))
            if record['role']!='heldout':continue
            assert all(arms[a]['truth']['status']=='complete' and arms[a]['truth']['dt']==300. for a in ('low','high'))
            gain={k:arms['low']['truth'][k]-arms['high']['truth'][k] for k in ('wait','ride','cost')}
            predictions={}
            for name,reader in readers.items():
                value=reader.predict(record['features'])
                predictions[name]=model_choice(value['gain']['cost'],gain['cost']) if value['status']=='supported' else None
            predictions['fixed0']=model_choice(0.,gain['cost'])
            predictions['fixedhigh']=model_choice(1.,gain['cost'])
            feature=record['features'][3]
            training=[r['features'][3] for r in train if r['line_id']==record['line_id']]
            case['heldout'].append(dict(id=record['id'],issue=record['issued_at'],features=record['features'],
                gain=gain,predictions=predictions,training_headway_range=[min(training),max(training)],
                headway_extrapolation=not min(training)<=feature<=max(training),
                train_label_age=[record['issued_at']-r['issued_at'] for r in train if r['line_id']==record['line_id']]))
        for row in case['pair_inputs']:
            row.pop('next_memory_feature_names');row.pop('next_memory_line_ids')
        case['coverage_by_line']={line:dict(train=sum(r['line_id']==line for r in train),
            heldout=sum(r['line_id']==line and r['role']=='heldout' for r in records)) for line in ('7X','102X','705X')}
        result['cases'][scenario]=case
    payload=json.dumps(result,separators=(',',':'),ensure_ascii=False)
    assert len(payload.encode())<=24000,len(payload.encode())
    print(payload)


if __name__=='__main__':run()
