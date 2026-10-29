"""Read existing v37 outputs in place; print only the integrity/coverage summary."""
import json
from pathlib import Path

ROOT = Path('/home/zhengliang01/scheduleurm_work/Thesis/experiments/mcwm_closed_loop_paired_coverage_v37_20261024')


def read(path):
    return json.loads(path.read_text())


def run():
    result = dict(cases={}, final_summary_exists=(ROOT/'paired_coverage_summary.json').exists(),
        pair_columns=['id','role','index','issue','sequence','holds','admitted','available','admitted_at',
            'gain_wait_ride_cost','next_event_low_high','next_states_equal','next_memory_equal','physical',
            'pair_integrity','overlap'], next_event_columns=['dt','r','wait','ride','cost','zero_duration','terminal','applied_hold'],
        physical_columns=['status','physical_separable','departure_gap_seconds','next_same_vehicle_arrival_gap_seconds'])
    for scenario in ('stable','local_speed_drop'):
        out = ROOT/'runs'/'collection'/scenario
        case = dict(exists=out.exists(), shadow_exists=(ROOT/'runs'/'shadow'/scenario).exists())
        for name in ('failure.json','preflight_gate.json'):
            if (out/name).exists(): case[name.removesuffix('.json')] = read(out/name)
        if not (out/'summary.json').exists():
            result['cases'][scenario]=case
            continue
        value = read(out/'summary.json')
        case.update(status=value['status'],integrity=value['integrity'],
            seed=value['native_external_seed'], initialization_seed=value['initialization_seed'],
            full_prefix_trace=value['full_prefix_trace'], selection=value['selection'],
            wall=value['wall'],context_integrity=value['context'].get('integrity'),
            budget={k:v for k,v in value['budget'].items() if isinstance(v,(int,float))},
            feedback=dict(packets=len(value['receipts']),
                maximum_processed_clock=max(r['processed_at'] for r in value['receipts']),
                slot_count=sum(r['records'] for r in value['receipts']),
                paired_slots=sum(r['paired_records'] for r in value['receipts']),
                all_pair_atomic=all(r['paired_records'] in (0,2) for r in value['receipts']),
                max_pair_available=max((a for r in value['receipts'] for a in r['pair_available_at']),default=None)))
        records=read(out/'pair_truth.json')
        small=[]
        for row in records:
            arms=row['outcomes']
            td={a:arms[a]['transition'] for a in ('low','high')}
            gain={k:arms['low']['truth'][k]-arms['high']['truth'][k] for k in ('wait','ride','cost')}
            small.append(dict(id=row['id'],role=row['role'],index=row['index'],line=row['line_id'],
                issue=row['issued_at'],sequence=row['source_sequence'],holds=row['holds'],
                admitted=row.get('selected_for_training'),available=row['available_at'],
                admitted_at=row.get('admitted_at'),gain=gain,
                next_event={a:dict(dt=td[a]['dt'] if td[a] else None,r=td[a]['r'] if td[a] else None,
                    truth=arms[a]['next_event'],applied_hold=arms[a]['execution'].get('native_applied_hold_seconds')) for a in ('low','high')},
                next_states_equal=td['low']['ns']==td['high']['ns'] if all(td.values()) else None,
                next_memory_equal=td['low']['memory_ns']==td['high']['memory_ns'] if all(td.values()) else None,
                physical=row['comparison']['physical_separation'],
                pair_integrity=row['pair_receipt']['all_pass'],overlap=row['pair_receipt']['overlap_seconds']))
        case['pairs']=[[r['id'],r['role'],r['index'],r['issue'],r['sequence'],r['holds'],r['admitted'],r['available'],r['admitted_at'],
            [r['gain'][k] for k in ('wait','ride','cost')],
            [[r['next_event'][a]['dt'],r['next_event'][a]['r'],
                *[r['next_event'][a]['truth'].get(k) for k in ('wait','ride','cost','zero_duration','terminal')],
                r['next_event'][a]['applied_hold']] for a in ('low','high')],
            r['next_states_equal'],r['next_memory_equal'],
            [r['physical'].get(k) for k in result['physical_columns']],r['pair_integrity'],r['overlap']] for r in small]
        dataset=read(out/'dataset.json')
        case['dataset_metadata']=dataset['metadata']
        case['dataset_proof']=dict(all_pass=dataset['proof']['all_pass'],failed_checks=[k for k,v in dataset['proof']['checks'].items() if not v])
        case['worker_summaries']={arm:read(ROOT/'runs'/'shadow'/scenario/arm/'summary.json')
            for arm in ('zero_memory','per_head_level','shared_level')
            if (ROOT/'runs'/'shadow'/scenario/arm/'summary.json').exists()}
        result['cases'][scenario]=case
    payload=json.dumps(result,ensure_ascii=False,separators=(',',':'))
    assert len(payload.encode())<=24000,len(payload.encode())
    print(payload)


if __name__=='__main__':run()
