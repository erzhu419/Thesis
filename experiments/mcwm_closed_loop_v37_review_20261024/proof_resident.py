"""Only completed recovery checks and the one decision disagreement; no models."""
from pathlib import Path
import json

ROOT=Path('/home/zhengliang01/scheduleurm_work/Thesis/experiments/mcwm_closed_loop_paired_recovery_v37r1_20261024/runs')
ARMS=('zero_memory','per_head_level','shared_level')


def read(path):return json.loads(path.read_text())


def run():
    result=dict(cases={},critical_state={})
    initializers=[]
    for scenario in ('stable','local_speed_drop'):
        parent=read(ROOT/scenario/'summary.json')
        workers={arm:read(ROOT/scenario/arm/'summary.json') for arm in ARMS}
        for worker in workers.values():initializers.append(worker['initializer'])
        result['cases'][scenario]=dict(parent_checks=parent['integrity']['checks'],
            workers={arm:dict(checks=v['integrity']['checks'],
                initial_parity=v['initial_preflight']['all_pass'],
                proof_scope=v['factual_proof']['scope'],
                source_full_integrity=v['source_full_integrity'],
                budget=v['budget'],heldout=v['heldout']) for arm,v in workers.items()})
    result['same_initializer_all_six']=len(set(initializers))==1
    result['initializer']=initializers[0]
    for arm in ARMS:
        rows=[json.loads(s) for s in (ROOT/'local_speed_drop'/arm/'heldout.jsonl').read_text().splitlines()]
        r=next(r for r in rows if r['slot']=='705X/early')
        result['critical_state'][arm]=dict(prediction=r['prediction'],
            variants={k:{name:v[name] for name in ('delta_min','choice','min_heads')} for k,v in r['variants'].items()},
            components=r['components'])
    payload=json.dumps(result,separators=(',',':'))
    assert len(payload.encode())<=12000,len(payload.encode())
    print(payload)


if __name__=='__main__':run()
