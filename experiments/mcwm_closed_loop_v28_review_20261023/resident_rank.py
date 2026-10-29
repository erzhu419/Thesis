"""Read-only rank/alias counts from resident cumulative shadow records."""
import json
from collections import Counter,defaultdict
from pathlib import Path
import sys

def analyze(root):
    groups=defaultdict(Counter);largest=[];final=[];checks=defaultdict(Counter);configs=[];walls=[]
    sign=lambda v:int(v>0)-int(v<0)
    for seed in (6201,6202):
        for scenario in ('stable','local_speed_drop'):
            path=root/'runs'/f'seed{seed}'/scenario
            summary=json.loads((path/'summary.json').read_text())
            assert summary['status']=='complete' and summary['integrity']['all_pass']
            config=json.loads((path/'protocol.json').read_text())['policy_config']
            configs.append(dict(seed=seed,scenario=scenario,config={k:config.get(k) for k in ('actor_q_reduction','target_subset_size','backup_entropy','online_backup')}))
            walls.append(dict(seed=seed,scenario=scenario,**summary['execution_wall']))
            records=[json.loads(t) for t in (path/'shadows.jsonl').read_text().splitlines()]
            peak=None
            for record in records:
                deadline=record['metadata']['deadline']
                for name,passed in record['fresh_main_replay']['checks'].items():
                    checks['fresh_main'][name]+=passed is True
                for name,passed in record['integrity'].items():
                    checks['shadow'][name]+=passed is True
                for row in record['probe_rows']:
                    line=row['line_id'];h=row['holding_seconds']
                    delta=h[1]-h[2]
                    entry=dict(seed=seed,scenario=scenario,deadline=deadline,bank_id=row['bank_id'],line=line,
                        started_at=row['started_at'],holding_fresh=h[1],holding_no=h[2],delta_seconds=delta)
                    if peak is None or abs(delta)>abs(peak['delta_seconds']):peak=entry
                    for contrast,a,b in (('actor_candidates',1,2),('fixed_7.5_zero',4,3)):
                        for head in ('mean','min'):
                            g=groups[seed,scenario,line,head,contrast]
                            g['scored']+=1
                            if h[a]==h[b]:g['same_executed_action']+=1;continue
                            g['distinct_executed_actions']+=1
                            f=row['q_'+head]['fresh'][a]-row['q_'+head]['fresh'][b]
                            n=row['q_'+head]['no_fresh'][a]-row['q_'+head]['no_fresh'][b]
                            sf,sn=sign(f),sign(n)
                            if sf and sn:g['agree_nonzero' if sf==sn else 'disagree_nonzero']+=1
                            elif not sf and not sn:g['both_exact_tie']+=1
                            else:g['fresh_exact_tie' if not sf else 'no_exact_tie']+=1
                            if deadline==18000 and contrast=='fixed_7.5_zero':
                                final.append([seed,scenario,line,row['bank_id'],head,h[3],h[4],f,n,sf,sn])
            largest.append(peak)
    return dict(group_columns=['seed','scenario','line','head','contrast'],
        groups=[dict(key=list(key),counts=dict(v)) for key,v in sorted(groups.items())],
        largest_holding_difference=largest,final_fixed_candidates=final,
        final_columns=['seed','scenario','line','bank_id','head','holding_zero','holding_7_5','fresh_advantage','no_advantage','fresh_sign','no_sign'],
        policy_configs=configs,execution_walls=walls,
        exact_check_pass_counts={k:dict(v) for k,v in checks.items()},
        interpretation='Exact floating-point ties are separated. Unequal executed actions may have tiny Q margins; signs are descriptive, not realized control returns. Actor-candidate endpoints are shared between critics. Fixed0/7.5 are proposals passed through F.')

if __name__=='__main__':
    result=analyze(Path(sys.argv[1]))
    encoded=json.dumps(result,separators=(',',':'),allow_nan=False)
    assert len(encoded.encode())<=25000,len(encoded.encode())
    print(encoded)
