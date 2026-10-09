"""Report every planned scenario, pairing tests before aggregating training seeds."""

from alignment_common import HERE, P, label, write
import json
import statistics


def summarize(base=HERE, p=P):
    records, missing, trains = {}, [], 0
    for scene in p['scenarios']:
        scenario=scene['name']
        for arm in p['arms']:
            for seed in p['training_seeds']:
                folder=base/'runs'/label(arm,scenario,seed)/'results'
                path=folder/'ready.json'
                if path.exists():
                    row=json.loads(path.read_text())
                    if (row['total_policy_updates'],row['total_critic_updates']) != (70000,70000):
                        raise ValueError('Unexpected learner budget: '+str(path))
                    trains+=1
                else:
                    missing.append(str(path.relative_to(base)))
                for case in range(10):
                    path=folder/f'eval_{case:02d}.json'
                    if not path.exists():
                        missing.append(str(path.relative_to(base)));continue
                    row=json.loads(path.read_text())
                    block=p['evaluation_blocks'][case%5]
                    if (row['arm'],row['scenario'],row['training_seed'],row['case'],row['demand_seed'],row['sumo_seed'],row['deployment_hold_scale']) != (arm,scenario,seed,case,block['demand_seed'],block['sumo_seed'],1.0):
                        raise ValueError('Unexpected test identity: '+str(path))
                    records[scenario,arm,seed,case]=row
    result=dict(complete=not missing, training_complete=trains,evaluation_complete=len(records),missing=missing)
    if missing:
        return result
    summaries={}
    for scene in p['scenarios']:
        scenario=scene['name'];summaries[scenario]={}
        for i,scale in enumerate(p['evaluation_demand_scales']):
            paired=[]
            for seed in p['training_seeds']:
                values={}
                for arm in p['arms']:
                    rows=[records[scenario,arm,seed,c] for c in range(i*5,(i+1)*5)]
                    values[arm]=dict(cost=statistics.mean(r['generalized_seconds_per_departed_passenger'] for r in rows),
                                     completion=statistics.mean(r['passengers']['completion_rate'] for r in rows))
                paired.append(dict(seed=seed,**values,cost_difference=values['mcwm']['cost']-values['h2o_matched']['cost']))
            summaries[scenario][str(scale)]=dict(by_seed=paired,
                mcwm_mean=statistics.mean(r['mcwm']['cost'] for r in paired),
                h2o_mean=statistics.mean(r['h2o_matched']['cost'] for r in paired),
                improved_seeds=sum(r['cost_difference']<0 for r in paired))
    result['results']=summaries
    return result


if __name__=='__main__':
    result=summarize();write(HERE/'summary.json',result);print(json.dumps(result,indent=2))
