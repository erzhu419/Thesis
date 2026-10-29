"""Audit existing support and bridge cached soft-Q to observed300s costs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:
        json.dump(value,handle,separators=(',',':'),allow_nan=False);handle.write('\n')


def rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cost(start,stop):
    return 2*(stop['accrued_wait_seconds']-start['accrued_wait_seconds'])+stop['accrued_in_vehicle_seconds']-start['accrued_in_vehicle_seconds']


def run(scenario):
    import numpy as np
    from experiments.mcwm_closed_loop_objective_alignment_v58_20261025.support import (
        inventory,validate_chain,validate_pair)
    from experiments.mcwm_closed_loop_objective_alignment_v58_20261025.objective import (
        objective_contract,bridge_chain,pair_bridge)
    p=read(HERE/'protocol.json');out=HERE/'runs'/scenario
    out.mkdir(parents=True,exist_ok=False);started=time.time()
    roots={key:HERE.parent/value/'runs'/scenario for key,value in p['sources'].items()}
    inv=inventory(HERE.parent,versions=tuple(p['sources']),scenarios=(scenario,))
    checks={};current=[];historical=[];short=[]
    current_parent=read(roots['v55']/'summary.json')
    checks['current_v55_parent_integrity']=current_parent['integrity']['all_pass']
    for slot in p['current_slots']:
        state=roots['v55']/'states'/slot.replace('/','_');record=read(state/'state_summary.json')
        meta={arm:read(state/arm/'summary.json') for arm in ('low','high')}
        lo,hi=meta['low'],meta['high']
        paired=(record['pair_receipt']['all_pass'] and lo['selected']==hi['selected']
            and lo['external_seed']==hi['external_seed']
            and lo['native']['other_batch_actions']==hi['native']['other_batch_actions']
            and all(m['counters']['before']==m['counters']['after'] for m in meta.values())
            and all(m['integrity']['all_pass'] for m in meta.values()))
        checks['v55_'+slot+'_paired_integrity']=paired
        first={arm:validate_chain([m['global_transition']],expected_root=m['selected'],
            counters=m['counters'],source_kind='summary_first_transition') for arm,m in meta.items()}
        full={arm:(state/arm/'events.jsonl').is_file() for arm in meta}
        if any(full.values()):
            raise RuntimeError('v55 has a completed-event writer; include its actual chain in this version before interpretation')
        low300,high300=lo['truth']['fixed300'],hi['truth']['fixed300']
        if not (low300['status']==high300['status']=='complete' and low300['dt']==high300['dt']==300.):
            raise RuntimeError('The existing exact300 ledger is not complete')
        benefit=low300['cost']-high300['cost']
        predictions={arm:record['prediction'][arm]['q_min'][1]-record['prediction'][arm]['q_min'][0]
            for arm in ('one_step','trace')}
        current.append(dict(slot=slot,issue=record['selected']['started_at'],same_root_seed_continuation=paired,
            holds=record['holds'],benefit300=benefit,scaled_benefit300=p['reward_scale']*benefit,
            root_Q_contrast=predictions,
            unidentified_non300_remainder={arm:value-p['reward_scale']*benefit for arm,value in predictions.items()},
            first_transition_support=first,full_event_chain_exists=full,
            future_reward_entropy_tail_split_supported=False,
            reason='Only first physical TD and fixed300 ledger saved; dynamic successor memory and all subsequent TDs are absent.'))
    parent=read(roots['v38']/'summary.json')
    scoring=read(roots['v38']/'scored/summary.json')
    checks['v38_completed_native_and_cached_score_gates']=(parent['integrity']['all_pass'] and scoring['integrity']['all_pass'])
    checks['v38_original285_456']=(parent['budget']['frozen_training_slots']==285 and
        parent['budget']['frozen_updates_per_model']==456)
    for slot in p['old_slots']:
        native=roots['v38']/'native/states'/slot.replace('/','_')
        cached=roots['v38']/'scored/states'/slot.replace('/','_')
        meta={arm:read(native/arm/'summary.json') for arm in ('low','high')}
        events={arm:rows(cached/arm/'events.jsonl') for arm in ('low','high')}
        support=validate_pair(events['low'],events['high'],meta['low'],meta['high'])
        checks['v38_'+slot+'_legal_pair']=support['paired_root_supported'] and support['both_chains_reach_300']
        endpoint_checks={};j300={}
        for arm in ('low','high'):
            m=meta[arm];last=events[arm][-1]['transition'];endpoint=m['real_endpoint'];boundary=m['boundary300']
            j300[arm]=cost(m['selected']['issue_ledger'],boundary['ledger'])
            endpoint_checks[arm]=(m['integrity']['all_pass'] and len(events[arm])==m['chain']['transition_count']
                and boundary['clock']==m['selected']['started_at']+300.
                and j300[arm]==m['truth']['fixed300']['cost']
                and endpoint['status']=='observed' and last['completed_at']==endpoint['clock']
                and last['ns']==endpoint['s'] and last['memory_ns']==endpoint['memory']
                and endpoint['issued'] is False and endpoint['executed'] is False)
        checks['v38_'+slot+'_real_endpoint_and_ledger']=all(endpoint_checks.values())
        models={}
        for model in p['cached_models']:
            bridges={}
            for arm,chain in events.items():
                sc=[r['scores'][model] for r in chain]
                exact=all(score['sequence']==r['transition']['sequence'] and
                    score['executed_action']==r['transition']['a'][0] and
                    score['reward_scaled']==float(np.float32(r['transition']['r'])*np.float32(p['reward_scale']))
                    for r,score in zip(chain,sc))
                if not exact:
                    raise RuntimeError('Cached score identity or reward units differ: '+slot+'/'+model+'/'+arm)
                arrays={key:[score[field] for score in sc] for key,field in
                    [('native','target'),('reward','reward_scaled'),('bootstrap','bootstrap_contribution'),('q_exec','q_min')]}
                arrays['entropy']=[.9*score['sac_entropy_bonus'] for score in sc]
                arrays['matching_q']=[score['target_q_min'] for score in sc[1:]]
                bridges[arm]=bridge_chain(chain,arrays,J300=j300[arm],reward_scale=p['reward_scale'])
                if abs(bridges[arm]['fixed300_Q_bridge']['identity_error'])>1e-7:
                    raise RuntimeError('Q/cost/entropy/tail additive bridge did not close')
            pair=pair_bridge(bridges['low'],bridges['high'])
            pair['target_lag_contrast']=(bridges['high']['Q_telescope']['target_Q_minus_online_Q_gap']-
                bridges['low']['Q_telescope']['target_Q_minus_online_Q_gap'])
            pair['mixed_vs_recorded_target_Q_contrast']=(bridges['high']['Q_telescope']['mixed_vs_recorded_target_Q_gap']-
                bridges['low']['Q_telescope']['mixed_vs_recorded_target_Q_gap'])
            models[model]=dict(pair=pair,arms=bridges)
        historical.append(dict(slot=slot,issue=meta['low']['selected']['started_at'],support=support,
            holds={arm:m['execution']['executed_hold_seconds'] for arm,m in meta.items()},
            benefit300=j300['low']-j300['high'],models=models,
            model_cohort='v37r1 cached M1 mixed-Q; original50000 initializer continuation, distinct from v54/v55 mu14'))
    for line in ('7X','102X','705X'):
        root=roots['v43']/'branches'/line
        meta={arm:read(root/arm/'summary.json') for arm in ('low','high')}
        events={arm:rows(root/arm/'events.jsonl') for arm in ('low','high')}
        pair=validate_pair(events['low'],events['high'],meta['low'],meta['high'])
        checks['v43_'+line+'_legal_short_pair']=pair['paired_root_supported']
        short.append(dict(line=line,support=pair,cohort='v43 frozen mu24; short45-event chains, no supplied300s outcome'))
    fact=rows(roots['v53']/'continuous_feedback.jsonl')
    factual=validate_chain(fact)
    checks['v53_real_factual_chain']=factual['legal'] and len(fact)==271
    value=dict(status='complete' if all(checks.values()) else 'integrity_failed',scenario=scenario,
        integrity=dict(all_pass=all(checks.values()),checks=checks),contract=objective_contract(),
        inventory=inv,current=current,historical=historical,short_pairs=short,
        factual_chain=dict(audit=factual,same_state_alternative_action=False,cohort='v53 mu14 one admitted factual continuation'),
        budget=dict(source_initializer_updates=50000,source_feedback_slots=285,source_optimizer_updates=456,
            horizon_seconds=24000,new_training_labels=0,new_optimizer_updates=0,new_native_branches=0,
            truth_admitted=0,checkpoints_loaded=0),wall=dict(started=started,finished=time.time()))
    write(out/'details.json',value)
    from experiments.mcwm_closed_loop_objective_alignment_v58_20261025.analyze import compact_case
    write(out/'summary.json',compact_case(value))
    if not all(checks.values()):
        raise RuntimeError('Source artifact integrity failed; preserve unsupported data rather than score it')
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario',choices=('stable','local_speed_drop'),required=True)
    args=parser.parse_args()
    try:
        value=run(args.scenario);print(json.dumps(dict(status=value['status'],scenario=args.scenario)))
    except BaseException as error:
        write(HERE/'runs'/args.scenario/'failure.json',dict(type=type(error).__name__,message=str(error),
            new_optimizer_updates=0,new_native_branches=0))
        raise
