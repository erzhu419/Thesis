"""Complete stable/shared v39, retaining five models and eleven finished chains."""
from pathlib import Path
import json
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from experiments.mcwm_closed_loop_trace_recovery_v39r1_20261024.evaluate import (
    read,write,jsonlines,evaluate_chain,budget_chains,
)
from experiments.mcwm_closed_loop_trace_support_v39_20261024.budget import build_budget_index
from experiments.mcwm_closed_loop_trace_support_v39_20261024.analyze import analyze
from experiments.mcwm_closed_loop_multievent_credit_v38_20261024.frozen import load_frozen,frozen_state,unchanged
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.learning import load_learner
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.frozen import freeze
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _rng


def run():
    source=HERE.parent/'mcwm_closed_loop_trace_support_v39_20261024'
    p=read(source/'protocol.json')
    v37=HERE.parent/p['source_v37_directory'];v38=HERE.parent/p['source_v38_directory']
    data=v37/'runs/collection/stable'
    old=source/'runs/stable/shared_level'
    failure=read(old/'failure.json')
    if failure['message']!='Identical recorded actions differ in the native two-action target batch':
        raise RuntimeError('Recovery source is not the diagnosed native batch-position stop')
    out=HERE/'runs/stable/shared_level';out.mkdir(parents=True,exist_ok=False)
    started=time.time()
    learners,contract,stations,model_gate=load_frozen('stable',HERE.parent/p['source_v37r1_directory']/'runs',v37,p['diagnostic_seed'])
    learner=learners['shared_level']
    behavior,replay,bc,bs,_=load_learner(p['seed'],arm='fresh');freeze(behavior)
    if behavior.total_updates!=50000 or replay.target_received or bc!=contract or (bs!=stations).any():
        raise RuntimeError('The deterministic source behavior changed')
    index=build_budget_index(read(data/'dataset.json'),pair_truth=read(data/'pair_truth.json'),prefix_id=str(data))
    before,bb,rng=frozen_state(learner),frozen_state(behavior),_rng()
    heldout=[];reused=0;new_chains=[]
    for state in read(v38/'runs/stable/native/summary.json')['states']:
        record=read(state['path']);slot=record['slot'];native_root=Path(state['path']).parent
        identity=read(v38/'analysis/stable'/(slot.replace('/','_')+'.json'))['analysis']['semantic_alignment']['first_differences']['r']['identity']
        for arm in ('low','high'):
            branch=read(native_root/arm/'summary.json')
            relative=native_root.relative_to(v38/'runs/stable/native')/arm/'events.jsonl'
            rows=jsonlines(v38/'runs/stable/scored'/relative)
            name=slot.replace('/','_')+'_'+arm+'.jsonl';saved=old/name
            if not saved.exists() and (slot,arm)!=('705X/late','high'):
                raise RuntimeError('An additional completed source chain is missing')
            result=evaluate_chain(learner,behavior,contract,stations,rows,
                dict(scenario='stable',prefix_id=str(data),kind='heldout',branch_id=slot,arm=arm),
                index,cached=True,selected=branch['selected'],execution=branch['execution'],
                reward_identity=identity,out=out/name,seed=p['diagnostic_seed'],
                reused_events=saved if saved.exists() else None)
            result.update(slot=slot,arm=arm);heldout.append(result)
            reused+=saved.exists()
            if not saved.exists():new_chains.append(slot+'/'+arm)
    admitted=[]
    for number,group in enumerate(budget_chains(index)):
        result=evaluate_chain(learner,behavior,contract,stations,group['rows'],group['origin'],index,
            cached=False,out=out/('budget_'+str(number)+'.jsonl'),seed=p['diagnostic_seed'])
        result.update(id=group['id']);admitted.append(result)
    proof=unchanged(learner,before,rng);bp=unchanged(behavior,bb,rng)
    checks=dict(model_frozen=proof['all_pass'],behavior_frozen=bp['all_pass'],
        all_old_chain_native_targets_exact=all(c['cached_v38_parts_exact'] for c in heldout),
        lambda_zero_all_exact=all(c['lambda_zero_bitwise_exact'] for c in heldout+admitted),
        all_real_native_targets=all(c['all_native_targets_equal_actual_learner'] for c in heldout+admitted),
        all_subset_links_consistent=all(c['subset_consistent'] for c in heldout+admitted),
        all_285_physical_sources_authorized=sum(c['source_budget']['authorized_rows'] for c in admitted)==285,
        all_heldout_feedback_excluded=all(c['source_budget']['authorized_rows']==0 for c in heldout),
        all12_suffix_logs=len(heldout)==12,reused11=reused==11,
        only_known_missing_chain=new_chains==['705X/late/high'])
    complete=all(checks.values())
    write(out/'summary.json',dict(status='complete' if complete else 'integrity_failed',
        integrity=dict(all_pass=complete,checks=checks),model='shared_level',heldout=heldout,admitted=admitted,
        frozen=proof,behavior_frozen=bp,source_slots=285,source_updates=456,
        new_updates=0,new_feedback=0,new_native_seconds=0,
        wall=dict(started=started,finished=time.time()),
        recovery=dict(source_failure=str(old/'failure.json'),reused_chains=reused,new_chains=new_chains,
            new_heldout_probe_rows=sum(c['events'] for c in heldout if c['reused_events'] is None),
            new_source_budget_probe_rows=sum(c['events'] for c in admitted))))
    for scenario in p['scenarios']:
        target=HERE/'runs'/scenario;target.mkdir(parents=True,exist_ok=True)
        write(target/'input_gate.json',read(source/'runs'/scenario/'input_gate.json'))
        for name in p['models']:
            if scenario=='stable' and name=='shared_level':continue
            original=source/'runs'/scenario/name/'summary.json';value=read(original)
            if not value['integrity']['all_pass']:raise RuntimeError('An intended reused model is incomplete')
            value['reused_from']=str(original);write(target/name/'summary.json',value)
        original=read(source/'runs'/scenario/'summary.json')
        write(target/'summary.json',dict(status='complete' if complete else 'integrity_failed',
            integrity=dict(all_pass=complete),scenario=scenario,
            actual_three_model_overlap_seconds=original['actual_three_model_overlap_seconds'],
            original_result=original,recovery_single_model=(scenario=='stable'),
            new_updates=0,new_feedback=0,new_native_seconds=0))
    summary=analyze(HERE/'runs',HERE/'analysis')
    numerical=[c for c in heldout+admitted if c['duplicate_slot_differences']]
    write(HERE/'recovery_proof.json',dict(integrity=dict(all_pass=complete),
        models_reused=5,completed_chains_reused=reused,new_chains=new_chains,
        diagnostic_scope='fixed native two-action batch position0, same target subset/min; repeated call exact',
        affected_chains=len(numerical),max_slot_output_ulp=max((c['maximum_duplicate_slot_ulp'] for c in numerical),default=0),
        first_position_difference=next((c['duplicate_slot_witness'] for c in numerical),None),
        old_v39_results_preserved=True,new_native_seconds=0,new_training_labels=0,new_optimizer_updates=0,
        summary_status=summary['status']))
    return complete


if __name__=='__main__':
    try:
        if not run():sys.exit(1)
    except BaseException as error:
        if not isinstance(error,SystemExit):
            write(HERE/'failure.json',dict(type=type(error).__name__,message=str(error),new_native_seconds=0,new_updates=0))
        raise
