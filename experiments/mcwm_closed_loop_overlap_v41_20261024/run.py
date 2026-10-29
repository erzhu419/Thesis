"""Resident-only frozen parameters; two scenario workers, no simulation/SGD."""
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
import numpy as np
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.learning import load_learner
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.frozen import freeze
from experiments.mcwm_closed_loop_multievent_credit_v38_20261024.frozen import frozen_state,unchanged
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _rng
from experiments.mcwm_closed_loop_trace_support_v39_20261024.context import behavior_context
from experiments.mcwm_closed_loop_trace_design_v40_20261024.behavior import actor_parameters,execution_measure
from experiments.mcwm_closed_loop_trace_design_v40_20261024.run import target_parameters
from experiments.mcwm_closed_loop_overlap_v41_20261024.attribution import decompose_log_c
from experiments.mcwm_closed_loop_overlap_v41_20261024.overlap import geometry_at,conditional_overlap


def read(path): return json.loads(Path(path).read_text())


def lines(path):
    with Path(path).open() as handle:return [json.loads(r) for r in handle if r.strip()]


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:
        json.dump(value,handle,separators=(',',':'),allow_nan=False);handle.write('\n')


def prefix_metrics(rows,count):
    chosen=rows[:count]
    old=math.fsum(r['attribution']['actual_log_c'] for r in chosen)
    geom=math.fsum(r['overlap']['expected_log_c_under_behavior'] for r in chosen)
    arith=math.fsum(r['overlap']['log_overlap'] for r in chosen)
    terms={key:math.fsum(r['attribution']['shapley_log_c'][key] for r in chosen)
           for key in ('gate','mean','std')}
    input_term=math.fsum(r['attribution']['baseline_log_c'] for r in chosen)
    overlaps=[r['overlap']['total_overlap'] for r in chosen]
    return dict(links=count,L_old=old,L_geom=geom,L_arith=arith,
        old_minus_geom=old-geom,jensen_gap=arith-geom,
        input_log_c=input_term,shapley=terms,
        closure_error=math.fsum([input_term,*terms.values()])-old,
        discount_log_weight=count*math.log(.9),
        old_total_log_weight=count*math.log(.9)+old,
        fixed_state_geom_total_log_weight=count*math.log(.9)+geom,
        fixed_state_arith_total_log_weight=count*math.log(.9)+arith,
        overlap_min=min(overlaps),overlap_mean=math.fsum(overlaps)/count,
        overlap_max=max(overlaps),
        atom_overlap_mean=math.fsum(r['overlap']['atom_overlap'] for r in chosen)/count,
        continuous_overlap_mean=math.fsum(r['overlap']['continuous_overlap'] for r in chosen)/count)


def worker(scenario,reference,contract,stations,p,out):
    out.mkdir(parents=True,exist_ok=False)
    started=time.time();before,rng=frozen_state(reference),_rng()
    v40=HERE.parent/p['source_v40'];v38=HERE.parent/p['source_v38']
    v39=HERE.parent/p['source_v39'];recovered=HERE.parent/p['source_v39r1']
    source=read(v40/'analysis/summary.json')
    if not source['integrity']['all_pass'] or not read(v40/'review/summary.json')['integrity']['all_pass']:
        raise RuntimeError('Completed v40 design and credit evidence required')
    proofs=Counter();maxima=Counter();models={name:[] for name in p['models']}
    for state in read(v38/'runs'/scenario/'native/summary.json')['states']:
        slot=read(state['path'])['slot'];native=Path(state['path']).parent
        for arm in ('low','high'):
            branch=read(native/arm/'summary.json')
            relative=native.relative_to(v38/'runs'/scenario/'native')/arm/'events.jsonl'
            raw=[r['transition'] for r in lines(v38/'runs'/scenario/'scored'/relative)[:45]]
            contexts=[];actual=[];visible=[];geometries=[];self_errors=[]
            for row in raw[1:]:
                context=behavior_context(row,branch['selected'],branch['execution'],contract,stations)
                if not context['known']:raise RuntimeError('Actual old behavior input is missing')
                same=np.array_equal(context['state'],np.asarray(row['s'],np.float32))
                ap=actor_parameters(reference,context['state'])
                vp=ap if same else actor_parameters(reference,row['s'])
                geometry=geometry_at(reference.action_transform,context['state'])
                target_geometry=geometry if same else geometry_at(reference.action_transform,row['s'])
                if geometry['map_key']!=target_geometry['map_key']:
                    raise RuntimeError('Actual input change also changed F; same-map analytic diagnosis unavailable')
                check=conditional_overlap(reference.action_transform,context['state'],ap,ap,geometry=geometry)
                self_errors.append(max(abs(check['total_overlap']-1.),abs(check['expected_log_c_under_behavior'])))
                contexts.append({k:v for k,v in context.items() if k!='state'})
                actual.append(ap);visible.append(vp);geometries.append(geometry)
                proofs['physical_links']+=1;proofs['reconstructed_inputs']+=not same
            maxima['self_overlap_error']=max(maxima['self_overlap_error'],*self_errors)
            for model in p['models']:
                filename=slot.replace('/','_')+'_'+arm+'.jsonl'
                cache_path=recovered/'runs'/scenario/model/filename
                if not cache_path.exists():cache_path=v39/'runs'/scenario/model/filename
                cache=lines(cache_path)[:45]
                previous=lines(v40/'runs'/scenario/model/filename)
                output=[]
                for i,row in enumerate(raw[1:],1):
                    saved=cache[i]['support'];target=target_parameters(saved)
                    if cache[i]['sequence']!=row['sequence'] or previous[i-1]['sequence']!=row['sequence']:
                        raise RuntimeError('Source sequence and scored state are not aligned')
                    attr=decompose_log_c(reference.action_transform,row['s'],row['a'],visible[i-1],target,
                        geometry=saved['geometry'],denominator_params=actual[i-1])
                    overlap=conditional_overlap(reference.action_transform,row['s'],actual[i-1],target,
                        geometry=geometries[i-1])
                    reverse=conditional_overlap(reference.action_transform,row['s'],target,actual[i-1],
                        geometry=geometries[i-1])
                    if attr['attribution_status']!='complete' or overlap['expected_log_c_under_behavior'] is None:
                        write(out/'unsupported.json',dict(slot=slot,arm=arm,model=model,event=i,attribution=attr,overlap=overlap))
                        raise RuntimeError('Unresolved measure is not a zero contribution')
                    old=previous[i-1]['ratio']
                    exact=all(attr[key]==old[field] for key,field in (
                        ('actual_log_c','log_c'),('actual_log_ratio','log_ratio'),('actual_c','c')))
                    proofs['model_links']+=1;proofs['v40_exact_links']+=exact
                    maxima['shapley_closure_error']=max(maxima['shapley_closure_error'],abs(attr['closure_error']['log_c']))
                    maxima['normalization_error']=max(maxima['normalization_error'],overlap['normalization_error'])
                    maxima['overlap_symmetry_error']=max(maxima['overlap_symmetry_error'],
                        abs(overlap['total_overlap']-reverse['total_overlap']))
                    maxima['expectation_identity_error']=max(maxima['expectation_identity_error'],overlap['expectation_identity_error'])
                    maxima['jensen_violation']=max(maxima['jensen_violation'],-overlap['jensen_gap'],overlap['log_overlap'])
                    output.append(dict(event=i,sequence=row['sequence'],dt=row['dt'],
                        behavior_context=contexts[i-1],actual_parameters=actual[i-1],visible_parameters=visible[i-1],
                        target_parameters=target,attribution=attr,overlap=overlap,v40_logc_exact=exact,
                        diagnostic_only=True,feedback_admitted=0))
                # Join the existing reward ordinal only after all44 state/action diagnostics.
                old_model=source['cases'][scenario]['models'][model]
                ordinal=next(r[3] for r in old_model['rows'] if r[0]==slot and r[1]==arm)
                if ordinal is None or not 0<ordinal<=len(output):
                    raise RuntimeError('Existing reward ordinal is outside fixed diagnostic rows')
                first=prefix_metrics(output,ordinal);full=prefix_metrics(output,len(output))
                maxima['cumulative_closure_error']=max(maxima['cumulative_closure_error'],abs(first['closure_error']),abs(full['closure_error']))
                result=dict(slot=slot,arm=arm,first_reward=first,full44=full,
                    dt0=sum(row['dt']==0 for row in raw),input_changes=sum(c['reconstructed'] for c in contexts))
                models[model].append(result)
                path=out/model/filename;path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('x') as handle:
                    for record in output:
                        json.dump(record,handle,separators=(',',':'),allow_nan=False);handle.write('\n')
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='overlap',scenario=scenario,
                slot=slot,arm=arm,completed=len(models[p['models'][0]]),total=12)),flush=True)
    proof=unchanged(reference,before,rng)
    checks=dict(reference_frozen=proof['all_pass'],exact_v40=proofs['model_links']==proofs['v40_exact_links'],
        links_complete=proofs['physical_links']==528 and proofs['model_links']==1584,
        math_precision=all(value<=5e-12 for value in maxima.values()),
        all12_chains=all(len(rows)==12 for rows in models.values()))
    write(out/'summary.json',dict(status='complete',scenario=scenario,models=models,
        integrity=dict(all_pass=all(checks.values()),checks=checks),counts=dict(proofs),errors=dict(maxima),
        frozen=proof,wall=dict(started=started,finished=time.time()),
        new_native_seconds=0,new_updates=0,new_feedback=0))


def main():
    p=read(HERE/'protocol.json');runs=HERE/'runs';runs.mkdir(exist_ok=False)
    release=read(HERE.parent/p['source_v40']/'release_plan.json')
    if release!=read(HERE/'release_plan.json') or not release['all_pass']:
        raise RuntimeError('v40 final release plan changed')
    reference,replay,contract,stations,_=load_learner(p['seed'],arm='fresh');freeze(reference)
    if reference.total_updates!=50000 or replay.target_received:
        raise RuntimeError('Original initializer was modified')
    before,rng=frozen_state(reference),_rng();children={}
    sys.stdout.flush();sys.stderr.flush()
    for scenario in p['scenarios']:
        pid=os.fork()
        if pid==0:
            log=os.open(runs/(scenario+'.log'),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            os.dup2(log,1);os.dup2(log,2);os.close(log)
            try:
                worker(scenario,reference,contract,stations,p,runs/scenario)
                sys.stdout.flush();sys.stderr.flush();os._exit(0)
            except BaseException as error:
                write(runs/scenario/'failure.json',dict(type=type(error).__name__,message=str(error)))
                import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
        children[scenario]=pid
    exits={key:os.waitstatus_to_exitcode(os.waitpid(pid,0)[1]) for key,pid in children.items()}
    parent=unchanged(reference,before,rng)
    from experiments.mcwm_closed_loop_overlap_v41_20261024.analyze import analyze
    value=analyze(runs,HERE/'analysis',parent,exits)
    print(json.dumps(dict(status=value['status'],integrity=value['integrity'],decision=value['decision'])),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        if not isinstance(error,SystemExit):
            write(HERE/'failure.json',dict(type=type(error).__name__,message=str(error),new_updates=0,new_native_seconds=0))
        raise
