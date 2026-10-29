"""Existing journals and cached distribution controls; no weights are loaded."""
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
import torch
from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
from experiments.mcwm_closed_loop_overlap_v41_20261024.overlap import geometry_at
from experiments.mcwm_closed_loop_sync_v42_20261024.controls import oracle_gate_behavior,score,summarize
from experiments.mcwm_closed_loop_sync_v42_20261024.journal import summarize_model


def read(path):return json.loads(Path(path).read_text())


def lines(path):
    with Path(path).open() as handle:return [json.loads(row) for row in handle if row.strip()]


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:
        json.dump(value,handle,separators=(',',':'),allow_nan=False);handle.write('\n')


def inventory(p):
    result={}
    for directory in p['inventory_versions']:
        root=HERE.parent/directory
        files=[dict(path=str(path.relative_to(root)),bytes=path.stat().st_size)
               for path in sorted(root.rglob('*')) if path.is_file()
               and path.suffix in ('.pt','.pth','.ckpt','.pkl')]
        result[directory]=dict(files=files,
            saved50024_name_candidates=[r['path'] for r in files if '50024' in r['path']],
            source_scope='named historical experiment directory only')
    initial=HERE.parent/p['source_inputs']/'h2oplus_seed6202.pt'
    return dict(versions=result,initializer=dict(path=str(initial),bytes=initial.stat().st_size,
        updates=50000,source='original initializer already certified by v40/v41'),
        actual_post24_snapshot_available=False,
        snapshot_limits='Only original50000 and completed50456 policies are saved; summaries do not reconstruct intermediate weights.')


def worker(scenario,p,transform,out):
    out.mkdir(parents=True,exist_ok=False);started=time.time();rng=torch.get_rng_state().clone()
    v41=HERE.parent/p['source_v41'];v38=HERE.parent/p['source_v38']
    source=read(v41/'analysis/summary.json')
    if not source['integrity']['all_pass']:raise RuntimeError('Complete v41 evidence required')
    journals={model:summarize_model(HERE.parent/p['source_training']/'runs'/scenario/model)
              for model in p['models']}
    models={model:[] for model in p['models']};counts=Counter();errors=Counter()
    for state in read(v38/'runs'/scenario/'native/summary.json')['states']:
        slot=read(state['path'])['slot'];native=Path(state['path']).parent
        for arm in ('low','high'):
            relative=native.relative_to(v38/'runs'/scenario/'native')/arm/'events.jsonl'
            raw=[r['transition'] for r in lines(v38/'runs'/scenario/'scored'/relative)[:45]]
            geometries=[geometry_at(transform,row['s']) for row in raw[1:]]
            for model in p['models']:
                filename=slot.replace('/','_')+'_'+arm+'.jsonl'
                cached=lines(v41/'runs'/scenario/model/filename)
                values={'baseline':[],'oracle_gate_alignment':[]}
                for i,row in enumerate(raw[1:],1):
                    old=cached[i-1]
                    if old['sequence']!=row['sequence'] or old['overlap']['map_key']!=geometries[i-1]['map_key']:
                        raise RuntimeError('Cached distribution/state or native F map changed')
                    actual,target=old['actual_parameters'],old['target_parameters']
                    kwargs=(transform,row['s'],row['a'])
                    point=old['attribution']['geometry']
                    baseline=score(*kwargs,actual,target,point,geometries[i-1])
                    control=score(*kwargs,oracle_gate_behavior(actual,target),target,point,geometries[i-1])
                    exact=all(baseline[key]==old['attribution'][field]
                              for key,field in (('log_c','actual_log_c'),('log_ratio','actual_log_ratio'),('c','actual_c')))
                    counts['links']+=1;counts['exact_v41_links']+=exact
                    counts['input_reconstructed_links']+=old['behavior_context']['reconstructed']
                    for key,field in (('expected_log_c','expected_log_c_under_behavior'),('overlap','total_overlap')):
                        errors['v41_overlap_replay_error']=max(errors['v41_overlap_replay_error'],
                            abs(baseline[key]-old['overlap'][field]))
                    for record in (baseline,control):
                        errors['normalization_error']=max(errors['normalization_error'],record['normalization_error'])
                        errors['jensen_violation']=max(errors['jensen_violation'],-record['jensen_gap'])
                    values['baseline'].append(baseline);values['oracle_gate_alignment'].append(control)
                # Existing reward ordinal only scopes results after all44 links.
                old_rows=source['cases'][scenario]['models'][model]['first_reward_rows']
                k=next(row[2] for row in old_rows if row[0]==slot and row[1]==arm)
                results={key:dict(first_reward=summarize(rows,k),full44=summarize(rows,44))
                         for key,rows in values.items()}
                models[model].append(dict(slot=slot,arm=arm,comparisons=results,
                    actual_post24_comparison=dict(status='unavailable',overlap=None,credit=None),
                    oracle_is_issue_visible=False,training_authorized=False))
                write(out/model/(slot.replace('/','_')+'_'+arm+'.json'),dict(
                    links=[dict(event=i+1,sequence=raw[i+1]['sequence'],
                        baseline=values['baseline'][i],oracle_gate_alignment=values['oracle_gate_alignment'][i])
                        for i in range(44)],diagnostic_only=True))
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='sync_diagnostic',scenario=scenario,
                slot=slot,arm=arm,completed=len(models[p['models'][0]]),total=12)),flush=True)
    checks=dict(exact_v41=counts['links']==counts['exact_v41_links']==1584,
        all57_packet_logs=all(value['packets']==57 and value['updates']==456 for value in journals.values()),
        math_precision=all(value<=5e-12 for value in errors.values()),
        rng_unchanged=torch.equal(rng,torch.get_rng_state()))
    write(out/'summary.json',dict(status='complete',integrity=dict(all_pass=all(checks.values()),checks=checks),
        scenario=scenario,journals=journals,models=models,counts=dict(counts),errors=dict(errors),
        wall=dict(started=started,finished=time.time()),new_updates=0,new_native_seconds=0,new_feedback=0,
        intermediate_snapshot_reconstructed=False,checkpoints_loaded=0))


def main():
    torch.set_num_threads(1);p=read(HERE/'protocol.json');runs=HERE/'runs';runs.mkdir(exist_ok=False)
    if read(HERE/'release_plan.json')!=read(HERE.parent/p['source_v40']/'release_plan.json'):
        raise RuntimeError('v40 final budget/release contract changed')
    inv=inventory(p);write(runs/'snapshot_inventory.json',inv)
    if any(value['saved50024_name_candidates'] for value in inv['versions'].values()):
        raise RuntimeError('A new actual50024 file needs provenance review before the declared unavailable interpretation')
    metadata=read(HERE.parent/p['source_inputs']/'history_metadata.json')
    transform=ExecutionAction(metadata['contract']);children={}
    sys.stdout.flush();sys.stderr.flush()
    for scenario in p['scenarios']:
        pid=os.fork()
        if pid==0:
            log=os.open(runs/(scenario+'.log'),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            os.dup2(log,1);os.dup2(log,2);os.close(log)
            try:
                worker(scenario,p,transform,runs/scenario)
                sys.stdout.flush();sys.stderr.flush();os._exit(0)
            except BaseException as error:
                write(runs/scenario/'failure.json',dict(type=type(error).__name__,message=str(error)))
                import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
        children[scenario]=pid
    exits={key:os.waitstatus_to_exitcode(os.waitpid(pid,0)[1]) for key,pid in children.items()}
    from experiments.mcwm_closed_loop_sync_v42_20261024.analyze import analyze
    value=analyze(runs,HERE/'analysis',inv,exits,read(HERE/'sync_plan.json'))
    print(json.dumps(dict(status=value['status'],integrity=value['integrity'],decision=value['decision'])),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        if not isinstance(error,SystemExit):
            write(HERE/'failure.json',dict(type=type(error).__name__,message=str(error),new_updates=0,new_native_seconds=0))
        raise
