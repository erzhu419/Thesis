"""Ordinary dripping development and fixed independent confirmation."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'mcwm_closed_loop_control_campaign_v60_20261025'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def cells(p,stage):
    return [dict(seed=seed,scenario=scenario,arm=arm,
        relative=f'runs/recent30/{stage}/seed{seed}/{scenario}/{arm}')
        for seed in p[stage+'_seeds'] for scenario in p['scenarios']
        for arm in ('fresh','no_fresh')]


def native_jobs(p,stage,infra):
    return [infra.task(cell['relative'],SOURCE/'control.py',
        ['--candidate','recent30','--seed',cell['seed'],'--scenario',cell['scenario'],
         '--arm',cell['arm'],'--out',infra.remote(HERE/cell['relative'])])
        for cell in cells(p,stage)]


def _advance(read_only):
    infra=module('v61_launch_infra',HERE/'infra.py')
    ready=HERE/'execution_ready.json'
    if not ready.exists() or json.loads(ready.read_text()).get('ready') is not True:
        return dict(stage='preparing_ordinary_control',done=False,goal_achieved=False)
    p=json.loads((HERE/'protocol.json').read_text())
    for stage in ('development','confirmation'):
        batch='recent30_'+stage;receipt_path=HERE/(batch+'_submission.json')
        receipt=(json.loads(receipt_path.read_text()) if receipt_path.exists()
            else infra.submit(native_jobs(p,stage,infra),batch))
        expected=cells(p,stage)
        paths=[cell['relative']+'/'+name for cell in expected for name in ('summary.json','failure.json')]
        found=read_only(infra.presence,paths)
        pending,failures,complete=[],[],[]
        ids={r['signature'].split('/runs/',1)[1]:r['id'] for r in receipt['submitted']}
        for cell in expected:
            base=cell['relative'];identifier=ids[base.removeprefix('runs/')]
            if found[base+'/failure.json']:
                failure=read_only(infra.collect,base+'/failure.json')
                failures.append(dict(cell=cell,id=identifier,failure=failure))
            elif found[base+'/summary.json']:
                summary=read_only(infra.collect,base+'/summary.json')
                (complete if summary['status']=='complete' else failures).append(dict(cell=cell,id=identifier))
            else:pending.append(dict(cell=cell,id=identifier))
        if pending:
            statuses={r['id']:r for r in read_only(infra.status,[r['id'] for r in pending])}
            still=[]
            for row in pending:
                status=statuses[row['id']]
                if status['status'] in ('done','failed','cancelled'):
                    failures.append(dict(row,scheduler_status=status['status']))
                else:still.append(dict(row,scheduler_status=status['status'],node=status.get('node')))
            if still:return dict(stage='ordinary_control_'+stage,done=False,goal_achieved=False,
                completed=len(complete),failures=failures,pending=still)
        relative='results/recent30/'+stage+'.json'
        analysis_batch=batch+'_analysis';analysis_receipt=HERE/(analysis_batch+'_submission.json')
        if not analysis_receipt.exists():
            infra.submit([infra.task('cost/'+stage,'summarize.py',
                ['--stage',stage,'--out',infra.remote(HERE/relative)])],analysis_batch)
        if not read_only(infra.presence,[relative])[relative]:
            receipt=json.loads(analysis_receipt.read_text());identifier=receipt['submitted'][0]['id']
            current=read_only(infra.status,[identifier])[0]
            return dict(stage='ordinary_summary_'+stage,done=False,goal_achieved=False,
                attention=current if current['status'] in ('done','failed','cancelled') else None)
        result=read_only(infra.collect,relative)
        if not result['integrity']['all_pass']:
            return dict(stage='ordinary_technical_attention',done=False,goal_achieved=False,
                result=relative,failed_checks=result['integrity'].get('failed_checks',[]),
                failures=result.get('failures',[]))
        decision=result['decision']
        if stage=='development' and not decision['advance_to_confirmation']:
            return dict(stage='ordinary_candidate_closed_requires_next_control',done=False,
                goal_achieved=False,result=relative,decision=decision)
        if stage=='confirmation':
            achieved=decision.get('ordinary_dripping_goal_achieved') is True
            return dict(stage='ordinary_gain_requires_root_review' if achieved
                    else 'ordinary_candidate_closed_after_confirmation',
                done=False,goal_achieved=achieved,result=relative,decision=decision,
                requires_agent_review=True)
    raise RuntimeError('Fixed ordinary-control plan has no decision')


def advance():
    helper=module('v61_v60_readonly',SOURCE/'supervisor.py')
    try:return _advance(helper.read_only)
    except helper.ReadUnavailable as error:
        return dict(stage='waiting_transient_ordinary_read',done=False,goal_achieved=False,message=str(error))
