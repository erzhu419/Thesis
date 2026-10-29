"""Fixed actual-control candidates, development then untouched confirmation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent


def module(name):
    spec=importlib.util.spec_from_file_location('v60_'+name,HERE/(name+'.py'))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def choose(summary):
    gate=summary.get('next_control_gate',{})
    complete=summary.get('status')=='complete' and summary.get('integrity',{}).get('all_pass')
    if complete and gate.get('passed'):
        eligible=gate.get('eligible_models',[])
        model='readout_line' if 'readout_line' in eligible else 'readout_lowdim'
        if model not in eligible:raise ValueError('Passed v59 gate has no eligible predeclared model')
        return dict(candidate='paired_readout',model=model,v59_disposition='admitted',
            ordinary_single_trajectory_dripping=False)
    return dict(candidate='recent30',model=None,
        v59_disposition='closed_failed_gate' if complete else 'closed_incomplete_admission',
        ordinary_single_trajectory_dripping=True)


def cells(protocol,selection,stage):
    candidate=selection['candidate']
    if candidate=='recent30':
        seeds=protocol['development_seeds' if stage=='development' else 'confirmation_seeds']
        return [dict(seed=seed,scenario=scenario,arm=arm,
            relative=f'runs/recent30/{stage}/seed{seed}/{scenario}/{arm}')
            for seed in seeds for scenario in protocol['scenarios']
            for arm in protocol['recent30']['arms']]
    seeds=protocol['paired_readout'][stage+'_external_seeds']
    return [dict(external_seed=seed,scenario=scenario,arm=arm,
        relative=f'runs/paired_readout/{stage}/external{seed}/{scenario}/{arm}')
        for seed in seeds for scenario in protocol['scenarios']
        for arm in protocol['paired_readout']['arms']]


def jobs(protocol,selection,stage,infra):
    result=[]
    for cell in cells(protocol,selection,stage):
        argv=['--scenario',cell['scenario'],'--arm',cell['arm'],
            '--out',infra.remote(HERE/cell['relative'])]
        if selection['candidate']=='recent30':
            runner='control.py';argv+=['--candidate','recent30','--seed',cell['seed']]
        else:
            runner='readout_control.py';argv+=['--external-seed',cell['external_seed'],
                '--model',selection['model']]
        result.append(infra.task(cell['relative'],runner,argv))
    return result


def watch(receipt,expected,infra,read_only):
    paths=[cell['relative']+'/'+name for cell in expected for name in ('summary.json','failure.json')]
    found=read_only(infra.presence,paths)
    statuses={};missing=[];results=[]
    indexed={row['signature'].split('/runs/',1)[-1]:row['id'] for row in receipt['submitted']}
    for cell in expected:
        stem=cell['relative'];identifier=indexed.get(stem.removeprefix('runs/'))
        if identifier is None:raise ValueError('Receipted job does not match a declared cell: '+stem)
        if found[stem+'/failure.json']:
            failure=read_only(infra.collect,stem+'/failure.json')
            results.append(dict(cell=cell,id=identifier,state='failed',failure=failure))
        elif found[stem+'/summary.json']:
            summary=read_only(infra.collect,stem+'/summary.json')
            results.append(dict(cell=cell,id=identifier,
                state='complete' if summary['status']=='complete' else 'failed'))
        else:missing.append((cell,identifier))
    if missing:
        statuses={row['id']:row for row in read_only(infra.status,[value[1] for value in missing])}
    for cell,identifier in missing:
        row=statuses[identifier]
        results.append(dict(cell=cell,id=identifier,
            state='failed' if row['status'] in ('done','failed','cancelled') else 'pending',
            scheduler_status=row['status'],node=row.get('node')))
    return dict(all_terminal=all(row['state']!='pending' for row in results),cells=results)


def _advance(summary,read_only):
    infra=module('infra')
    p=json.loads((HERE/'protocol.json').read_text())
    selection=choose(summary)
    selection_path=HERE/'candidate_selection.json'
    if selection_path.exists():
        if json.loads(selection_path.read_text())!=selection:
            raise ValueError('An already selected control candidate cannot change')
    else:infra.write(selection_path,selection)
    if not (HERE/'execution_ready.json').exists():
        return dict(stage='preparing_actual_control_code',done=False,selection=selection)
    staged=HERE/'control_staging.json'
    if not staged.exists():infra.write(staged,infra.stage())
    for stage in ('development','confirmation'):
        candidate=selection['candidate'];batch=candidate+'_'+stage
        receipt_path=HERE/(batch+'_submission.json')
        receipt=(json.loads(receipt_path.read_text()) if receipt_path.exists()
            else infra.submit(jobs(p,selection,stage,infra),batch))
        current=watch(receipt,cells(p,selection,stage),infra,read_only)
        if not current['all_terminal']:
            return dict(stage='actual_control_'+stage,done=False,selection=selection,**current)
        analysis_relative=f'results/{candidate}/{stage}.json'
        analysis_batch=batch+'_analysis'
        analysis_receipt=HERE/(analysis_batch+'_submission.json')
        if not analysis_receipt.exists():
            infra.submit([infra.task('analyze/'+batch,'control_analyze.py',
                ['--candidate',candidate,'--stage',stage,'--out',infra.remote(HERE/analysis_relative)])],analysis_batch)
        if not read_only(infra.presence,[analysis_relative])[analysis_relative]:
            submitted=json.loads(analysis_receipt.read_text())['submitted']
            status=read_only(infra.status,[submitted[0]['id']])[0]
            if status['status'] in ('done','failed','cancelled'):
                return dict(stage='actual_cost_summary_attention',done=False,selection=selection,
                    task=status,attention='Terminal summary task has no declared result; do not repeat control days')
            return dict(stage='waiting_actual_cost_'+stage,done=False,selection=selection)
        result=read_only(infra.collect,analysis_relative)
        if not result['integrity']['all_pass']:
            return dict(stage='actual_control_technical_failure',done=False,selection=selection,
                result=analysis_relative,attention='Resolve incurred-work technical failure without retuning or dropping cells')
        if stage=='development' and not result['decision']['advance_to_confirmation']:
            return dict(stage='candidate_closed_next_control_candidate_required',done=False,
                selection=selection,result=analysis_relative,decision=result['decision'])
        if stage=='confirmation':
            confirmed=result['decision'].get('confirmed_control_improvement',False)
            return dict(stage=('confirmed_ordinary_dripping_gain' if confirmed and candidate=='recent30'
                    else 'confirmed_counterfactual_assisted_gain' if confirmed else 'candidate_closed_after_confirmation'),
                done=False,selection=selection,result=analysis_relative,decision=result['decision'],
                requires_agent_review=True)
    raise RuntimeError('Control campaign exhausted without a terminal decision')


def advance(summary):
    supervisor=module('supervisor')
    try:
        return _advance(summary,supervisor.read_only)
    except supervisor.ReadUnavailable as error:
        return dict(stage='waiting_transient_control_read',done=False,message=str(error))
