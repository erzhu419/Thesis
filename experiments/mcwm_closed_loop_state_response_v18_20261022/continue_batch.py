"""Reuse the full native preflight day within the four-day timing pilot."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import time

import infra
import launch

HERE=Path(__file__).resolve().parent


def log(stage,**values):
    print(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),stage=stage,**values),
                     ensure_ascii=False,allow_nan=False),flush=True)


def metadata():
    days=[str(launch.day_path(cell)) for cell in launch.cells()]
    pref=str(launch.day_path(launch.cells(preflight=True)[0]))
    pref_pairs=[str(launch.evaluation_path(cell,arm)) for cell in launch.evaluation_cells(preflight=True) for arm in ('low','high')]
    pairs=[str(launch.evaluation_path(cell,arm)) for cell in launch.evaluation_cells() for arm in ('low','high')]
    script='''import json
from pathlib import Path
p=Path(__HERE__)
def state(relative):
 q=p/relative
 if (q/'failure.json').is_file():return dict(status='failed',all_pass=False)
 f=q/'summary.json'
 if not f.is_file():return dict(status='pending',all_pass=None)
 v=json.loads(f.read_text())
 return dict(status=v.get('status'),all_pass=v.get('integrity',{}).get('all_pass'))
pref=state(__PREF__)
pref_pairs=[state(relative) for relative in __PREFPAIRS__]
days=[state(relative) for relative in __DAYS__]
pairs=[state(relative) for relative in __PAIRS__]
failures=[str(f.relative_to(p)) for f in (p/'summary_failure.json',) if f.is_file()]
bad=lambda rows:[i for i,row in enumerate(rows) if row['status']!='pending' and (row['status'] not in ('complete','selection_missing') or row['all_pass'] is False)]
out=dict(preflight_day=pref,preflight_day_gate=(p/__PREF__/'gate.json').is_file(),
 preflight_pairs_terminal=sum(row['status']!='pending' for row in pref_pairs),
 preflight_pairs_bad=bad(pref_pairs),preflight_gate=(p/'preflight_gate.json').is_file(),
 days_complete=sum(row['status']!='pending' for row in days),days_passed=sum(row['status']=='complete' and row['all_pass'] is True for row in days),
 days_bad=bad(days),pairs_terminal=sum(row['status']!='pending' for row in pairs),pairs_bad=bad(pairs),
 failures=failures,result_exists=(p/__RESULT__).is_file())
if out['result_exists']:
 f=p/__RESULT__;v=json.loads(f.read_text());out['result']=dict(status=v.get('status'),integrity=v.get('integrity'),bytes=f.stat().st_size)
print(json.dumps(out))
'''.replace('__HERE__',repr(infra.remote(HERE))).replace('__PREF__',repr(pref)).replace('__PREFPAIRS__',repr(pref_pairs)).replace('__DAYS__',repr(days)).replace('__PAIRS__',repr(pairs)).replace('__RESULT__',repr(launch.RESULT_NAME))
    return json.loads(infra.scheduler.run_on('node001',shlex.join([infra.PYTHON,'-c',script]),timeout=45,check=True)[1])


def wait_until(stage,predicate,*,keep_failed_evaluations=False):
    previous=None
    while True:
        value=metadata()
        if value!=previous:
            log(stage,**value);previous=value
        pref=value['preflight_day']
        if (pref['status']!='pending' and (pref['status']!='complete' or pref['all_pass'] is not True)) or value['preflight_pairs_bad'] or value['days_bad'] or value['failures']:
            raise RuntimeError('Preserve and review native budget/preflight/day/analysis failure: '+str(value))
        if value['pairs_bad'] and not keep_failed_evaluations:
            raise RuntimeError('Native validation branch failed; preserve its evidence')
        if predicate(value):return value
        time.sleep(60)


def exists(batch):return (HERE/(batch+'_submission.json')).is_file()


def submitted(stage,receipt):
    log(stage,ids=[row['id'] for row in receipt['submitted']],priority='high')


def main():
    log('continuation_started',priority='high',main_days=len(launch.cells()),
        additional_days=len(launch.remaining_cells()),preflight_reused=True,
        evaluation_states=len(launch.evaluation_cells()),evaluation_branches=2*len(launch.evaluation_cells()),
        paired_outcomes_replace_ordinary_slots=True,independent_confirmation='not_submitted')
    if not exists('preflight'):raise RuntimeError('Submit native preflight day first')
    if not exists('preflight_pairs'):
        wait_until('waiting_native_budgeted_preflight_day',lambda v:v['preflight_day_gate'])
        submitted('preflight_evaluation_pairs_submitted',launch.submit_evaluations(preflight=True))
    if not exists('preflight_finalize'):
        submitted('preflight_finalizer_submitted',launch.submit_finalizer(preflight=True))
    if not exists('diagnostic'):
        wait_until('waiting_budgeted_paired_preflight',lambda v:v['preflight_gate'])
        submitted('remaining_three_days_submitted',launch.submit_days())
    if not exists('paired'):
        value=wait_until('waiting_new_common_prefix_days',lambda v:v['days_complete']==len(launch.cells()))
        if value['days_passed']!=len(launch.cells()):raise RuntimeError('All common prefix days must pass')
        submitted('remaining_36_heldout_branches_submitted',launch.submit_evaluations())
    if not exists('diagnostic_finalize'):
        submitted('learning_finalizer_submitted',launch.submit_finalizer())
    value=wait_until('waiting_native_validation_and_summary',lambda v:v['result_exists'],keep_failed_evaluations=True)
    if value['result']['bytes']>launch.SUMMARY_MAX_BYTES:raise RuntimeError('Reduce summary on server before collection')
    if not (HERE/launch.RESULT_NAME).is_file():infra.collect_json(launch.RESULT_NAME,max_bytes=launch.SUMMARY_MAX_BYTES)
    log('diagnostic_finished',**value['result'],collected_bytes=(HERE/launch.RESULT_NAME).stat().st_size,
        next_action='Review fixed candidate versus shared/route/zero on new external seeds, label ages,300s direction/regret and stable effects; no automatic confirmation or deployment')


if __name__=='__main__':
    try:main()
    except Exception as error:
        log('stopped',error_type=type(error).__name__,error=str(error));raise
