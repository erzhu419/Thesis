"""Reuse the first tail branch, expand only after exact native replay passes."""
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
    paths=[str(launch.branch_path(line,arm)) for line,arm in launch.branches()]
    first=str(launch.branch_path(*launch.first_branch()))
    script='''import json
from pathlib import Path
p=Path(__ROOT__)
def state(relative):
 q=p/relative
 if (q/'failure.json').is_file():return dict(status='failed',all_pass=False)
 f=q/'summary.json'
 if not f.is_file():return dict(status='pending',all_pass=None)
 v=json.loads(f.read_text());return dict(status=v.get('status'),all_pass=v.get('integrity',{}).get('all_pass'))
rows=[state(path) for path in __PATHS__]
out=dict(first=state(__FIRST__),branches_terminal=sum(r['status']!='pending' for r in rows),
 bad=[i for i,r in enumerate(rows) if r['status']!='pending' and (r['status']!='complete' or r['all_pass'] is not True)],
 preflight_gate=(p/'preflight_gate.json').is_file(),result_exists=(p/'intervention_summary.json').is_file(),
 failures=[f.name for f in (p/'analysis_failure.json',p/'summary_failure.json') if f.is_file()])
if out['result_exists']:
 f=p/'intervention_summary.json';v=json.loads(f.read_text());out['result']=dict(status=v.get('status'),integrity=v.get('integrity'),bytes=f.stat().st_size)
print(json.dumps(out))
'''.replace('__ROOT__',repr(infra.remote(HERE))).replace('__PATHS__',repr(paths)).replace('__FIRST__',repr(first))
    return json.loads(infra.scheduler.run_on('node001',shlex.join([infra.PYTHON,'-c',script]),timeout=45,check=True)[1])


def wait_until(stage,predicate):
    previous=None
    while True:
        value=metadata()
        if value!=previous:log(stage,**value);previous=value
        if value['bad'] or value['failures']:
            raise RuntimeError('Preserve and review native execution-phase failure: '+str(value))
        if predicate(value):return value
        time.sleep(60)


def exists(batch):return (HERE/(batch+'_submission.json')).is_file()


def submitted(stage,receipt):
    log(stage,ids=[row['id'] for row in receipt['submitted']],priority='high')


def main():
    if not exists('preflight_retest'):raise RuntimeError('Submit the corrected shadow-gated first branch before continuation')
    log('continuation_started',new_branches=6,first_branch_reused=True,new_suffix_seconds=1800,
        new_training_labels=0,priority='high',independent_confirmation='not_submitted')
    if not exists('preflight_finalize'):
        wait_until('waiting_exact_prefix_preflight',lambda v:v['first']['status']=='complete')
        submitted('preflight_finalizer_submitted',launch.submit_finalizer(preflight=True))
    if not exists('intervention'):
        wait_until('waiting_native_comparison_gate',lambda v:v['preflight_gate'])
        submitted('remaining_five_tail_branches_submitted',launch.submit_remaining())
    if not exists('intervention_finalize'):
        submitted('intervention_finalizer_submitted',launch.submit_finalizer())
    value=wait_until('waiting_six_tail_branches',lambda v:v['result_exists'])
    if value['result']['bytes']>30000:raise RuntimeError('Reduce the resident result before collection')
    if not (HERE/'intervention_summary.json').is_file():infra.collect_json('intervention_summary.json',max_bytes=30000)
    log('diagnostic_finished',**value['result'],next_action='Review native/tail300s effects before any new model or time-window experiment.')


if __name__=='__main__':
    try:main()
    except Exception as error:log('stopped',error_type=type(error).__name__,error=str(error));raise
