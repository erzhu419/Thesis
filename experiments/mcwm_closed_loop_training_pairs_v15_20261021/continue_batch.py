"""Run paired preflight before training-period branches; collect small output only."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import time

import infra
import launch

HERE = Path(__file__).resolve().parent


def log(stage, **values):
    print(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(), stage=stage, **values),
                     ensure_ascii=False, allow_nan=False), flush=True)


def metadata():
    paths = [str(launch.branch_path(cell, arm)) for cell in launch.cells() for arm in ('low','high')]
    script = '''import json
from pathlib import Path
p=Path(__HERE__)
def state(relative):
 q=p/relative
 if (q/'failure.json').is_file():return 'failed'
 if not (q/'summary.json').is_file():return 'pending'
 return json.loads((q/'summary.json').read_text()).get('status')
pref=[state('preflight/pair/'+arm) for arm in ('low','high')]
main=[state(relative) for relative in __PATHS__]
failures=[str(f.relative_to(p)) for f in [p/'summary_failure.json',p/'preflight/summary_failure.json',p/'selection_failure.json',p/'preflight/selection_failure.json'] if f.is_file()]
bad=[relative for relative,s in zip(__PATHS__,main) if s not in ('pending','complete','selection_missing')]
out=dict(preflight_selection=(p/'preflight/selection_summary.json').is_file(),
 selection=(p/'selection_summary.json').is_file(),
 preflight_branches=sum(s!='pending' for s in pref),preflight_bad=[s for s in pref if s not in ('pending','complete')],
 preflight_gate=(p/'preflight/gate.json').is_file(),
 branches_terminal=sum(s!='pending' for s in main),bad_branches=bad,failures=failures,
 result_exists=(p/__RESULT__).is_file())
if out['result_exists']:
 f=p/__RESULT__;v=json.loads(f.read_text())
 out['result']=dict(status=v.get('status'),bytes=f.stat().st_size,integrity=v.get('integrity'))
print(json.dumps(out))
'''.replace('__HERE__',repr(infra.remote(HERE))).replace('__PATHS__',repr(paths)).replace('__RESULT__',repr(launch.RESULT_NAME))
    return json.loads(infra.scheduler.run_on('node001', shlex.join([infra.PYTHON,'-c',script]), timeout=45,check=True)[1])


def wait_until(stage, predicate, *, retain_failed_main=False):
    previous = None
    while True:
        value = metadata()
        if value != previous:
            log(stage, **value)
            previous = value
        if value['failures'] or value['preflight_bad']:
            raise RuntimeError('Preserved native selection/preflight/summary failure: '+str(value))
        if value['bad_branches'] and not retain_failed_main:
            raise RuntimeError('Native branch failed; retain and review its evidence')
        if predicate(value):
            return value
        time.sleep(60)


def exists(batch):
    return (HERE/(batch+'_submission.json')).is_file()


def submitted(stage, receipt):
    log(stage, ids=[row['id'] for row in receipt['submitted']],priority='high')


def main():
    log('continuation_started',priority='high',design_states=12,branches=24,
        diagnostic_labels_enter_fit=0,future_learning='not_submitted')
    if not exists('selection'):
        raise RuntimeError('Submit resident selection first')
    if not exists('preflight_pairs'):
        wait_until('waiting_preflight_selection',lambda v:v['preflight_selection'])
        submitted('preflight_pairs_submitted',launch.submit_preflight())
    if not exists('preflight_finalize'):
        submitted('preflight_finalizer_submitted',launch.submit_finalizer(preflight=True))
    if not exists('paired'):
        wait_until('waiting_native_preflight_and_selection',lambda v:v['preflight_gate'] and v['selection'])
        launch.require_gate()
        submitted('training_pairs_submitted',launch.submit_pairs())
    if not exists('diagnostic_finalize'):
        submitted('finalizer_submitted',launch.submit_finalizer())
    value=wait_until('waiting_training_pairs_and_summary',lambda v:v['result_exists'],retain_failed_main=True)
    if value['result']['bytes']>launch.SUMMARY_MAX_BYTES:
        raise RuntimeError('Reduce resident output before collection')
    if not (HERE/launch.RESULT_NAME).is_file():
        infra.collect_json(launch.RESULT_NAME,max_bytes=launch.SUMMARY_MAX_BYTES)
    log('diagnostic_finished',**value['result'],collected_bytes=(HERE/launch.RESULT_NAME).stat().st_size,
        next_action='Review training-period paired signal and coverage before budgeted differential learning')


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        log('stopped',error_type=type(error).__name__,error=str(error))
        raise
