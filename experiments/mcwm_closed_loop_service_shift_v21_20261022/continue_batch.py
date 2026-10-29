"""Expand once after scientific service-history admission; preserve failures."""
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
        ensure_ascii=False), flush=True)


def metadata():
    first = launch.cells(preflight=True)[0]
    a = str(launch.day_path(first, preflight=True))
    ap = [str(launch.evaluation_path(c, arm, preflight=True))
        for c in launch.evaluation_cells(preflight=True) for arm in ('low', 'high')]
    b = [str(launch.day_path(c)) for c in launch.cells()]
    bp = [str(launch.evaluation_path(c, arm))
        for c in launch.evaluation_cells() for arm in ('low', 'high')]
    code = '''from pathlib import Path
import json
root=Path(ROOT)
def state(relative):
 p=root/relative
 f=p/'failure.json'
 if f.is_file():
  v=json.loads(f.read_text());return dict(path=relative,status='failed',failure={k:v.get(k) for k in ('type','message','time')})
 f=p/'summary.json'
 if not f.is_file():return dict(path=relative,status='pending')
 v=json.loads(f.read_text());return dict(path=relative,status=v.get('status'),all_pass=v.get('integrity',{}).get('all_pass'),coverage=v.get('design_coverage',{}).get('all_pass'))
def brief(name):
 f=root/name
 if not f.is_file():return None
 v=json.loads(f.read_text());return dict(status=v.get('status'),integrity=v.get('integrity'),scientific_gate=v.get('scientific_gate'),bytes=f.stat().st_size)
first=state(FIRST)
ap=[state(x) for x in APAIRS];days=[state(x) for x in DAYS];bp=[state(x) for x in BPAIRS]
bad=lambda rows:[v for v in rows if v['status']!='pending' and (v['status'] not in ('complete','selection_missing') or v.get('all_pass') is not True)]
out=dict(first=first,first_gate=(root/FIRST/'gate.json').is_file(),service_pairs_terminal=sum(v['status']!='pending' for v in ap),service_pairs_bad=bad(ap),response_days_terminal=sum(v['status']!='pending' for v in days),response_days_bad=bad(days),response_days_passed=sum(v['status']=='complete' and v.get('all_pass') is True and v.get('coverage') is True for v in days),response_pairs_terminal=sum(v['status']!='pending' for v in bp),response_pairs_bad=bad(bp),analysis_failures=[json.loads(p.read_text()) for p in (root/'analysis_failure.json',root/'summary_failure.json') if p.is_file()],service_gate=brief('service_gate.json'),response_summary=brief('response_summary.json'))
print(json.dumps(out))
'''.replace('ROOT', repr(infra.remote(HERE))).replace('FIRST', repr(a)).replace('APAIRS', repr(ap)).replace('DAYS', repr(b)).replace('BPAIRS', repr(bp))
    return json.loads(infra.scheduler.run_on('node001', shlex.join([infra.PYTHON, '-c', code]),
        timeout=45, check=True)[1])


def wait_until(stage, predicate):
    previous = None
    while True:
        value = metadata()
        if value != previous:
            log(stage, **value)
            previous = value
        first = value['first']
        if (first['status'] != 'pending' and (first['status'] != 'complete' or first.get('all_pass') is not True)) or any(
                value[k] for k in ('service_pairs_bad', 'response_days_bad', 'response_pairs_bad', 'analysis_failures')):
            raise RuntimeError('Preserve and review failed native evidence: '+str(value))
        for key in ('service_gate', 'response_summary'):
            row = value[key]
            if row is not None and (row['status'] != 'complete' or row.get('integrity', {}).get('all_pass') is not True):
                raise RuntimeError('Resident analysis integrity failed: '+key)
        if predicate(value):
            return value
        time.sleep(45)


def exists(batch):
    return (HERE/(batch+'_submission.json')).is_file()


def submitted(stage, receipt):
    log(stage, ids=[row['id'] for row in receipt['submitted']], priority='high')


def main():
    if not exists('preflight'):
        raise RuntimeError('Submit the service native preflight day first')
    log('started', priority='high', service_days=1, conditional_response_days=3,
        total_days_maximum=4, labels_per_day=285, service_heldout_branches=12,
        conditional_response_heldout_branches=36, service_preflight_reused=True,
        independent_confirmation='not_submitted')
    if not exists('service_pairs'):
        wait_until('waiting_service_day', lambda v:v['first_gate'])
        submitted('service_pairs_submitted', launch.submit_service_pairs())
    if not exists('service_finalize'):
        submitted('service_finalizer_submitted', launch.submit_finalizer(preflight=True))
    wait_until('waiting_service_scientific_gate', lambda v:v['service_gate'] is not None)
    gate = infra.collect('service_gate.json', max_bytes=launch.GATE_MAX_BYTES)
    if gate.get('scientific_gate', {}).get('passed') is not True:
        log('scientific_gate_stopped', scientific_gate=gate.get('scientific_gate'),
            response_days_submitted=False, independent_confirmation='not_submitted')
        return
    if not exists('response_days'):
        submitted('response_days_submitted', launch.submit_response_days())
    if not exists('response_pairs'):
        wait_until('waiting_response_days', lambda v:v['response_days_terminal'] == len(launch.cells()))
        submitted('response_pairs_submitted', launch.submit_response_pairs())
    if not exists('response_finalize'):
        submitted('response_finalizer_submitted', launch.submit_finalizer())
    value = wait_until('waiting_response_pairs', lambda v:v['response_summary'] is not None)
    infra.collect('response_summary.json', max_bytes=launch.SUMMARY_MAX_BYTES)
    log('finished', result=value['response_summary'], independent_confirmation='not_submitted',
        next='Review300-second direction, regret and stable degradation before further experiments.')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        log('stopped', error_type=type(error).__name__, error=str(error))
        raise
