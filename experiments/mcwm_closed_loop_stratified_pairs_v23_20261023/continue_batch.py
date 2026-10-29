"""Resume read-only polls; validate one service day without submitting B."""
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


def readonly_remote(code):
    for attempt in range(4):
        try:
            return infra.scheduler.run_on('node001', shlex.join([infra.PYTHON, '-c', code]),
                timeout=45, check=True)[1]
        except RuntimeError as error:
            if 'rc=255' not in str(error) or attempt == 3:
                raise
            log('readonly_ssh_retry', retry=attempt+1, maximum_retries=3,
                delay_seconds=45, error=str(error)[:400])
            time.sleep(45)


def collect_small(relative, limit=launch.GATE_MAX_BYTES):
    target = HERE/relative
    if target.is_file():
        return json.loads(target.read_text())
    code = 'from pathlib import Path; p=Path('+repr(infra.remote(target))+'); assert p.stat().st_size<='+str(limit)+'; print(p.read_text(),end="")'
    value = json.loads(readonly_remote(code))
    infra.write(target, value)
    return value


def metadata():
    paths = [str(launch.evaluation_path(c, arm))
        for c in launch.evaluation_cells() for arm in ('low', 'high')]
    pairs = [str(launch.evaluation_path(c, 'low').parent) for c in launch.evaluation_cells()]
    code = '''from pathlib import Path
import json
root=Path(ROOT)
def state(relative):
 p=root/relative
 if (p/'failure.json').is_file():return dict(path=relative,status='failed',all_pass=False)
 f=p/'summary.json'
 if not f.is_file():return dict(path=relative,status='pending')
 v=json.loads(f.read_text());return dict(path=relative,status=v.get('status'),all_pass=v.get('integrity',{}).get('all_pass'),coverage=v.get('design_coverage',{}).get('all_pass'))
def gate(relative):
 f=root/relative
 if not f.is_file():return None
 v=json.loads(f.read_text());return dict(status=v.get('status'),all_pass=v.get('integrity',{}).get('all_pass'),coverage=v.get('design_coverage',{}).get('all_pass'),training_passed=v.get('training_gate',{}).get('passed'),service_validation_eligible=v.get('service_validation_eligible'),scientific_passed=v.get('scientific_gate',{}).get('passed'),bytes=f.stat().st_size)
rows=[state(x) for x in PAIRS]
bad=[v for v in rows if v['status']!='pending' and (v['status'] not in ('complete','selection_missing') or v.get('all_pass') is not True)]
for relative in PAIRROOTS:
 f=root/relative/'failure.json'
 if f.is_file():
  v=json.loads(f.read_text());bad.append(dict(path=relative,status='failed',type=v.get('type'),message=v.get('message')))
failures=[]
for name in ('analysis_failure.json','summary_failure.json'):
 f=root/name
 if f.is_file():
  v=json.loads(f.read_text());failures.append(dict(path=name,type=v.get('type'),message=v.get('message')))
print(json.dumps(dict(day=state(DAY),day_gate=gate(DAY+'/gate.json'),training_gate=gate('training_gate.json'),service_pairs=dict(terminal=sum(v['status']!='pending' for v in rows),passed=sum(v['status']=='complete' and v.get('all_pass') is True for v in rows),bad=bad),service_gate=gate('service_gate.json'),analysis_failures=failures)))
'''.replace('PAIRROOTS', repr(pairs)).replace('PAIRS', repr(paths)).replace('ROOT', repr(infra.remote(HERE))).replace('DAY', repr(str(launch.day_path())))
    return json.loads(readonly_remote(code))


def wait_until(stage, predicate):
    previous = None
    while True:
        value = metadata()
        if value != previous:
            log(stage, **value)
            previous = value
        day = value['day']
        if (day['status'] != 'pending' and (day['status'] != 'complete' or day.get('all_pass') is not True)) or value['service_pairs']['bad'] or value['analysis_failures']:
            raise RuntimeError('Preserve native failures; no automatic simulation retry: '+str(value))
        for key in ('day_gate', 'training_gate', 'service_gate'):
            gate = value[key]
            if gate is not None and (gate['status'] != 'complete' or gate['all_pass'] is not True):
                raise RuntimeError('Resident native integrity failed: '+key)
        if predicate(value):
            return value
        time.sleep(45)


def exists(batch):
    return (HERE/(batch+'_submission.json')).is_file()


def submitted(stage, receipt):
    log(stage, ids=[row['id'] for row in receipt['submitted']], priority='high')


def finish(status, **values):
    result = dict(status=status, priority='high', response_days_submitted=False,
        independent_confirmation='not_submitted', **values)
    infra.write(HERE/'continuation_result.json', result)
    log(status, **values)
    return result


def main():
    terminal = HERE/'continuation_result.json'
    if terminal.is_file():
        log('previous_terminal_state', result=json.loads(terminal.read_text()))
        return
    if not exists('preflight'):
        raise RuntimeError('Submit the single fresh preflight day first')
    log('started', priority='high', main_days=1, training_pairs=9,
        training_branch_processes=18, labels=285, conditional_service_branches=12,
        conditional_service_pair_tasks=6,
        same_pair_arms='two native children launched before wait', response_days_submitted=False)
    wait_until('waiting_native_day', lambda v:v['day_gate'] is not None)
    if not exists('training_finalize'):
        submitted('training_finalizer_submitted', launch.submit_finalizer(stage='train'))
    wait_until('waiting_training_gate', lambda v:v['training_gate'] is not None)
    gate = collect_small('training_gate.json')
    if gate.get('training_gate', {}).get('passed') is not True or gate.get('service_validation_eligible') is not True:
        finish('stopped_training_gate', training_gate=gate.get('training_gate'),
            service_validation_eligible=gate.get('service_validation_eligible'), service_pairs_submitted=False)
        return
    if not exists('service_pairs'):
        submitted('service_pairs_submitted', launch.submit_service_pairs())
    if not exists('service_finalize'):
        submitted('service_finalizer_submitted', launch.submit_finalizer())
    wait_until('waiting_service_validation', lambda v:v['service_gate'] is not None)
    result = collect_small('service_gate.json')
    finish('complete_service_validation', scientific_gate=result.get('scientific_gate'),
        service_pairs_submitted=True,
        next='Review the new-prefix service evidence before any B decision.')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        value = dict(status='failed', type=type(error).__name__, message=str(error))
        path = HERE/'continuation_failure.json'
        if not path.exists():
            infra.write(path, value)
        log('stopped', **value)
        raise
