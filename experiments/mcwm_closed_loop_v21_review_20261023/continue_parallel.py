"""Resume small read-only polls; admit new work only after both native gates."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import time

import infra
import launch_parallel as launch

HERE = Path(__file__).resolve().parent
ORIGINAL = launch.ORIGINAL
GATE_MAX_BYTES = 20000
SUMMARY_MAX_BYTES = 50000


def log(stage, **values):
    print(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(), stage=stage, **values),
        ensure_ascii=False), flush=True)


def readonly_remote(code):
    """Only SSH rc255 may retry; JSON and remote analysis failures propagate."""
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


def collect_small(source, relative, limit):
    target = HERE/relative
    if target.is_file():
        return json.loads(target.read_text())
    code = 'from pathlib import Path; p=Path('+repr(infra.remote(source))+'); assert p.stat().st_size<='+str(limit)+'; print(p.read_text(),end="")'
    value = json.loads(readonly_remote(code))
    infra.write(target, value)
    return value


def metadata():
    p = launch.protocol()
    a = [str(Path('stage_a/pairs/stable/seed6201')/line/window['name']/arm)
        for line in p['target_lines'] for window in p['heldout_windows'] for arm in ('low', 'high')]
    days = [str(Path('stage_b/runs')/scene/f'seed{seed}') for seed, scene in launch.CELLS]
    b = [str(Path('stage_b/pairs')/scene/f'seed{seed}'/line/window['name']/arm)
        for seed, scene in launch.CELLS for line in p['target_lines']
        for window in p['heldout_windows'] for arm in ('low', 'high')]
    code = '''from pathlib import Path
import json
root,old=Path(ROOT),Path(OLD)
def state(base,relative):
 p=base/relative
 if (p/'failure.json').is_file():return dict(path=relative,status='failed',all_pass=False)
 f=p/'summary.json'
 if not f.is_file():return dict(path=relative,status='pending')
 v=json.loads(f.read_text());return dict(path=relative,status=v.get('status'),all_pass=v.get('integrity',{}).get('all_pass'),coverage=v.get('design_coverage',{}).get('all_pass'))
def counts(rows):
 return dict(terminal=sum(v['status']!='pending' for v in rows),passed=sum(v['status']=='complete' and v.get('all_pass') is True for v in rows),covered=sum(v.get('coverage') is True for v in rows),bad=[v for v in rows if v['status']!='pending' and (v['status'] not in ('complete','selection_missing') or v.get('all_pass') is not True)])
def gate(path):
 if not path.is_file():return None
 v=json.loads(path.read_text());return dict(status=v.get('status'),all_pass=v.get('integrity',{}).get('all_pass'),scientific_passed=v.get('scientific_gate',{}).get('passed'),bytes=path.stat().st_size)
failures=[]
for base,name in ((old,'analysis_failure.json'),(root,'analysis_failure.json'),(root,'summary_failure.json')):
 f=base/name
 if f.is_file():
  v=json.loads(f.read_text());failures.append(dict(path=str(f),type=v.get('type'),message=v.get('message')))
out=dict(service_day=state(old,'stage_a/runs/stable/seed6201'),service_pairs=counts([state(old,x) for x in APAIRS]),parity=counts([state(root,'parity/'+a) for a in ('low','high')]),response_days=counts([state(root,x) for x in DAYS]),response_pairs=counts([state(root,x) for x in BPAIRS]),analysis_failures=failures,service_gate=gate(old/'service_gate.json'),parity_gate=gate(root/'parity_gate.json'),response_summary=gate(root/'response_summary.json'))
print(json.dumps(out))
'''.replace('ROOT', repr(infra.remote(HERE))).replace('OLD', repr(infra.remote(ORIGINAL))).replace('APAIRS', repr(a)).replace('DAYS', repr(days)).replace('BPAIRS', repr(b))
    return json.loads(readonly_remote(code))


def wait_until(stage, predicate):
    previous = None
    while True:
        value = metadata()
        if value != previous:
            log(stage, **value)
            previous = value
        first = value['service_day']
        if (first['status'] != 'pending' and (first['status'] != 'complete' or first.get('all_pass') is not True)) or value['analysis_failures'] or any(
                value[k]['bad'] for k in ('service_pairs', 'parity', 'response_days', 'response_pairs')):
            raise RuntimeError('Preserve native/analysis failures; no automatic simulation retry: '+str(value))
        for key in ('service_gate', 'parity_gate', 'response_summary'):
            gate = value[key]
            if gate is not None and (gate['status'] != 'complete' or gate['all_pass'] is not True):
                raise RuntimeError('Resident integrity gate failed: '+key)
        if predicate(value):
            return value
        time.sleep(45)


def exists(batch):
    return (HERE/(batch+'_submission.json')).is_file()


def submit_response_pairs():
    p = launch.protocol()
    rows = []
    for seed, scene in launch.CELLS:
        source = HERE/'stage_b/runs'/scene/f'seed{seed}'
        for line in p['target_lines']:
            for window in p['heldout_windows']:
                for arm in ('low', 'high'):
                    out = HERE/'stage_b/pairs'/scene/f'seed{seed}'/line/window['name']/arm
                    rows.append(launch.branch_job(seed, scene, source,
                        f"heldout/{line}/{window['name']}", out))
    return infra.submit(rows, 'response_pairs')


def submit_response_finalizer():
    args = ['--root', infra.remote(HERE), '--stage', 'response', '--out',
        infra.remote(HERE/'response_summary.json'), '--wait']
    return infra.submit([infra.task(launch.protocol(), 'parallel/response_finalize', args,
        runner='analyze.py', ram_mb=2048)], 'response_finalize')


def submitted(stage, receipt):
    log(stage, ids=[r['id'] for r in receipt['submitted']], priority='high')


def finish(status, **values):
    result = dict(status=status, priority='high', independent_confirmation='not_submitted', **values)
    infra.write(HERE/'continuation_result.json', result)
    log(status, **values)
    return result


def main():
    terminal = HERE/'continuation_result.json'
    if terminal.is_file():
        log('previous_terminal_state', result=json.loads(terminal.read_text()))
        return
    log('started', priority='high', service_source=str(ORIGINAL),
        new_parity_suffixes=2, new_parity_suffix_seconds=600, parity_truth_not_fit=True,
        conditional_response_days=3, conditional_heldout_branches=36)
    wait_until('waiting_service_and_parallel_gates',
        lambda v:v['service_gate'] is not None and v['parity_gate'] is not None)
    gate = collect_small(ORIGINAL/'service_gate.json', 'service_gate.json', GATE_MAX_BYTES)
    parity = collect_small(HERE/'parity_gate.json', 'parity_gate.json', GATE_MAX_BYTES)
    if not (gate.get('status') == 'complete' and gate.get('integrity', {}).get('all_pass') is True and
            parity.get('status') == 'complete' and parity.get('integrity', {}).get('all_pass') is True):
        raise RuntimeError('Native service and parallel parity integrity must both pass')
    if gate.get('scientific_gate', {}).get('passed') is not True:
        finish('stopped_scientific_gate', scientific_gate=gate.get('scientific_gate'),
            response_days_submitted=False, parallel_parity_passed=True)
        return
    if not exists('response_days'):
        submitted('response_days_submitted', launch.submit_response_days())
    if not exists('response_pairs'):
        value = wait_until('waiting_parallel_response_days',
            lambda v:v['response_days']['terminal'] == len(launch.CELLS))
        if value['response_days']['passed'] != len(launch.CELLS) or value['response_days']['covered'] != len(launch.CELLS):
            raise RuntimeError('All three response days need native integrity and coverage')
        submitted('response_pairs_submitted', submit_response_pairs())
    if not exists('response_finalize'):
        submitted('response_finalizer_submitted', submit_response_finalizer())
    wait_until('waiting_parallel_response_summary', lambda v:v['response_summary'] is not None)
    value = collect_small(HERE/'response_summary.json', 'response_summary.json', SUMMARY_MAX_BYTES)
    finish('complete', response_status=value.get('status'),
        response_integrity=value.get('integrity'), parallel_parity_passed=True,
        next='Review300-second direction, regret and stable degradation before expansion.')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        failure = dict(status='failed', type=type(error).__name__, message=str(error))
        path = HERE/'continuation_failure.json'
        if not path.exists():
            infra.write(path, failure)
        log('stopped', **failure)
        raise
