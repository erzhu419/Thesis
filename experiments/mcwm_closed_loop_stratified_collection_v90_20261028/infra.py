"""Submit the fixed native cohort and summarize bounded JSON locally."""
from pathlib import Path
import json
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
V89 = HERE.parent/'mcwm_closed_loop_load_coverage_v89_20261028'
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_execution_phase_v19_20261022.infra import (
    CPUS, PYTHON, SCHEDULER, environment, remote, scheduler)

SCENARIOS = ('stable', 'local_speed_drop')
NODES = ['node004', *[node for node in CPUS if node != 'node004']]


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name+'.writing')
    with temporary.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    temporary.rename(path)


def stage():
    if not (HERE/'strata_input.json').exists():
        write(HERE/'strata_input.json', json.loads((V89/'frozen_strata.json').read_text()))
    paths = [*sorted(HERE.glob('*.py')), HERE/'protocol.json', HERE/'strata_input.json']
    scheduler.run_on('node001', 'mkdir -p '+shlex.quote(remote(HERE)), timeout=45, check=True)
    subprocess.run(['rsync', '-a', '--timeout=30', '-e', scheduler._ssh_rsync_shell_for_node('node001'),
        *map(str, paths), scheduler._ssh_target_for_node('node001')+':'+remote(HERE)+'/'], timeout=90, check=True)
    value = dict(files=len(paths), uploaded_bytes=sum(path.stat().st_size for path in paths), old_paths_written=[])
    write(HERE/'staging.json', value)
    return value


def resident_gate():
    command = shlex.join(['env', *[key+'='+value for key, value in environment().items()],
        PYTHON, '-u', remote(HERE/'freeze_audit.py'), '--out', remote(HERE/'freeze_audit.json')])
    rc, raw, error = scheduler.run_on('node001', command, timeout=90, check=False)
    if len(raw.encode()) > 16000:
        raise ValueError('Source/reuse audit exceeds16KB')
    value = json.loads(raw)
    write(HERE/'freeze_audit.json', value)
    if rc or not value['all_pass']:
        raise RuntimeError('Frozen source/reusable pair held: '+error[-1500:])
    return value


def task(label, runner, argv, cpu, ram_mb):
    name = json.loads((HERE/'protocol.json').read_text())['name']
    return dict(project='Thesis', description=name+' '+label, signature=name+'/'+label,
        cwd=str(HERE), cmd=shlex.join([PYTHON, '-u', remote(HERE/runner), *map(str, argv)]),
        priority='high', allowed_nodes=NODES, cpu=cpu, ram_mb=ram_mb, vram=0, env=environment(),
        resource_family=name+'/'+runner, ram_resource_family=name+'/'+runner,
        allow_cpu_training=True,
        cpu_training_justification='Fixed service-response collection only: two frozen prefixes, '
            'eleven new action pairs with22 parallel suffixes; reuse the audited old pair without '
            'querying it again. New cap289200 native seconds; zero fits, SGD, admission or v87 truth.',
        allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True, no_auto_requeue=True)


def submit(batch, rows):
    path, attempt = HERE/(batch+'_submission.json'), HERE/(batch+'_submission_attempt.json')
    if path.exists():
        return json.loads(path.read_text())
    if attempt.exists():
        raise RuntimeError('Inspect existing signatures before submitting again')
    jobs = HERE/(batch+'_jobs.json')
    write(jobs, rows)
    write(attempt, dict(signatures=[row['signature'] for row in rows]))
    result = subprocess.run([sys.executable, str(SCHEDULER), 'submit-jsonl', '--file', str(jobs),
        '--trusted', '--json', '--lock-timeout', '300'], capture_output=True, text=True, timeout=360)
    if result.returncode:
        write(HERE/(batch+'_failure.json'), dict(returncode=result.returncode,
            stdout=result.stdout[-5000:], stderr=result.stderr[-3000:]))
        raise RuntimeError('Submission failed; known attempt retained')
    value = json.loads(result.stdout)
    write(path, value)
    dispatched = subprocess.run([sys.executable, str(SCHEDULER), 'dispatch', '--no-bulk-window',
        '--lock-timeout', '300', *[item for row in value['submitted'] for item in ('--task-id', row['id'])]],
        capture_output=True, text=True, timeout=360)
    write(HERE/(batch+'_dispatch.json'), dict(returncode=dispatched.returncode,
        stdout=dispatched.stdout[-7000:], stderr=dispatched.stderr[-2000:]))
    if dispatched.returncode:
        raise RuntimeError('Task IDs known; inspect before redispatch')
    return value


def submit_collection():
    audit = json.loads((HERE/'freeze_audit.json').read_text())
    if not (audit['all_pass'] and all(audit['checks'].values()) and all(audit[key] == 0
            for key in ('new_fits', 'new_neural_updates', 'new_suffixes', 'new_labels', 'v87_truth_used', 'native_seconds'))):
        raise RuntimeError('Frozen source/reuse audit held')
    rows = [task('collection/'+scene, 'run.py',
        ['--scenario', scene, '--out', remote(HERE/'runs'/scene)], 2, 8000) for scene in SCENARIOS]
    rows[1]['wait_for_files'] = [str(HERE/'first_pair_gate.json')]
    return submit('collection', rows)


def submit_summary():
    if not (HERE/'collection_submission.json').exists():
        raise RuntimeError('Submit the two declared native collectors first')
    row = task('summary', 'infra.py', ['local-summary'], 1, 1000)
    row.update(cmd=shlex.join([sys.executable, '-u', str(HERE/'infra.py'), 'local-summary']),
        cwd=str(HERE), allowed_nodes=['local'], require_node='local', preferred_node='local',
        env=dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'),
        cpu_training_justification='Local bounded-JSON readout only. Relay the first native technical '
            'gate to the scheduler file prerequisite, then read the two compact collectors; no '
            'simulation, coefficient fit, network update or truth admission.')
    return submit('summary', [row])


def _pull(relative, limit):
    allowed = {'first_pair_gate.json', 'runs/stable/collector_summary.json',
               'runs/local_speed_drop/collector_summary.json', 'freeze_audit.json'}
    if relative not in allowed:
        raise ValueError('Only compact declared JSON receipts may be pulled')
    source = remote(HERE/relative)
    code = '''import json
from pathlib import Path
p=Path(SOURCE)
if p.exists():
    body=json.loads(p.read_text())
    raw=json.dumps(body,separators=(',',':'),allow_nan=False)
    if len(raw.encode())>LIMIT: raise ValueError('Compact receipt exceeds its bounded transfer')
    result=dict(ready=True,body=body,body_bytes=len(raw.encode()))
else:
    failure=p.parent/'failure.json'
    result=dict(ready=False,failure=json.loads(failure.read_text()) if failure.exists() else None)
print(json.dumps(result,separators=(',',':'),allow_nan=False))
'''.replace('SOURCE', repr(source)).replace('LIMIT', str(limit))
    raw = scheduler.run_on('node001', shlex.join([PYTHON, '-c', code]), timeout=45, check=True)[1]
    return json.loads(raw)


def local_summary():
    from experiments.mcwm_closed_loop_stratified_collection_v90_20261028.analysis import analyze
    deadline, transferred = time.monotonic()+7200, 0
    audit = json.loads((HERE/'freeze_audit.json').read_text())
    transferred += len(json.dumps(audit, separators=(',', ':')).encode())
    protocol = json.loads((HERE/'protocol.json').read_text())
    inputs = HERE/'local_inputs'
    inputs.mkdir(exist_ok=True)
    scenes, failures, gate = {}, {}, None
    while time.monotonic() < deadline:
        if gate is None:
            if (HERE/'first_pair_gate.json').exists():
                gate = json.loads((HERE/'first_pair_gate.json').read_text())
            else:
                value = _pull('first_pair_gate.json', 8000)
                if value['ready']:
                    gate = value['body']
                    transferred += value['body_bytes']
                    write(HERE/'first_pair_gate.json', gate)
        for scenario in SCENARIOS:
            if scenario in scenes or scenario in failures:
                continue
            # Do not pull large native artifacts: only the final compact JSON.
            value = _pull('runs/'+scenario+'/collector_summary.json', 64000)
            if value['ready']:
                scenes[scenario] = value['body']
                transferred += value['body_bytes']
                write(inputs/(scenario+'.json'), value['body'])
            elif value['failure'] is not None:
                failures[scenario] = value['failure']
                transferred += len(json.dumps(value['failure'], separators=(',', ':')).encode())
                write(inputs/(scenario+'_failure.json'), value['failure'])
        if transferred > 128000:
            raise ValueError('Combined compact collectors/gate/audit exceed128KB')
        gate_failed = gate is not None and not gate['all_pass']
        if gate_failed or ('stable' in failures and gate is None):
            failures.setdefault('local_speed_drop', dict(reason='First stable pair held; no shock prefix reset',
                consumed_budget=dict(prefixes=0, suffixes=0, total_native_seconds=0)))
        if len(scenes)+len(failures) == 2:
            result = analyze(list(scenes.values()), protocol, audit)
            if failures:
                result['native_failures'] = failures
                result['budget']['failed_or_not_started'] = {scene: row.get('consumed_budget')
                    for scene, row in failures.items()}
            encoded = json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'
            if len(encoded.encode()) > 128000:
                raise ValueError('Local collection summary exceeds128KB')
            write(HERE/'summary.json', result)
            write(HERE/'local_summary_receipt.json', dict(execution_node='local', source_node='node001',
                transferred_bytes=transferred, summary_bytes=len(encoded.encode()),
                new_native_seconds=0, new_fits=0, new_neural_updates=0, compact_inputs_only=True))
            print(json.dumps(dict(status=result['status'], collection=result['collection_gate']['passed'],
                summary=str(HERE/'summary.json'), transferred_bytes=transferred)), flush=True)
            return result
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='waiting_for_native_gate_and_collectors',
            current=len(scenes)+len(failures), total=2, unit='compact_collectors')), flush=True)
        time.sleep(30)
    raise TimeoutError('Fixed native collection did not finish within the two-hour readout wait')


def collect(relative):
    if relative not in ('freeze_audit.json', 'first_pair_gate.json', 'summary.json'):
        raise ValueError('Only the bounded audit, technical gate or local result may be collected')
    path = HERE/relative
    if path.exists():
        return json.loads(path.read_text())
    if relative == 'summary.json':
        raise FileNotFoundError('The summary runs locally; inspect its existing task before retrying')
    value = _pull(relative, 16000 if relative == 'freeze_audit.json' else 8000)
    if not value['ready']:
        raise FileNotFoundError(relative)
    write(path, value['body'])
    return value['body']


def status(batch):
    ids = ','.join(row['id'] for row in json.loads((HERE/(batch+'_submission.json')).read_text())['submitted'])
    result = subprocess.run([sys.executable, str(SCHEDULER), 'status', '--readonly', '--json',
        '--lock-timeout', '30', '--ids', ids], capture_output=True, text=True, timeout=50)
    if result.returncode:
        raise RuntimeError(result.stderr[-1500:])
    return [{key: row.get(key) for key in ('id', 'status', 'priority', 'node', 'signature')}
            for row in json.loads(result.stdout)['tasks']]


if __name__ == '__main__':
    if sys.argv[1:] != ['local-summary']:
        raise SystemExit('Use local-summary for the existing scheduled readout task')
    local_summary()
