"""One bounded completion query and local JSON-only collection analysis."""
from pathlib import Path
import json
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
V91 = HERE.parent/'mcwm_closed_loop_issue_conditions_v91_20261028'
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_execution_phase_v19_20261022.infra import (
    CPUS, PYTHON, SCHEDULER, environment, remote, scheduler)

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
    if not (HERE/'source_protocol.json').exists():
        write(HERE/'source_protocol.json', json.loads((V91/'protocol.json').read_text()))
    paths = [*sorted(HERE.glob('*.py')), HERE/'protocol.json', HERE/'source_protocol.json']
    scheduler.run_on('node001', 'mkdir -p '+shlex.quote(remote(HERE)), timeout=45, check=True)
    subprocess.run(['rsync', '-a', '--timeout=30', '-e', scheduler._ssh_rsync_shell_for_node('node001'),
        *map(str, paths), scheduler._ssh_target_for_node('node001')+':'+remote(HERE)+'/'], timeout=90, check=True)
    result = dict(files=len(paths), uploaded_bytes=sum(path.stat().st_size for path in paths), old_paths_written=[])
    write(HERE/'staging.json', result)
    return result


def resident_gate():
    command = shlex.join(['env', *[key+'='+value for key, value in environment().items()],
        PYTHON, '-u', remote(HERE/'freeze_audit.py'), '--out', remote(HERE/'freeze_audit.json')])
    rc, raw, error = scheduler.run_on('node001', command, timeout=120, check=False)
    if not raw.strip():
        raise RuntimeError('Historical source audit failed: '+error[-3000:])
    if len(raw.encode()) > 48000:
        raise ValueError('Thin source manifest exceeds48KB')
    value = json.loads(raw)
    write(HERE/'freeze_audit.json', value)
    if rc or not value['all_pass']:
        raise RuntimeError('Frozen old pairs/conditions held: '+error[-1500:])
    return value


def task(label, runner, argv, cpu, ram_mb):
    name = json.loads((HERE/'protocol.json').read_text())['name']
    return dict(project='Thesis', description=name+' '+label, signature=name+'/'+label,
        cwd=str(HERE), cmd=shlex.join([PYTHON, '-u', remote(HERE/runner), *map(str, argv)]),
        priority='high', allowed_nodes=NODES, cpu=cpu, ram_mb=ram_mb, vram=0, env=environment(),
        resource_family=name+'/'+runner, ram_resource_family=name+'/'+runner,
        allow_cpu_training=True,
        cpu_training_justification='One frozen shock prefix through12900 and at most one new native '
            'pair with two actually parallel suffixes. Reuse all eleven audited old pairs without '
            'requerying their outcomes; new cap36900 native seconds, zero fits, SGD or v87 truth.',
        allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True, no_auto_requeue=True)


def submit(batch, rows):
    path, attempt = HERE/(batch+'_submission.json'), HERE/(batch+'_submission_attempt.json')
    if path.exists():
        return json.loads(path.read_text())
    if attempt.exists():
        raise RuntimeError('Inspect existing task signatures before submitting again')
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
    dispatch = subprocess.run([sys.executable, str(SCHEDULER), 'dispatch', '--no-bulk-window',
        '--lock-timeout', '300', *[item for row in value['submitted'] for item in ('--task-id', row['id'])]],
        capture_output=True, text=True, timeout=360)
    write(HERE/(batch+'_dispatch.json'), dict(returncode=dispatch.returncode,
        stdout=dispatch.stdout[-7000:], stderr=dispatch.stderr[-2000:]))
    if dispatch.returncode:
        raise RuntimeError('Task IDs known; inspect before redispatch')
    return value


def submit_collection():
    audit = json.loads((HERE/'freeze_audit.json').read_text())
    if not (audit['all_pass'] and all(audit['checks'].values()) and audit['reused_pairs'] == 11
            and audit['reused_labels'] == 22 and all(audit[key] == 0 for key in
            ('new_fits', 'new_neural_updates', 'new_suffixes', 'new_labels', 'v87_truth_used', 'native_seconds'))):
        raise RuntimeError('The fixed eleven-pair reuse manifest has not passed')
    return submit('collection', [task('collection', 'run.py',
        ['--out', remote(HERE/'runs'/'collection')], 2, 8000)])


def submit_summary():
    if not (HERE/'collection_submission.json').exists():
        raise RuntimeError('Submit the one declared native collector first')
    row = task('summary', 'infra.py', ['local-summary'], 1, 1000)
    row.update(cmd=shlex.join([sys.executable, '-u', str(HERE/'infra.py'), 'local-summary']),
        cwd=str(HERE), allowed_nodes=['local'], require_node='local', preferred_node='local',
        env=dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'),
        cpu_training_justification='Local bounded-JSON completion readout only: combine one compact '
            'collector with the existing thin eleven-pair source manifest; no native execution, '
            'response fit, neural update or feedback admission.')
    return submit('summary', [row])


def _pull_collector():
    code = '''import json
from pathlib import Path
p=Path(SOURCE)
if p.exists():
    body=json.loads(p.read_text())
    raw=json.dumps(body,separators=(',',':'),allow_nan=False)
    if len(raw.encode())>32000: raise ValueError('Completion collector exceeds32KB')
    result=dict(ready=True,body=body,body_bytes=len(raw.encode()))
else:
    failure=p.parent/'failure.json'
    body=json.loads(failure.read_text()) if failure.exists() else None
    result=dict(ready=False,failure=body)
print(json.dumps(result,separators=(',',':'),allow_nan=False))
'''.replace('SOURCE', repr(remote(HERE/'runs'/'collection'/'collector_summary.json')))
    raw = scheduler.run_on('node001', shlex.join([PYTHON, '-c', code]), timeout=45, check=True)[1]
    if len(raw.encode()) > 33000:
        raise ValueError('Completion receipt exceeds the bounded collector transfer')
    return json.loads(raw)


def local_summary():
    from experiments.mcwm_closed_loop_completion_v92_20261028.analysis import analyze
    audit = json.loads((HERE/'freeze_audit.json').read_text())
    audit_bytes = len(json.dumps(audit, separators=(',', ':')).encode())
    if audit_bytes > 48000:
        raise ValueError('Thin eleven-pair source manifest exceeds48KB')
    protocol = json.loads((HERE/'protocol.json').read_text())
    deadline = time.monotonic()+7200
    while time.monotonic() < deadline:
        value = _pull_collector()
        if value['ready'] or value['failure'] is not None:
            folder = HERE/'local_inputs'
            folder.mkdir(exist_ok=True)
            if value['ready']:
                write(folder/'collector_summary.json', value['body'])
                result = analyze(value['body'], protocol, audit)
                transferred = value['body_bytes']
            else:
                write(folder/'failure.json', value['failure'])
                result = dict(status='hold', integrity=dict(all_pass=False),
                    collection_gate=dict(passed=False, decision='Native completion failed; no retry or fitting'),
                    native_failure=value['failure'],
                    budget=dict(current_new=value['failure'].get('consumed_budget'),
                        historical_native_seconds=314090, reused_pairs=11, reused_labels=22),
                    new_fits=0, new_neural_updates=0, v87_truth_used=0)
                transferred = len(json.dumps(value['failure'], separators=(',', ':')).encode())
            encoded = json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'
            if len(encoded.encode()) > 96000:
                raise ValueError('Local completion summary exceeds96KB')
            write(HERE/'summary.json', result)
            write(HERE/'local_summary_receipt.json', dict(execution_node='local', source_node='node001',
                collector_transferred_bytes=transferred, existing_audit_bytes=audit_bytes,
                summary_bytes=len(encoded.encode()), new_native_seconds=0, new_fits=0,
                new_neural_updates=0, compact_inputs_only=True))
            print(json.dumps(dict(status=result['status'], summary=str(HERE/'summary.json'),
                collector_transferred_bytes=transferred)), flush=True)
            return result
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='waiting_for_completion_collector',
            current=0, total=1, unit='compact_collector')), flush=True)
        time.sleep(30)
    raise TimeoutError('Native completion did not finish within the fixed two-hour readout wait')


def collect(relative):
    if relative not in ('freeze_audit.json', 'summary.json'):
        raise ValueError('Only the thin audit and local result are collected')
    path = HERE/relative
    if not path.exists():
        raise FileNotFoundError('Inspect the existing audit or local readout task: '+relative)
    return json.loads(path.read_text())


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
