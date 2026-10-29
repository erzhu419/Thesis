"""One frozen coverage observation, with no suffixes, labels or fitting."""
from pathlib import Path
import json
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
V88 = HERE.parent/'mcwm_closed_loop_service_response_v88_20261028'
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_execution_phase_v19_20261022.infra import (
    CPUS, PYTHON, SCHEDULER, environment, remote, scheduler)

NODES = ['node004', *[node for node in CPUS if node != 'node004']]


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def stage():
    paths = [*sorted(HERE.glob('*.py')), HERE/'protocol.json']
    scheduler.run_on('node001', 'mkdir -p '+shlex.quote(remote(HERE)), timeout=45, check=True)
    subprocess.run(['rsync', '-a', '--timeout=30', '-e', scheduler._ssh_rsync_shell_for_node('node001'),
        *map(str, paths), scheduler._ssh_target_for_node('node001')+':'+remote(HERE)+'/'],
        timeout=90, check=True)
    result = dict(files=len(paths), uploaded_bytes=sum(path.stat().st_size for path in paths), old_paths_written=[])
    write(HERE/'staging.json', result)
    return result


def resident_gate():
    """Load only the stable prior actor, six old issue inputs and v88's audit."""
    code = '''import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,REPO)
from experiments.mcwm_closed_loop_cost_advantage_v85_20261027.native import load_reference,V84
repo=Path(REPO)
v85=repo/'experiments/mcwm_closed_loop_cost_advantage_v85_20261027'
v88=repo/'experiments/mcwm_closed_loop_service_response_v88_20261028'
old_audit=json.loads((v88/'freeze_audit.json').read_text())
trace=v88/'runs/preflight/prefix_trace.jsonl'
controller,contract,stations,metadata=load_reference('stable')
checks=dict(prior_audit_pass=old_audit['all_pass'] and all(old_audit['checks'].values()),
    prior_initializer_6207=old_audit['source_initializer']==metadata['source_training_seed']==6207,
    same_reference_actor=old_audit['reference_source_paths']['stable']==metadata['source_actor_path'],
    reference_trace_available=trace.is_file() and trace.stat().st_size>0)
roots=0
for cohort,path in (('v84',V84),('v85',v85)):
    for line in ('7X','102X','705X'):
        row=json.loads((path/'runs/stable/pairs'/line/'root.json').read_text())
        action=controller(np.asarray(row['root']['s'],np.float32))
        checks[cohort+'/'+line]=bool(
            np.array_equal(action,np.asarray(row['choice']['reference_action'],np.float32))
            and np.array_equal(controller.last_proposal,np.asarray(row['root']['reference_proposal'],np.float32)))
        roots+=1
checks['six_original_reference_inputs']=roots==6
audit=dict(all_pass=all(checks.values()),checks=checks,source_initializer=6207,
    external_seed=716207,scenario='stable',reference_roots=roots,
    source_actor_path=metadata['source_actor_path'],reference_trace_path=str(trace),
    new_fits=0,new_neural_updates=0,new_suffixes=0,new_labels=0,v87_truth_used=0,native_seconds=0)
with Path(AUDIT_PATH).open('x') as handle:
    json.dump(audit,handle,ensure_ascii=False,indent=2,allow_nan=False)
    handle.write('\\n')
print(json.dumps(audit,ensure_ascii=False,allow_nan=False))
'''.replace('REPO', repr(remote(HERE.parents[1]))).replace('AUDIT_PATH', repr(remote(HERE/'freeze_audit.json')))
    command = shlex.join(['env', *[key+'='+value for key, value in environment().items()], PYTHON, '-c', code])
    raw = scheduler.run_on('node001', command, timeout=60, check=True)[1]
    if len(raw.encode()) > 16000:
        raise ValueError('Freeze audit exceeds16KB')
    result = json.loads(raw)
    write(HERE/'freeze_audit.json', result)
    if not result['all_pass']:
        raise RuntimeError('Prior actor or observation-reference audit failed')
    return result


def task(label, runner, argv, ram_mb):
    name = json.loads((HERE/'protocol.json').read_text())['name']
    return dict(project='Thesis', description=name+' '+label, signature=name+'/'+label,
        cwd=str(HERE), cmd=shlex.join([PYTHON, '-u', remote(HERE/runner), *map(str, argv)]),
        priority='high', allowed_nodes=NODES, cpu=1, ram_mb=ram_mb, vram=0, env=environment(),
        resource_family=name+'/'+runner, ram_resource_family=name+'/'+runner,
        allow_cpu_training=True,
        cpu_training_justification='Frozen stable external716207 coverage observation from0 to12600; '
            'one native prefix, no forks, suffixes, labels, fits or SGD. Read issue-visible '
            'service state without changing actor, F, trust or the factual trajectory; no v87 data.',
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
        raise RuntimeError('Submission failed; existing attempt retained')
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


def submit_observation():
    audit = collect('freeze_audit.json')
    if not (audit['all_pass'] and all(audit['checks'].values())
            and audit['source_initializer'] == 6207 and audit['external_seed'] == 716207
            and audit['scenario'] == 'stable' and audit['reference_roots'] == 6
            and all(audit[key] == 0 for key in ('new_fits', 'new_neural_updates', 'new_suffixes',
                'new_labels', 'v87_truth_used', 'native_seconds'))):
        raise RuntimeError('The fixed original-actor observation audit has not passed')
    return submit('observation', [task('observation', 'run.py',
        ['--out', remote(HERE/'runs'/'observation')], 6000)])


def submit_summary():
    if not (HERE/'observation_submission.json').exists():
        raise RuntimeError('Submit the single declared observation collector first')
    collector = remote(HERE/'runs'/'observation'/'collector_summary.json')
    row = task('summary', 'analysis.py', ['--collector', collector,
        '--protocol', remote(HERE/'protocol.json'), '--audit', remote(HERE/'freeze_audit.json'),
        '--out', remote(HERE/'summary.json')], 1000)
    wait = 'until [ -s '+shlex.quote(collector)+' ]; do sleep 30; done\nexec '+row['cmd']
    row['cmd'] = shlex.join(['timeout', '7200', '/bin/bash', '-c', wait])
    row['cpu_training_justification'] = 'Read one coverage collector and audit; no native simulation, fitting or training.'
    return submit('summary', [row])


def collect(relative):
    if relative not in ('freeze_audit.json', 'summary.json'):
        raise ValueError('Only the bounded audit and coverage summary are collected')
    path = HERE/relative
    if path.exists():
        return json.loads(path.read_text())
    code = ('from pathlib import Path;import json;p=Path('+repr(remote(path))+');'
        'raw=json.dumps(json.loads(p.read_text()),separators=(",",":"),allow_nan=False);'
        'assert len(raw.encode())<=16000;print(raw)')
    raw = scheduler.run_on('node001', shlex.join([PYTHON, '-c', code]), timeout=45, check=True)[1]
    if len(raw.encode()) > 16000:
        raise ValueError('Collected summary exceeds16KB')
    value = json.loads(raw)
    write(path, value)
    return value


def status(batch):
    ids = ','.join(row['id'] for row in json.loads((HERE/(batch+'_submission.json')).read_text())['submitted'])
    result = subprocess.run([sys.executable, str(SCHEDULER), 'status', '--readonly', '--json',
        '--lock-timeout', '30', '--ids', ids], capture_output=True, text=True, timeout=50)
    if result.returncode:
        raise RuntimeError(result.stderr[-1500:])
    return [{key: row.get(key) for key in ('id', 'status', 'priority', 'node', 'signature')}
            for row in json.loads(result.stdout)['tasks']]
