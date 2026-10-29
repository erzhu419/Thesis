"""One fixed native service preflight; no fits, training or v87 evidence reads."""
from pathlib import Path
import json
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
V86 = HERE.parent/'mcwm_closed_loop_fresh_response_v86_20261027'
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_execution_phase_v19_20261022.infra import (
    CPUS, PYTHON, SCHEDULER, environment, remote, scheduler)


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
    result = dict(files=len(paths), uploaded_bytes=sum(path.stat().st_size for path in paths),
                  old_paths_written=[])
    write(HERE/'staging.json', result)
    return result


def resident_gate():
    """Read the prior model and replay only v84/v85 issue inputs through actor/F."""
    expected = json.loads((V86/'model.json').read_text())
    code = '''import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,REPO)
from experiments.mcwm_closed_loop_cost_advantage_v85_20261027.native import load_reference,V84
repo=Path(REPO)
v85=repo/'experiments/mcwm_closed_loop_cost_advantage_v85_20261027'
v86=repo/'experiments/mcwm_closed_loop_fresh_response_v86_20261027'
expected=json.loads(sys.argv[1])
model=json.loads((v86/'model.json').read_text())
checks=dict(source_model_exact=model==expected,
    source_model_gate=model['gate']['all_pass'],
    original_four_fits=model['fits']==4,
    source_initializer_6207=all(row['training_scope']['source_seed']==6207 for row in model['arms'].values()),
    source_zero_neural_updates=all(model[key]==0 for key in
        ('actor_updates','critic_updates','value_updates','B_updates','world_updates','trust_updates')))
paths={}
roots=0
for scene in ('stable','local_speed_drop'):
    controller,contract,stations,metadata=load_reference(scene)
    paths[scene]=metadata['source_actor_path']
    checks[scene+'/reference_actor_source']=paths[scene]==model['source_reference_paths'][scene]
    for cohort,path in (('v84',V84),('v85',v85)):
        for line in ('7X','102X','705X'):
            row=json.loads((path/'runs'/scene/'pairs'/line/'root.json').read_text())
            action=controller(np.asarray(row['root']['s'],np.float32))
            checks[cohort+'/'+scene+'/'+line]=bool(
                np.array_equal(action,np.asarray(row['choice']['reference_action'],np.float32))
                and np.array_equal(controller.last_proposal,np.asarray(row['root']['reference_proposal'],np.float32)))
            roots+=1
checks['twelve_original_issue_inputs']=roots==12
audit=dict(all_pass=all(checks.values()),checks=checks,
    source_initializer=6207,external_seed=716207,scenario='stable',reference_roots=roots,
    reference_source_paths=paths,source_model_path=str(v86/'model.json'),
    original_training_fits=model['fits'],new_fits=0,new_neural_updates=0,
    v87_truth_used=0,native_seconds=0)
with Path(AUDIT_PATH).open('x') as handle:
    json.dump(audit,handle,ensure_ascii=False,indent=2,allow_nan=False)
    handle.write('\\n')
print(json.dumps(audit,ensure_ascii=False,allow_nan=False))
'''.replace('REPO', repr(remote(HERE.parents[1]))).replace('AUDIT_PATH', repr(remote(HERE/'freeze_audit.json')))
    command = shlex.join(['env', *[key+'='+value for key, value in environment().items()],
                         PYTHON, '-c', code, json.dumps(expected, separators=(',', ':'), allow_nan=False)])
    raw = scheduler.run_on('node001', command, timeout=60, check=True)[1]
    if len(raw.encode()) > 64000:
        raise ValueError('Freeze audit exceeds64KB')
    result = json.loads(raw)
    write(HERE/'freeze_audit.json', result)
    if not result['all_pass']:
        raise RuntimeError('Prior actor/F or frozen model gate failed; no preflight submitted')
    return result


def task(label, runner, argv, cpu, ram_mb):
    protocol = json.loads((HERE/'protocol.json').read_text())
    return dict(project='Thesis', description=protocol['name']+' '+label,
        signature=protocol['name']+'/'+label, cwd=str(HERE),
        cmd=shlex.join([PYTHON, '-u', remote(HERE/runner), *map(str, argv)]),
        priority='high', allowed_nodes=CPUS, cpu=cpu, ram_mb=ram_mb, vram=0,
        env=environment(), resource_family=protocol['name']+'/'+runner,
        ram_resource_family=protocol['name']+'/'+runner,
        allow_cpu_training=True,
        cpu_training_justification='Fixed issue-visible service preflight: one stable external716207 '
            'reference prefix, six covariate slots and only two predeclared7X action pairs '
            'with four actually parallel suffixes. At most60600 simulation seconds, zero fits, '
            'SGD or v87 truth use. Original actor/F/trust remain frozen.',
        allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True, no_auto_requeue=True)


def submit(batch, rows):
    path = HERE/(batch+'_submission.json')
    if path.exists():
        return json.loads(path.read_text())
    attempt = HERE/(batch+'_submission_attempt.json')
    if attempt.exists():
        raise RuntimeError('Inspect existing task signatures before submitting again')
    jobs = HERE/(batch+'_jobs.json')
    write(jobs, rows)
    write(attempt, dict(signatures=[row['signature'] for row in rows]))
    submitted = subprocess.run([sys.executable, str(SCHEDULER), 'submit-jsonl', '--file', str(jobs),
        '--trusted', '--json', '--lock-timeout', '300'], capture_output=True, text=True, timeout=360)
    if submitted.returncode:
        write(HERE/(batch+'_failure.json'), dict(returncode=submitted.returncode,
            stdout=submitted.stdout[-5000:], stderr=submitted.stderr[-3000:]))
        raise RuntimeError('Submission failed; existing attempt retained')
    result = json.loads(submitted.stdout)
    write(path, result)
    dispatched = subprocess.run([sys.executable, str(SCHEDULER), 'dispatch', '--no-bulk-window',
        '--lock-timeout', '300', *[item for row in result['submitted'] for item in ('--task-id', row['id'])]],
        capture_output=True, text=True, timeout=360)
    write(HERE/(batch+'_dispatch.json'), dict(returncode=dispatched.returncode,
        stdout=dispatched.stdout[-7000:], stderr=dispatched.stderr[-2000:]))
    if dispatched.returncode:
        raise RuntimeError('Task IDs known; inspect before redispatch')
    return result


def submit_preflight():
    audit = collect('freeze_audit.json')
    if not (audit['all_pass'] and all(audit['checks'].values())
            and audit['source_initializer'] == 6207 and audit['external_seed'] == 716207
            and audit['scenario'] == 'stable' and audit['reference_roots'] == 12
            and audit['new_fits'] == audit['new_neural_updates'] == audit['v87_truth_used'] == audit['native_seconds'] == 0):
        raise RuntimeError('The fixed original-actor audit has not passed')
    return submit('preflight', [task('preflight', 'run.py',
        ['--out', remote(HERE/'runs'/'preflight')], 2, 8000)])


def submit_summary():
    if not (HERE/'preflight_submission.json').exists():
        raise RuntimeError('Submit the single declared preflight collector first')
    collector = remote(HERE/'runs'/'preflight'/'collector_summary.json')
    row = task('summary', 'analysis.py', ['--collector', collector,
        '--protocol', remote(HERE/'protocol.json'), '--audit', remote(HERE/'freeze_audit.json'),
        '--out', remote(HERE/'summary.json')], 1, 1500)
    wait = 'until [ -s '+shlex.quote(collector)+' ]; do sleep 30; done\nexec '+row['cmd']
    row['cmd'] = shlex.join(['timeout', '7200', '/bin/bash', '-c', wait])
    row['cpu_training_justification'] = ('Read the one native collector, fixed protocol and original-actor '
        'freeze audit; zero fits, training or new branches. Only covariate coverage, timing and '
        'native physical separation are evaluated; no cost or prediction selection.')
    return submit('summary', [row])


def collect(relative):
    if relative not in ('freeze_audit.json', 'summary.json'):
        raise ValueError('Only the bounded freeze audit and final preflight summary are collected')
    path = HERE/relative
    if path.exists():
        return json.loads(path.read_text())
    code = ('from pathlib import Path;import json;p=Path('+repr(remote(path))+');'
        'raw=json.dumps(json.loads(p.read_text()),separators=(",",":"),allow_nan=False);'
        'assert len(raw.encode())<=64000;print(raw)')
    raw = scheduler.run_on('node001', shlex.join([PYTHON, '-c', code]), timeout=45, check=True)[1]
    if len(raw.encode()) > 64000:
        raise ValueError('Collected summary exceeds64KB')
    result = json.loads(raw)
    write(path, result)
    return result


def status(batch):
    ids = ','.join(row['id'] for row in json.loads((HERE/(batch+'_submission.json')).read_text())['submitted'])
    result = subprocess.run([sys.executable, str(SCHEDULER), 'status', '--readonly', '--json',
        '--lock-timeout', '30', '--ids', ids], capture_output=True, text=True, timeout=50)
    if result.returncode:
        raise RuntimeError(result.stderr[-1500:])
    return [{key: row.get(key) for key in ('id', 'status', 'priority', 'node', 'signature')}
            for row in json.loads(result.stdout)['tasks']]
