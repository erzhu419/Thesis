"""High-priority fixed-cohort fitting and heldout jobs; bounded JSON only."""
from pathlib import Path
import json
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
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
        *map(str, paths), scheduler._ssh_target_for_node('node001')+':'+remote(HERE)+'/'], timeout=90, check=True)
    value = dict(files=len(paths), uploaded_bytes=sum(path.stat().st_size for path in paths), old_paths_written=[])
    write(HERE/'staging.json', value)
    return value


def resident_gate():
    code = '''import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,REPO)
from experiments.mcwm_closed_loop_cost_advantage_v85_20261027.native import load_reference,V84
V85=Path(REPO)/'experiments/mcwm_closed_loop_cost_advantage_v85_20261027'
checks={}
for scene in ('stable','local_speed_drop'):
    controller,contract,stations,metadata=load_reference(scene)
    metadata['heldout_external_seed']=316207
    checks[scene+'/heldout_external_seed']=metadata['heldout_external_seed']==316207
    for cohort,path in (('v84_stale',V84),('v85_fresh',V85)):
        for line in ('7X','102X','705X'):
            row=json.loads((path/'runs'/scene/'pairs'/line/'root.json').read_text())
            actual=controller(np.asarray(row['root']['s'],np.float32))
            checks[cohort+'/'+scene+'/'+line]=bool(
                np.array_equal(actual,np.asarray(row['choice']['reference_action'],np.float32))
                and np.array_equal(controller.last_proposal,np.asarray(row['root']['reference_proposal'],np.float32)))
print(json.dumps(dict(all_pass=all(checks.values()),checks=checks,reference_roots=12,
    heldout_external_seed=316207,native_seconds=0,new_fits=0,replay_loaded=False)))
'''.replace('REPO', repr(remote(HERE.parents[1])))
    command = shlex.join(['env', *[key+'='+value for key, value in environment().items()], PYTHON, '-c', code])
    result = json.loads(scheduler.run_on('node001', command, timeout=60, check=True)[1])
    write(HERE/'resident_gate.json', result)
    if not result['all_pass']:
        raise RuntimeError('Prior reference reproduction failed; no jobs submitted')
    return result


def task(label, runner, argv, cpu, ram_mb):
    protocol = json.loads((HERE/'protocol.json').read_text())
    return dict(project='Thesis', description=protocol['name']+' '+label, signature=protocol['name']+'/'+label,
        cwd=str(HERE), cmd=shlex.join([PYTHON, '-u', remote(HERE/runner), *map(str, argv)]),
        priority='high', allowed_nodes=CPUS, cpu=cpu, ram_mb=ram_mb, vram=0, env=environment(),
        resource_family=protocol['name']+'/'+runner, ram_resource_family=protocol['name']+'/'+runner,
        allow_cpu_training=True,
        cpu_training_justification='Four closed-form 3/4-parameter D fits on two fixed six-pair cohorts '
            '(v84 stale external106207 and v85 fresh external216207); then actor-only frozen-reference '
            'heldout with new external316207, two prefixes and twelve suffixes within169200 native seconds. '
            'No initializer/replay reload or prefix/suffix neural SGD; paired native arms actually concurrent; '
            'new heldout truth excluded from fitting; actor/Q/V/B/F/trust frozen.',
        allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True, no_auto_requeue=True)


def submit(batch, rows):
    path = HERE/(batch+'_submission.json')
    if path.exists():
        return json.loads(path.read_text())
    attempt = HERE/(batch+'_submission_attempt.json')
    if attempt.exists():
        raise RuntimeError('Inspect existing task signatures; no duplicate submission')
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
    dispatch = subprocess.run([sys.executable, str(SCHEDULER), 'dispatch', '--no-bulk-window', '--lock-timeout', '300',
        *[arg for row in value['submitted'] for arg in ('--task-id', row['id'])]],
        capture_output=True, text=True, timeout=360)
    write(HERE/(batch+'_dispatch.json'), dict(returncode=dispatch.returncode,
        stdout=dispatch.stdout[-7000:], stderr=dispatch.stderr[-2000:]))
    if dispatch.returncode:
        raise RuntimeError('Tasks known; inspect existing IDs before redispatch')
    return value


def submit_training():
    return submit('training', [task('training', 'train.py', [], 1, 2000)])


def submit_heldout():
    training = collect('train_summary.json')
    if not training['gate']['all_pass']:
        raise RuntimeError('Fixed training gate held; no new heldout')
    return submit('heldout', [task('heldout/'+scene, 'run.py',
        ['--scenario', scene, '--out', remote(HERE/'runs'/scene)], 2, 8000)
        for scene in ('stable', 'local_speed_drop')])


def collect(relative):
    if relative not in ('model.json', 'train_summary.json', 'summary.json'):
        raise ValueError('Only the three small model/result JSONs are collected')
    path = HERE/relative
    if path.exists():
        return json.loads(path.read_text())
    code = ('from pathlib import Path;import json;p=Path('+repr(remote(path))+');'
        'raw=json.dumps(json.loads(p.read_text()),separators=(",",":"),allow_nan=False);'
        'assert len(raw.encode())<=64000;print(raw)')
    raw = scheduler.run_on('node001', shlex.join([PYTHON, '-c', code]), timeout=45, check=True)[1]
    value = json.loads(raw)
    write(path, value)
    return value


def submit_summary():
    """Wait on the two resident collectors, then score once without new simulation."""
    if not (HERE/'heldout_submission.json').exists():
        raise RuntimeError('Submit the known two native collectors first')
    stable = remote(HERE/'runs/stable/collector_summary.json')
    shock = remote(HERE/'runs/local_speed_drop/collector_summary.json')
    argv = ['--stable', stable, '--shock', shock, '--model', remote(HERE/'model.json'),
            '--out', remote(HERE/'summary.json')]
    row = task('summary', 'analysis.py', argv, 1, 1500)
    wait = ('until [ -s '+shlex.quote(stable)+' ] && [ -s '+shlex.quote(shock)
        +' ]; do sleep 30; done\nexec '+row['cmd'])
    row['cmd'] = shlex.join(['timeout', '7200', '/bin/bash', '-c', wait])
    row['cpu_training_justification'] = ('Read two completed resident collector summaries and the frozen '
        'coefficient JSON once. No fits, native branches, neural updates or label feedback; raw logs stay resident.')
    return submit('summary', [row])


def analyze_resident():
    command = shlex.join([PYTHON, remote(HERE/'analysis.py'),
        '--stable', remote(HERE/'runs/stable/collector_summary.json'),
        '--shock', remote(HERE/'runs/local_speed_drop/collector_summary.json'),
        '--model', remote(HERE/'model.json'), '--out', remote(HERE/'summary.json')])
    return scheduler.run_on('node001', command, timeout=45, check=True)[1]


def status(batch):
    ids = ','.join(row['id'] for row in json.loads((HERE/(batch+'_submission.json')).read_text())['submitted'])
    result = subprocess.run([sys.executable, str(SCHEDULER), 'status', '--readonly', '--json',
        '--lock-timeout', '30', '--ids', ids], capture_output=True, text=True, timeout=50)
    if result.returncode:
        raise RuntimeError(result.stderr[-1500:])
    return [{key: row.get(key) for key in ('id', 'status', 'priority', 'node', 'signature')}
            for row in json.loads(result.stdout)['tasks']]
