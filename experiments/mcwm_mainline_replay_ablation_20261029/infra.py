"""Run one reused production preflight, then five parallel frozen ablation days."""
from pathlib import Path
import json
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_execution_phase_v19_20261022.infra import (
    CPUS, PYTHON, SCHEDULER, environment, remote, scheduler)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def stage():
    paths = [*sorted(HERE.glob('*.py')), HERE/'protocol.json', HERE/'reference_tables.json']
    scheduler.run_on('node001', 'mkdir -p '+shlex.quote(remote(HERE)), timeout=45, check=True)
    subprocess.run(['rsync', '-a', '--timeout=30', '-e', scheduler._ssh_rsync_shell_for_node('node001'),
        *map(str, paths), scheduler._ssh_target_for_node('node001')+':'+remote(HERE)+'/'], timeout=90, check=True)
    result = dict(files=len(paths), uploaded_bytes=sum(path.stat().st_size for path in paths),
                  old_paths_written=[])
    write(HERE/'staging.json', result)
    return result


def task(seed, scene):
    name = HERE.name
    return dict(project='Thesis', description=name+f' coupled seed{seed} {scene}',
        signature=name+f'/coupled/{seed}/{scene}', cwd=str(HERE),
        cmd=shlex.join([PYTHON, '-u', remote(HERE/'control.py'), '--seed', str(seed),
            '--scenario', scene, '--arm', 'fresh', '--external-seed', str(seed+100000),
            '--out', remote(HERE/'runs'/'coupled'/f'seed{seed}'/scene/'fresh')]),
        priority='high', allowed_nodes=CPUS, preferred_node='node004', cpu=1, ram_mb=8000,
        vram=0, env=environment(), allow_cpu_training=True,
        cpu_training_justification='Fixed mainline component ablation: reuse existing50000-step '
            'initializers; six coupled-risk fresh days compared with cached v75 decoupled days. '
            'Exactly285 factual labels and456 updates per optimizer per day, no world/trust '
            'training or counterfactual queries. Native budget144000seconds for the whole batch.',
        allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True, no_auto_requeue=True)


def submit(batch, rows):
    if (HERE/(batch+'_submission.json')).exists():
        return json.loads((HERE/(batch+'_submission.json')).read_text())
    if (HERE/(batch+'_submission_attempt.json')).exists():
        raise RuntimeError('Inspect existing signatures before another submission attempt')
    jobs = HERE/(batch+'_jobs.json')
    write(jobs, rows)
    write(HERE/(batch+'_submission_attempt.json'), dict(signatures=[row['signature'] for row in rows]))
    process = subprocess.run([sys.executable, str(SCHEDULER), 'submit-jsonl', '--file', str(jobs),
        '--trusted', '--json', '--lock-timeout', '300'], capture_output=True, text=True, timeout=360)
    if process.returncode:
        write(HERE/(batch+'_submission_failure.json'), dict(returncode=process.returncode,
            stdout=process.stdout[-4500:], stderr=process.stderr[-2000:]))
        raise RuntimeError('Submission failed; attempt retained for inspection')
    result = json.loads(process.stdout)
    write(HERE/(batch+'_submission.json'), result)
    dispatch = subprocess.run([sys.executable, str(SCHEDULER), 'dispatch', '--no-bulk-window',
        '--lock-timeout', '300', *[part for row in result['submitted'] for part in ('--task-id', row['id'])]],
        capture_output=True, text=True, timeout=360)
    write(HERE/(batch+'_dispatch.json'), dict(returncode=dispatch.returncode,
        stdout=dispatch.stdout[-7000:], stderr=dispatch.stderr[-2000:]))
    if dispatch.returncode:
        raise RuntimeError('Task IDs known; inspect before redispatching')
    return result


def submit_first():
    first = task(6206, 'stable')
    continuation = dict(first,
        signature=HERE.name+'/continuation_summary', description=HERE.name+' finite continuation and compact analysis',
        cmd=shlex.join([sys.executable, '-u', str(HERE/'infra.py'), 'continue']),
        allowed_nodes=['local'], require_node='local', preferred_node='local', cpu=1, ram_mb=1000,
        env=dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'),
        cpu_training_justification='A bounded controller for six predeclared production days: '
            'read the reused first8-step technical preflight, submit only the remaining five '
            'days if valid, then retrieve one compact server-side analysis. No direct training.')
    return submit('first', [first, continuation])


def resident_status():
    root = remote(HERE/'runs'/'coupled')
    code = '''from pathlib import Path
import json
root=Path(ROOT)
out={}
for seed in (6206,6207,6208):
 for scene in ('stable','local_speed_drop'):
  p=root/f'seed{seed}'/scene/'fresh'
  out[str(seed)+'/'+scene]={name:(p/name).exists() for name in ('summary.json','failure.json','preflight.json')}
print(json.dumps(out))
'''.replace('ROOT', repr(root))
    return json.loads(scheduler.run_on('node001', shlex.join([PYTHON, '-c', code]), timeout=45, check=True)[1])


def first_gate():
    new = remote(HERE/'runs'/'coupled'/'seed6206'/'stable'/'fresh'/'preflight.json')
    old = remote(HERE.parent/'mcwm_closed_loop_frozen_validation_v75_20261027'/'runs'/'executed_cql'/'validation'/'seed6206'/'stable'/'fresh'/'preflight.json')
    code = '''from pathlib import Path
import json
new=json.loads(Path(NEW).read_text());old=json.loads(Path(OLD).read_text())
a=new['prefix_before_first_update'];b=old['prefix_before_first_update']
checks=dict(technical=new['technical_gate_pass'] and all(new['checks'].values()),
 initializer=new['initializer']['path']==old['initializer']['path'],
 rows=a['rows']==b['rows'] and a['rows']>0,clock=a['clock']==b['clock'],
 initial_updates=a['initial_updates']==b['initial_updates']==50000)
count=0;difference=None
with Path(a['trace_path']).open() as left,Path(b['trace_path']).open() as right:
 for index in range(a['rows']):
  x=left.readline();y=right.readline()
  if not x or not y or json.loads(x)!=json.loads(y):
   difference=index;break
  count+=1
checks['full_pre_update_prefix_exact']=count==a['rows'] and difference is None
out=dict(all_pass=all(checks.values()),checks=checks,compared_rows=count,difference=difference,
 reused_production_day=True,cost_used_for_continuation=False,native_seconds_added=0)
print(json.dumps(out,separators=(',',':')))
'''.replace('NEW', repr(new)).replace('OLD', repr(old))
    raw = scheduler.run_on('node001', shlex.join([PYTHON, '-c', code]), timeout=60, check=True)[1]
    if len(raw.encode()) > 12000:
        raise ValueError('First technical gate exceeds12KB')
    return json.loads(raw)


def finish():
    argv = [PYTHON, '-u', remote(HERE/'analysis.py'), '--root', remote(HERE/'runs'),
        '--reference', remote(HERE/'reference_tables.json'), '--protocol', remote(HERE/'protocol.json'),
        '--out', remote(HERE/'summary.json')]
    scheduler.run_on('node001', shlex.join(['env', *[k+'='+v for k,v in environment().items()], *argv]), timeout=120, check=True)
    code = ('from pathlib import Path;import json;raw=Path('+repr(remote(HERE/'summary.json'))+
            ').read_text();assert len(raw.encode())<=65536;print(raw)')
    raw = scheduler.run_on('node001', shlex.join([PYTHON, '-c', code]), timeout=45, check=True)[1]
    if len(raw.encode()) > 65536:
        raise ValueError('Final readout exceeds64KB')
    value = json.loads(raw)
    write(HERE/'summary.json', value)
    write(HERE/'retrieval.json', dict(summary_bytes=len(raw.encode()), raw_trajectories_pulled=0,
        csv_pulled=0, checkpoints_pulled=0, new_native_seconds=0))
    return value


def continuation():
    deadline = time.monotonic()+21600
    gate = None
    remaining_submitted = False
    while time.monotonic() < deadline:
        status = resident_status()
        first = status['6206/stable']
        if gate is None and first['preflight.json']:
            gate = first_gate()
            write(HERE/'preflight_gate.json', gate)
            if gate['all_pass']:
                rows = [task(seed, scene) for seed in (6206,6207,6208)
                        for scene in ('stable','local_speed_drop') if (seed,scene)!=(6206,'stable')]
                submit('remaining', rows)
                remaining_submitted = True
        if ((gate is not None and not gate['all_pass']) or
                (gate is None and first['failure.json'])) and (first['summary.json'] or first['failure.json']):
            print(json.dumps(dict(status='technical_hold', remaining_submitted=False)), flush=True)
            return finish()
        done = sum(value['summary.json'] or value['failure.json'] for value in status.values())
        if remaining_submitted and done == 6:
            return finish()
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='fixed_six_day_ablation',
            current=done,total=6,unit='native_days',technical_gate=None if gate is None else gate['all_pass'])),flush=True)
        time.sleep(20)
    raise TimeoutError('Fixed ablation did not finish within six hours; inspect existing tasks')


if __name__ == '__main__':
    if sys.argv[1:] != ['continue']:
        raise SystemExit('Use continue for the existing finite continuation task')
    continuation()
