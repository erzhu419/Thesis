"""Fixed historical critic coverage study; resident data and small receipts only."""
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.mcwm_closed_loop_execution_phase_v19_20261022.infra import environment as native_environment
from experiments.mcwm_full_state_recovery_v105_20261029.infra import (
    scheduler, remote, PYTHON, SCHEDULER, environment as original_environment, write)


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def environment():
    env = native_environment()
    env.update(original_environment())
    return env


def stage():
    paths = sorted([*HERE.glob('*.py'), HERE/'protocol.json'])
    scheduler.run_on('node001', 'mkdir -p '+shlex.quote(remote(HERE)), timeout=45, check=True)
    subprocess.run(['rsync', '-a', '--timeout=30', '-e', scheduler._ssh_rsync_shell_for_node('node001'),
        *map(str, paths), scheduler._ssh_target_for_node('node001')+':'+remote(HERE)+'/'],
        timeout=90, check=True)
    write(HERE/'staging.json', dict(files=[p.name for p in paths],
        uploaded_bytes=sum(p.stat().st_size for p in paths)))


def submit_pair(scene):
    p = protocol()
    recovered = json.loads((ROOT/'experiments/mcwm_endpoint_restore_v106_20261029/summary.json').read_text())
    if recovered['status'] != 'complete':
        raise RuntimeError('The recovered endpoint must pass before native work')
    if scene == 'local_speed_drop':
        gate = json.loads((HERE/'preflight_gate.json').read_text())
        if not gate['all_pass']:
            raise RuntimeError('First-packet technical gate must pass before shock pair')
    elif scene != 'stable':
        raise ValueError(scene)
    rows = []
    for arm in p['arms']:
        argv = [PYTHON, '-B', '-u', remote(HERE/'control.py'), '--thesis-root',remote(ROOT),
            '--scenario',scene,'--arm',arm,'--out',remote(HERE/'runs'/scene/arm)]
        rows.append(dict(project='Thesis', description=p['name']+' '+scene+' '+arm,
            signature=p['name']+'/'+scene+'/'+arm, cwd=str(HERE), cmd=shlex.join(argv),
            priority='high', allowed_nodes=p['runtime']['allowed_nodes'], cpu=1,
            ram_mb=p['runtime']['ram_mb'], vram=0, env=environment(),
            resource_family=p['name']+'/native', ram_resource_family=p['name']+'/native',
            skip_launch_staging=True, no_auto_requeue=True, allow_no_resume=True,
            allow_no_ckpt=True, allow_cpu_training=True,
            cpu_training_justification='Fixed complete MC-WM endpoint6201, paired24000-second target day; '
                '456 fixed-weight mixed TD updates from57 packets. Fresh critic covers128 historical plus64 target rows; '
                'V,actor,temperature use original batches. Added64 critic rows and256 bootstrap forward rows per fresh update. '
                'No initialization,source collection or model refit.'))
    write(HERE/(scene+'_jobs.json'), rows)
    write(HERE/(scene+'_submission_attempt.json'), dict(signatures=[r['signature'] for r in rows]))
    proc = subprocess.run([sys.executable,str(SCHEDULER),'submit-jsonl','--file',
        str(HERE/(scene+'_jobs.json')),'--trusted','--json','--lock-timeout','300'],
        capture_output=True,text=True,timeout=360)
    if proc.returncode:
        write(HERE/(scene+'_submission_error.json'),dict(stdout=proc.stdout,stderr=proc.stderr))
        raise RuntimeError('Inspect failed submission; never resubmit blindly')
    result = json.loads(proc.stdout)
    write(HERE/(scene+'_submission.json'),result)
    print(json.dumps(result),flush=True)
    proc = subprocess.run([sys.executable,str(SCHEDULER),'dispatch','--no-bulk-window',
        '--lock-timeout','300',*[item for row in result['submitted'] for item in ('--task-id',row['id'])]],
        capture_output=True,text=True,timeout=360)
    write(HERE/(scene+'_dispatch.json'),dict(returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr))
    return result


def poll():
    # Prefixes and full summaries are examined in place; only the small verdict leaves the server.
    submitted = 4 if (HERE/'local_speed_drop_submission.json').exists() else 2
    have_gate = (HERE/'preflight_gate.json').exists()
    code = ('from pathlib import Path\nimport sys,json\n'
        'here=Path('+repr(remote(HERE))+')\nroot=Path('+repr(remote(ROOT))+')\n'
        'sys.path.insert(0,str(here))\nimport analyze\n'
        'gate=analyze.preflight(root,here)\n'
        'files=list((here/"runs").glob("*/*/summary.json"))\n'
        'failed=[str(p.relative_to(here)) for p in files if json.loads(p.read_text()).get("status")!="complete"]\n'
        'held=bool(failed) or (gate.get("status")!="pending" and not gate.get("all_pass"))\n'
        'final=analyze.analyze(root) if len(files)==4 or (held and len(files)>='+str(submitted)+') else None\n'+
        ('if gate.get("status")!="pending": gate={k:gate[k] for k in ("status","all_pass")}\n' if have_gate else '')+
        'value=dict(gate=gate,final=final,finished_days=len(files),failed=failed)\n'
        'raw=json.dumps(value,separators=(",",":"),allow_nan=False)\n'
        'assert len(raw.encode())<=64000\nprint(raw)\n')
    raw = scheduler.run_on('node001',shlex.join([PYTHON,'-B','-c',code]),timeout=45,check=True)[1]
    return json.loads(raw),len(raw.encode())


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'stage': stage()
    elif mode in ('stable','local_speed_drop'): submit_pair(mode)
    elif mode == 'poll': print(json.dumps(poll()[0]),flush=True)
    else: raise SystemExit('Use stage, stable, local_speed_drop or poll')
