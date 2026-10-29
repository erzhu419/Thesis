"""Run only the two declared live-state contrasts; retain native logs remotely."""
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.mcwm_behavior_replay_v113_20261029.infra import (
    scheduler, remote, PYTHON, SCHEDULER, environment, write)


def stage():
    paths = sorted([*HERE.glob('*.py'), HERE/'protocol.json', HERE/'anchor_binding.json'])
    scheduler.run_on('node001', 'mkdir -p '+shlex.quote(remote(HERE)), timeout=45, check=True)
    subprocess.run(['rsync', '-a', '--timeout=30', '-e', scheduler._ssh_rsync_shell_for_node('node001'),
        *map(str, paths), scheduler._ssh_target_for_node('node001')+':'+remote(HERE)+'/'],
        timeout=90, check=True)
    write(HERE/'staging.json', dict(files=[p.name for p in paths],
        uploaded_bytes=sum(p.stat().st_size for p in paths)))


def submit():
    p = json.loads((HERE/'protocol.json').read_text())
    gate = json.loads((HERE/'implementation_checks.json').read_text())
    if not gate['all_pass']:
        raise RuntimeError('Finish the CPU fork and analysis checks before native execution')
    rows = []
    for scene in p['scenarios']:
        argv = [PYTHON, '-B', '-u', remote(HERE/'control.py'), '--thesis-root', remote(ROOT),
            '--scenario', scene, '--arm', 'fresh', '--out', remote(HERE/'runs'/scene)]
        rows.append(dict(project='Thesis', description=p['name']+' '+scene,
            signature=p['name']+'/'+scene, cwd=str(HERE), cmd=shlex.join(argv),
            priority='high', require_node='node001', allowed_nodes=['node001'], cpu=2,
            ram_mb=p['runtime']['ram_mb'], vram=0, env=environment(),
            resource_family=p['name']+'/live-fork', ram_resource_family=p['name']+'/live-fork',
            skip_launch_staging=True, no_auto_requeue=True, allow_no_resume=True,
            allow_no_ckpt=True, allow_cpu_training=True,
            cpu_training_justification='One original v111 fresh prefix then two CPU1 live-memory children '
                'from one complete libsumo/learner/Adam/RNG state. Change only the fixed root holding proposal '
                'through original F; preserve subsequent learning and full24000-second objective. '
                'Reserve2 CPUs for concurrent suffixes. Across both scenes unique64218 native seconds '
                'and1032 optimizer calls per type, no initialization/source collection/world refit.'))
    write(HERE/'jobs.json', rows)
    write(HERE/'submission_attempt.json', dict(signatures=[r['signature'] for r in rows]))
    proc = subprocess.run([sys.executable, str(SCHEDULER), 'submit-jsonl', '--file',
        str(HERE/'jobs.json'), '--trusted', '--json', '--lock-timeout', '300'],
        capture_output=True, text=True, timeout=360)
    if proc.returncode:
        write(HERE/'submission_error.json', dict(stdout=proc.stdout, stderr=proc.stderr))
        raise RuntimeError('Inspect the submission record; do not resubmit automatically')
    result = json.loads(proc.stdout)
    write(HERE/'submission.json', result)
    print(json.dumps(result), flush=True)
    proc = subprocess.run([sys.executable, str(SCHEDULER), 'dispatch', '--no-bulk-window',
        '--lock-timeout', '300', *[item for row in result['submitted'] for item in ('--task-id', row['id'])]],
        capture_output=True, text=True, timeout=360)
    write(HERE/'dispatch.json', dict(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr))


def poll():
    code = ('from pathlib import Path\nimport sys,json\n'
        'here=Path('+repr(remote(HERE))+')\nroot=Path('+repr(remote(ROOT))+')\n'
        'sys.path.insert(0,str(here))\nimport analyze\n'
        'parents=list((here/"runs").glob("*/summary.json"))\n'
        'children=list((here/"runs").glob("*/branches/*/summary.json"))\n'
        'final=analyze.analyze(root) if len(parents)==2 else None\n'
        'value=dict(finished_prefixes=len(parents),finished_suffixes=len(children),final=final)\n'
        'raw=json.dumps(value,separators=(",",":"),allow_nan=False)\n'
        'assert len(raw.encode())<=64000\nprint(raw)\n')
    raw = scheduler.run_on('node001', shlex.join([PYTHON, '-B', '-c', code]), timeout=60, check=True)[1]
    return json.loads(raw), len(raw.encode())


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'stage': stage()
    elif mode == 'submit': submit()
    elif mode == 'poll': print(json.dumps(poll()[0]), flush=True)
    else: raise SystemExit('Use stage, submit or poll')
