"""v104 frozen historical actor evaluation and compact resident summaries."""
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def stage(label):
    paths = sorted([*HERE.glob('*.py'), HERE/'protocol.json'])
    scheduler.run_on('node001', 'mkdir -p '+shlex.quote(remote(HERE)), timeout=45, check=True)
    subprocess.run(['rsync', '-a', '--timeout=30', '-e', scheduler._ssh_rsync_shell_for_node('node001'),
                    *map(str, paths), scheduler._ssh_target_for_node('node001')+':'+remote(HERE)+'/'],
                   timeout=90, check=True)
    receipt = dict(files=[p.name for p in paths], uploaded_bytes=sum(p.stat().st_size for p in paths))
    write(HERE/(label+'_staging.json'), receipt)
    return receipt


def task(label, runner, argv, cpu=1, ram_mb=4000):
    name = HERE.name
    return dict(project='Thesis', description=name+' '+label, signature=name+'/'+label,
                cwd=str(HERE), cmd=shlex.join([PYTHON, '-B', '-u', remote(HERE/runner), *map(str, argv)]),
                priority='high', allowed_nodes=CPUS, cpu=cpu, ram_mb=ram_mb, vram=0,
                env=environment(), resource_family=name+'/native', ram_resource_family=name+'/native',
                allow_cpu_training=True, cpu_training_justification='Frozen historical MC-WM actors, zero optimizer or data-fitting updates; '
                'five actors crossed with six fixed v75 environment days. Cap720000 native seconds. '
                'First formal day reused; no retry, extra probe or validation-label release.',
                allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True, no_auto_requeue=True)


def submit(label, rows):
    write(HERE/(label+'_jobs.json'), rows)
    write(HERE/(label+'_submission_attempt.json'), dict(signatures=[r['signature'] for r in rows]))
    process = subprocess.run([sys.executable, str(SCHEDULER), 'submit-jsonl', '--file',
                              str(HERE/(label+'_jobs.json')), '--trusted', '--json', '--lock-timeout', '300'],
                             capture_output=True, text=True, timeout=360)
    if process.returncode:
        write(HERE/(label+'_submission_error.json'), dict(stdout=process.stdout, stderr=process.stderr))
        raise RuntimeError('Submission failed; inspect saved receipt')
    result = json.loads(process.stdout)
    write(HERE/(label+'_submission.json'), result)
    dispatch = subprocess.run([sys.executable, str(SCHEDULER), 'dispatch', '--no-bulk-window',
                               '--lock-timeout', '300', *[s for r in result['submitted'] for s in ('--task-id', r['id'])]],
                              capture_output=True, text=True, timeout=360)
    write(HERE/(label+'_dispatch.json'), dict(returncode=dispatch.returncode,stdout=dispatch.stdout,stderr=dispatch.stderr))
    return result


def pull(relative, limit=65536):
    code = 'from pathlib import Path\nimport json\np=Path('+repr(remote(HERE/relative))+')\n'
    code += 'raw=json.dumps(json.loads(p.read_text()),separators=(",",":"))\n'
    code += 'assert len(raw.encode())<='+str(limit)+'\nprint(raw)\n'
    raw = scheduler.run_on('node001', shlex.join([PYTHON,'-B','-c',code]),timeout=45,check=True)[1]
    value = json.loads(raw)
    write(HERE/relative,value)
    return value, len(raw.encode())


def rows(first=False):
    p = json.loads((HERE/'protocol.json').read_text())
    cells = [dict(training_seed=seed, external_seed=external, scenario=scene)
             for seed in p['historical_training_seeds']
             for external in p['external_seeds'] for scene in p['scenarios']]
    chosen = [c for c in cells if (c == p['first_formal_day']) == first]
    return [task('/'.join(map(str, c.values())), 'run.py', [
        '--training-seed',c['training_seed'],'--external-seed',c['external_seed'],
        '--scenario',c['scenario'],'--out',remote(HERE/'runs'/str(c['training_seed'])/
                                                str(c['external_seed'])/c['scenario'])]) for c in chosen]


if __name__ == '__main__':
    mode = sys.argv[1:]
    if mode == ['first']:
        print(stage('initial'),flush=True)
        print(submit('first',rows(first=True)))
    elif mode == ['remaining']:
        p = json.loads((HERE/'protocol.json').read_text())
        c = p['first_formal_day']
        result = json.loads((HERE/'runs'/str(c['training_seed'])/str(c['external_seed'])/
                             c['scenario']/'summary.json').read_text())
        if result['status'] != 'complete' or not result['integrity']['all_pass']:
            raise RuntimeError('First formal day failed the technical gate')
        print(submit('remaining',rows(first=False)))
    else:
        raise SystemExit('Use first or remaining')
