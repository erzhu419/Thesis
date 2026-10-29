"""Frozen-response confirmation jobs; no fitting and bounded JSON collection."""
from pathlib import Path
import json
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'mcwm_closed_loop_fresh_response_v86_20261027'
SEEDS = (416207, 516207, 616207)
SCENARIOS = ('stable', 'local_speed_drop')
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
    """Compare the small frozen artifact and replay 18 known actor inputs only."""
    expected = json.loads((SOURCE/'model.json').read_text())
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
training=json.loads((v86/'train_summary.json').read_text())
checks=dict(source_model_exact=model==expected,
    source_model_gate_pass=model['gate']['all_pass'],
    source_training_gate_pass=training['gate']['all_pass'],
    original_four_fits=model['fits']==4,
    source_zero_neural_updates=all(model[key]==0 for key in
        ('actor_updates','critic_updates','value_updates','B_updates','world_updates','trust_updates')),
    source_initializer_6207=all(value['training_scope']['source_seed']==6207 for value in model['arms'].values()))
roots=0
for scene in ('stable','local_speed_drop'):
    controller,contract,stations,metadata=load_reference(scene)
    checks[scene+'/reference_actor_source']=metadata['source_actor_path']==model['source_reference_paths'][scene]
    for cohort,path in (('v84_stale',V84),('v85_fresh',v85),('v86_exposed',v86)):
        for line in ('7X','102X','705X'):
            row=json.loads((path/'runs'/scene/'pairs'/line/'root.json').read_text())
            actual=controller(np.asarray(row['root']['s'],np.float32))
            checks[cohort+'/'+scene+'/'+line]=bool(
                np.array_equal(actual,np.asarray(row['choice']['reference_action'],np.float32))
                and np.array_equal(controller.last_proposal,np.asarray(row['root']['reference_proposal'],np.float32)))
            roots+=1
frozen={key:model[key] for key in ('action_support','legacy_action_support','ridge','denominator','hold_scale','feature_names')}
frozen.update(decision_threshold=0.,policy='first_supported_negative_once_then_reference',
    coefficients={arm:{kind:value['models'][kind]['coef'] for kind in ('line','state')}
                  for arm,value in model['arms'].items()})
audit=dict(all_pass=all(checks.values()),checks=checks,source_model_path=str(v86/'model.json'),
    source_training_summary_path=str(v86/'train_summary.json'),source_initializer=6207,
    confirmation_seeds=[416207,516207,616207],heldout_external_seeds=[416207,516207,616207],
    original_training_fits=model['fits'],new_fits=0,new_neural_updates=0,heldout_truth_used=0,
    source_reference_paths=model['source_reference_paths'],frozen=frozen,reference_roots=roots,
    native_seconds=0,actor_updates=0,critic_updates=0,value_updates=0,B_updates=0,
    world_updates=0,trust_updates=0,initialization_updates=0,prefix_training_updates=0,
    suffix_training_updates=0,replay_loaded=False)
with Path(AUDIT_PATH).open('x') as handle:
    json.dump(audit,handle,ensure_ascii=False,indent=2,allow_nan=False)
    handle.write('\\n')
print(json.dumps(audit,ensure_ascii=False,allow_nan=False))
'''.replace('REPO', repr(remote(HERE.parents[1]))).replace('AUDIT_PATH', repr(remote(HERE/'freeze_audit.json')))
    command = shlex.join(['env', *[key+'='+value for key, value in environment().items()],
                         PYTHON, '-c', code, json.dumps(expected, separators=(',', ':'), allow_nan=False)])
    result = json.loads(scheduler.run_on('node001', command, timeout=60, check=True)[1])
    write(HERE/'freeze_audit.json', result)
    if not result['all_pass']:
        raise RuntimeError('Frozen source or reference reproduction failed; no jobs submitted')
    return result


def task(label, runner, argv, cpu, ram_mb):
    protocol = json.loads((HERE/'protocol.json').read_text())
    return dict(project='Thesis', description=protocol['name']+' '+label, signature=protocol['name']+'/'+label,
        cwd=str(HERE), cmd=shlex.join([PYTHON, '-u', remote(HERE/runner), *map(str, argv)]),
        priority='high', allowed_nodes=CPUS, cpu=cpu, ram_mb=ram_mb, vram=0, env=environment(),
        resource_family=protocol['name']+'/'+runner, ram_resource_family=protocol['name']+'/'+runner,
        allow_cpu_training=True,
        cpu_training_justification='Frozen v86 D readouts only: zero new fits, neural updates or initializer '
            'reloads. Three independent external seeds416207/516207/616207 across two scenarios, '
            'six native prefixes and36 actually parallel paired suffixes within507600 simulation seconds. '
            'Reference actor/Q/V/B/F/trust remain unchanged; all confirmation truth stays out of learning.',
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


def submit_heldout():
    audit = collect('freeze_audit.json')
    if not (audit['all_pass'] and audit['new_fits'] == 0 and audit['new_neural_updates'] == 0
            and audit['heldout_truth_used'] == 0 and audit['confirmation_seeds'] == list(SEEDS)):
        raise RuntimeError('Frozen-model gate held; no new confirmation')
    return submit('heldout', [task(f'heldout/external{seed}/'+scene, 'run.py',
        ['--external-seed', seed, '--scenario', scene,
         '--out', remote(HERE/'runs'/f'external{seed}'/scene)], 2, 8000)
        for seed in SEEDS for scene in SCENARIOS])


def submit_summary():
    if not (HERE/'heldout_submission.json').exists():
        raise RuntimeError('Submit the six known native collectors first')
    paths = [remote(HERE/'runs'/f'external{seed}'/scene/'collector_summary.json')
             for seed in SEEDS for scene in SCENARIOS]
    row = task('summary', 'analysis.py', ['--protocol', remote(HERE/'protocol.json'),
        '--audit', remote(HERE/'freeze_audit.json'), '--model', remote(SOURCE/'model.json'),
        '--out', remote(HERE/'summary.json')], 1, 2000)
    condition = ' && '.join('[ -s '+shlex.quote(path)+' ]' for path in paths)
    wait = 'until '+condition+'; do sleep 30; done\nexec '+row['cmd']
    row['cmd'] = shlex.join(['timeout', '10800', '/bin/bash', '-c', wait])
    row['cpu_training_justification'] = ('Read six completed confirmation collector summaries, '
        'the freeze audit and source v86 coefficient JSON once; zero fits, neural updates, native '
        'branches or label feedback. All raw logs remain resident.')
    return submit('summary', [row])


def collect(relative):
    if relative not in ('freeze_audit.json', 'summary.json'):
        raise ValueError('Only the small freeze audit and result summary are collected')
    path = HERE/relative
    if path.exists():
        return json.loads(path.read_text())
    code = ('from pathlib import Path;import json;p=Path('+repr(remote(path))+');'
        'raw=json.dumps(json.loads(p.read_text()),separators=(",",":"),allow_nan=False);'
        'assert len(raw.encode())<=128000;print(raw)')
    raw = scheduler.run_on('node001', shlex.join([PYTHON, '-c', code]), timeout=45, check=True)[1]
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
