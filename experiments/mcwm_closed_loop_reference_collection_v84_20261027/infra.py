"""Two high native collectors and bounded server-side analysis; no response training."""
from pathlib import Path
import json
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from experiments.mcwm_closed_loop_execution_phase_v19_20261022.infra import (
    CPUS, PYTHON, SCHEDULER, environment, remote, scheduler)


def write(path,value):
    with Path(path).open('x') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write('\n')


def stage():
    paths=[*sorted(HERE.glob('*.py')),HERE/'protocol.json']
    scheduler.run_on('node001','mkdir -p '+shlex.quote(remote(HERE)),timeout=45,check=True)
    subprocess.run(['rsync','-a','--timeout=30','-e',scheduler._ssh_rsync_shell_for_node('node001'),
        *map(str,paths),scheduler._ssh_target_for_node('node001')+':'+remote(HERE)+'/'],timeout=90,check=True)
    result=dict(files=len(paths),uploaded_bytes=sum(p.stat().st_size for p in paths),
                destination=remote(HERE),old_paths_written=[])
    write(HERE/'staging.json',result)
    return result


def resident_gate():
    source=HERE.parent/'mcwm_closed_loop_frozen_validation_v75_20261027'
    paths=[source/'inputs/h2oplus_seed6207.pt']
    for scene in ('stable','local_speed_drop'):
        paths += [source/f'runs/executed_cql/validation/seed6207/{scene}/fresh'/name
                  for name in ('summary.json','prefix_trace.jsonl','updates.jsonl')]
    code=('import sys,json;from pathlib import Path;sys.path.insert(0,'+repr(remote(HERE.parents[1]))+');'
        'import experiments.mcwm_closed_loop_reference_collection_v84_20261027.run as run;'
        'checks={p:Path(p).is_file() for p in '+repr([remote(p) for p in paths])+'};'
        'print(json.dumps(dict(all_pass=all(checks.values()),runtime_import=True,files=checks)))')
    command=shlex.join(['env',*[key+'='+value for key,value in environment().items()],PYTHON,'-c',code])
    value=json.loads(scheduler.run_on('node001',command,timeout=60,check=True)[1])
    write(HERE/'resident_gate.json',value)
    if not value['all_pass']: raise RuntimeError('Missing resident source; no collector submitted')
    return value


def task(scene):
    p=json.loads((HERE/'protocol.json').read_text())
    return dict(project='Thesis',description=p['name']+' '+scene+' three frozen native pairs',
        signature=p['name']+'/'+scene,cwd=str(HERE),
        cmd=shlex.join([PYTHON,'-u',remote(HERE/'run.py'),'--scenario',scene,
                       '--out',remote(HERE/'runs'/scene)]),
        priority='high',allowed_nodes=CPUS,cpu=2,ram_mb=24000,vram=0,env=environment(),
        resource_family=p['name']+'/native',ram_resource_family=p['name']+'/native',
        allow_cpu_training=True,cpu_training_justification='Restore matching v75 fresh packet37 only:185covariates185outcomes296steps per optimizer; '
            'existing50000-step initializer,zero new initializer. Freeze at12000,first eligible each of3lines12000-12600; '
            'each root forks2actually concurrent reference/candidate suffixes to24000,then parent resumes unchanged. '
            'Per scene max84600native seconds;global169200. No suffix training/admission/truth feedback, no response training.',
        allow_no_ckpt=True,allow_no_resume=True,skip_launch_staging=True,no_auto_requeue=True)


def submit():
    receipt=HERE/'collection_submission.json'
    if receipt.exists(): return json.loads(receipt.read_text())
    if (HERE/'collection_submission_attempt.json').exists():
        raise RuntimeError('Inspect known submission attempt before another one')
    rows=[task(scene) for scene in ('stable','local_speed_drop')]
    jobs=HERE/'collection_jobs.json';write(jobs,rows)
    write(HERE/'collection_submission_attempt.json',dict(signatures=[r['signature'] for r in rows]))
    result=subprocess.run([sys.executable,str(SCHEDULER),'submit-jsonl','--file',str(jobs),
        '--trusted','--json','--lock-timeout','300'],capture_output=True,text=True,timeout=360)
    if result.returncode:
        write(HERE/'submission_failure.json',dict(returncode=result.returncode,stdout=result.stdout[-5000:],stderr=result.stderr[-3000:]))
        raise RuntimeError('Submission rejected; no duplicate retry')
    value=json.loads(result.stdout);write(receipt,value)
    dispatch=subprocess.run([sys.executable,str(SCHEDULER),'dispatch','--no-bulk-window','--lock-timeout','300',
        *[arg for row in value['submitted'] for arg in ('--task-id',row['id'])]],capture_output=True,text=True,timeout=360)
    write(HERE/'collection_dispatch.json',dict(returncode=dispatch.returncode,stdout=dispatch.stdout[-7000:],stderr=dispatch.stderr[-2000:]))
    if dispatch.returncode: raise RuntimeError('Known tasks submitted; inspect their IDs, do not resubmit')
    return value


def analyze_resident():
    command=shlex.join([PYTHON,remote(HERE/'analysis.py'),
        '--stable',remote(HERE/'runs/stable/collector_summary.json'),
        '--shock',remote(HERE/'runs/local_speed_drop/collector_summary.json'),
        '--out',remote(HERE/'summary.json')])
    return scheduler.run_on('node001',command,timeout=45,check=True)[1]


def collect():
    path=HERE/'summary.json'
    if path.exists(): return json.loads(path.read_text())
    code=('from pathlib import Path;import json;p=Path('+repr(remote(path))+');'
        'raw=json.dumps(json.loads(p.read_text()),separators=(",",":"),allow_nan=False);'
        'assert len(raw.encode())<=48000;print(raw)')
    raw=scheduler.run_on('node001',shlex.join([PYTHON,'-c',code]),timeout=45,check=True)[1]
    value=json.loads(raw);write(path,value)
    return value


def status():
    receipt=json.loads((HERE/'collection_submission.json').read_text())
    ids=','.join(r['id'] for r in receipt['submitted'])
    result=subprocess.run([sys.executable,str(SCHEDULER),'status','--readonly','--json','--lock-timeout','30','--ids',ids],capture_output=True,text=True,timeout=50)
    if result.returncode: raise RuntimeError(result.stderr[-1500:])
    return [{k:r.get(k) for k in ('id','status','priority','node','signature')}
            for r in json.loads(result.stdout)['tasks']]
