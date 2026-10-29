"""High-priority frozen capacity decomposition; bounded resident JSON; small JSON only."""
from pathlib import Path
import json
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'mcwm_closed_loop_capacity_control_v81_20261027'
sys.path.insert(0,str(HERE.parents[1]))
from experiments.mcwm_closed_loop_execution_phase_v19_20261022.infra import (
    CPUS,PYTHON,SCHEDULER,environment,remote,scheduler)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def stage():
    paths=[*sorted(HERE.glob('*.py')),HERE/'protocol.json']
    scheduler.run_on('node001','mkdir -p '+shlex.quote(remote(HERE)),timeout=45,check=True)
    subprocess.run(['rsync','-a','--timeout=30','-e',scheduler._ssh_rsync_shell_for_node('node001'),
        *map(str,paths),scheduler._ssh_target_for_node('node001')+':'+remote(HERE)+'/'],timeout=90,check=True)
    return dict(files=len(paths),bytes=sum(p.stat().st_size for p in paths),
        reference_implementation_unchanged=remote(SOURCE/'control.py'),
        frozen_runner=remote(HERE/'run.py'))


def task(seed):
    p=json.loads((HERE/'protocol.json').read_text())
    return dict(project='Thesis',description='v82 frozen capacity contribution readout seed'+str(seed),
        signature=p['name']+'/seed'+str(seed),cwd=str(HERE),
        cmd=shlex.join([PYTHON,'-u',remote(HERE/'run.py'),'--seed',str(seed),
            '--out',remote(HERE/'runs'/('seed'+str(seed)))]),
        priority='high',allowed_nodes=CPUS,cpu=1,ram_mb=8000,vram=0,env=environment(),
        allow_no_ckpt=True,allow_no_resume=True,skip_launch_staging=True,no_auto_requeue=True)


def submit(rows,batch):
    receipt=HERE/(batch+'_submission.json')
    if receipt.exists():return json.loads(receipt.read_text())
    jobs=HERE/(batch+'_jobs.json');attempt=HERE/(batch+'_submission_attempt.json')
    if attempt.exists():raise RuntimeError('Unresolved submission; inspect existing task signatures: '+batch)
    if jobs.exists():
        if json.loads(jobs.read_text())!=rows:raise RuntimeError('Fixed batch changed: '+batch)
    else:write(jobs,rows)
    write(attempt,dict(batch=batch,signatures=[r['signature'] for r in rows]))
    result=subprocess.run([sys.executable,str(SCHEDULER),'submit-jsonl','--file',str(jobs),
        '--trusted','--json','--lock-timeout','300'],capture_output=True,text=True,timeout=360)
    if result.returncode:
        write(HERE/(batch+'_submission_failure.json'),dict(returncode=result.returncode,
            stdout=result.stdout,stderr=result.stderr))
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    value=json.loads(result.stdout);write(receipt,value)
    dispatch=subprocess.run([sys.executable,str(SCHEDULER),'dispatch','--no-bulk-window','--lock-timeout','300',
        *[arg for row in value['submitted'] for arg in ('--task-id',row['id'])]],
        capture_output=True,text=True,timeout=360)
    write(HERE/(batch+'_dispatch.json'),dict(returncode=dispatch.returncode,
        stdout=dispatch.stdout[-16000:],stderr=dispatch.stderr[-2000:]))
    if dispatch.returncode:raise RuntimeError('Known submitted tasks need dispatch review; do not resubmit')
    return value


def presence(relative_paths):
    paths=[(HERE/p).resolve() for p in relative_paths]
    for p in paths:p.relative_to(HERE)
    code='from pathlib import Path;import json;print(json.dumps({p:Path(p).exists() for p in '+repr([remote(p) for p in paths])+'}))'
    value=json.loads(scheduler.run_on('node001',shlex.join([PYTHON,'-c',code]),timeout=45,check=True)[1])
    return {name:value[remote(path)] for name,path in zip(relative_paths,paths)}


def collect(relative):
    path=(HERE/relative).resolve();path.relative_to(HERE)
    if path.suffix!='.json':raise ValueError('Small JSON results only')
    if path.exists():return json.loads(path.read_text())
    code='from pathlib import Path;import json;p=Path('+repr(remote(path))+');raw=json.dumps(json.loads(p.read_text()),ensure_ascii=False,separators=(",",":"),allow_nan=False);assert len(raw.encode())<=48000;print(raw,end="")'
    raw=scheduler.run_on('node001',shlex.join([PYTHON,'-c',code]),timeout=45,check=True)[1]
    if len(raw.encode())>48000:raise ValueError('Received result exceeds48KB')
    value=json.loads(raw);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:f.write(raw)
    return value


def status(ids):
    result=subprocess.run([sys.executable,str(SCHEDULER),'status','--readonly','--json',
        '--lock-timeout','30','--ids',','.join(ids)],text=True,capture_output=True,timeout=50)
    if result.returncode:raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return [{k:r.get(k) for k in ('id','status','priority','node','signature')}
        for r in json.loads(result.stdout)['tasks']]
