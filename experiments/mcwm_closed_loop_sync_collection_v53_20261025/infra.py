"""Only v53 code uploads, high native tasks and bounded result reads."""
from pathlib import Path
import base64
import json
import shlex
import subprocess
import sys
import zlib

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from experiments.mcwm_closed_loop_execution_phase_v19_20261022.infra import (
    CPUS,PYTHON,SCHEDULER,environment,remote,scheduler,
)


def write(path,value):
    with Path(path).open('x') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write('\n')


def stage(names):
    paths=[HERE/name for name in names]
    for path in paths: path.resolve().relative_to(HERE)
    payload=base64.b64encode(zlib.compress(json.dumps({p.name:p.read_text() for p in paths}).encode())).decode()
    code=('from pathlib import Path;import base64,json,zlib;root=Path('+repr(remote(HERE))+');'
        'root.mkdir(parents=True,exist_ok=True);files=json.loads(zlib.decompress(base64.b64decode('+repr(payload)+')));'
        '[(root/name).write_text(value) for name,value in files.items()]')
    scheduler.run_on('node005',shlex.join([PYTHON,'-c',code]),timeout=40,check=True)
    return dict(files=len(paths),bytes=sum(path.stat().st_size for path in paths))


def task(case):
    p=json.loads((HERE/'protocol.json').read_text())
    return dict(project='Thesis',description='v53 '+case+' same-prefix warmup and continuous acquisition',
        signature=p['name']+'/native/'+case,cwd=str(HERE),
        cmd=shlex.join([PYTHON,'-u',remote(HERE/'run.py'),'--scenario',case,'--seed','6202',
            '--out',remote(HERE/'runs'/case)]),priority='high',allowed_nodes=CPUS,
        cpu=1,ram_mb=16000,vram=0,env=environment(),allow_cpu_training=True,
        cpu_training_justification='One factual SUMO prefix,14 new warmup labels and24 authorized original-budget CPU updates; acquire271 subsequent feedback with zero response training. No fork branches;432 of456 updates remain reserved.',
        resource_family=p['name']+'/native',ram_resource_family=p['name']+'/native',
        allow_no_ckpt=True,allow_no_resume=True,skip_launch_staging=True,no_auto_requeue=True)


def submit():
    path=HERE/'native_jobs.json'
    p=json.loads((HERE/'protocol.json').read_text())
    write(path,[task(case) for case in p['scenarios']])
    result=subprocess.run(['python3',SCHEDULER,'submit-jsonl','--file',str(path),'--trusted','--json',
        '--lock-timeout','30'],capture_output=True,text=True,timeout=180)
    record=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    write(HERE/('native_submission.json' if result.returncode==0 else 'submission_failure.json'),record)
    return record


def collect(relative,max_bytes=24000):
    path=HERE/relative
    code='from pathlib import Path;p=Path('+repr(remote(path))+');assert p.stat().st_size<='+str(max_bytes)+';print(p.read_text(),end="")'
    raw=scheduler.run_on('node005',shlex.join([PYTHON,'-c',code]),timeout=40,check=True)[1]
    value=json.loads(raw)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists(): path.write_text(raw)
    return value
