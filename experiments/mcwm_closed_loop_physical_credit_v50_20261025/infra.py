"""Only v50 uploads and bounded summaries; no native or training task exists."""
from pathlib import Path
import json
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from experiments.mcwm_closed_loop_execution_phase_v19_20261022.infra import (
    PYTHON,SCHEDULER,environment,remote,scheduler,
)


def stage(names):
    paths=[HERE/name for name in names]
    for p in paths: p.resolve().relative_to(HERE)
    scheduler.run_on('node001','mkdir -p '+shlex.quote(remote(HERE)),timeout=40,check=True)
    subprocess.run(['rsync','-a','--timeout=30','-e',scheduler._ssh_rsync_shell_for_node('node001'),
        *map(str,paths),scheduler._ssh_target_for_node('node001')+':'+remote(HERE)+'/'],check=True,timeout=60)
    return dict(files=len(paths),bytes=sum(p.stat().st_size for p in paths))


def job():
    return dict(project='Thesis',description='v50 physical-time mixed residual target precheck',
        signature='thesis-mcwm-physical-credit-v50-20261025/readonly',cwd=str(HERE),
        cmd=shlex.join([PYTHON,'-u',remote(HERE/'run.py')]),priority='high',
        allowed_nodes=['node004','node005','node006'],cpu=1,ram_mb=4000,vram=0,
        env=environment(),allow_no_ckpt=True,allow_no_resume=True,
        skip_launch_staging=True,no_auto_requeue=True)


def collect():
    p=HERE/'summary.json'
    limit=json.loads((HERE/'protocol.json').read_text())['summary_max_bytes']
    code='from pathlib import Path;p=Path('+repr(remote(p))+');assert p.stat().st_size<='+str(limit)+';print(p.read_text(),end="")'
    raw=scheduler.run_on('node001',shlex.join([PYTHON,'-c',code]),timeout=40,check=True)[1]
    value=json.loads(raw)
    with p.open('x') as handle: handle.write(raw)
    return dict(status=value['status'],integrity=value['integrity'],bytes=len(raw.encode()))
