"""Upload only this version and collect only its bounded JSON summary."""
from pathlib import Path
import base64
import json
import shlex
import sys
import zlib

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from experiments.mcwm_closed_loop_execution_phase_v19_20261022.infra import (
    PYTHON,SCHEDULER,environment,remote,scheduler,
)


def stage(names):
    paths=[HERE/name for name in names]
    for path in paths: path.resolve().relative_to(HERE)
    files={path.name:path.read_text() for path in paths}
    payload=base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
    code=('from pathlib import Path;import base64,json,zlib;root=Path('+repr(remote(HERE))+');'
          'root.mkdir(parents=True,exist_ok=True);files=json.loads(zlib.decompress(base64.b64decode('+repr(payload)+')));'
          '[(root/name).write_text(value) for name,value in files.items()]')
    scheduler.run_on('node005',shlex.join([PYTHON,'-c',code]),timeout=40,check=True)
    return dict(files=len(paths),bytes=sum(path.stat().st_size for path in paths))


def job():
    return dict(project='Thesis',description='v51 bounded physical-time residual target precheck',
        signature='thesis-mcwm-bounded-credit-v51-20261025/readonly',cwd=str(HERE),
        cmd=shlex.join([PYTHON,'-u',remote(HERE/'run.py')]),priority='high',
        allowed_nodes=['node004','node005','node006'],cpu=1,ram_mb=4000,vram=0,
        env=environment(),allow_no_ckpt=True,allow_no_resume=True,
        skip_launch_staging=True,no_auto_requeue=True)


def collect():
    path=HERE/'summary.json'
    limit=json.loads((HERE/'protocol.json').read_text())['summary_max_bytes']
    code='from pathlib import Path;p=Path('+repr(remote(path))+');assert p.stat().st_size<='+str(limit)+';print(p.read_text(),end="")'
    raw=scheduler.run_on('node005',shlex.join([PYTHON,'-c',code]),timeout=40,check=True)[1]
    result=json.loads(raw)
    with path.open('x') as handle: handle.write(raw)
    return dict(status=result['status'],integrity=result['integrity'],bytes=len(raw.encode()))
