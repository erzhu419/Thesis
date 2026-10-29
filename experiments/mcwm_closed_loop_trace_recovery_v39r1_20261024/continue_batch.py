"""Collect the bounded recovery proof and merged summary; no further jobs."""
import json
from pathlib import Path
import shlex
import time
import infra

HERE=Path(__file__).resolve().parent


def main():
    code=('from pathlib import Path;import json;p=Path('+repr(infra.remote(HERE))+');'
        'f=p/"failure.json";print(json.dumps({"ready":(p/"analysis/summary.json").exists() '
        'and (p/"recovery_proof.json").exists(),"failure":json.loads(f.read_text()) if f.exists() else None}))')
    while True:
        rc,out,error=infra.scheduler.run_on('node001',shlex.join([infra.PYTHON,'-c',code]),timeout=45,check=False)
        if rc: raise RuntimeError(error[-1800:])
        state=json.loads(out)
        if state['failure'] is not None:
            infra.write(HERE/'resident_failure.json',state['failure']);return
        if state['ready']:
            summary=infra.collect('analysis/summary.json',max_bytes=24000)
            proof=infra.collect('recovery_proof.json',max_bytes=12000)
            result=dict(status=summary['status'],integrity=summary['integrity'],recovery=proof['integrity'],
                decision=summary['decision'],new_native_seconds=0,new_updates=0)
            infra.write(HERE/'continuation_result.json',result)
            print(json.dumps(result),flush=True);return
        time.sleep(45)


if __name__=='__main__':
    try:main()
    except Exception as error:
        infra.write(HERE/'continuation_failure.json',dict(type=type(error).__name__,message=str(error)));raise
