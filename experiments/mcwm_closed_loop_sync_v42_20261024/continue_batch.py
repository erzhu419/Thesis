"""Collect v42 only; never submit training or native collection."""
import json
from pathlib import Path
import shlex
import time
import infra
import launch

HERE=Path(__file__).resolve().parent


def main():
    receipt=HERE/'precheck_submission.json'
    submitted=json.loads(receipt.read_text()) if receipt.exists() else launch.submit()
    print(json.dumps(dict(stage='submitted',priority='high',ids=[r['id'] for r in submitted['submitted']])),flush=True)
    code=('from pathlib import Path;import json;p=Path('+repr(infra.remote(HERE))+');'
        'print(json.dumps({"ready":(p/"analysis/summary.json").exists(),'
        '"failure":json.loads((p/"failure.json").read_text()) if (p/"failure.json").exists() else None}))')
    while True:
        _,out,_=infra.scheduler.run_on('node001',shlex.join([infra.PYTHON,'-c',code]),timeout=45,check=True)
        state=json.loads(out)
        if state['failure'] is not None:
            infra.write(HERE/'resident_failure.json',state['failure']);return
        if state['ready']:
            summary=infra.collect('analysis/summary.json',max_bytes=24000)
            result=dict(status=summary['status'],integrity=summary['integrity'],decision=summary['decision'])
            infra.write(HERE/'continuation_result.json',result);print(json.dumps(result),flush=True);return
        time.sleep(30)


if __name__=='__main__':
    try:main()
    except Exception as error:
        infra.write(HERE/'continuation_failure.json',dict(type=type(error).__name__,message=str(error)));raise
