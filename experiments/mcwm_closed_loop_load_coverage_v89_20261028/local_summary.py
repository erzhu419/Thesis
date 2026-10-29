"""Local readout of the small completed native observation, without raw assets."""
import json
from pathlib import Path
import shlex
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from experiments.mcwm_closed_loop_load_coverage_v89_20261028 import infra
from experiments.mcwm_closed_loop_load_coverage_v89_20261028.analysis import analyze


def run():
    source=infra.remote(HERE/'runs/observation/collector_summary.json')
    code='''import json
from pathlib import Path
p=Path(SOURCE)
if p.exists():
    raw=p.read_bytes()
    if len(raw)>48000: raise ValueError('Native observation JSON exceeds48KB; do not pull raw assets')
    result=dict(ready=True,collector=json.loads(raw))
else:
    failure=p.parent/'failure.json'
    result=dict(ready=False,failure=json.loads(failure.read_text()) if failure.exists() else None)
print(json.dumps(result,separators=(',',':'),allow_nan=False))
'''.replace('SOURCE',repr(source))
    deadline=time.monotonic()+7200
    while time.monotonic()<deadline:
        raw=infra.scheduler.run_on('node004',shlex.join([infra.PYTHON,'-c',code]),timeout=45,check=True)[1]
        value=json.loads(raw)
        if value['ready']:
            collector=value['collector']
            folder=HERE/'local_inputs';folder.mkdir(exist_ok=True)
            infra.write(folder/'collector_summary.json',collector)
            protocol=json.loads((HERE/'protocol.json').read_text())
            audit=json.loads((HERE/'freeze_audit.json').read_text())
            result=analyze(collector,protocol,audit)
            encoded=json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'
            if len(encoded.encode())>16000: raise ValueError('Local summary exceeds16KB')
            with (HERE/'summary.json').open('x') as handle:handle.write(encoded)
            infra.write(HERE/'local_summary_receipt.json',dict(task_id='t212062',execution_node='local',
                source_node='node004',transferred_bytes=len(raw.encode()),
                summary_bytes=len(encoded.encode()),new_native_seconds=0,new_fits=0,new_labels=0))
            print(json.dumps(dict(status=result['status'],coverage=result['coverage_gate']['passed'],
                summary=str(HERE/'summary.json'),transferred_bytes=len(raw.encode()))),flush=True)
            return
        if value['failure'] is not None:
            raise RuntimeError('Native observation failed: '+json.dumps(value['failure']))
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='waiting_for_native_observation',
            current=0,total=1,unit='small_collector_json')),flush=True)
        time.sleep(30)
    raise TimeoutError('Native observation did not finish within the original two-hour wait')


if __name__=='__main__':run()
