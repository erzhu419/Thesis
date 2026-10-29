"""Proceed from runtime checks to two native days, then bounded summary only."""
import json
from pathlib import Path
import shlex
import time
import infra
import launch

HERE = Path(__file__).resolve().parent


def wait_for(paths):
    paths = [[str(infra.remote(HERE/name)) for name in group] for group in paths]
    code = ('from pathlib import Path;import json;p='+repr(paths)+';'
        'print(json.dumps([[name for name in group if Path(name).exists()] for group in p]))')
    while True:
        found = json.loads(infra.scheduler.run_on('node001', shlex.join([infra.PYTHON, '-c', code]), timeout=45, check=True)[1])
        if all(found):
            return found
        time.sleep(30)


def main():
    runtime = HERE/'resident_preflight_submission.json'
    if not runtime.exists():
        launch.stage()
        launch.runtime()
    wait_for([['resident_gate.json']])
    gate = infra.collect('resident_gate.json', max_bytes=4000)
    if not gate['all_pass']:
        infra.write(HERE/'continuation_result.json', dict(status='runtime_failed', gate=gate,
            native_launched=False))
        return
    receipt = HERE/'native_submission.json'
    value = json.loads(receipt.read_text()) if receipt.exists() else launch.days()
    print(json.dumps(dict(stage='native_submitted', priority='high',
        ids=[row['id'] for row in value['submitted']], two_scenarios=True)), flush=True)
    wait_for([[f'runs/{case}/pilot_summary.json', f'runs/{case}/pilot_failure.json']
        for case in launch.protocol()['scenarios']])
    receipt = HERE/'finalize_submission.json'
    final = json.loads(receipt.read_text()) if receipt.exists() else launch.finalize()
    print(json.dumps(dict(stage='finalize_submitted', ids=[row['id'] for row in final['submitted']])), flush=True)
    wait_for([['analysis/summary.json']])
    summary = infra.collect('analysis/summary.json', max_bytes=24000)
    result = dict(status=summary['status'], integrity=summary['integrity'], decision=summary['decision'])
    infra.write(HERE/'continuation_result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        infra.write(HERE/'continuation_failure.json', dict(type=type(error).__name__, message=str(error)))
        raise
