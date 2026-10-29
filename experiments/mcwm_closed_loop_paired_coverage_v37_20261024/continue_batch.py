"""Advance each completed new prefix immediately to its matched shadow triplet."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import time
import infra
import launch

HERE = Path(__file__).resolve().parent


def log(stage, **values):
    print(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(), stage=stage, **values)), flush=True)


def resident(code):
    while True:
        try:
            rc, out, error = infra.scheduler.run_on('node001',
                shlex.join([infra.PYTHON, '-c', code]), timeout=45, check=False)
        except subprocess.TimeoutExpired:
            log('transport_retry', reason='SSH read timeout')
            time.sleep(30)
            continue
        if rc == 0:
            return json.loads(out)
        if any(value in error for value in ('kex_exchange_identification', 'Connection closed',
                'Connection timed out', 'Connection reset')):
            log('transport_retry', reason=error[-500:])
            time.sleep(30)
            continue
        raise RuntimeError(error[-1800:])


def submitted(batch, callback):
    path = HERE / (batch + '_submission.json')
    value = json.loads(path.read_text()) if path.exists() else callback()
    log('submitted', batch=batch, ids=[row['id'] for row in value['submitted']], priority='high')


def snapshot():
    root = infra.remote(HERE / 'runs')
    return resident('''from pathlib import Path
import json
root = Path(''' + repr(root) + ''')
scenarios = ''' + repr(launch.protocol()['scenarios']) + '''
result = {}
for scenario in scenarios:
    result[scenario] = {}
    for stage in ('collection', 'shadow'):
        p = root/stage/scenario
        if (p/'failure.json').is_file():
            result[scenario][stage] = {'status':'failed','failure':json.loads((p/'failure.json').read_text())}
        elif (p/'summary.json').is_file():
            s = json.loads((p/'summary.json').read_text())
            result[scenario][stage] = {'status':s['status'], 'integrity':s.get('integrity',s.get('proof',{}))}
        else:
            result[scenario][stage] = None
print(json.dumps(result))
''')


def main():
    submitted('collection', launch.submit_collect)
    previous = None
    while True:
        states = snapshot()
        if states != previous:
            log('pipeline', states=states)
            previous = states
        failed = []
        for scenario, state in states.items():
            collection = state['collection']
            if collection and collection['status'] != 'complete':
                failed.append((scenario, 'collection'))
            if state['shadow'] and state['shadow']['status'] != 'complete':
                failed.append((scenario, 'shadow'))
            if collection and collection['status'] == 'complete' and not (HERE / ('shadow_' + scenario + '_submission.json')).exists():
                submitted('shadow_' + scenario, lambda scenario=scenario: launch.submit_shadow(scenario))
        if failed:
            result = dict(status='stopped_integrity', failed=failed, states=states)
            break
        if all(s['shadow'] and s['shadow']['status'] == 'complete' for s in states.values()):
            infra.write(HERE / 'diagnostic_completion.json', states)
            submitted('finalize', launch.submit_finalize)
            limit = launch.protocol().get('summary_max_bytes', 24000)
            while True:
                value = resident('from pathlib import Path;import json;root=Path(' + repr(infra.remote(HERE)) + ');'
                    'p=root/"paired_coverage_summary.json";f=root/"paired_coverage_summary_failure.json";'
                    'assert not p.exists() or p.stat().st_size<=' + str(limit) + ';'
                    'print(p.read_text() if p.exists() else json.dumps({"status":"analysis_failed",'
                    '"failure":json.loads(f.read_text())}) if f.exists() else "null")')
                if value is not None:
                    name = 'paired_coverage_summary_failure.json' if value['status'] == 'analysis_failed' else 'paired_coverage_summary.json'
                    infra.write(HERE / name, value)
                    result = dict(status='diagnosis_complete' if value.get('integrity', {}).get('all_pass') else 'stopped_analysis_integrity',
                        summary_status=value['status'])
                    break
                time.sleep(30)
            break
        time.sleep(45)
    result['priority'] = 'high'
    infra.write(HERE / 'continuation_result.json', result)
    log('finished', **result)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        value = dict(status='failed', type=type(error).__name__, message=str(error))
        infra.write(HERE / 'continuation_failure.json', value)
        log('failed', **value)
        raise
