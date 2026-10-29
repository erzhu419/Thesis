"""Continue the three unsubmitted v27 days; preserve the completed first day."""
from pathlib import Path
import json
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent/'mcwm_closed_loop_update_attribution_v27_20261023'
sys.path.insert(0, str(ROOT))
import infra
import launch


def resident(code):
    while True:
        rc, out, error = infra.scheduler.run_on('node001',
            shlex.join([infra.PYTHON, '-c', code]), timeout=45, check=False)
        if rc == 0:
            return json.loads(out)
        if any(text in error for text in ('kex_exchange_identification',
                'Connection closed', 'Connection timed out', 'Connection reset')):
            print(json.dumps(dict(stage='transport_retry', error=error[-600:])), flush=True)
            time.sleep(30)
            continue
        raise RuntimeError(error[-1800:])


def submit(rows, name):
    receipt = HERE/(name+'_submission.json')
    if receipt.exists():
        return json.loads(receipt.read_text())
    jobs = HERE/(name+'_jobs.json')
    infra.write(jobs, rows)
    result = json.loads(subprocess.check_output([sys.executable, str(infra.SCHEDULER),
        'submit-jsonl', '--file', str(jobs), '--trusted', '--json'], text=True))
    infra.write(receipt, result)
    print(json.dumps(dict(stage='submitted', batch=name,
        ids=[row['id'] for row in result['submitted']], priority='high')), flush=True)
    return result


def main():
    p = launch.protocol()
    code = ('from pathlib import Path;import json;root=Path('+repr(infra.remote(ROOT))+');'
        'gate=json.loads((root/"preflight_gate.json").read_text());'
        'day=json.loads((root/"runs/seed6202/local_speed_drop/summary.json").read_text());'
        'print(json.dumps(dict(gate=gate,status=day["status"],integrity=day["integrity"],budget=day["budget"])))')
    first = resident(code)
    if not (first['gate']['integrity']['all_pass'] and first['status']=='complete'
            and first['integrity']['all_pass']):
        raise RuntimeError('Completed first replay did not pass')
    if not (HERE/'reused_first_day.json').exists():
        infra.write(HERE/'reused_first_day.json', first)
    rows = []
    for seed, scenario in ((6201,'stable'), (6201,'local_speed_drop'), (6202,'stable')):
        rows.append(infra.task(p, f'replay/{seed}/{scenario}', ['--seed', seed,
            '--scenario', scenario, '--out', infra.remote(ROOT/'runs'/f'seed{seed}'/scenario)],
            runner='run.py', ram_mb=12000))
    submit(rows, 'remaining')
    code = ('from pathlib import Path;import json;root=Path('+repr(infra.remote(ROOT))+');'
        'print(json.dumps({str(seed)+"/"+scenario:json.loads((root/"runs"/("seed"+str(seed))/scenario/"summary.json").read_text())["status"] '
        'if (root/"runs"/("seed"+str(seed))/scenario/"summary.json").exists() else "failed" '
        'if (root/"runs"/("seed"+str(seed))/scenario/"failure.json").exists() else None '
        'for seed in (6201,6202) for scenario in ("stable","local_speed_drop")}))')
    previous = None
    while True:
        states = resident(code)
        if states != previous:
            print(json.dumps(dict(stage='replays', states=states)), flush=True)
            previous = states
        if all(value is not None for value in states.values()):
            break
        time.sleep(45)
    infra.write(HERE/'replay_completion.json', states)
    submit([infra.task(p, 'shadow/finalize', ['--root', infra.remote(ROOT),
        '--out', infra.remote(ROOT/'shadow_summary.json')], runner='analyze.py', ram_mb=2048)], 'finalize')
    code = ('from pathlib import Path;p=Path('+repr(infra.remote(ROOT/'shadow_summary.json'))+');'
        'assert not p.exists() or p.stat().st_size<=60000;print(p.read_text() if p.exists() else "null")')
    while True:
        result = resident(code)
        if result is not None:
            infra.write(HERE/'resident_summary.json', result)
            print(json.dumps(dict(stage='finished', status=result['status'],
                integrity=result['integrity']['all_pass'])), flush=True)
            break
        time.sleep(30)
    if not result['integrity']['all_pass']:
        return
    task = infra.task(p, 'shadow/endpoint-detail', [], ram_mb=2048)
    task['cmd'] = shlex.join([infra.PYTHON, '-u', infra.remote(HERE/'endpoint_detail.py'),
        '--root', infra.remote(ROOT), '--out', infra.remote(HERE/'endpoint_detail.json')])
    submit([task], 'endpoint_detail')
    code = ('from pathlib import Path;p=Path('+repr(infra.remote(HERE/'endpoint_detail.json'))+');'
        'assert not p.exists() or p.stat().st_size<=50000;print(p.read_text() if p.exists() else "null")')
    while True:
        detail = resident(code)
        if detail is not None:
            infra.write(HERE/'endpoint_detail.json', detail)
            print(json.dumps(dict(stage='endpoint_detail_finished',
                anchors=detail['counts']['anchors'])), flush=True)
            break
        time.sleep(30)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        infra.write(HERE/'recovery_failure.json', dict(type=type(error).__name__, message=str(error)))
        raise
