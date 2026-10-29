"""Retry immutable v24r1 with a longer wait, then validate and adopt if exact."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_fast_branch_v24r1_20261023 import infra
CODE = HERE.parent/'mcwm_closed_loop_fast_branch_v24r1_20261023'
SOURCE = HERE.parent/'mcwm_closed_loop_stratified_pairs_v23_20261023/stage_a/runs/stable/seed6201'


def log(stage, **values):
    print(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(), stage=stage, **values)), flush=True)


def submit(batch, runner, args, cpu, ram_mb):
    receipt = HERE/(batch+'_submission.json')
    if receipt.exists(): return json.loads(receipt.read_text())
    protocol = json.loads((CODE/'protocol.json').read_text())
    row = infra.task(protocol, 'retry/'+batch, args, runner=runner, ram_mb=ram_mb)
    row['cpu'] = cpu
    row['allowed_nodes'] = [node for node in row['allowed_nodes'] if node != 'node004']
    jobs = HERE/(batch+'_jobs.json')
    infra.write(jobs, [row])
    result = json.loads(subprocess.check_output([sys.executable, str(infra.SCHEDULER),
        'submit-jsonl', '--file', str(jobs), '--trusted', '--json'], text=True))
    infra.write(receipt, result)
    log('submitted', batch=batch, ids=[r['id'] for r in result['submitted']], priority='high')
    return result


def wait_result(name):
    log('waiting', file=name)
    local = HERE/name
    while True:
        if local.exists(): return json.loads(local.read_text())
        code = 'from pathlib import Path;p=Path('+repr(infra.remote(local))+');assert not p.is_file() or p.stat().st_size<=20000;print(p.read_text() if p.is_file() else "null")'
        value = json.loads(infra.scheduler.run_on('node001', shlex.join([infra.PYTHON, '-c', code]), timeout=45, check=True)[1])
        if value is not None:
            infra.write(local, value)
            return value
        time.sleep(30)


def adopt():
    doc = HERE.parent/'MCWM_CLOSED_LOOP.md'
    marker = HERE.name
    if marker not in doc.read_text():
        with doc.open('a') as output:
            output.write('\n后续新增主日训练配对默认采用 [v24r1加速执行器]('+marker+'/README.md)：'
                '复制原v24r1的执行器及支持代码，保留新版本自己的protocol/HERE。'
                '同种子完整流、全部预测真值、9对285和后缀零训练已通过，证据见该目录day_equivalence_gate.json。'
                '保持high；既有任务与结果不迁移；留出后缀仍按原重放流程。\n')
    value = dict(status='adopted_for_future_new_main_days', runner=str(CODE/'fast_day.py'),
        support=['fork_state.py','route_streams.py'], evidence=str(HERE/'day_equivalence_gate.json'),
        priority='high', existing_versions_changed=False, heldout_backend_changed=False)
    infra.write(HERE/'adoption.json', value)
    return value


def main():
    if (HERE/'continuation_result.json').exists(): return
    cfg = Path(infra.environment()['THESIS_SUMO_ROOT'])/'online_control/control_sim_traci_period.sumocfg'
    submit('native_stream_probe', 'native_fork_probe.py', ['--sumo-cfg', str(cfg),
        '--prefix-ticks', 1000, '--suffix-ticks', 3000, '--timeout-seconds', 1200,
        '--out', infra.remote(HERE/'native_stream_probe.json')], 2, 8000)
    native = wait_result('native_stream_probe.json')
    if native.get('all_pass') is not True:
        result = dict(status='stopped_native_stream_gate', stage_b='closed')
    else:
        submit('day_benchmark', 'fast_day.py', ['--mode', 'day', '--seed', 6201,
            '--scenario', 'stable', '--preflight', '--out', infra.remote(HERE/'benchmark_day')], 2, 24000)
        submit('day_equivalence_finalize', 'day_compare.py', ['--fast', infra.remote(HERE/'benchmark_day'),
            '--source', infra.remote(SOURCE), '--out', infra.remote(HERE/'day_equivalence_gate.json'), '--wait'], 1, 2048)
        gate = wait_result('day_equivalence_gate.json')
        passed = gate.get('integrity', {}).get('all_pass') is True
        result = dict(status='complete' if passed else 'stopped_day_equivalence',
            integrity=gate.get('integrity'), decision=gate.get('decision'), stage_b='closed')
        if passed: result['backend'] = adopt()
    infra.write(HERE/'continuation_result.json', result)
    log('finished', **result)


if __name__ == '__main__':
    try: main()
    except Exception as error:
        value = dict(status='failed', type=type(error).__name__, message=str(error))
        if not (HERE/'continuation_failure.json').exists(): infra.write(HERE/'continuation_failure.json', value)
        log('failed', **value)
        raise
