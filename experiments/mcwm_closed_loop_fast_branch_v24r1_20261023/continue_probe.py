"""Advance the bounded execution probes only after each exact-equivalence gate."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import time
import infra
import launch

HERE=Path(__file__).resolve().parent


def log(stage,**values):
    print(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),stage=stage,**values)),flush=True)


def wait_file(name):
    log('waiting',file=name)
    target=HERE/name
    while True:
        if target.is_file():return json.loads(target.read_text())
        code='from pathlib import Path;import json;p=Path('+repr(infra.remote(target))+');assert not p.is_file() or p.stat().st_size<=20000;print(p.read_text() if p.is_file() else "null")'
        text=infra.scheduler.run_on('node001',shlex.join([infra.PYTHON,'-c',code]),timeout=45,check=True)[1]
        value=json.loads(text)
        if value is not None:
            infra.write(target,value)
            return value
        time.sleep(30)


def submitted(batch,callback):
    receipt=HERE/(batch+'_submission.json')
    value=json.loads(receipt.read_text()) if receipt.is_file() else callback()
    log('submitted',batch=batch,ids=[r['id'] for r in value['submitted']],priority='high')


def adopt_backend():
    # User authorized changing future experiments only after full exact parity.
    doc = HERE.parent/'MCWM_CLOSED_LOOP.md'
    marker = 'mcwm_closed_loop_fast_branch_v24r1_20261023'
    text = doc.read_text()
    if marker not in text:
        with doc.open('a') as output:
            output.write('\n后续新增实验的主日训练配对默认采用 ['+marker+']('+marker+
                '/README.md) 的 `fast_day.py`、`fork_state.py` 和 `route_streams.py`；'
                '整日同种子完整轨迹、全部预测真值及9对285预算精确同值已通过。'
                '新版本须一并复制执行器支持代码并保留自己的 protocol/HERE，保持 high；'
                '既有任务和结果不迁移。当前加速范围为主日训练配对，留出后缀流程尚未改为免重放。\n')
    result = dict(status='adopted_for_future_new_main_days',
        runner=str(HERE/'fast_day.py'),support=['fork_state.py','route_streams.py'],
        evidence=str(HERE/'day_equivalence_gate.json'),priority='high',
        existing_experiments_modified=False,heldout_backend_changed=False)
    if not (HERE/'adoption.json').exists():infra.write(HERE/'adoption.json',result)
    return result


def main():
    if (HERE/'continuation_result.json').is_file():return
    native=wait_file(launch.NATIVE_RESULT)
    if native.get('all_pass') is not True:
        result=dict(status='stopped_native_stream_gate',stage_b='closed')
    else:
        submitted('day_benchmark',launch.submit_day)
        submitted('day_equivalence_finalize',launch.submit_day_compare)
        gate=wait_file('day_equivalence_gate.json')
        passed=gate.get('integrity',{}).get('all_pass') is True
        result=dict(status='complete' if passed else 'stopped_day_equivalence',
            integrity=gate.get('integrity'),decision=gate.get('decision'),stage_b='closed')
        if passed:
            result['backend']=adopt_backend()
    infra.write(HERE/'continuation_result.json',result)
    log('finished',**result)


if __name__=='__main__':
    try:main()
    except Exception as error:
        value=dict(status='failed',type=type(error).__name__,message=str(error))
        if not (HERE/'continuation_failure.json').exists():infra.write(HERE/'continuation_failure.json',value)
        log('failed',**value)
        raise
