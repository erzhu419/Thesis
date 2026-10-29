"""Collect the single fixed reconstruction receipt; never submit or retry training."""
import json
from pathlib import Path
import shlex
import time

import infra

HERE = Path(__file__).resolve().parent


def main():
    if (HERE/'recovery_summary.json').exists():
        return
    p = infra.protocol()
    deadline = time.monotonic()+10800
    previous = None
    while time.monotonic() < deadline:
        code = ('from pathlib import Path\nimport json\nroot=Path('
            +repr(infra.remote(HERE/'run/seed6201'))+')\n'
            'final=root/"recovery_summary.json"\n'
            'if final.exists():\n'
            ' try:\n'
            '  value=json.loads(final.read_text())\n'
            '  raw=json.dumps(value,separators=(",",":"))\n'
            '  assert len(raw.encode())<='+str(p['small_summary_limit_bytes'])+'\n'
            '  print(json.dumps(dict(kind="final",value=value),separators=(",",":")))\n'
            ' except json.JSONDecodeError: print(json.dumps(dict(kind="writing_final")))\n'
            'else:\n'
            ' progress=root/"results/progress.json"\n'
            ' if progress.exists():\n'
            '  try:\n'
            '   v=json.loads(progress.read_text())\n'
            '   print(json.dumps(dict(kind="progress",source_events=v["source_events"],'
            'updates=v["continuation_updates"],elapsed_seconds=v["elapsed_seconds"])))\n'
            '  except json.JSONDecodeError: print(json.dumps(dict(kind="writing_progress")))\n'
            ' else: print(json.dumps(dict(kind="waiting_for_first_training_progress")))\n')
        raw = infra.scheduler.run_on('node005', shlex.join([infra.PYTHON,'-B','-c',code]),
                                     timeout=45, check=True)[1]
        item = json.loads(raw)
        if item['kind'] == 'final':
            value = item['value']
            infra.write(HERE/'recovery_summary.json', value)
            infra.write(HERE/'collection_receipt.json', dict(downloaded_bytes=len(raw.encode()),
                checkpoint_downloaded=False, collected_wall=time.time()))
            passed = value['status'] == 'complete'
            lines = ['# v105 完整学习态重建结果', '',
                '固定6201：'+('原actor端点和恢复续步检查均通过。' if passed else
                             '未通过固定恢复门，保留本次结果并停止扩展。'), '',
                '决定：`'+value.get('decision','UNKNOWN')+'`。',
                '预算：'+json.dumps(value['budget'],ensure_ascii=False)+'。', '',
                '完整checkpoint留在服务器：`'+str(value.get('checkpoint'))+'`。',
                '本轮没有target日控制或fresh收益结果，也未启动其他种子。', '',
                '原历史Q/world内存状态已经丢失；本轮核对的是按原流程重建后的actor身份，以及新快照自身的保存/恢复与续步等价。', '',
                '[详细小摘要]('+str(HERE/'recovery_summary.json')+')。', '']
            with (HERE/'RESULTS_AND_NEXT.md').open('x') as handle:
                handle.write('\n'.join(lines))
            print(json.dumps(dict(status=value['status'], decision=value.get('decision'),
                                  downloaded_bytes=len(raw.encode()))),flush=True)
            return
        if item != previous:
            print(json.dumps(item),flush=True)
            previous = item
        time.sleep(45)
    raise TimeoutError('Three-hour receipt deadline; no training was retried or cancelled')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        path = HERE/'collection_error.json'
        if not path.exists():
            infra.write(path, dict(type=type(error).__name__, message=str(error),
                                  training_retried=False, training_cancelled=False))
        raise
