"""Continue only the predeclared shock pair, then collect the fixed study."""
import json
import time

import infra


def report(value):
    def number(delta):
        return '缺失' if delta is None else f'{delta["cost_per_passenger"]:+.3f}'
    lines=['# v107：完整 MC-WM＋dripping 固定试验','',
        '状态：`'+value['status']+'`；决定：`'+value['decision']['state']+'`。',
        '成本为(2×等待＋乘车)/15079，单位加权秒／乘客；负差值更好。','',
        '| 场景 | fresh−匹配no_fresh | fresh−冻结MC-WM | no_fresh−冻结MC-WM |',
        '| --- | ---: | ---: | ---: |']
    for cell in value['cells']:
        d=cell['comparisons']['full_day']
        lines.append('| '+cell['scenario']+' | '+ ' | '.join(number(d[k]) for k in
            ('fresh_minus_matched','fresh_minus_frozen','no_fresh_minus_frozen'))+' |')
    lines.extend(['','matched全天平均：'+number(value['means']['full_day']['fresh_minus_matched'])+
        '；停更后平均：'+number(value['means']['post_budget']['fresh_minus_matched'])+'。',
        '实际预算和各窗等待／乘车分解见 summary.json；不将缺失格记作零。','',
        '下一步：'+('保留单种子pilot结果，另行预声明独立确认。' if value['decision']['cost_gate_pass'] else
            '固定方案收口，先依据已记录的差异分析；不自动改窗口、份额或扩种子。'),
        '限制：一组训练种子、一个环境种子；本轮不是独立确认或逐格零退化保证。',''])
    with (infra.HERE/'RESULTS_AND_NEXT.md').open('x') as handle:
        handle.write('\n'.join(lines))


def main():
    downloaded = 0
    previous = None
    deadline = time.monotonic()+14400
    while time.monotonic()<deadline:
        value,size = infra.poll()
        downloaded += size
        gate,final = value['gate'],value['final']
        if gate and gate.get('status') != 'pending':
            if not (infra.HERE/'preflight_gate.json').exists():
                infra.write(infra.HERE/'preflight_gate.json',gate)
        held = bool(value['failed']) or bool(gate and gate.get('status') != 'pending' and not gate.get('all_pass'))
        if not held and gate and gate.get('all_pass') and not (infra.HERE/'local_speed_drop_submission.json').exists():
            infra.submit_pair('local_speed_drop')
        if final and final.get('status') != 'pending':
            infra.write(infra.HERE/'summary.json',final)
            report(final)
            infra.write(infra.HERE/'continuation_result.json',dict(status='complete',
                downloaded_bytes=downloaded,checkpoint_downloaded=False))
            print(json.dumps(final),flush=True)
            return
        state = dict(finished_days=value['finished_days'],first_packet_pass=bool(gate and gate.get('all_pass')),
            remaining_launch_held=held,failed=value['failed'])
        if state != previous:
            print(json.dumps(state),flush=True)
            previous=state
        time.sleep(45)
    raise TimeoutError('Four-hour collection deadline; no native tasks retried or cancelled')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        infra.write(infra.HERE/'continuation_error.json',dict(type=type(error).__name__,
            message=str(error),automatic_retries=False))
        raise
