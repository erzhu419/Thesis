"""Complete only the authorized exact-replay batch and collect a bounded verdict."""
import json
import time
import infra

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
            infra.write(infra.HERE/'collection_receipt.json',dict(status='complete',downloaded_bytes=downloaded,
                full_logs_downloaded=False,checkpoint_downloaded=False))
            print(json.dumps(dict(status=final['status'],summary=str(infra.HERE/'summary.json'))),flush=True)
            return
        state=dict(finished_days=value['finished_days'],first_packet_pass=bool(gate and gate.get('all_pass')),
            remaining_launch_held=held,failed=value['failed'])
        if state != previous:
            print(json.dumps(state),flush=True)
            previous=state
        time.sleep(45)
    raise TimeoutError('Four-hour collection bound; preserve all tasks and do not retry automatically')

if __name__=='__main__':
    try:
        main()
    except Exception as error:
        infra.write(infra.HERE/'continuation_error.json',dict(type=type(error).__name__,message=str(error)))
        raise
