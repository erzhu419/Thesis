"""Collect only this fixed pair of scene jobs, without launching follow-ups."""
import json
import time
import infra


def main():
    downloaded, previous = 0, None
    deadline = time.monotonic()+14400
    while time.monotonic() < deadline:
        value, size = infra.poll()
        downloaded += size
        final = value.pop('final')
        if final is not None and final.get('status') != 'pending':
            infra.write(infra.HERE/'summary.json', final)
            infra.write(infra.HERE/'collection_receipt.json', dict(status='complete',
                downloaded_bytes=downloaded, full_logs_downloaded=False, checkpoint_downloaded=False))
            print(json.dumps(dict(status=final['status'], summary=str(infra.HERE/'summary.json'))), flush=True)
            return
        if value != previous:
            print(json.dumps(value), flush=True)
            previous = value
        time.sleep(45)
    raise TimeoutError('Collection timeout; retain existing tasks, never retry native work automatically')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        infra.write(infra.HERE/'continuation_error.json', dict(type=type(error).__name__, message=str(error)))
        raise
