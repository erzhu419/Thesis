"""Collect completion receipts only; never launch or repeat native work."""
import json
import time
from infra import HERE, poll, write

downloaded = 0
for attempt in range(180):
    value, size = poll()
    downloaded += size
    if value['finished'] == 2:
        write(HERE/'summary.json', value)
        write(HERE/'collection_receipt.json', dict(downloaded_stdout_bytes=downloaded,
            polls=attempt+1, full_logs_downloaded=False, checkpoint_downloaded=False))
        print(json.dumps(dict(status='collected', summary=str(HERE/'summary.json'))), flush=True)
        break
    time.sleep(30)
else:
    write(HERE/'collection_pending.json', dict(downloaded_stdout_bytes=downloaded,
        status='No repeat or cancellation; inspect the existing tasks.'))
