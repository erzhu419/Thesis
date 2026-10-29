"""Retrieve compact progress and a single final summary; raw logs stay remote."""
import json
import time
from infra import HERE, poll, write


def collect():
    receipt = HERE/'collection_receipt.json'
    previous = json.loads(receipt.read_text()) if receipt.exists() else {}
    total_bytes = previous.get('downloaded_stdout_bytes', 0)
    polls = previous.get('polls', 0)
    unknown_upper = previous.get('interrupted_pull_upper_bound_bytes', 0)
    while True:
        value, size = poll()
        polls += 1
        total_bytes += size
        receipt.write_text(json.dumps(dict(downloaded_stdout_bytes=total_bytes,
            polls=polls, interrupted_pull_upper_bound_bytes=unknown_upper,
            total_download_upper_bound_bytes=total_bytes+unknown_upper,
            full_logs_downloaded=False, checkpoints_downloaded=False), indent=2)+'\n')
        if value['all_summaries_present']:
            write(HERE/'summary.json', value)
            print(json.dumps(dict(finished=True, status=value['status'],
                budget=value['budget'], shared_prefix=value['shared_prefix'],
                observation={s:c.get('observation') for s,c in value['cases'].items()},
                execution=value.get('execution'))), flush=True)
            return
        print(json.dumps(value), flush=True)
        time.sleep(35)


if __name__ == '__main__':
    collect()
