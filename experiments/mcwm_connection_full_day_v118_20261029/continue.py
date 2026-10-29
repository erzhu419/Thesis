"""Read this fixed native task; never submit a follow-up or retry."""
import json
import time
import infra


def main():
    downloaded = 0
    for _ in range(120):
        value, size = infra.poll()
        downloaded += size
        if value['complete']:
            infra.write(infra.HERE/'summary.json', value['summary'])
            infra.write(infra.HERE/'collection_receipt.json', dict(downloaded_stdout_bytes=downloaded,
                complete_logs_downloaded=False, checkpoint_downloaded=False))
            print(json.dumps(dict(status=value['summary']['status'], summary=str(infra.HERE/'summary.json'))), flush=True)
            return
        time.sleep(30)
    raise TimeoutError('Keep existing task; inspect progress without resubmission')


if __name__ == '__main__':
    main()
