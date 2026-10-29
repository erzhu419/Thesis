"""Collect this fixed diagnostic only; never launch another native task."""
import json
import time
import infra


def main():
    downloaded, previous = 0, None
    deadline = time.monotonic()+7200
    while time.monotonic() < deadline:
        value, size = infra.poll()
        downloaded += size
        if value['finished'] == 2:
            cases = value['cases']
            passed = all(row['status']=='complete' and row['integrity']['all_pass'] for row in cases.values())
            native = [row['budget']['native_seconds'] for row in cases.values()]
            known = [seconds for seconds in native if seconds is not None]
            result = dict(status='complete' if passed else 'technical_hold', all_pass=passed, cases=cases,
                budget=dict(native_seconds=sum(known) if len(known)==2 else None,
                    known_native_seconds=sum(known), native_seconds_is_lower_bound=any(
                        row['budget']['native_seconds_is_lower_bound'] for row in cases.values()),
                    optimizer_updates=0, feedback_admitted=0, learner_created=False),
                performance_evidence=False, automatic_followup=False)
            infra.write(infra.HERE/'summary.json', result)
            infra.write(infra.HERE/'collection_receipt.json', dict(downloaded_bytes=downloaded,
                complete_logs_downloaded=False, checkpoint_downloaded=False))
            print(json.dumps(dict(status=result['status'], summary=str(infra.HERE/'summary.json'))), flush=True)
            return
        if value['finished'] != previous:
            print(json.dumps(dict(finished=value['finished'])), flush=True)
            previous = value['finished']
        time.sleep(40)
    raise TimeoutError('Readout timed out; retain tasks and costs without retries')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        infra.write(infra.HERE/'collection_error.json', dict(type=type(error).__name__, message=str(error)))
        raise
