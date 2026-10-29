"""Resume v59 receipts without repeated prefixes, pairs, or learning workers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import infra
import launch

HERE = Path(__file__).resolve().parent
TERMINAL = {'done', 'failed', 'cancelled'}


def read_receipt(batch):
    path = HERE/(batch+'_submission.json')
    return json.loads(path.read_text()) if path.exists() else None


def task_map(receipt):
    return {row['signature']: row['id'] for row in receipt['submitted']}


def output_names(relative):
    return [relative+'/summary.json', relative+'/failure.json']


def outcome(task_id, relative, statuses, found):
    summary_path, failure_path = output_names(relative)
    row = statuses[task_id]
    scheduler_status = row['status']
    base = dict(id=task_id, scheduler_status=scheduler_status, node=row.get('node'))
    if found.get(failure_path):
        failure = infra.collect(failure_path)
        return dict(base, state='failed', output=failure_path,
            message=failure.get('message', str(failure)), failure_budget=failure.get('failure_budget'))
    if found.get(summary_path):
        summary = infra.collect(summary_path)
        complete = summary.get('status') == 'complete' and scheduler_status not in ('failed', 'cancelled')
        return dict(base, state='complete' if complete else 'failed', output=summary_path,
            result_status=summary.get('status'), message=None if complete else 'Output or scheduler task failed')
    if scheduler_status in TERMINAL:
        return dict(base, state='failed', output=None, message='Terminal task without declared output')
    return dict(base, state='pending', output=None)


def snapshot(receipts, paths):
    ids = [row['id'] for receipt in receipts for row in receipt['submitted']]
    statuses = {row['id']: row for row in infra.status(ids)}
    if set(statuses) != set(ids):
        raise RuntimeError('Scheduler did not return all requested task IDs; no task will be resubmitted')
    found = infra.presence(paths)
    # A worker can finish between these reads. Only a terminal missing-output
    # snapshot gets one additional existence read before being recorded failed.
    if any(row['status'] in TERMINAL for row in statuses.values()):
        missing = [path for path, present in found.items() if not present]
        if missing:
            found.update(infra.presence(missing))
    return statuses, found


def step():
    p = launch.protocol()
    native = read_receipt('native')
    if native is None:
        raise RuntimeError('Run launch.py start before continuation; no implicit native submission')
    receipts = [native]
    paths = []
    learning = {}
    for case in p['scenarios']:
        paths += output_names('runs/collection/'+case)
        paths += ['runs/collection/'+case+'/preflight_gate.json']
        receipt = read_receipt('learning_'+case)
        learning[case] = receipt
        if receipt:
            receipts.append(receipt)
            for arm in launch.ARMS:
                paths += output_names('runs/learning/'+case+'/'+arm)
    final = read_receipt('finalize')
    if final:
        receipts.append(final)
        paths += output_names('analysis')
    statuses, found = snapshot(receipts, paths)
    native_ids = task_map(native)
    cases = {}
    newly_submitted = []
    for case in p['scenarios']:
        native_id = native_ids[p['name']+'/collection/'+case]
        collection = outcome(native_id, 'runs/collection/'+case, statuses, found)
        gate_path = 'runs/collection/'+case+'/preflight_gate.json'
        if found.get(gate_path):
            gate = infra.collect(gate_path, max_bytes=16000)
            collection['preflight'] = gate
        workers = {}
        receipt = learning[case]
        if collection['state'] == 'complete' and receipt is None:
            receipt = launch.learning(case)
            learning[case] = receipt
            newly_submitted += [row['id'] for row in receipt['submitted']]
            workers = {arm: dict(state='pending', id=task_map(receipt)[p['name']+'/learning/'+case+'/'+arm],
                scheduler_status='submitted') for arm in launch.ARMS}
        elif receipt:
            worker_ids = task_map(receipt)
            workers = {arm: outcome(worker_ids[p['name']+'/learning/'+case+'/'+arm],
                'runs/learning/'+case+'/'+arm, statuses, found) for arm in launch.ARMS}
        elif collection['state'] == 'failed':
            workers = {arm: dict(state='skipped', reason='Native collection failed; no retry')
                for arm in launch.ARMS}
        cases[case] = dict(collection=collection, learning=workers)
    all_terminal = all(value['collection']['state'] in ('complete', 'failed') and
        len(value['learning']) == len(launch.ARMS) and
        all(row['state'] in ('complete', 'failed', 'skipped') for row in value['learning'].values())
        for value in cases.values())
    lifecycle = dict(protocol=p['name'], cases=cases, all_terminal=all_terminal,
        feedback_slots_per_arm=285, training_updates_per_arm=456,
        preflight_reused=True, automatic_retries=0)
    if all_terminal and final is None:
        infra.publish_state(lifecycle)
        final = launch.finalize()
        newly_submitted += [row['id'] for row in final['submitted']]
        return dict(stage='waiting_small_summary', all_terminal=True, cases=cases,
            submitted_ids=newly_submitted, done=False)
    if final:
        summary_id = final['submitted'][0]['id']
        if summary_id not in statuses:
            return dict(stage='waiting_small_summary', all_terminal=all_terminal, cases=cases,
                submitted_ids=newly_submitted, done=False)
        analysis = outcome(summary_id, 'analysis', statuses, found)
        if analysis['state'] == 'complete':
            summary = infra.collect('analysis/summary.json')
            result = dict(stage='collected', done=True, status=summary.get('status'),
                integrity=summary.get('integrity'), cases=cases,
                native_task_ids=[row['id'] for row in native['submitted']],
                finalize_task_ids=[summary_id])
            if not (HERE/'continuation_result.json').exists():
                infra.write(HERE/'continuation_result.json', result)
            return result
        if analysis['state'] == 'failed':
            # A partial science summary may deliberately have status='partial'.
            # Retain it as a completed analysis instead of retrying the job.
            path = 'analysis/summary.json'
            if found.get(path) and statuses[summary_id]['status'] == 'done':
                summary = infra.collect(path)
                result = dict(stage='collected', done=True, status=summary.get('status'),
                    integrity=summary.get('integrity'), cases=cases, analysis=analysis)
                if not (HERE/'continuation_result.json').exists():
                    infra.write(HERE/'continuation_result.json', result)
                return result
            return dict(stage='analysis_failed', done=True, cases=cases, analysis=analysis)
    return dict(stage='waiting_collection_or_learning', cases=cases,
        all_terminal=all_terminal, submitted_ids=newly_submitted, done=False)


def main(watch=False, interval=30):
    deadline = time.monotonic()+launch.protocol().get('continuation_timeout_seconds', 172800)
    previous = None
    while True:
        current = step()
        (HERE/'continuation_state.json').write_text(json.dumps(current, ensure_ascii=False, indent=2)+'\n')
        if current != previous:
            print(json.dumps(current, ensure_ascii=False), flush=True)
            previous = current
        if current['done'] or not watch:
            return current
        if time.monotonic() >= deadline:
            raise TimeoutError('Continuation wait expired; existing receipts remain reusable')
        time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=int, default=30, choices=range(1, 61))
    args = parser.parse_args()
    with (HERE/'continuation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Existing continuation is alive; reuse its scheduler receipts.', flush=True)
            raise SystemExit(0)
        (HERE/'continuation_process.json').write_text(json.dumps(
            dict(pid=os.getpid(), started_unix=time.time()), indent=2)+'\n')
        try:
            main(args.watch, args.interval)
        except Exception as error:
            failure = dict(status='failed', type=type(error).__name__, message=str(error),
                automatic_retries=0)
            if not (HERE/'continuation_failure.json').exists():
                infra.write(HERE/'continuation_failure.json', failure)
            print(json.dumps(failure), flush=True)
            raise
