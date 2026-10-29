"""Finish the fixed v104 first day, remaining submission, and small readout once."""
import json
from pathlib import Path
import shlex
import time
import traceback

import infra
import analyze

HERE = Path(__file__).resolve().parent
RUNS = HERE/'runs_runtime_fix'


def main():
    if (HERE/'summary.json').exists():
        print('v104 already has its final summary; no action', flush=True)
        return
    p = json.loads((HERE/'protocol.json').read_text())
    cases = analyze.expected_runs(p)
    first = p['first_formal_day']
    first_key = (first['training_seed'], first['external_seed'], first['scenario'])
    relative = {key: str(Path('runs_runtime_fix')/str(key[0])/str(key[1])/key[2]/'summary.json')
                for key in cases}
    receipt_path = HERE/'finish_retrieval.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else dict(files=[], total_bytes=0)
    stage, status = 'waiting_first_formal_day', {}
    next_poll = next_log = 0.
    deadline = time.monotonic()+3*3600

    def pull(key):
        path = HERE/relative[key]
        if path.exists():
            return json.loads(path.read_text())
        value, size = infra.pull(relative[key], limit=p['summary_max_bytes_per_day'])
        receipt['files'].append(dict(path=relative[key], bytes=size))
        receipt['total_bytes'] += size
        receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
        return value

    def pull_batch():
        missing = [relative[key] for key in cases if not (HERE/relative[key]).exists()]
        if not missing: return
        code = ('from pathlib import Path\nimport json\nroot=Path('+repr(infra.remote(HERE))+')\n'
            'values={name:json.loads((root/name).read_text()) for name in '+repr(missing)+'}\n'
            'assert all(len(json.dumps(v,separators=(",",":")).encode())+1<='
            +str(p['summary_max_bytes_per_day'])+' for v in values.values())\n'
            'raw=json.dumps(values,separators=(",",":"))\nassert len(raw.encode())+1<='
            +str(p['summary_total_max_bytes']-receipt['total_bytes'])+'\nprint(raw)\n')
        raw = infra.scheduler.run_on('node001', shlex.join([infra.PYTHON,'-B','-c',code]),
                                    timeout=45, check=True)[1]
        values = json.loads(raw)
        for name in missing:
            infra.write(HERE/name, values[name])
            receipt['files'].append(dict(path=name, retrieved_in_batch=True,
                compact_json_bytes=len(json.dumps(values[name], separators=(',', ':')).encode())))
        receipt['batch_transfer_bytes'] = len(raw.encode())
        receipt['total_bytes'] += len(raw.encode())
        receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')

    def finish():
        runs = {key: json.loads((HERE/relative[key]).read_text())
                if (HERE/relative[key]).exists() else None for key in cases}
        validation_path = HERE.parents[1]/p['v75_results_path']
        result = analyze.analyze(p, json.loads(validation_path.read_text()), runs)
        result['sources'] = dict(protocol=str(HERE/'protocol.json'),
            v75_validation=str(validation_path), runs=str(RUNS), retrieval=str(receipt_path))
        infra.write(HERE/'summary.json', result)
        print(json.dumps(dict(stage='finished', status=result['status'], decision=result['decision'],
                              budget=result['budget'])), flush=True)

    while True:
        now = time.monotonic()
        if now >= deadline:
            infra.write(HERE/'finish_timeout.json', dict(stage=stage, deadline_seconds=10800,
                last_remote_status=status, unknown_missing=[name for name in relative.values()
                    if status.get(name, {}).get('status', 'missing') == 'missing'],
                tasks_retried=False))
            raise TimeoutError('Three-hour v104 monitor deadline reached; missing results remain unknown; no task rerun')
        if now >= next_poll:
            paths = [relative[key] for key in cases]
            code = ('from pathlib import Path\nimport json\nroot=Path('+repr(infra.remote(HERE))+')\n'
                'rows={}\nfor name in '+repr(paths)+':\n'
                ' p=root/name\n'
                ' if not p.exists(): rows[name]={"status":"missing"}; continue\n'
                ' v=json.loads(p.read_text())\n'
                ' rows[name]={"status":v.get("status"),"all_pass":v.get("integrity",{}).get("all_pass"),'
                '"bytes":len(json.dumps(v,separators=(",",":")).encode())+1}\n'
                'print(json.dumps(rows,separators=(",",":")))\n')
            raw = infra.scheduler.run_on('node001', shlex.join([infra.PYTHON,'-B','-c',code]),
                                         timeout=45, check=True)[1]
            status = json.loads(raw)
            if time.monotonic() >= deadline: continue
            first_status = status[relative[first_key]]
            if first_status['status'] != 'missing':
                if first_status['bytes'] > p['summary_max_bytes_per_day']:
                    raise RuntimeError('First-day summary exceeds declared small retrieval bound')
                first_result = pull(first_key)
                if first_result['status'] != 'complete' or first_result['integrity']['all_pass'] is not True:
                    stage = 'first_formal_day_failed_no_remaining_submission'
                    finish()
                    return
                submission = HERE/'remaining_submission.json'
                if not submission.exists():
                    if (HERE/'remaining_submission_attempt.json').exists():
                        raise RuntimeError('Remaining submission attempt has no receipt; root must resolve it; not resubmitting')
                    stage = 'submitting_remaining_once'
                    print(json.dumps(dict(stage=stage)), flush=True)
                    infra.submit('remaining', infra.rows(False))
                stage = 'waiting_fixed_30_summaries'
                if all(row['status'] != 'missing' for row in status.values()):
                    sizes = [row['bytes'] for row in status.values()]
                    if max(sizes) > p['summary_max_bytes_per_day'] or sum(sizes) > p['summary_total_max_bytes']:
                        raise RuntimeError('Fixed panel summaries exceed declared retrieval bounds')
                    pull_batch()
                    finish()
                    return
            next_poll = time.monotonic()+45.
        now = time.monotonic()
        if now >= next_log:
            complete = sum(row.get('status') == 'complete' and row.get('all_pass') is True for row in status.values())
            missing = sum(row.get('status') == 'missing' for row in status.values())
            print(json.dumps(dict(stage=stage, complete=complete,
                failed=len(status)-complete-missing, missing=missing)), flush=True)
            next_log = now+60.
        time.sleep(max(.01, min(next_poll, next_log, deadline)-time.monotonic()))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        path = HERE/'finish_error.json'
        if not path.exists():
            infra.write(path, dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        raise
