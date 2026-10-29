"""Finish the approved frozen-parameter control batch after its native preflight."""

from datetime import datetime, timezone
import json
import shlex
import time

import infra
import launch

HERE = infra.HERE


def log(stage, **fields):
    print(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(), stage=stage, **fields)), flush=True)


def metadata():
    script = '''import json
from pathlib import Path
p=Path(__ROOT__)
q=p/'preflight_frozen_world/control/local_speed_drop/frozen_world/seed6201'
f=p/'frozen_world_summary.json'
v=dict(preflight_started=q.is_dir(),preflight_complete=(q/'summary.json').is_file(),
 preflight_failed=(q/'failure.json').is_file(),
 failures=[str(x.relative_to(p)) for x in (p/'frozen_runs/control').glob('*/*/seed*/failure.json')],
 result_exists=f.is_file())
if f.is_file():
 x=json.loads(f.read_text());v['result']=dict(status=x.get('status'),bytes=f.stat().st_size,
  complete=x.get('integrity',{}).get('complete_runs'),errors=len(x.get('integrity',{}).get('errors',[])))
print(json.dumps(v))
'''.replace('__ROOT__', repr(infra.remote(HERE)))
    try:
        _, output, _ = infra.scheduler.run_on('node001', shlex.join([infra.PYTHON, '-c', script]),
                                            timeout=45, check=True)
        return json.loads(output)
    except RuntimeError as error:
        if 'cmd failed (rc=255)' not in str(error):
            raise
        log('metadata_connection_interrupted', error=str(error), retry_seconds=60)
        return None


def main():
    if not (HERE / 'preflight_frozen_world_submission.json').is_file():
        raise RuntimeError('Submit the approved frozen-world preflight first')
    previous = None
    while True:
        value = metadata()
        if value is not None:
            if value != previous:
                log('frozen_batch_status', **value)
                previous = value
            if value['preflight_failed'] or value['failures']:
                raise RuntimeError('Frozen-world run failed; retain resident failure evidence')
            if value['preflight_complete']:
                if not (HERE / 'stage_b_submission.json').exists():
                    result = launch.submit_frozen()
                    log('stage_b_submitted', ids=[row['id'] for row in result['submitted']])
                if not (HERE / 'stage_b_finalize_submission.json').exists():
                    result = launch.submit_finalizer('stage_b')
                    log('finalizer_submitted', ids=[row['id'] for row in result['submitted']])
            if value['result_exists']:
                result = value['result']
                if result['status'] != 'complete' or result['complete'] != 4 or result['errors']:
                    raise RuntimeError('Frozen-world summary is incomplete; inspect resident evidence')
                if not (HERE / 'frozen_world_summary.json').exists():
                    infra.collect_json('frozen_world_summary.json')
                log('stage_b_complete', collected_bytes=(HERE / 'frozen_world_summary.json').stat().st_size)
                return
        time.sleep(60)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        log('stopped', error_type=type(error).__name__, error=str(error))
        raise
