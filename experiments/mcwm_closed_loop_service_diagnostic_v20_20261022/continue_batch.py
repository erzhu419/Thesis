"""Stop on any failed observation; never duplicate the first of six suffixes."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import time

import infra
import launch

HERE = Path(__file__).resolve().parent


def log(stage, **values):
    print(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(), stage=stage, **values),
        ensure_ascii=False), flush=True)


def metadata():
    script = '''from pathlib import Path
import json
r=Path(ROOT)
rows=[]
for line in ('7X','102X','705X'):
 for arm in ('low','high'):
  p=r/'branches/native'/line/arm
  if (p/'failure.json').exists():
   d=json.loads((p/'failure.json').read_text());rows.append({'line':line,'arm':arm,'status':'failed','failure':d});continue
  p=p/'summary.json'
  if not p.exists():rows.append({'line':line,'arm':arm,'status':'pending'});continue
  d=json.loads(p.read_text());rows.append({'line':line,'arm':arm,'status':d.get('status'),'all_pass':d.get('integrity',{}).get('all_pass')})
out={'rows':rows,'first':rows[0],'bad':[x for x in rows if x['status']!='pending' and (x['status']!='complete' or x.get('all_pass') is not True)],
 'analysis_failures':[json.loads(p.read_text()) for p in (r/'analysis_failure.json',r/'summary_failure.json') if p.exists()]}
for name in ('preflight_gate','service_summary'):
 p=r/(name+'.json')
 if p.exists():
  d=json.loads(p.read_text());out[name]={'status':d.get('status'),'all_pass':d.get('integrity',{}).get('all_pass'),'bytes':p.stat().st_size}
 else:out[name]=None
print(json.dumps(out))
'''.replace('ROOT', repr(infra.remote(HERE)))
    return json.loads(infra.scheduler.run_on('node001', shlex.join([infra.PYTHON, '-c', script]),
        timeout=45, check=True)[1])


def wait_until(stage, predicate):
    previous = None
    while True:
        value = metadata()
        if value != previous:
            log(stage, **value)
            previous = value
        if value['bad']:
            raise RuntimeError('Preserve and review failed v20 prefix/suffix: '+str(value['bad']))
        if value['analysis_failures']:
            raise RuntimeError('Preserve failed resident analysis: '+str(value['analysis_failures']))
        for name in ('preflight_gate', 'service_summary'):
            result = value[name]
            if result is not None and (result['status'] != 'complete' or result['all_pass'] is not True):
                raise RuntimeError('Service diagnostic integrity failed: '+name)
        if predicate(value):
            return value
        time.sleep(45)


def exists(name):
    return (HERE/(name+'_submission.json')).exists()


def main():
    if not exists('preflight'):
        raise RuntimeError('Submit the single first arm before continuation')
    log('started', total_new_suffixes=6, suffix_seconds=1800, first_reused=True, priority='high')
    if not exists('preflight_finalize'):
        wait_until('waiting_first_arm', lambda d: d['first']['status'] == 'complete')
        log('preflight_finalizer', receipt=launch.submit_finalizer(preflight=True))
    if not exists('remaining_five'):
        wait_until('waiting_preflight_gate', lambda d: d['preflight_gate'] is not None)
        log('remaining_five', receipt=launch.submit_remaining())
    if not exists('service_finalize'):
        log('service_finalizer', receipt=launch.submit_finalizer())
    value = wait_until('waiting_service_pairs', lambda d: d['service_summary'] is not None)
    infra.collect('service_summary.json', max_bytes=30000)
    log('finished', summary=value['service_summary'], next='Review service evidence before choosing one model change.')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        log('stopped', error_type=type(error).__name__, error=str(error))
        raise
