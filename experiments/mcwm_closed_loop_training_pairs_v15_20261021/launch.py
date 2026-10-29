"""High-priority resident selection, native preflight, then24 diagnostic branches."""
import argparse
from itertools import product
import json
from pathlib import Path
import shlex

import infra

HERE = Path(__file__).resolve().parent
RESULT_NAME = 'training_pair_summary.json'
SUMMARY_MAX_BYTES = 100000
SMALL_MAX_BYTES = 20000


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def cells(*, preflight=False):
    p = protocol()
    if preflight:
        return [dict(seed=p['preflight']['seed'], scenario=p['preflight']['scenario'],
                     target_line=p['preflight']['target_line'], window='preflight')]
    return [dict(seed=p['paired_seed'], scenario=scene, target_line=line, window=window['name'])
            for scene, line, window in product(p['scenarios'], p['target_lines'], p['training_windows'])]


def branch_path(cell, branch, *, preflight=False):
    if preflight:
        return Path('preflight/pair')/branch
    return Path('pairs')/cell['scenario']/cell['target_line']/f"seed{cell['seed']}"/cell['window']/branch


def manifest_path(*, preflight=False):
    return Path('preflight/selection_manifest.json' if preflight else 'selection_manifest.json')


def selection_summary(*, preflight=False):
    return Path('preflight/selection_summary.json' if preflight else 'selection_summary.json')


def source_ready():
    p = protocol()
    source = (HERE/p['reference_v14_root']).resolve()
    script = '''import json
from pathlib import Path
root=Path(__ROOT__)
p=json.loads(Path(__PROTOCOL__).read_text())
for key in __KEYS__:
 assert p[key]==__EXPECTED__[key],key
for path in ('runs/stable/seed6201','runs/local_speed_drop/seed6201','preflight/day'):
 d=root/path
 v=json.loads((d/'summary.json').read_text())
 assert v['status']=='complete' and v['integrity']['all_pass']
 assert (d/'prefix_trace.jsonl').is_file()
for name in ('innovation.py','run.py'):
 assert (root/name).is_file()
q=Path(__INPUTS__)
assert (q/'h2oplus_seed6201.pt').is_file() and (q/'history.npz').is_file()
assert (q/'history_metadata.json').is_file() and (q/'source_9101.npz').is_file()
assert len(json.loads(Path(__PHYSICAL__).read_text()))==102
print(json.dumps(dict(reference_days=3,source_intervals=102,budget_configuration_equal=True)))
'''
    keys = ['initializer_updates','history_rows','source_rows','horizon_seconds','target_record_budget',
            'policy_update_budget','model_init_updates','model_refit_updates','feedback_delay_seconds',
            'target_seed_offset','paired_suffix_seconds']
    script = script.replace('__ROOT__', repr(infra.remote(source))).replace('__PROTOCOL__', repr(infra.remote(source/'protocol.json')))
    script = script.replace('__KEYS__', repr(keys)).replace('__EXPECTED__', repr({k:p[k] for k in keys}))
    script = script.replace('__INPUTS__', repr(str(infra.RESIDENT_INPUTS))).replace('__PHYSICAL__', repr(str(infra.RESIDENT_SOURCE/'source_intervals.json')))
    return json.loads(infra.scheduler.run_on('node001', shlex.join([infra.PYTHON, '-c', script]), timeout=45, check=True)[1])


def make_branch(cell, branch, *, preflight=False):
    directory = branch_path(cell, branch, preflight=preflight)
    argv = ['--seed', str(cell['seed']), '--scenario', cell['scenario'], '--target-line', cell['target_line'],
            '--window', cell['window'], '--branch', branch, '--manifest', infra.remote(HERE/manifest_path(preflight=preflight)),
            '--out', infra.remote(HERE/directory)]
    if preflight:
        argv.append('--preflight')
    return infra.task(protocol(), str(directory), argv)


def submit_selection():
    source_ready()
    tasks = []
    for pref in (True, False):
        argv = ['--out', infra.remote(HERE/manifest_path(preflight=pref)),
                '--summary', infra.remote(HERE/selection_summary(preflight=pref))]
        if pref:
            argv.append('--preflight')
        tasks.append(infra.task(protocol(), 'preflight/select' if pref else 'diagnostic/select', argv,
                                runner='select_states.py', ram_mb=1024))
    return infra.submit(tasks, 'selection')


def require_selection(*, preflight=False):
    relative = selection_summary(preflight=preflight)
    local = HERE/relative
    value = json.loads(local.read_text()) if local.is_file() else infra.collect_json(str(relative), max_bytes=SMALL_MAX_BYTES)
    if value.get('status') != 'complete':
        raise RuntimeError('Resident selection must complete before native branches')
    if preflight and not all(row.get('status') == 'selected' for row in value['states']):
        raise RuntimeError('Preflight requires an executable training-period state')
    return value


def submit_preflight():
    require_selection(preflight=True)
    return infra.submit([make_branch(cells(preflight=True)[0], arm, preflight=True)
                         for arm in ('low','high')], 'preflight_pairs')


def require_gate():
    relative = 'preflight/gate.json'
    value = infra.collect_json(relative, max_bytes=SMALL_MAX_BYTES) if not (HERE/relative).is_file() else json.loads((HERE/relative).read_text())
    if value.get('status') != 'complete' or value.get('integrity', {}).get('all_pass') is not True:
        raise RuntimeError('Native paired preflight must pass before main diagnostic')
    return value


def submit_pairs():
    require_gate()
    require_selection()
    return infra.submit([make_branch(cell, arm) for cell in cells() for arm in ('low','high')], 'paired')


def submit_finalizer(*, preflight=False):
    batch = 'preflight_pairs' if preflight else 'paired'
    json.loads((HERE/(batch+'_submission.json')).read_text())
    root = HERE/'preflight' if preflight else HERE
    out = root/'gate.json' if preflight else root/RESULT_NAME
    argv = ['--root', infra.remote(root), '--manifest', infra.remote(HERE/manifest_path(preflight=preflight)),
            '--out', infra.remote(out), '--wait']
    if preflight:
        argv.append('--preflight')
    return infra.submit([infra.task(protocol(), 'preflight/finalize' if preflight else 'diagnostic/finalize',
                                   argv, runner='analyze.py', ram_mb=1024)],
                        'preflight_finalize' if preflight else 'diagnostic_finalize')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('stage','select','preflight','preflight-finalize','pairs','finalize','status','collect'))
    parser.add_argument('--batch', default='paired')
    args = parser.parse_args()
    if args.operation == 'stage':
        paths = [path for path in HERE.glob('*.py') if not path.name.startswith('test_')]
        value = infra.stage([*sorted(paths), HERE/'protocol.json'])
    elif args.operation == 'select':
        value = submit_selection()
    elif args.operation == 'preflight':
        value = submit_preflight()
    elif args.operation == 'preflight-finalize':
        value = submit_finalizer(preflight=True)
    elif args.operation == 'pairs':
        value = submit_pairs()
    elif args.operation == 'finalize':
        value = submit_finalizer()
    elif args.operation == 'status':
        rows = json.loads((HERE/(args.batch+'_submission.json')).read_text())['submitted']
        value = infra.status([row['id'] for row in rows])
    else:
        v = infra.collect_json(RESULT_NAME, max_bytes=SUMMARY_MAX_BYTES)
        value = dict(status=v['status'],bytes=(HERE/RESULT_NAME).stat().st_size,integrity=v['integrity'])
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
