"""Service-history gate first; three fresh response days only after it passes."""
import argparse
from itertools import product
import json
from pathlib import Path
import shlex

import infra

HERE = Path(__file__).resolve().parent
GATE_MAX_BYTES = 20000
SUMMARY_MAX_BYTES = 50000


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def cells(*, preflight=False):
    p = protocol()
    first = dict(seed=p['preflight']['seed'], scenario=p['preflight']['scenario'])
    if preflight:
        return [first]
    return [dict(seed=seed, scenario=scene)
        for seed, scene in product(p['training_seeds'], p['scenarios'])
        if dict(seed=seed, scenario=scene) != first]


def day_path(cell, *, preflight=False):
    return Path('stage_a' if preflight else 'stage_b')/'runs'/cell['scenario']/f"seed{cell['seed']}"


def evaluation_cells(*, preflight=False):
    p = protocol()
    return [dict(**cell, line=line, window=window['name'])
        for cell in cells(preflight=preflight)
        for line, window in product(p['target_lines'], p['heldout_windows'])]


def evaluation_path(cell, arm, *, preflight=False):
    return Path('stage_a' if preflight else 'stage_b')/'pairs'/cell['scenario']/f"seed{cell['seed']}"/cell['line']/cell['window']/arm


def day_job(cell, *, preflight=False):
    path = day_path(cell, preflight=preflight)
    args = ['--mode', 'day', '--seed', cell['seed'], '--scenario', cell['scenario'],
        '--out', infra.remote(HERE/path)]
    if preflight:
        args.append('--preflight')
    return infra.task(protocol(), str(path), args, ram_mb=16000)


def evaluation_job(cell, arm, *, preflight=False):
    path = evaluation_path(cell, arm, preflight=preflight)
    args = ['--mode', 'branch', '--seed', cell['seed'], '--scenario', cell['scenario'],
        '--source', infra.remote(HERE/day_path(cell, preflight=preflight)),
        '--target-id', f"heldout/{cell['line']}/{cell['window']}", '--branch', arm,
        '--out', infra.remote(HERE/path)]
    if preflight:
        args.append('--preflight')
    return infra.task(protocol(), str(path), args, ram_mb=8000)


def source_ready():
    p = protocol()
    keys = ('initializer_updates', 'history_rows', 'source_rows', 'horizon_seconds',
        'target_record_budget', 'policy_update_budget', 'model_init_updates',
        'model_refit_updates', 'feedback_delay_seconds')
    reference = HERE.parent/'mcwm_closed_loop_state_response_v18_20261022/protocol.json'
    inputs = HERE.parent/'mcwm_closed_loop_formal_20261020/inputs'
    physical = HERE.parent/'mcwm_closed_loop_diagnostic_v2_20261020/inputs/source_intervals.json'
    expected = {key:p[key] for key in keys}
    offsets = [p['preflight']['target_seed_offset'], p['target_seed_offset']]
    code = '''from pathlib import Path
import json
p=json.loads(Path(REFERENCE).read_text())
for key,value in EXPECTED.items():
 assert p[key]==value,key
q=Path(INPUTS)
for seed in SEEDS:assert (q/('h2oplus_seed%d.pt'%seed)).is_file()
for name in ('history.npz','history_metadata.json','source_9101.npz'):assert (q/name).is_file()
assert len(json.loads(Path(PHYSICAL).read_text()))==102
assert len(set(OFFSETS+[p['target_seed_offset']]))==3
assert p['training_seeds']==SEEDS
print(json.dumps(dict(initialization_and_budget_equal=True,source_intervals=102,service_external_seed=SEEDS[0]+OFFSETS[0],response_external_seeds=[s+OFFSETS[1] for s in SEEDS])))
'''.replace('REFERENCE', repr(infra.remote(reference))).replace('EXPECTED', repr(expected)).replace('INPUTS', repr(infra.remote(inputs))).replace('SEEDS', repr(p['training_seeds'])).replace('PHYSICAL', repr(infra.remote(physical))).replace('OFFSETS', repr(offsets))
    return json.loads(infra.scheduler.run_on('node001', shlex.join([infra.PYTHON, '-c', code]),
        timeout=45, check=True)[1])


def require_gate(relative, *, coverage=False, scientific=False):
    value = infra.collect(str(relative), max_bytes=GATE_MAX_BYTES)
    if value.get('status') != 'complete' or value.get('integrity', {}).get('all_pass') is not True:
        raise RuntimeError('Native integrity gate must pass: '+str(relative))
    if coverage and value.get('design_coverage', {}).get('all_pass') is not True:
        raise RuntimeError('Training and heldout coverage must pass: '+str(relative))
    if scientific and value.get('scientific_gate', {}).get('passed') is not True:
        raise RuntimeError('Service predictability scientific gate must pass before response training')
    return value


def submit_preflight():
    source_ready()
    return infra.submit([day_job(cells(preflight=True)[0], preflight=True)], 'preflight')


def submit_service_pairs():
    require_gate(day_path(cells(preflight=True)[0], preflight=True)/'gate.json', coverage=True)
    return infra.submit([evaluation_job(cell, arm, preflight=True)
        for cell in evaluation_cells(preflight=True) for arm in ('low', 'high')], 'service_pairs')


def submit_finalizer(*, preflight=False):
    batch = 'service_pairs' if preflight else 'response_pairs'
    json.loads((HERE/(batch+'_submission.json')).read_text())
    stage = 'service' if preflight else 'response'
    output = 'service_gate.json' if preflight else 'response_summary.json'
    args = ['--root', infra.remote(HERE), '--out', infra.remote(HERE/output), '--stage', stage, '--wait']
    return infra.submit([infra.task(protocol(), stage+'/finalize', args, runner='analyze.py', ram_mb=2048)],
        stage+'_finalize')


def submit_response_days():
    require_gate('service_gate.json', scientific=True)
    return infra.submit([day_job(cell) for cell in cells()], 'response_days')


def days_status():
    paths = [str(day_path(cell)) for cell in cells()]
    code = '''from pathlib import Path
import json
root=Path(ROOT)
rows=[]
for relative in PATHS:
 p=root/relative
 if (p/'failure.json').is_file():rows.append(dict(path=relative,status='failed'));continue
 f=p/'summary.json'
 if not f.is_file():rows.append(dict(path=relative,status='pending'));continue
 d=json.loads(f.read_text());rows.append(dict(path=relative,status=d.get('status'),all_pass=d.get('integrity',{}).get('all_pass'),coverage=d.get('design_coverage',{}).get('all_pass')))
print(json.dumps(rows))
'''.replace('ROOT', repr(infra.remote(HERE))).replace('PATHS', repr(paths))
    return json.loads(infra.scheduler.run_on('node001', shlex.join([infra.PYTHON, '-c', code]),
        timeout=45, check=True)[1])


def submit_response_pairs():
    require_gate('service_gate.json', scientific=True)
    rows = days_status()
    if len(rows) != len(cells()) or not all(row.get('status') == 'complete' and
            row.get('all_pass') is True and row.get('coverage') is True for row in rows):
        raise RuntimeError('All response days must pass integrity and coverage before response branches')
    return infra.submit([evaluation_job(cell, arm)
        for cell in evaluation_cells() for arm in ('low', 'high')], 'response_pairs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('stage', 'preflight', 'service-pairs', 'service-finalize',
        'response-days', 'response-pairs', 'response-finalize', 'collect'))
    args = parser.parse_args()
    if args.operation == 'stage':
        names = ('bootstrap.py', 'infra.py', 'launch.py', 'continue_batch.py', 'run.py',
            'analyze.py', 'history.py', 'shift_model.py', 'candidate.py', 'protocol.json')
        value = infra.stage([HERE/name for name in names])
    elif args.operation == 'preflight': value = submit_preflight()
    elif args.operation == 'service-pairs': value = submit_service_pairs()
    elif args.operation == 'service-finalize': value = submit_finalizer(preflight=True)
    elif args.operation == 'response-days': value = submit_response_days()
    elif args.operation == 'response-pairs': value = submit_response_pairs()
    elif args.operation == 'response-finalize': value = submit_finalizer()
    else:
        value = infra.collect('response_summary.json', max_bytes=SUMMARY_MAX_BYTES)
        value = {key:value.get(key) for key in ('status', 'integrity', 'decision')}
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
