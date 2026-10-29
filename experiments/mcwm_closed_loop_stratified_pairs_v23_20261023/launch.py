"""One new-prefix service preflight, then conditional held-out suffixes only."""
import argparse
from itertools import product
import json
from pathlib import Path
import shlex

import infra

HERE = Path(__file__).resolve().parent
GATE_MAX_BYTES = 40000


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def cell():
    p = protocol()['preflight']
    return dict(seed=p['seed'], scenario=p['scenario'])


def day_path():
    c = cell()
    return Path('stage_a/runs')/c['scenario']/f"seed{c['seed']}"


def evaluation_cells():
    p = protocol()
    return [dict(**cell(), line=line, window=window['name'])
        for line, window in product(p['target_lines'], p['heldout_windows'])]


def evaluation_path(c, arm):
    return Path('stage_a/pairs')/c['scenario']/f"seed{c['seed']}"/c['line']/c['window']/arm


def day_job():
    c = cell()
    args = ['--mode', 'day', '--seed', c['seed'], '--scenario', c['scenario'],
        '--out', infra.remote(HERE/day_path()), '--preflight']
    row = infra.task(protocol(), str(day_path()), args, ram_mb=24000)
    row['cpu'] = 2
    row['cpu_training_justification'] = (
        'Paused native parent launches both single-thread 0/7.5-second action siblings '
        'before waiting; deterministic low/high label admission preserves285 outcomes.')
    return row


def service_pair_job(c):
    path = evaluation_path(c, 'low').parent
    args = ['--seed', c['seed'], '--scenario', c['scenario'],
        '--source', infra.remote(HERE/day_path()),
        '--target-id', f"heldout/{c['line']}/{c['window']}",
        '--out', infra.remote(HERE/path), '--preflight']
    row = infra.task(protocol(), str(path), args, runner='run_pair.py', ram_mb=24000)
    row['cpu'] = 2
    row['cpu_training_justification'] = (
        'Both frozen single-thread native service suffixes launch before waiting; '
        'their measured native phases must overlap. No service truth enters training.')
    return row


def source_ready():
    p = protocol()
    assert p['target_seed_offset'] == p['preflight']['target_seed_offset'] == 90000
    assert cell() == dict(seed=6201, scenario='stable')
    keys = ('initializer_updates', 'history_rows', 'source_rows', 'horizon_seconds',
        'target_record_budget', 'policy_update_budget', 'model_init_updates',
        'model_refit_updates', 'feedback_delay_seconds')
    reference = HERE.parent/'mcwm_closed_loop_state_response_v18_20261022/protocol.json'
    inputs = HERE.parent/'mcwm_closed_loop_formal_20261020/inputs'
    physical = HERE.parent/'mcwm_closed_loop_diagnostic_v2_20261020/inputs/source_intervals.json'
    code = '''from pathlib import Path
import json
p=json.loads(Path(REFERENCE).read_text())
for key,value in EXPECTED.items():assert p[key]==value,key
q=Path(INPUTS)
assert (q/'h2oplus_seed6201.pt').is_file()
for name in ('history.npz','history_metadata.json','source_9101.npz'):assert (q/name).is_file()
assert len(json.loads(Path(PHYSICAL).read_text()))==102
assert p['target_seed_offset']!=90000
print(json.dumps(dict(initialization_and_budget_equal=True,source_intervals=102,external_seed=96201,old_truth_loaded=False)))
'''.replace('REFERENCE', repr(infra.remote(reference))).replace('EXPECTED', repr({k:p[k] for k in keys})).replace('INPUTS', repr(infra.remote(inputs))).replace('PHYSICAL', repr(infra.remote(physical)))
    return json.loads(infra.scheduler.run_on('node001', shlex.join([infra.PYTHON, '-c', code]),
        timeout=45, check=True)[1])


def require_day_gate(*, coverage=True):
    gate = infra.collect(str(day_path()/'gate.json'), max_bytes=GATE_MAX_BYTES)
    if gate.get('status') != 'complete' or gate.get('integrity', {}).get('all_pass') is not True:
        raise RuntimeError('Native day integrity must pass before held-out validation')
    if coverage and gate.get('design_coverage', {}).get('all_pass') is not True:
        raise RuntimeError('New day coverage is incomplete; retain missing cells')
    return gate


def require_training_gate():
    require_day_gate()
    gate = infra.collect('training_gate.json', max_bytes=GATE_MAX_BYTES)
    if gate.get('status') != 'complete' or gate.get('integrity', {}).get('all_pass') is not True:
        raise RuntimeError('Training analysis integrity must pass')
    if gate.get('training_gate', {}).get('passed') is not True or gate.get('service_validation_eligible') is not True:
        raise RuntimeError('Training coverage and scoreable reversal admission must pass')
    return gate


def submit_preflight():
    source_ready()
    return infra.submit([day_job()], 'preflight')


def submit_service_pairs():
    require_training_gate()
    return infra.submit([service_pair_job(c) for c in evaluation_cells()], 'service_pairs')


def submit_finalizer(*, stage='service'):
    if stage == 'train':
        require_day_gate(coverage=False)
        output, batch = 'training_gate.json', 'training_finalize'
    else:
        json.loads((HERE/'service_pairs_submission.json').read_text())
        output, batch = 'service_gate.json', 'service_finalize'
    args = ['--root', infra.remote(HERE), '--out', infra.remote(HERE/output),
        '--stage', stage, '--wait']
    return infra.submit([infra.task(protocol(), stage+'/finalize', args,
        runner='analyze.py', ram_mb=2048)], batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('stage', 'preflight', 'training-finalize',
        'service-pairs', 'service-finalize', 'collect'))
    args = parser.parse_args()
    if args.operation == 'stage':
        names = ('bootstrap.py', 'infra.py', 'launch.py', 'continue_batch.py', 'run.py',
            'analyze.py', 'history.py', 'shift_model.py', 'candidate.py', 'sampling.py',
            'run_pair.py', 'protocol.json')
        value = infra.stage([HERE/name for name in names])
    elif args.operation == 'preflight':
        value = submit_preflight()
    elif args.operation == 'training-finalize':
        value = submit_finalizer(stage='train')
    elif args.operation == 'service-pairs':
        value = submit_service_pairs()
    elif args.operation == 'service-finalize':
        value = submit_finalizer()
    else:
        value = infra.collect('service_gate.json', max_bytes=GATE_MAX_BYTES)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
