"""One service-observer preflight, reuse it, then five branches within budget."""
import argparse
import json
from pathlib import Path

import infra

HERE = Path(__file__).resolve().parent


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def source_root():
    return HERE.parent/protocol()['source_version']


def branches():
    return [(line, arm) for line in protocol()['fixed_lines'] for arm in ('low', 'high')]


def branch_path(line, arm):
    return Path('branches/native')/line/arm


def branch_job(line, arm):
    p = protocol()
    args = ['--mode', 'branch', '--seed', str(p['fixed_seed']), '--scenario', p['fixed_scenario'],
        '--source', infra.remote(source_root()/'runs/stable/seed6201'),
        '--target-id', p['anchors'][line]['id'], '--branch', arm,
        '--out', infra.remote(HERE/branch_path(line, arm))]
    return infra.task(p, str(branch_path(line, arm)), args)


def submit_preflight():
    return infra.submit([branch_job('7X', 'low')], 'preflight')


def submit_remaining():
    gate = infra.collect('preflight_gate.json', max_bytes=12000)
    if gate.get('status') != 'complete' or gate.get('integrity', {}).get('all_pass') is not True:
        raise RuntimeError('Exact native truth and service-observation preflight must pass')
    return infra.submit([branch_job(line, arm) for line, arm in branches()
                         if (line, arm) != ('7X', 'low')], 'remaining_five')


def submit_finalizer(*, preflight=False):
    required = HERE/(('preflight' if preflight else 'remaining_five')+'_submission.json')
    json.loads(required.read_text())
    name = 'preflight_finalize' if preflight else 'service_finalize'
    output = 'preflight_gate.json' if preflight else 'service_summary.json'
    args = ['--source', infra.remote(source_root()), '--root', infra.remote(HERE),
        '--out', infra.remote(HERE/output), '--wait']
    if preflight:
        args.append('--preflight')
    return infra.submit([infra.task(protocol(), name, args, runner='analyze.py', ram_mb=2048)], name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('stage', 'preflight', 'remaining', 'finalize', 'collect'))
    operation = parser.parse_args().operation
    if operation == 'stage':
        value = infra.stage([HERE/name for name in ('protocol.json', 'bootstrap.py', 'run.py',
            'service_trace.py', 'analyze.py', 'infra.py', 'launch.py', 'continue_batch.py')])
    elif operation == 'preflight': value = submit_preflight()
    elif operation == 'remaining': value = submit_remaining()
    elif operation == 'finalize': value = submit_finalizer()
    else: value = infra.collect('service_summary.json', max_bytes=30000)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
