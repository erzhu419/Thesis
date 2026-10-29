"""High-priority factual-outcome attribution pilot; no counterfactual tasks."""
import argparse
import json
from pathlib import Path
import infra

HERE = Path(__file__).resolve().parent


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def submit_pair(scenario):
    if scenario == 'local_speed_drop':
        gate = infra.collect('stable_gate.json', max_bytes=30000)
        if gate.get('integrity', {}).get('all_pass') is not True:
            raise RuntimeError('Stable native implementation gate must pass; no effect-sign selection')
    args = ['--seed', 6201, '--scenario', scenario,
        '--out', infra.remote(HERE/'runs'/scenario/'seed6201')]
    row = infra.task(protocol(), 'factual_pair/'+scenario, args,
        runner='run_pair.py', ram_mb=16000)
    row['cpu'] = 2
    row['cpu_training_justification'] = (
        'Two independent single-thread factual control days start before waiting; '
        'matched optimizer budget, no world or target action-branch queries.')
    return infra.submit([row], 'stable_pair' if scenario == 'stable' else 'shock_pair')


def submit_finalize(stage):
    output = 'stable_gate.json' if stage == 'stable' else 'pilot_summary.json'
    args = ['--root', infra.remote(HERE), '--stage', stage,
        '--out', infra.remote(HERE/output), '--wait']
    return infra.submit([infra.task(protocol(), stage+'/finalize', args,
        runner='analyze.py', ram_mb=2048)], stage+'_finalize')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('stage', 'stable', 'stable-finalize',
        'shock', 'pilot-finalize', 'collect-stable', 'collect-pilot'))
    args = parser.parse_args()
    if args.operation == 'stage':
        names = ('bootstrap.py', 'infra.py', 'learning.py', 'issue_audit.py', 'run.py',
            'run_pair.py', 'analyze.py', 'audit_summary.py', 'launch.py', 'protocol.json')
        value = infra.stage([HERE/name for name in names])
    elif args.operation == 'stable': value = submit_pair('stable')
    elif args.operation == 'shock': value = submit_pair('local_speed_drop')
    elif args.operation == 'stable-finalize': value = submit_finalize('stable')
    elif args.operation == 'pilot-finalize': value = submit_finalize('pilot')
    else: value = infra.collect('stable_gate.json' if args.operation == 'collect-stable'
                                else 'pilot_summary.json', max_bytes=30000)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
