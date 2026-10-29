"""Two concurrent ordinary-day prechecks; no new control branches or learning."""
import argparse
import json
from pathlib import Path
import infra

HERE = Path(__file__).resolve().parent


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def stage():
    names = ('bootstrap.py', 'infra.py', 'run.py', 'memory.py', 'critic.py',
             'probes.py', 'analyze.py', 'launch.py', 'protocol.json')
    return infra.stage([HERE/name for name in names])


def submit_days():
    p = protocol()
    rows = []
    for seed in p['training_seeds']:
        for scenario in p['scenarios']:
            row = infra.task(p, f'coverage/{seed}/{scenario}', ['--seed', seed,
                '--scenario', scenario, '--arm', 'fresh', '--out',
                infra.remote(HERE/'runs'/f'seed{seed}'/scenario)], ram_mb=12000)
            rows.append(row)
    return infra.submit(rows, 'paired')


def submit_finalize():
    return infra.submit([infra.task(protocol(), 'coverage/finalize',
        ['--root', infra.remote(HERE), '--out', infra.remote(HERE/'coverage_summary.json')],
        runner='analyze.py', ram_mb=2048)], 'finalize')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('stage', 'days', 'finalize', 'collect'))
    op = parser.parse_args().operation
    value = stage() if op == 'stage' else submit_days() if op == 'days' else \
        submit_finalize() if op == 'finalize' else \
        infra.collect('coverage_summary.json', max_bytes=protocol()['summary_max_bytes'])
    print(json.dumps(value, ensure_ascii=False, indent=2))
