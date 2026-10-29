"""High-priority v37 collection, matched learning, and resident analysis."""
import argparse
import json
from pathlib import Path
import infra

HERE = Path(__file__).resolve().parent


def protocol():
    return json.loads((HERE / 'protocol.json').read_text())


def stage():
    names = ('bootstrap.py', 'infra.py', 'collect.py', 'difference.py', 'run.py',
             'analyze.py', 'launch.py', 'protocol.json', 'README.md')
    return infra.stage([HERE / name for name in names])


def submit_collect():
    rows = []
    for scenario in protocol()['scenarios']:
        row = infra.task(protocol(), 'collect/6202/' + scenario,
            ['--seed', 6202, '--scenario', scenario, '--out',
             infra.remote(HERE / 'runs' / 'collection' / scenario)],
            runner='collect.py', ram_mb=18000)
        row['cpu'] = 2
        rows.append(row)
    return infra.submit(rows, 'collection')


def submit_shadow(scenario):
    source = HERE / 'runs' / 'collection' / scenario
    row = infra.task(protocol(), 'triplet/6202/' + scenario,
        ['--seed', 6202, '--scenario', scenario,
         '--dataset', infra.remote(source / 'dataset.json'),
         '--heldout', infra.remote(source / 'heldout_prediction_inputs.json'),
         '--out', infra.remote(HERE / 'runs' / 'shadow' / scenario)], ram_mb=24000)
    row['cpu'] = 3
    return infra.submit([row], 'shadow_' + scenario)


def submit_finalize():
    row = infra.task(protocol(), 'paired-coverage/finalize',
        ['--root', infra.remote(HERE / 'runs'), '--out',
         infra.remote(HERE / 'paired_coverage_summary.json')], runner='analyze.py', ram_mb=2048)
    return infra.submit([row], 'finalize')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('stage', 'collect-days', 'finalize', 'fetch'))
    op = parser.parse_args().operation
    value = stage() if op == 'stage' else submit_collect() if op == 'collect-days' else \
        submit_finalize() if op == 'finalize' else infra.collect('paired_coverage_summary.json', max_bytes=24000)
    print(json.dumps(value, ensure_ascii=False, indent=2))
