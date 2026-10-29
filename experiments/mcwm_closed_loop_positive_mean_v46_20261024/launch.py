"""Four high-priority frozen scoring shards; no learning or native launch."""
import argparse
import json
from pathlib import Path
import infra

HERE = Path(__file__).resolve().parent


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def stage():
    return infra.stage([HERE/name for name in ('protocol.json', 'README.md', 'infra.py',
        'estimator.py', 'analysis.py', 'summary.py', 'run.py', 'analyze.py',
        'launch.py', 'continue_batch.py', 'native_summary.py')])


def shards():
    p = protocol()
    rows = [infra.task(p, 'frozen/'+case+'/'+checkpoint,
        ['--scenario', case, '--checkpoint', checkpoint])
        for case in p['scenarios'] for checkpoint in p['checkpoints']]
    return infra.submit(rows, 'frozen')


def finalize():
    return infra.submit([infra.task(protocol(), 'small-summary', runner='analyze.py',
        ram_mb=2000)], 'finalize')


def native_summary():
    return infra.submit([infra.task(protocol(), 'native-level-summary',
        runner='native_summary.py', ram_mb=2000)], 'native')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('stage', 'shards', 'finalize', 'collect',
                                              'native', 'collect-native'))
    op = parser.parse_args().operation
    value = (stage() if op == 'stage' else shards() if op == 'shards' else
        finalize() if op == 'finalize' else native_summary() if op == 'native' else
        infra.collect('analysis/native_summary.json', max_bytes=5000) if op == 'collect-native' else
        infra.collect('analysis/summary.json', max_bytes=protocol()['summary_max_bytes']))
    print(json.dumps(value, ensure_ascii=False, indent=2))
