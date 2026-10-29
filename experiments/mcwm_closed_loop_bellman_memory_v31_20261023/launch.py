"""High-priority v31 native pairs, with all three children running in parallel."""
import argparse
import json
from pathlib import Path
import infra

HERE = Path(__file__).resolve().parent


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def stage():
    names = ('bootstrap.py','infra.py','run.py','fork_state.py','suffix.py','visibility.py','bellman.py','analyze.py',
        'launch.py','protocol.json','selected_reference.json')
    return infra.stage([HERE/name for name in names])


def submit_pairs():
    p = protocol()
    rows = []
    for seed in p['training_seeds']:
        for scenario in p['native_pair_scenarios']:
            args = ['--seed',seed,'--scenario',scenario,'--out',
                infra.remote(HERE/'runs'/f'seed{seed}'/scenario)]
            row = infra.task(p,f'pair/{seed}/{scenario}',args,runner='run.py',ram_mb=24000)
            row['cpu'] = 3
            row['cpu_training_justification'] = (
                'One exact factual parent, paused while three native fork children run concurrently; '
                'unchanged285 feedback slots/456 main updates; frozen Bellman/memory inference only, '
                'diagnostic truth never admitted.')
            rows.append(row)
    return infra.submit(rows,'paired')


def submit_finalize():
    return infra.submit([infra.task(protocol(),'truth/finalize',
        ['--root',infra.remote(HERE),'--out',infra.remote(HERE/'truth_summary.json')],
        runner='analyze.py',ram_mb=2048)],'finalize')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('stage','pairs','finalize','collect'))
    op = parser.parse_args().operation
    value = stage() if op == 'stage' else submit_pairs() if op == 'pairs' \
        else submit_finalize() if op == 'finalize' \
        else infra.collect('truth_summary.json',max_bytes=protocol()['summary_max_bytes'])
    print(json.dumps(value,ensure_ascii=False,indent=2))
