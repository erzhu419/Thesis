"""Aggregate only four bounded resident shards; never load model weights."""
from __future__ import annotations

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments.mcwm_closed_loop_positive_mean_v46_20261024.summary import encode_bounded


SHARED_KEYS = ('schema', 'labels', 'columns', 'bank', 'preserved_contract',
               'new_budget', 'scope', 'rounding')


def aggregate_summaries(cases, protocol):
    """Combine ``{'scenario/checkpoint': compact_result(...)}`` without rescoring."""
    expected = [f'{scenario}/{checkpoint}' for scenario in protocol['scenarios']
                for checkpoint in protocol['checkpoints']]
    if set(cases) != set(expected):
        raise ValueError('All four predeclared scenario/checkpoint shards are required')
    structured = [cases[key] for key in expected if 'schema' in cases[key]]
    first = structured[0] if structured else None
    for key in expected:
        source = cases[key]
        scenario, checkpoint = key.split('/')
        if source['scenario'] != scenario or source['checkpoint'] != checkpoint:
            raise ValueError('Shard identity differs from its declared scenario/checkpoint')
        encode_bounded(source, protocol['summary_max_bytes_per_shard'])
        if 'schema' in source and any(
                source[name] != first[name] for name in SHARED_KEYS):
            raise ValueError('Resident shard export schemas or preserved contracts disagree')
    complete = all(cases[key]['status'] == 'complete' and cases[key]['proof']['all_pass']
                   for key in expected)
    value = dict(status='complete' if complete else 'failed', priority='high',
        **({name: first[name] for name in SHARED_KEYS} if first else {}),
        cases={key: {name: item for name, item in cases[key].items()
                      if name not in SHARED_KEYS} for key in expected},
        integrity=dict(all_pass=complete, completed_shards=sum(
            cases[key]['status'] == 'complete' for key in expected), expected_shards=len(expected)),
        decision=dict(training_started=False, target_change_installed=False,
            next_gate='Review matchedM1/M8 noise, signmix and remaining bootstrap-minus-Q mean; no automatic shadow training.'))
    encode_bounded(value, protocol['summary_max_bytes'])
    return value


def main():
    protocol = json.loads((HERE/'protocol.json').read_text())
    cases = {}
    for scenario in protocol['scenarios']:
        for checkpoint in protocol['checkpoints']:
            root = HERE/'runs'/scenario/checkpoint
            source = root/'summary.json'
            if source.exists():
                case = json.loads(source.read_text())
            else:
                case = dict(status='failed', scenario=scenario, checkpoint=checkpoint,
                            failure=json.loads((root/'failure.json').read_text()))
            cases[f'{scenario}/{checkpoint}'] = case
    value = aggregate_summaries(cases, protocol)
    encoded = encode_bounded(value, protocol['summary_max_bytes'])
    out = HERE/'analysis'
    out.mkdir(exist_ok=False)
    with (out/'summary.json').open('x') as handle:
        handle.write(encoded)
    print(json.dumps(dict(status=value['status'], integrity=value['integrity'],
        bytes=len(encoded.encode()))), flush=True)


if __name__ == '__main__':
    main()
