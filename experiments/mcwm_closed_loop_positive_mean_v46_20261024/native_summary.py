"""Read existing v46 blocks to separate single-step level and paired noise.

No model is loaded or scored. Float64 reassembly uses already saved float32
model outputs; it is not a float64 model evaluation or a replacement target.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
ARMS = ('low', 'high')
COLUMNS = ['case', 'line', 'low_m1_mean', 'low_m1_std', 'low_m8_mean', 'low_m8_std',
    'high_m1_mean', 'high_m1_std', 'high_m8_mean', 'high_m8_std',
    'pair_m1_mean', 'pair_m1_std', 'pair_m8_mean', 'pair_m8_std',
    'pair_bootstrap64_m1_mean', 'pair_bootstrap64_m1_std',
    'pair_bootstrap64_m8_mean', 'pair_bootstrap64_m8_std']


def _moments(values):
    return [statistics.fmean(values), statistics.stdev(values)]


def _bootstrap64(row, positive_joint=None):
    if row['terminal']:
        return 0.
    joint = row['positive_joint'] if positive_joint is None else positive_joint
    sac = math.fsum((row['zero_probability']*row['zero_joint'],
                     row['hold_probability']*joint))
    return row.get('gamma', 1.)*math.fsum((.1*row['next_value'], .9*sac))


def _line_moments(blocks):
    levels = {arm: {'m1': [], 'm8': []} for arm in ARMS}
    pair = {'m1': [], 'm8': []}
    bootstrap = {'m1': [], 'm8': []}
    for data in blocks:
        raw = data['raw_scores']
        roots = {arm: [chain[0] for chain in raw[arm]['single']] for arm in ARMS}
        means = {arm: raw[arm]['mean8'][0] for arm in ARMS}
        if any(len(roots[arm]) != 8 for arm in ARMS):
            raise ValueError('All eight saved positive draws are required')
        for arm in ARMS:
            levels[arm]['m1'].extend(row['native_target'] for row in roots[arm])
            levels[arm]['m8'].append(means[arm]['native_target'])
        for index in range(8):
            low, high = roots['low'][index], roots['high'][index]
            pair['m1'].append(high['native_target']-low['native_target'])
            bootstrap['m1'].append(_bootstrap64(high)-_bootstrap64(low))
        pair['m8'].append(means['high']['native_target']-means['low']['native_target'])
        reassembled = {arm: _bootstrap64(means[arm],
            statistics.fmean(row['positive_joint'] for row in roots[arm])) for arm in ARMS}
        bootstrap['m8'].append(reassembled['high']-reassembled['low'])
    return [value for arm in ARMS for estimator in ('m1', 'm8')
            for value in _moments(levels[arm][estimator])] + [
        value for source in (pair, bootstrap) for estimator in ('m1', 'm8')
        for value in _moments(source[estimator])]


def _compact(value):
    if isinstance(value, float):
        return float(f'{value:.9g}')
    if isinstance(value, list):
        return [_compact(item) for item in value]
    if isinstance(value, dict):
        return {key: _compact(item) for key, item in value.items()}
    return value


def main():
    protocol = json.loads((HERE/'protocol.json').read_text())
    cases = [(scenario, checkpoint) for scenario in protocol['scenarios']
             for checkpoint in protocol['checkpoints']]
    rows = []
    for case_index, (scenario, checkpoint) in enumerate(cases):
        root = HERE/'runs'/scenario/checkpoint
        source = json.loads((root/'summary.json').read_text())
        if source['status'] != 'complete' or not source['proof']['all_pass']:
            raise ValueError('The completed v46 scoring shard is required')
        for line_index, line in enumerate(protocol['lines']):
            def read_blocks():
                for declared in protocol['blocks']:
                    path = root/'detail'/f"{line}_block{declared['block_id']:02d}.json"
                    data = json.loads(path.read_text())
                    if data['block'] != declared:
                        raise ValueError('Resident block identifiers or declared seeds differ')
                    yield data
            rows.append([case_index, line_index, *_line_moments(read_blocks())])
    value = _compact(dict(status='complete', priority='high', columns=COLUMNS, rows=rows,
        cases=[f'{scenario}/{checkpoint}' for scenario, checkpoint in cases],
        lines=protocol['lines'], block_count=protocol['block_count'],
        m1_draws=protocol['block_count']*protocol['draws_per_block'],
        m8_scores=protocol['block_count'], std_ddof=1, direction='high_minus_low',
        source='existing_v46_raw_scores_root_event_only',
        bootstrap64='float64_reassembly_of_saved_float32_parts_with_same8_positive_joint_draws_not_native64_model',
        preserved_contract=dict(initializer_updates=protocol['initializer_updates'],
            feedback_slots=protocol['source_feedback_slots'], updates=protocol['source_optimizer_updates'],
            horizon_seconds=protocol['horizon_seconds'], F_unchanged=True, trust_unchanged=True),
        new_budget=dict(scoring_calls=0, training_updates=0, feedback_slots=0, native_branches=0),
        scope='estimator_stability_only_no300s_truth_or_policy_benefit'))
    encoded = json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n'
    size = len(encoded.encode())
    if size > 5000:
        raise RuntimeError(f'Native-only summary is {size} bytes; limit is5000')
    out = HERE/'analysis'
    out.mkdir(exist_ok=True)
    with (out/'native_summary.json').open('x') as handle:
        handle.write(encoded)
    print(json.dumps(dict(status='complete', bytes=size, rows=len(rows))), flush=True)


if __name__ == '__main__':
    main()
