"""Small paired summaries; full event decompositions remain resident."""
import math
import numpy as np

from .trace import decompose_trace


def sign(value):
    return 1 if value > 0. else -1 if value < 0. else 0


def product_difference(low, high, value_key):
    """Symmetric identity, separating changed values from changed weights.

    These are event-indexed algebraic terms; the two actual clocks can differ.
    They must not be described as a common-clock causal reward comparison.
    """
    values, weights = [], []
    for left, right in zip(low['per_event'], high['per_event']):
        wl, wh = left['reach_weight'], right['reach_weight']
        vl, vh = left[value_key], right[value_key]
        values.append(.5*(wl+wh)*(vh-vl))
        weights.append(.5*(vl+vh)*(wh-wl))
    return dict(common_weight_value_difference=math.fsum(values),
                common_value_weight_difference=math.fsum(weights))


def paired_decomposition(low, high):
    left, right = low['root'], high['root']
    components = {key: right['components'][key]-left['components'][key]
                  for key in left['components']}
    delta = right['trace_target']-left['trace_target']
    reward_split = product_difference(low, high, 'reward_scaled')
    split = {key: right['bootstrap_split'][key]-left['bootstrap_split'][key]
             for key in left.get('bootstrap_split', {})}
    return dict(native_difference=right['native_target']-left['native_target'],
        trace_difference=delta, correction_difference=right['correction']-left['correction'],
        components=components, bootstrap_split=split,
        bootstrap_minus_matching_Q=components['bootstrap']+components['minus_matching_target_q'],
        reconstruction_residual=delta-math.fsum(components.values()),
        reward_value_vs_weight=reward_split,
        terminal_bootstrap_difference=(high['terminal_dependency']['weighted_bootstrap']-
                                       low['terminal_dependency']['weighted_bootstrap']),
        terminal_bootstrap_coefficient=[low['terminal_dependency']['coefficient'],
                                        high['terminal_dependency']['coefficient']],
        root_levels=[left['trace_target'], right['trace_target']],
        root_corrections=[left['correction'], right['correction']])


REPLICATE_COLUMNS = ['seed', 'native_high_minus_low', 'trace_high_minus_low',
    'correction_high_minus_low', 'reward_high_minus_low', 'bootstrap_high_minus_low',
    'minus_matching_Q_high_minus_low', 'terminal_bootstrap_high_minus_low',
    'common_weight_reward_difference', 'common_reward_weight_difference']
PREFIX_COLUMNS = ['links', 'low_elapsed_seconds', 'high_elapsed_seconds',
    'low_reach_weight', 'high_reach_weight', 'cumulative_reward_high_minus_low',
    'cumulative_bootstrap_minus_matching_Q_high_minus_low']


def summarize_checkpoint(chains_by_seed, seeds):
    """Return small output plus full resident decompositions for both lambdas."""
    detail, summaries = {}, []
    for seed in seeds:
        chains = chains_by_seed[str(seed)]['chains']
        indexed = {}
        for chain in chains:
            rows, ratios = chain['rows'], chain['c']
            zero = decompose_trace(rows, ratios, 0.)
            full = decompose_trace(rows, ratios, 1.)
            if not zero['lambda_zero_native_exact']:
                raise RuntimeError('Lambda0 changed a native target')
            indexed[(chain['line_id'], chain['arm'])] = dict(chain=chain, zero=zero, full=full)
        detail[str(seed)] = indexed
    first = detail[str(seeds[0])]
    lines = sorted({line for line, arm in first})
    for line in lines:
        replicates = []
        deltas = []
        for seed in seeds:
            bank = detail[str(seed)]
            low, high = [bank[(line, arm)]['full'] for arm in ('low', 'high')]
            value = paired_decomposition(low, high)
            deltas.append(value)
            split = value['reward_value_vs_weight']
            replicates.append([seed, value['native_difference'], value['trace_difference'],
                value['correction_difference'], value['components']['reward'],
                value['components']['bootstrap'], value['components']['minus_matching_target_q'],
                value['terminal_bootstrap_difference'], split['common_weight_value_difference'],
                split['common_value_weight_difference']])
        low, high = [first[(line, arm)]['full'] for arm in ('low', 'high')]
        source_chains = [first[(line, arm)]['chain'] for arm in ('low', 'high')]
        elapsed = [np.cumsum([row['dt'] for row in value['rows']]).tolist()
                   for value in source_chains]
        prefixes = []
        for index in (8, 16, 24, 32, 44):
            def acc(trace, field):
                return math.fsum(row[field] for row in trace['per_event'][:index+1])
            delta_r = acc(high, 'reward_root_contribution')-acc(low, 'reward_root_contribution')
            delta_b = (acc(high, 'bootstrap_root_contribution')+
                acc(high, 'minus_matching_q_root_contribution')-
                acc(low, 'bootstrap_root_contribution')-acc(low, 'minus_matching_q_root_contribution'))
            prefixes.append([index, elapsed[0][index], elapsed[1][index],
                low['reach_weights'][index], high['reach_weights'][index], delta_r, delta_b])
        all_traces = [detail[str(seed)][(line, arm)]['full'] for seed in seeds for arm in ('low', 'high')]
        summaries.append(dict(line_id=line, primary=deltas[0], replicates=replicates,
            trace_direction_consistent_across_seeds=len({sign(row['trace_difference']) for row in deltas}) == 1,
            trace_difference_range=[min(row['trace_difference'] for row in deltas),
                                    max(row['trace_difference'] for row in deltas)],
            native_difference_range=[min(row['native_difference'] for row in deltas),
                                     max(row['native_difference'] for row in deltas)],
            roots=[dict(native_target=value['root']['native_target'], trace_target=value['root']['trace_target'],
                        correction=value['root']['correction']) for value in (low, high)],
            elapsed_seconds=[value[-1] for value in elapsed],
            zero_duration_events=[value['zero_duration_events'] for value in (low, high)],
            terminal_bootstrap=[value['terminal_dependency'] for value in (low, high)],
            all_full_chain_supported=all(value['full_chain_supported'] for value in all_traces),
            missing_links=[dict(seed=seed, arm=arm, stop=detail[str(seed)][(line, arm)]['full']['first_stop'])
                for seed in seeds for arm in ('low', 'high')
                if not detail[str(seed)][(line, arm)]['full']['full_chain_supported']],
            max_reconstruction_residual=max(abs(value['root']['reconstruction_residual']) for value in all_traces),
            prefix_attribution=prefixes))
    # JSON keys stay plain strings; the full arrays never enter the local pull.
    resident = {seed: {line+'/'+arm: value for (line, arm), value in bank.items()}
                for seed, bank in detail.items()}
    return summaries, resident


def compact(value):
    if isinstance(value, float):
        return float(f'{value:.9g}')
    if isinstance(value, list):
        return [compact(item) for item in value]
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()}
    return value
