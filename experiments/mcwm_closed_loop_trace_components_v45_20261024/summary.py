"""Bounded component tables; full45-event arrays remain resident."""
import math
import numpy as np

from .components import COMPONENTS, crossed_variance

CACHE_COLUMNS = ['line', 'seed', 'native_delta', 'trace_delta', 'correction_delta', *COMPONENTS]
CELL_COLUMNS = ['actor_seed', 'head_seed', 'native_delta', 'trace_delta', *COMPONENTS]
EVENT_COLUMNS = ['event', 'low_start', 'high_start', 'low_end', 'high_end',
                 'low_dt', 'high_dt', 'total', *COMPONENTS]
MAIN_COMPONENTS = ('reward', 'mixed_value', 'sac_q_without_entropy', 'entropy', 'minus_recorded_q')


def component_row(pair):
    return [pair['component_totals'][key] for key in COMPONENTS]


def weight_value_split(low, high):
    """Symmetric weighted-difference identity; not common-clock causal cost."""
    result = {}
    for key in MAIN_COMPONENTS:
        value_terms, weight_terms = [], []
        for left, right in zip(low['per_event'], high['per_event']):
            wl, wh = left['reach_weight'], right['reach_weight']
            if wl == 0. or wh == 0.:
                raise ValueError('The current full-supported45-event source must have positive reach')
            vl, vh = left['components'][key]/wl, right['components'][key]/wh
            value_terms.append(.5*(wl+wh)*(vh-vl))
            weight_terms.append(.5*(vl+vh)*(wh-wl))
        result[key] = dict(common_weight_value_difference=math.fsum(value_terms),
                           common_value_weight_difference=math.fsum(weight_terms))
    return result


def pair_small(pair, low, high):
    components = pair['component_totals']
    event_index = {row['event']: row for row in pair['per_event']}
    largest = [event_index[row['event']] for row in pair['absolute_event_order'][:6]]
    return dict(native_delta=pair['native_target_difference'],
        trace_delta=pair['trace_target_difference'], correction_delta=pair['correction_difference'],
        components=components,
        rounding_sum=math.fsum(components[key] for key in COMPONENTS if 'rounding' in key),
        fixed_weight_component_deletion={key: pair['trace_target_difference']-components[key]
            for key in ('mixed_value', 'sac_q_without_entropy', 'entropy', 'minus_recorded_q')},
        component_deletion_scope='algebra_only_not_retrained_or_validated_target',
        fixed_bins=[[name, value['events'], value['total'],
            *[value['components'][key] for key in COMPONENTS]]
            for name, value in pair['fixed_bins'].items()],
        zero_duration=[[name, value['events'], value['total'],
            *[value['components'][key] for key in COMPONENTS]]
            for name, value in pair['zero_duration'].items()],
        largest_events=[[row['event'], row['low_started_at'], row['high_started_at'],
            row['low_completed_at'], row['high_completed_at'], row['low_dt'], row['high_dt'],
            row['total'], *[row['components'][key] for key in COMPONENTS]] for row in largest],
        weight_value_split=weight_value_split(low, high),
        arm_trace_targets=[low['trace_target'], high['trace_target']],
        reconstruction_residual=pair['reconstruction_residual'])


def checkpoint_small(cached_pairs, factor_pairs, seeds):
    """Summarize exact old diagonals and the full declared factor grid."""
    cache = []
    for line, bank in cached_pairs.items():
        for seed in seeds:
            pair = bank[str(seed)]['pair']
            cache.append([line, seed, pair['native_target_difference'], pair['trace_target_difference'],
                          pair['correction_difference'], *component_row(pair)])
    cells = []
    for actor in seeds:
        for head in seeds:
            pair = factor_pairs[f'{actor}/{head}']['pair']
            cells.append([actor, head, pair['native_target_difference'], pair['trace_target_difference'],
                          *component_row(pair)])
    metrics = {'trace_delta': lambda pair: pair['trace_target_difference']}
    metrics.update({key: (lambda pair, key=key: pair['component_totals'][key]) for key in MAIN_COMPONENTS})
    noise = {}
    for metric, getter in metrics.items():
        matrix = [[getter(factor_pairs[f'{actor}/{head}']['pair']) for head in seeds] for actor in seeds]
        item = crossed_variance(matrix)
        noise[metric] = (item if metric == 'trace_delta' else
            {key: item[key] for key in ('grand_mean', 'sums_of_squares', 'fractions',
                'actor_draw_fixed_subset_ranges', 'subset_fixed_actor_ranges')})
    first = factor_pairs[f'{seeds[0]}/{seeds[0]}']
    primary = pair_small(first['pair'], first['low'], first['high'])
    return dict(cached_context=cache, focus705X=primary, crossed_cells=cells,
        crossed_noise=noise,
        native_diagonal_pair_deltas_exact=all(
            factor_pairs[f'{seed}/{seed}']['pair']['trace_target_difference'] ==
            cached_pairs['705X'][str(seed)]['pair']['trace_target_difference'] for seed in seeds),
        head_order_numerical_max_abs=max(noise['trace_delta']['subset_fixed_actor_ranges']),
        actor_draw_range_at_fixed_head=noise['trace_delta']['actor_draw_fixed_subset_ranges'],
        grand_component_means={key: value['grand_mean'] for key, value in noise.items()})


def compact(value):
    if isinstance(value, float):
        return float(f'{value:.9g}')
    if isinstance(value, list):
        return [compact(item) for item in value]
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()}
    return value
