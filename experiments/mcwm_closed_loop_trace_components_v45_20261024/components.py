"""Additive fixed-chain accounting and crossed scoring-noise descriptions.

The original v44 float32 recurrence is the only source of trace targets.  This
module splits its already-scored bootstrap into V, SAC-Q and entropy, keeping
every assembly residual.  It neither samples models nor chooses an endpoint.
"""
from __future__ import annotations

import math

import numpy as np

from experiments.mcwm_closed_loop_trace_target_v44_20261024.trace import decompose_trace


COMPONENTS = (
    'reward', 'mixed_value', 'sac_q_without_entropy', 'entropy',
    'minus_recorded_q', 'sac_split_rounding', 'bootstrap_split_rounding',
    'native_rounding', 'recurrence_rounding',
)
FIXED_BINS = (('root0', 0, 1), ('events1_8', 1, 9), ('events9_16', 9, 17),
              ('events17_24', 17, 25), ('events25_44', 25, 45))


def _sum_entries(entries):
    entries = list(entries)
    components = {name: math.fsum(row['components'][name] for row in entries)
                  for name in COMPONENTS}
    return dict(events=len(entries), components=components,
                total=math.fsum(components.values()))


def _bins(events):
    return {name: dict(first_event=start, last_event=stop-1,
                       **_sum_entries(events[start:stop]))
            for name, start, stop in FIXED_BINS}


def decompose_components(rows, c, lambda_trace=1., *, mixed_path_weight=.9):
    """Split one predeclared45-event trace while retaining v44 target bits.

    Row indices are event ordinals; they are not aligned wall-clock intervals.
    ``sac_entropy_bonus`` is the cached float32 difference between the native
    entropy-inclusive SAC value and its entropy-free counterpart.  Its split
    residual is retained rather than mislabelled as an entropy contribution.
    """
    if len(rows) != 45:
        raise ValueError('The prescribed complete45-event chain is required')
    native = decompose_trace(rows, c, lambda_trace, mixed_path_weight=mixed_path_weight)
    events = []
    for index, (row, previous) in enumerate(zip(rows, native['per_event'])):
        weight = previous['reach_weight']
        discount = np.float32(row.get('gamma', 1.))*np.float32(not row['terminal'])
        multiplier = np.float32(discount*np.float32(mixed_path_weight))
        q_part = float(np.float32(multiplier*np.float32(row['sac_q_without_entropy'])))
        entropy_part = float(np.float32(multiplier*np.float32(row['sac_entropy_bonus'])))
        value, sac = float(row['value_contribution']), float(row['sac_contribution'])
        components = dict(
            reward=previous['reward_root_contribution'], mixed_value=weight*value,
            sac_q_without_entropy=weight*q_part, entropy=weight*entropy_part,
            minus_recorded_q=previous['minus_matching_q_root_contribution'],
            sac_split_rounding=weight*(sac-q_part-entropy_part),
            bootstrap_split_rounding=weight*(float(row['bootstrap_contribution'])-value-sac),
            native_rounding=previous['native_rounding_root_contribution'],
            recurrence_rounding=previous['recurrence_rounding_root_contribution'])
        events.append(dict(event=index, sequence=row.get('sequence'),
            started_at=row.get('started_at'), completed_at=row.get('completed_at'),
            dt=float(row['dt']), zero_duration=float(row['dt']) == 0.,
            reach_weight=weight, components=components,
            total=math.fsum(components.values()),
            raw=dict(next_value=float(row['next_value']), value_contribution=value,
                sac_q_without_entropy=float(row['sac_q_without_entropy']),
                sac_entropy_bonus=float(row['sac_entropy_bonus']),
                sac_contribution=sac,
                previous_link_recorded_q=previous['previous_link_matching_target_q'],
                bootstrap_contribution=float(row['bootstrap_contribution']),
                native_target=float(row['native_target']))))
    total = _sum_entries(events)
    root_target = native['root']['trace_target']
    return dict(event_count=45, lambda_trace=float(lambda_trace),
        mixed_path_weight=float(mixed_path_weight),
        native_target=native['root']['native_target'], trace_target=root_target,
        correction=native['root']['correction'], component_totals=total['components'],
        reconstructed_target=total['total'],
        reconstruction_residual=root_target-total['total'],
        v44_reconstruction_residual=native['root']['reconstruction_residual'],
        target_bits_exact_with_v44=True,
        lambda_zero_native_exact=native['lambda_zero_native_exact'],
        per_event=events, fixed_bins=_bins(events),
        zero_duration=dict(zero=_sum_entries(row for row in events if row['zero_duration']),
                           positive=_sum_entries(row for row in events if not row['zero_duration'])),
        links=native['links'], terminal_dependency=native['terminal_dependency'],
        grouping='fixed_event_ordinals_not_common_clock_alignment',
        endpoint_selection=False, new_training_updates=0)


def compare_pair(low, high):
    """High-minus-low root attribution on two complete, fixed real chains."""
    if low['event_count'] != 45 or high['event_count'] != 45:
        raise ValueError('Paired comparison requires two complete45-event chains')
    events = []
    for left, right in zip(low['per_event'], high['per_event']):
        if left['event'] != right['event']:
            raise ValueError('Comparison indexes must be the same event ordinals')
        components = {name: right['components'][name]-left['components'][name]
                      for name in COMPONENTS}
        duration_class = ('both_zero' if left['zero_duration'] and right['zero_duration']
            else 'low_only_zero' if left['zero_duration'] else
            'high_only_zero' if right['zero_duration'] else 'both_positive')
        events.append(dict(event=left['event'], low_dt=left['dt'], high_dt=right['dt'],
            low_started_at=left['started_at'], high_started_at=right['started_at'],
            low_completed_at=left['completed_at'], high_completed_at=right['completed_at'],
            low_reach_weight=left['reach_weight'], high_reach_weight=right['reach_weight'],
            duration_class=duration_class, components=components,
            total=math.fsum(components.values())))
    total = _sum_entries(events)
    target_delta = high['trace_target']-low['trace_target']
    # All45 entries remain available; ordering is a description, not truncation.
    ranked = sorted(events, key=lambda row: (-abs(row['total']), row['event']))
    running = 0.
    largest = []
    for rank, event in enumerate(ranked, 1):
        running = math.fsum((running, event['total']))
        largest.append(dict(rank=rank, event=event['event'], total=event['total'],
            cumulative_total=running, components=event['components'],
            low_dt=event['low_dt'], high_dt=event['high_dt']))
    return dict(direction='high_minus_low', event_count=45,
        native_target_difference=high['native_target']-low['native_target'],
        trace_target_difference=target_delta,
        correction_difference=high['correction']-low['correction'],
        component_totals=total['components'], reconstructed_difference=total['total'],
        reconstruction_residual=target_delta-total['total'],
        per_event=events, fixed_bins=_bins(events),
        zero_duration={name: _sum_entries(row for row in events if row['duration_class'] == name)
            for name in ('both_zero', 'low_only_zero', 'high_only_zero', 'both_positive')},
        absolute_event_order=largest,
        ordinal_pairing_is_not_common_clock_alignment=True, endpoint_selection=False)


def crossed_variance(matrix):
    """Orthogonal3actor x3head-subset descriptive sums of squares.

    Rows are independently held actor draws, columns independently held target
    subsets.  These nine scoring cells are not independent policy outcomes and
    no standard errors, hypothesis tests or population claims are returned.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (3, 3) or not np.isfinite(values).all():
        raise ValueError('The predeclared complete finite3actor x3subset matrix is required')
    grand = float(values.mean())
    actor = values.mean(axis=1)-grand
    subset = values.mean(axis=0)-grand
    interaction = values-grand-actor[:, None]-subset[None, :]
    actor_ss = float(3.*np.square(actor).sum())
    subset_ss = float(3.*np.square(subset).sum())
    interaction_ss = float(np.square(interaction).sum())
    total_ss = float(np.square(values-grand).sum())
    signs = np.sign(values).astype(int)
    parts = dict(actor=actor_ss, target_subset=subset_ss, interaction=interaction_ss)
    return dict(matrix=values.tolist(), grand_mean=grand,
        actor_main_effect=actor.tolist(), target_subset_main_effect=subset.tolist(),
        interaction=interaction.tolist(), sums_of_squares=dict(**parts, total=total_ss),
        sums_of_squares_residual=total_ss-math.fsum(parts.values()),
        fractions={name: (value/total_ss if total_ss else None) for name, value in parts.items()},
        actor_draw_fixed_subset_ranges=np.ptp(values, axis=0).tolist(),
        subset_fixed_actor_ranges=np.ptp(values, axis=1).tolist(),
        actor_draw_fixed_subset_sign_flip=[bool((signs[:, i] < 0).any() and
            (signs[:, i] > 0).any()) for i in range(3)],
        subset_fixed_actor_sign_flip=[bool((signs[i] < 0).any() and
            (signs[i] > 0).any()) for i in range(3)],
        positive_cells=int((values > 0.).sum()), negative_cells=int((values < 0.).sum()),
        zero_cells=int((values == 0.).sum()), cell_count=9,
        scope='descriptive_fixed_crossed_scoring_cells_not_independent_policy_evidence')


def factorial_summary(cells, metric='value'):
    """Convenience adapter for records with actor_seed, subset_seed and metric."""
    actors = sorted({cell['actor_seed'] for cell in cells})
    subsets = sorted({cell['subset_seed'] for cell in cells})
    if len(cells) != 9 or len(actors) != 3 or len(subsets) != 3:
        raise ValueError('Exactly nine crossed scoring cells are required')
    indexed = {(cell['actor_seed'], cell['subset_seed']): cell[metric] for cell in cells}
    if len(indexed) != 9:
        raise ValueError('Actor/subset cells must be unique')
    result = crossed_variance([[indexed[(actor, subset)] for subset in subsets] for actor in actors])
    return dict(actor_seeds=actors, subset_seeds=subsets, metric=metric, **result)
