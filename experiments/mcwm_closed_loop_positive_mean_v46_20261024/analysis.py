"""Describe fixed M=1/M=8 scoring blocks without treating draws as new states.

Every block uses the same eight actor draws for both estimators.  M=8 is
assembled by the frozen target scorer, rather than by averaging root targets
here; their small discrepancy is reported explicitly.
"""
from __future__ import annotations

import math

import numpy as np

from experiments.mcwm_closed_loop_trace_components_v45_20261024.components import COMPONENTS


RESPONSE_COMPONENTS = ('mixed_value', 'sac_q_without_entropy', 'entropy',
                       'minus_recorded_q')
SPLIT_ROUNDING_COMPONENTS = ('sac_split_rounding', 'bootstrap_split_rounding')
ASSEMBLY_ROUNDING_COMPONENTS = ('native_rounding', 'recurrence_rounding')
METRICS = ('trace_target_difference', 'native_target_difference',
           'correction_difference', 'bootstrap_minus_q', 'reward',
           'split_rounding', 'assembly_rounding', 'reconstruction_residual')


def _values(pair):
    components = {name: float(pair['component_totals'][name]) for name in COMPONENTS}
    native, trace = float(pair['native_target_difference']), float(pair['trace_target_difference'])
    result = dict(trace_target_difference=trace, native_target_difference=native,
        correction_difference=trace-native,
        bootstrap_minus_q=math.fsum(components[name] for name in RESPONSE_COMPONENTS),
        reward=components['reward'],
        split_rounding=math.fsum(components[name] for name in SPLIT_ROUNDING_COMPONENTS),
        assembly_rounding=math.fsum(components[name] for name in ASSEMBLY_ROUNDING_COMPONENTS),
        reconstruction_residual=trace-math.fsum(components.values()))
    if not all(math.isfinite(value) for value in (*result.values(), *components.values())):
        raise ValueError('All fixed-bank target differences and components must be finite')
    return result, components


def _stats(values):
    array = np.asarray(values, dtype=np.float64)
    positive, negative, zero = (int((array > 0.).sum()), int((array < 0.).sum()),
                                int((array == 0.).sum()))
    counts = {1: positive, -1: negative, 0: zero}
    maximum = max(counts.values())
    winners = [sign for sign, count in counts.items() if count == maximum]
    return dict(n=len(array), mean=float(array.mean()),
        sample_std=float(array.std(ddof=1)) if len(array) > 1 else None,
        min=float(array.min()), max=float(array.max()),
        positive=positive, negative=negative, zero=zero,
        sign_flip=bool(positive and negative),
        dominant_sign=winners[0] if len(winners) == 1 else None,
        dominant_sign_fraction=maximum/len(array))


def _moments(values):
    array = np.asarray(values, dtype=np.float64)
    return dict(mean=float(array.mean()),
        sample_std=float(array.std(ddof=1)) if len(array) > 1 else None)


def _covariance(components):
    q = np.asarray([row['sac_q_without_entropy'] for row in components], dtype=np.float64)
    entropy = np.asarray([row['entropy'] for row in components], dtype=np.float64)
    q_variance, entropy_variance = float(q.var(ddof=1)), float(entropy.var(ddof=1))
    covariance = float(np.dot(q-q.mean(), entropy-entropy.mean())/(len(q)-1))
    joint_variance = float((q+entropy).var(ddof=1))
    denominator = math.sqrt(q_variance*entropy_variance)
    return dict(q_variance=q_variance, entropy_variance=entropy_variance,
        covariance=covariance, correlation=covariance/denominator if denominator else None,
        joint_variance=joint_variance,
        variance_identity_residual=joint_variance-q_variance-entropy_variance-2.*covariance,
        scope='paired_root_contribution_differences_not_independent_Q_and_entropy_draws')


def _summarize(rows, components):
    return dict(metrics={name: _stats([row[name] for row in rows]) for name in METRICS},
        components={name: _moments([row[name] for row in components]) for name in COMPONENTS},
        q_entropy_covariance=_covariance(components))


def analyze_bank(blocks):
    """Summarize declared blocks of eight paired M=1 and one paired M=8 target.

    ``blocks`` contains ``block_id``, ``single_pairs`` (eight v45 compare_pair
    results) and ``mean8_pair`` (the scorer's native float32 M=8 result).  This
    function does no scoring, sampling, model loading or endpoint selection.
    """
    blocks = list(blocks)
    if len(blocks) < 2:
        raise ValueError('At least two predeclared blocks are needed for across-block dispersion')
    if len({block['block_id'] for block in blocks}) != len(blocks):
        raise ValueError('Predeclared block identifiers must be unique')
    m1_rows, m1_components, m8_rows, m8_components = [], [], [], []
    block_data = []
    for block in blocks:
        if len(block['single_pairs']) != 8:
            raise ValueError('Each declared block must contain exactly eight M=1 paired scores')
        singles = [_values(pair) for pair in block['single_pairs']]
        mean8, mean8_components = _values(block['mean8_pair'])
        single_rows = [row for row, _ in singles]
        means = {name: math.fsum(row[name] for row in single_rows)/8 for name in METRICS}
        target_stats = _stats([row['trace_target_difference'] for row in single_rows])
        block_data.append(dict(block_id=block['block_id'], m1_mean=means,
            m8=mean8, m1_trace_stats=target_stats))
        m1_rows.extend(single_rows)
        m1_components.extend(component for _, component in singles)
        m8_rows.append(mean8)
        m8_components.append(mean8_components)

    m1, m8 = _summarize(m1_rows, m1_components), _summarize(m8_rows, m8_components)
    comparisons = {}
    for name in METRICS:
        original, averaged = m1['metrics'][name], m8['metrics'][name]
        differences = [row['m8'][name]-row['m1_mean'][name] for row in block_data]
        comparisons[name] = dict(m8_minus_m1_mean=averaged['mean']-original['mean'],
            m8_over_m1_sample_std=(averaged['sample_std']/original['sample_std']
                if original['sample_std'] else None),
            block_mean_assembly_difference_mean=float(np.mean(differences)),
            block_mean_assembly_difference_max_abs=max(abs(value) for value in differences))

    columns = ['block_id', 'm1_trace_mean', 'm1_trace_sample_std', 'm1_trace_min',
        'm1_trace_max', 'm1_positive', 'm1_negative', 'm1_zero', 'm1_sign_flip',
        'm8_trace', 'm8_native', 'm1_residual_mean', 'm8_residual',
        'm8_minus_m1_trace_mean', 'm1_reward_mean', 'm8_reward',
        'm8_split_rounding', 'm8_assembly_rounding', 'm8_reconstruction_residual']
    table = []
    for row in block_data:
        stats, averaged, means = row['m1_trace_stats'], row['m8'], row['m1_mean']
        table.append([row['block_id'], stats['mean'], stats['sample_std'], stats['min'],
            stats['max'], stats['positive'], stats['negative'], stats['zero'],
            int(stats['sign_flip']), averaged['trace_target_difference'],
            averaged['native_target_difference'], means['bootstrap_minus_q'],
            averaged['bootstrap_minus_q'], averaged['trace_target_difference']-stats['mean'],
            means['reward'], averaged['reward'], averaged['split_rounding'],
            averaged['assembly_rounding'], averaged['reconstruction_residual']])

    flips = sum(row['m1_trace_stats']['sign_flip'] for row in block_data)
    return dict(schema='v46_fixed_block_target_summary_v1', direction='high_minus_low',
        block_count=len(blocks), draws_per_block=8, underlying_actor_draws=len(m1_rows),
        m1_score_count=len(m1_rows), m8_score_count=len(m8_rows),
        m1=m1, m8=m8, matched_estimators=comparisons,
        sign_stability=dict(m1_within_block_flip_count=flips,
            m1_within_block_flip_fraction=flips/len(blocks),
            m8_across_blocks_sign_flip=m8['metrics']['trace_target_difference']['sign_flip']),
        per_block=dict(columns=columns, rows=table),
        accounting='trace_difference = reward + bootstrap_minus_q + split_rounding + assembly_rounding + reconstruction_residual',
        scope='fixed_states_and_predeclared_scoring_draws_estimator_stability_not_accuracy_or_independent_state_inference',
        no_300_second_truth=True, new_training_updates=0, new_native_branches=0)
