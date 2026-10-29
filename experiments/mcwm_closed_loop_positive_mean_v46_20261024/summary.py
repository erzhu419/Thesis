"""Bounded numeric exports of the resident fixed-draw target diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

from .analysis import METRICS
from experiments.mcwm_closed_loop_trace_components_v45_20261024.components import COMPONENTS


HERE = Path(__file__).resolve().parent
LINE_NAMES = ('7X', '102X', '705X')
ESTIMATORS = ('m1', 'm8')
ARMS = ('low', 'high')
STAT_METRICS = ('trace_target_difference', 'native_target_difference',
                'bootstrap_minus_q', 'correction_difference')
STAT_FIELDS = ('mean', 'sample_std', 'min', 'max', 'positive', 'negative', 'zero',
               'dominant_sign', 'dominant_sign_fraction')
COMPARISON_FIELDS = ('m8_minus_m1_mean', 'm8_over_m1_sample_std',
    'block_mean_assembly_difference_mean', 'block_mean_assembly_difference_max_abs')
COVARIANCE_FIELDS = ('q_variance', 'entropy_variance', 'covariance', 'correlation',
                     'joint_variance', 'variance_identity_residual')
FOCUS_BLOCK_COLUMNS = ('block_id', 'm1_trace_mean', 'm1_trace_sample_std',
    'm1_positive', 'm1_negative', 'm8_trace', 'm1_residual_mean', 'm8_residual')


def compact(value):
    """Retain nine significant figures; all exactness decisions precede export."""
    if isinstance(value, float):
        return float(f'{value:.9g}')
    if isinstance(value, (list, tuple)):
        return [compact(item) for item in value]
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()}
    return value


def encode_bounded(value, max_bytes):
    encoded = json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n'
    size = len(encoded.encode())
    if size > max_bytes:
        raise RuntimeError(f'Compact resident summary is {size} bytes; limit is {max_bytes}')
    return encoded


def compact_result(result):
    """Return one <=12KB shard; full per-line banks remain on the server.

    Required input is run.py's ``result_stats.json`` object: scenario,
    checkpoint, lines (full analyze_bank output plus arm_stats), proof and
    wall_seconds.  The 705X16-block table is retained; the other two lines keep
    their whole-bank moments and sign counts.
    """
    protocol = json.loads((HERE/'protocol.json').read_text())
    metric_rows, component_rows, comparison_rows, covariance_rows = [], [], [], []
    arm_rows, sign_rows = [], []
    for line_index, line in enumerate(LINE_NAMES):
        bank = result['lines'][line]
        if (bank['block_count'] != protocol['block_count']
                or bank['draws_per_block'] != protocol['draws_per_block']):
            raise ValueError('Export requires all predeclared fixed-draw blocks')
        for estimator_index, estimator in enumerate(ESTIMATORS):
            source = bank[estimator]
            for metric_index, metric in enumerate(STAT_METRICS):
                stats = source['metrics'][metric]
                metric_rows.append([line_index, estimator_index, metric_index,
                                    *[stats[name] for name in STAT_FIELDS]])
            covariance_rows.append([line_index, estimator_index,
                *[source['q_entropy_covariance'][name] for name in COVARIANCE_FIELDS]])
            for arm_index, arm in enumerate(ARMS):
                stats = bank['arm_stats'][arm][estimator]
                arm_rows.append([line_index, arm_index, estimator_index,
                                 *[stats[name] for name in STAT_FIELDS]])
        for component_index, component in enumerate(COMPONENTS):
            component_rows.append([line_index, component_index,
                bank['m1']['components'][component]['mean'],
                bank['m1']['components'][component]['sample_std'],
                bank['m8']['components'][component]['mean'],
                bank['m8']['components'][component]['sample_std']])
        for metric_index, metric in enumerate(METRICS):
            comparison_rows.append([line_index, metric_index,
                *[bank['matched_estimators'][metric][name] for name in COMPARISON_FIELDS]])
        stability = bank['sign_stability']
        sign_rows.append([line_index, stability['m1_within_block_flip_count'],
            stability['m1_within_block_flip_fraction'],
            int(stability['m8_across_blocks_sign_flip'])])

    focus = result['lines'][protocol['focus_line']]['per_block']
    focus_indexes = [focus['columns'].index(name) for name in FOCUS_BLOCK_COLUMNS]
    value = compact(dict(schema='v46_frozen_mean_compact_v1',
        status='complete' if result['proof']['all_pass'] else 'failed',
        scenario=result['scenario'], checkpoint=result['checkpoint'], priority='high',
        labels=dict(lines=list(LINE_NAMES), estimators=list(ESTIMATORS), arms=list(ARMS),
                    metrics=list(STAT_METRICS), comparison_metrics=list(METRICS),
                    components=list(COMPONENTS)),
        columns=dict(metric=['line', 'estimator', 'metric', *STAT_FIELDS],
            component=['line', 'component', 'm1_mean', 'm1_sample_std', 'm8_mean', 'm8_sample_std'],
            comparison=['line', 'metric', *COMPARISON_FIELDS],
            covariance=['line', 'estimator', *COVARIANCE_FIELDS],
            arm=['line', 'arm', 'estimator', *STAT_FIELDS],
            sign=['line', 'm1_within_block_flip_count', 'm1_within_block_flip_fraction',
                  'm8_across_blocks_sign_flip'], focus_block=list(FOCUS_BLOCK_COLUMNS)),
        bank=dict(block_count=protocol['block_count'], draws_per_block=protocol['draws_per_block'],
            m1_draws=protocol['block_count']*protocol['draws_per_block'],
            m8_scores=protocol['block_count'], focus_line=protocol['focus_line'],
            direction='high_minus_low', std_ddof=1),
        tables=dict(metric=metric_rows, component=component_rows,
            comparison=comparison_rows, covariance=covariance_rows, arm=arm_rows,
            sign=sign_rows, focus_block=[[row[index] for index in focus_indexes]
                                       for row in focus['rows']]),
        proof=result['proof'], wall_seconds=result['wall_seconds'],
        preserved_contract=dict(initializer_updates=protocol['initializer_updates'],
            training_seed=protocol['training_seed'], native_external_seed=protocol['native_external_seed'],
            source_feedback_slots=protocol['source_feedback_slots'],
            source_optimizer_updates=protocol['source_optimizer_updates'],
            horizon_seconds=protocol['horizon_seconds'], F_unchanged=True, trust_unchanged=True),
        new_budget=dict(training_updates=0, feedback_slots=0, native_branches=0,
            native_seconds=0, checkpoint_downloads=0),
        scope='fixed_draw_estimator_stability_not_accuracy_no300s_truth_no_independent_states',
        rounding='nine_significant_figures_for_export_only_exactness_uses_resident_values'))
    encode_bounded(value, protocol['summary_max_bytes_per_shard'])
    return value
