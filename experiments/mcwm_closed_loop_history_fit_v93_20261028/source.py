"""Validate the frozen twelve-pair source and rebuild covariate designs only."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

LINES = ('7X', '102X', '705X')
SCENARIOS = ('stable', 'local_speed_drop')
BINS = ('lighter', 'heavier')
CASES = {(scene, line, group) for scene in SCENARIOS for line in LINES for group in BINS}
NEW_CASE = ('local_speed_drop', '705X', 'lighter')
DENOMINATOR, HOLD_SCALE = 15079., 7.5
ZERO_BUDGET = ('new_initialization_updates', 'prefix_training_updates', 'suffix_updates',
               'labels_fed_back', 'new_fits', 'new_neural_updates', 'v87_truth_used')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _condition(q, issue, origin):
    require(q['valid'] is True and all(q['checks'].values()) and q['status'] == 'supported'
        and q['historical_only'] is True and np.isfinite(q['value']), 'Invalid historical condition')
    require(q['issue_clock'] == issue and q['latest_completed_at'] < issue
        and q['latest_observed_at'] < issue and 0 < issue-q['latest_completed_at']
        == q['latest_age_seconds'] <= 900, 'Condition is not strictly available before issue')
    require(set(q['baseline_counts']) == set(LINES) and min(q['baseline_counts'].values()) >= 3
        and set(q['baseline_medians_seconds']) == set(LINES)
        and all(np.isfinite(value) and value > 0 for value in q['baseline_medians_seconds'].values()),
        'Missing three-line historical baseline')
    provenance = q['baseline_provenance']
    if origin == 'old_v90_fixed':
        require(provenance['source_schema'] == 'v91_fixed_compact'
            and 0 <= provenance['baseline_last_observed_at'] < 9000,
            'Old compact baseline provenance differs')
    else:
        require(provenance['source_schema'] == 'native_completion_full' and provenance['baseline_before'] == 9000
            and set(provenance['baseline_latest_completed_at']) == set(LINES)
            and set(provenance['baseline_latest_observed_at']) == set(LINES)
            and all(0 <= provenance['baseline_latest_completed_at'][line] < 9000
                and 0 <= provenance['baseline_latest_observed_at'][line] < 9000 for line in LINES),
            'New full baseline provenance differs')
    sources = q['latest_sources']
    require(bool(sources) and all(row['line_id'] in LINES
        and row['completed_at'] == q['latest_completed_at'] < issue
        and row['observed_at'] < issue for row in sources)
        and max(row['observed_at'] for row in sources) == q['latest_observed_at'],
        'Condition contains a future or untied source')
    normalized = [(row['nonholding_ready_seconds']-row['baseline_nonholding_seconds'])/60 for row in sources]
    require(all(np.isfinite(value) and value == row['normalized_slowdown']
                and row['baseline_nonholding_seconds'] == q['baseline_medians_seconds'][row['line_id']]
                for value, row in zip(normalized, sources))
            and sum(normalized)/len(normalized) == q['value'], 'Condition value does not match its sources')


def build_training(summary, frozen_protocol):
    """Reject incomplete evidence; preserve the source row order for all models."""
    require(summary['status'] == 'complete' and summary['completion_gate']['passed'] is True
        and summary['integrity']['all_pass'] is True and all(summary['integrity']['checks'].values())
        and all(summary['audit_checks'].values()) and all(summary['protocol_checks'].values())
        and all(summary['first_pair_checks'].values()), 'The frozen collection did not pass')
    require(all(not value for value in summary['exceptions'].values()), 'The collection contains explicit missing or invalid evidence')
    reference, collection, fitting = (frozen_protocol[key] for key in ('reference', 'collection', 'fitting'))
    require(reference['initializer_seed'] == 6207 and reference['training_external_seed'] == 716207
        and reference['scenarios'] == list(SCENARIOS), 'Different initialization or external seed')
    require(collection['native_load_thresholds'] == {'7X': .13, '102X': .5, '705X': .5}
        and collection['terminal'] == 24000 and collection['label_available_at'] == 24030
        and collection['logical_pairs'] == 12 and collection['logical_physical_outcome_labels'] == 24,
        'Different frozen cohort, strata or label clocks')
    require(set(fitting['models']) == {'line3', 'service5', 'service_history6'}
        and fitting['ridge'] == 1. and fitting['closed_form_solves'] == 3
        and fitting['data_availability_clock'] == 24030, 'Different fixed readout design')
    budget, new = summary['budget'], summary['budget']['new']
    require(new['prefixes'] == 1 and new['suffixes'] == new['labels_observed'] == 2
        and new['reused_pairs'] == 11 and new['reused_labels'] == 22
        and new['unknown_suffix_costs'] == 0 and new['native_prefix_seconds'] == 12900
        and new['total_native_seconds'] == new['native_prefix_seconds']+new['suffix_native_seconds'] == 35220
        and budget['historical_native_seconds'] == 314090 and budget['cumulative_native_seconds'] == 349310
        and budget['cumulative_native_seconds'] == 314090+new['total_native_seconds']
        and budget['new_cap'] == 36900 and budget['cumulative_cap'] == 350990
        and budget['old_logical_pairs'] == 11 and budget['new_pairs'] == 1
        and budget['observed_logical_pairs'] == budget['valid_logical_pairs'] == 12
        and all(new[key] == 0 for key in ZERO_BUDGET), 'Source native or information budget differs')
    declarations = {(row['scenario'], row['line_id'], row['stratum']): row
                    for row in collection['reused_roots']}
    require(len(declarations) == 11 and set(declarations) == CASES-{NEW_CASE}, 'The eleven old cases differ')
    source_rows = summary['pairs']
    require(len(source_rows) == 12, 'Exactly twelve paired roots are required')
    rows, cases, physical_roots, old_paths = [], [], [], []
    for source in source_rows:
        scene, line, group = source['scenario'], source['line_id'], source['load_bin']
        case = (scene, line, group)
        require(case in CASES and source['slot'] == line+'/'+group and source['external_seed'] == 716207,
                'Undeclared source case')
        require(source['valid'] is True and all(source['checks'].values()), 'An invalid pair cannot enter fitting')
        root = source['root']
        origin = 'new_completion' if case == NEW_CASE else 'old_v90_fixed'
        require(source['origin'] == origin, 'Old and new label origins differ')
        require(source['physical_root_id'] == [scene, 716207, root['sequence']], 'Physical root identity differs')
        require(12000 <= root['started_at'] < (12900 if origin == 'new_completion' else 12600)
            and source['dt'] == 24000-root['started_at'] > 0, 'Source root is outside its declared window')
        if origin == 'old_v90_fixed':
            declared = declarations[case]
            require(root == declared['root'] and source['condition']['value'] == declared['q']
                and source['source_pair_path'] and source['outcome_source_pair_path']
                and source['source_pair_path'].endswith('/'+declared['source']+'/pair_summary.json'),
                'A fixed old root, historical condition or source path changed')
            old_paths.append(source['source_pair_path'])
        else:
            require(source['source_pair_path'] is None and source['outcome_source_pair_path'] is None,
                    'The new label cannot be relabeled as a reused query')
        h_ref, h_cand = source['reference_hold_seconds'], source['candidate_hold_seconds']
        require(np.isfinite([h_ref, h_cand]).all() and 3. <= h_ref < 3.75
            and abs(h_cand-7.5) <= 1e-5 and h_cand > h_ref, 'The actual actions leave the common domain')
        features = np.asarray(source['service_features'], dtype=np.float64)
        require(features.shape == (2,) and np.isfinite(features).all()
            and 0 <= features[0] <= 1 and -1 <= features[1] <= 1
            and ('lighter' if features[0] < collection['native_load_thresholds'][line] else 'heavier') == group,
            'Native service features or frozen load stratum differ')
        _condition(source['condition'], root['started_at'], origin)
        delta, per = source['label_delta'], source['label_delta_per_passenger']
        require(all(np.isfinite([delta[key], per[key]]).all() and delta[key]/DENOMINATOR == per[key]
                    for key in ('wait', 'ride', 'cost'))
            and delta['cost'] == 2*delta['wait']+delta['ride'], 'Raw and per-passenger labels disagree')
        cases.append(case)
        physical_roots.append(tuple(source['physical_root_id']))
        rows.append(dict(id='/'.join(case), scenario=scene, line_id=line, load_bin=group,
            root=deepcopy(root), external_seed=716207,
            features=[*features.tolist(), float(source['condition']['value'])],
            a=(h_cand-h_ref)/HOLD_SCALE, target=float(per['cost']), raw_delta_person_seconds=float(delta['cost'])))
    require(set(cases) == CASES and len(set(cases)) == 12 and len(set(physical_roots)) == 12
        and len(set(old_paths)) == 11, 'Duplicate or missing paired roots')

    a = np.asarray([row['a'] for row in rows], dtype=np.float64)
    x = np.asarray([row['features'] for row in rows], dtype=np.float64)
    line_names = [row['line_id'] for row in rows]
    centers = {}
    for line in LINES:
        mask = np.asarray([value == line for value in line_names])
        weights = a[mask]**2
        service = ((weights[:, None]*x[mask, :2]).sum(axis=0)/weights.sum()).tolist()
        values = [row for row in rows if row['line_id'] == line]
        q = sum(row['a']*row['a']*row['features'][2] for row in values)/sum(row['a']*row['a'] for row in values)
        centers[line] = [*service, q]
    line3 = a[:, None]*np.asarray([[float(line == target) for target in LINES] for line in line_names])
    centered_service = np.asarray([feature[:2]-np.asarray(centers[line][:2])
                                   for feature, line in zip(x, line_names)])
    service5 = np.column_stack((line3, a[:, None]*centered_service))
    history6 = np.column_stack((service5, [row['a']*(row['features'][2]-centers[row['line_id']][2]) for row in rows]))
    matrices = dict(line3=line3, service5=service5, service_history6=history6)
    design = summary['covariate_design']
    ranks = {name: int(np.linalg.matrix_rank(matrix)) for name, matrix in matrices.items()}
    checks = dict(twelve_fixed_valid_pairs=True, fixed_source_metadata=True, mature_source_labels=True,
        old_eleven_and_new_one=True, source_budget_separate=True, raw_target_scaling=True,
        strict_historical_conditions=True, preserved_source_row_order=True,
        exact_source_matrix6=np.array_equal(history6, np.asarray(design['matrix6'], dtype=np.float64)),
        exact_service_centers=all(centers[line][:2] == design['service_line_centers'][line] for line in LINES),
        exact_q_centers=all(centers[line][2] == design['q_line_centers'][line] for line in LINES),
        fixed_ranks=ranks == {'line3': 3, 'service5': 5, 'service_history6': 6}
            and design['ranks'] == {'line': 3, 'service5': 5, 'service_q6': 6},
        centered_columns_orthogonal=bool(np.allclose(line3.T@history6[:, 3:], 0., rtol=0., atol=1e-12)))
    require(all(checks.values()), 'Source matrix or centering differs: '+', '.join(key for key, value in checks.items() if not value))
    return dict(rows=rows, matrices=matrices, line_centers=centers, checks=checks, all_pass=True)


def load_training(summary_path, frozen_protocol_path):
    summary_path, frozen_protocol_path = Path(summary_path).resolve(), Path(frozen_protocol_path).resolve()
    result = build_training(json.loads(summary_path.read_text()), json.loads(frozen_protocol_path.read_text()))
    result.update(source_summary=str(summary_path), source_protocol=str(frozen_protocol_path))
    return result
