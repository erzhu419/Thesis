"""Resident, small paired-action truth audit at training-period native states."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import numpy as np

ENDPOINTS = ('global_next', 'same_vehicle_next', 'fixed300')
METRICS = ('dt', 'wait', 'ride', 'cost')
MODELS = ('baseline', 'residual', 'interaction', 'separated', 'separated_no_cost', 'innovation', 'innovation_line')
GLOBAL_MODELS = ('base_analytic', 'global58', *MODELS)
EFFECT_FIELDS = ('predicted_gain', 'gain_error', 'direction_correct', 'action_correct', 'regret')
SUMMARY_FIELDS = ('n', 'gain_mae', 'gain_rmse', 'direction_correct', 'action_correct', 'predicted_ties', 'regret_total')
MAX_BYTES = 100000
PREFLIGHT_MAX_BYTES = 20000


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def compact(x):
    if isinstance(x, float):
        return float(format(x, '.8g'))
    if isinstance(x, dict):
        return {key: compact(value) for key, value in x.items()}
    if isinstance(x, (list, tuple)):
        return [compact(value) for value in x]
    return x


def complete_truth(value, endpoint):
    row = value.get('truth', {}).get(endpoint)
    return row if isinstance(row, dict) and row.get('status') == 'complete' and all(
        finite(row.get(key)) for key in METRICS) else None


def prediction_endpoint(prediction, model, endpoint, action):
    if model in ('base_analytic', 'global58'):
        if endpoint != 'global_next':
            return None
        return prediction.get('global_comparator', {}).get('arms', {}).get(model, {}).get(action)
    return prediction.get(model, {}).get(action, {}).get('endpoints', {}).get(endpoint)


def gain(prediction, model, endpoint, *, raw=False):
    if model in ('base_analytic', 'global58'):
        if endpoint != 'global_next':
            return None
        name = 'raw_cost_gain_low_minus_high' if raw else 'cost_gain_low_minus_high'
        x = prediction.get('global_comparator', {}).get('arms', {}).get(model, {}).get(name)
    else:
        arm = prediction.get(model, {})
        x = arm.get('raw_global_next_gain') if raw else arm.get('gains', {}).get(endpoint)
    return x if finite(x) else None


def direction(x):
    return 1 if x > 0 else -1 if x < 0 else 0


def effect(predicted, actual):
    if not finite(predicted) or not finite(actual):
        return None
    regret = max(0., -actual) if predicted > 0 else max(0., actual)
    return [predicted, predicted-actual, direction(predicted) == direction(actual), regret == 0., regret]


def moments(xs):
    return dict(n=len(xs), mean=sum(xs)/len(xs) if xs else None,
        mean_absolute=sum(abs(x) for x in xs)/len(xs) if xs else None,
        rms=math.sqrt(sum(x*x for x in xs)/len(xs)) if xs else None)


def effect_summary(values):
    n = len(values)
    return [n, sum(abs(row[1]) for row in values)/n if n else None,
        math.sqrt(sum(row[1]**2 for row in values)/n) if n else None,
        sum(row[2] for row in values), sum(row[3] for row in values),
        sum(row[0] == 0 for row in values), sum(row[4] for row in values) if n else None]


def load_result(path):
    if (path/'summary.json').is_file():
        return json.loads((path/'summary.json').read_text())
    if (path/'failure.json').is_file():
        value = json.loads((path/'failure.json').read_text())
        return dict(status='failed', failure={key: value.get(key) for key in ('type', 'message', 'time')})
    return dict(status='missing')


def branch_path(root, state, action, *, preflight=False):
    if preflight:
        return Path(root)/'pair'/action
    cell = state['cell']
    return Path(root)/'pairs'/cell['scenario']/cell['target_line']/f"seed{cell['seed']}"/cell['window_name']/action


def zero_training(value):
    delta = value.get('suffix', {}).get('training_delta')
    return isinstance(delta, dict) and bool(delta) and all(finite(x) and x == 0 for x in delta.values())


def background_snapshot(context, endpoint):
    snap = context.get('innovation_background', {})
    row = snap.get('endpoints', {}).get(endpoint, {})
    mean = row.get('mean')
    names = ('wait', 'ride') if endpoint == 'fixed300' else ('dt', 'wait', 'ride')
    physical = None
    if isinstance(mean, list) and len(mean) == len(names) and all(finite(x) for x in mean):
        physical = dict(dt=0., wait=0., ride=0.)
        physical.update({name: x*(300. if name == 'dt' else 10000.) for name, x in zip(names, mean)})
        physical['cost'] = 2*physical['wait']+physical['ride']
    return dict(status=row.get('status', 'missing'), issued_at=snap.get('issued_at'),
        fit_rows=row.get('fit_rows'), max_available_at=row.get('max_available_at'),
        max_sequence=row.get('max_sequence'), physical_mean=physical)


def compare_source(low, high, state):
    source = low.get('source', {})
    hold = low.get('source_original_hold_seconds', source.get('nominal_hold_seconds', state.get('nominal_hold_seconds')))
    if not finite(hold):
        a = state.get('nominal_a')
        hold = 30*(1.+a[0]) if isinstance(a, list) and a and finite(a[0]) else None
    chosen = 'low' if finite(hold) and abs(hold) < 1e-5 else 'high' if finite(hold) and abs(hold-7.5) < 1e-5 else None
    result = dict(nominal_action=chosen, nominal_hold_seconds=hold, endpoints={})
    original = source.get('truth', state.get('source_truth', {}))
    frozen = low if chosen == 'low' else high if chosen == 'high' else {}
    for endpoint in ENDPOINTS:
        a, b = original.get(endpoint, {}), complete_truth(frozen, endpoint)
        ok = a.get('status') == 'complete' and all(finite(a.get(key)) for key in METRICS) and b is not None
        result['endpoints'][endpoint] = dict(status='complete' if ok else 'missing',
            source=[a[key] for key in METRICS] if ok else None,
            frozen_minus_source=[b[key]-a[key] for key in METRICS] if ok else None)
    return result


def analyze_pair(state, low, high):
    cell = state['cell']
    result = dict(state_id=state['state_id'], **{key: cell[key] for key in ('seed', 'scenario', 'target_line', 'window_name')},
        selection_status=state.get('status'), branch_status={name: value.get('status') for name, value in (('low', low), ('high', high))})
    if state.get('status') != 'selected':
        checks = dict(selection_missing_retained=low.get('status') == high.get('status') == 'selection_missing')
        result.update(status='selection_missing', integrity=dict(all_pass=all(checks.values()), checks=checks),
            truth={endpoint: dict(status='missing', branches=dict(low=None, high=None), gain=None) for endpoint in ENDPOINTS},
            effects={endpoint: {model: None for model in (GLOBAL_MODELS if endpoint == 'global_next' else MODELS)} for endpoint in ENDPOINTS},
            budget={name: value.get('budget', {}) for name, value in (('low', low), ('high', high))}, duration='missing_selection')
        return result
    prediction = low.get('prediction', {})
    prefix = low.get('prefix', {})
    selected = low.get('selected')
    source = state.get('source', {})
    both = (low, high)
    complete = all(value.get('status') == 'complete' for value in both)
    checks = dict(branches_complete=complete,
        branch_integrity=complete and all(value.get('integrity', {}).get('all_pass') is True for value in both),
        full_native_prefix_exact=all(value.get('integrity', {}).get('checks', {}).get('full_native_prefix_exact') is True for value in both),
        common_state=bool(selected) and selected == high.get('selected') == state.get('selected'),
        common_prefix=bool(prefix) and prefix == high.get('prefix'),
        v14_prediction_exact=bool(prediction) and prediction == high.get('prediction') == state.get('prediction'),
        common_context=bool(low.get('context')) and low.get('context') == high.get('context'),
        same_external_seed=prefix.get('external_seed') is not None and all(value.get('prefix', {}).get('external_seed') == source.get('external_seed') for value in both),
        initializer_unchanged=all(value.get('prefix', {}).get('initial_updates') == source.get('initializer_updates') for value in both),
        suffix_zero_training=all(zero_training(value) for value in both),
        suffix_exact300=all(value.get('suffix', {}).get('seconds') == 300. for value in both),
        actual_low_high_commands=all(value.get('execution', {}).get('requested_hold_seconds') == expected
            and value.get('execution', {}).get('executed_hold_seconds') == expected for value, expected in ((low, 0.), (high, 7.5))),
        same_batch_other_actions=low.get('native', {}).get('other_batch_actions') is not None
            and low.get('native', {}).get('other_batch_actions') == high.get('native', {}).get('other_batch_actions'),
        no_diagnostic_truth_training=all(value.get('budget', {}).get('truth_labels_enter_training') == 0 for value in both),
        prefix_budget_exact=all(value.get('budget', {}).get('replay_target_labels') == source.get('prefix_target_labels')
            and value.get('budget', {}).get('replay_policy_updates') == source.get('prefix_policy_updates') for value in both),
        prefix_clock_exact=all(value.get('budget', {}).get('prefix_sim_seconds') == state['selected']['started_at'] for value in both))
    clock_predictions = [prediction_endpoint(prediction, model, 'fixed300', action) for model in MODELS for action in ('low', 'high')]
    checks['predicted_fixed300_clock_exact'] = all(row is None or row.get('dt') == 300. for row in clock_predictions)
    truth, effects = {}, {}
    for endpoint in ENDPOINTS:
        actual = {name: complete_truth(value, endpoint) for name, value in (('low', low), ('high', high))}
        ok = all(row is not None for row in actual.values())
        differences = {key: actual['low'][key]-actual['high'][key] for key in METRICS} if ok else None
        truth[endpoint] = dict(status='complete' if ok else 'missing',
            branches={name: [row[key] for key in METRICS] if row is not None else None for name, row in actual.items()},
            gain=differences['cost'] if ok else None, low_minus_high=[differences[key] for key in METRICS] if ok else None,
            missing_reasons={name: value.get('truth', {}).get(endpoint, {}).get('reason') for name, value in (('low', low), ('high', high)) if actual[name] is None})
        effects[endpoint] = {model: effect(gain(prediction, model, endpoint), truth[endpoint]['gain'])
            for model in (GLOBAL_MODELS if endpoint == 'global_next' else MODELS)}
    fixed = truth['fixed300']
    checks['observed_fixed300_clock_exact'] = all(row is None or row[0] == 300. for row in fixed['branches'].values())
    context = low.get('context', {})
    snapshots = {name: background_snapshot(context, name) for name in ENDPOINTS}
    issued_at = state['selected']['started_at']
    sequence = state['selected']['sequence']
    checks['background_snapshot_scope'] = all(row['issued_at'] == issued_at and row['physical_mean'] is not None
        and (row['max_available_at'] is None or row['max_available_at'] <= issued_at)
        and (row['max_sequence'] is None or row['max_sequence'] < sequence) for row in snapshots.values())
    fixed_levels, residuals = {}, {}
    for model in MODELS:
        fixed_levels[model] = [prediction_endpoint(prediction, model, 'fixed300', action) for action in ('low', 'high')]
        fixed_levels[model] = [row['cost']-actual[3] if isinstance(row, dict) and finite(row.get('cost')) and actual is not None else None
            for row, actual in zip(fixed_levels[model], (fixed['branches']['low'], fixed['branches']['high']))]
    for endpoint in ENDPOINTS:
        rows = []
        for action in ('low', 'high'):
            actual = truth[endpoint]['branches'][action]
            base = prediction_endpoint(prediction, 'baseline', endpoint, action)
            m = snapshots[endpoint]['physical_mean']
            ok = actual is not None and isinstance(base, dict) and finite(base.get('cost')) and m is not None
            rows.append([base['cost'], m['cost'], actual[3]-base['cost'], actual[3]-base['cost']-m['cost']] if ok else None)
        residuals[endpoint] = rows
    gl = truth['global_next']['branches']
    duration = 'missing_global' if any(row is None for row in gl.values()) else 'zero_dt' if gl['low'][0] == gl['high'][0] == 0 else 'positive_dt' if gl['low'][0] > 0 and gl['high'][0] > 0 else 'mixed_dt'
    result.update(status='complete' if complete else 'missing_or_failed', integrity=dict(all_pass=all(checks.values()), checks=checks),
        selected={key: state['selected'][key] for key in ('sequence', 'started_at', 'line_id', 'vehicle_id', 'station_id')},
        prefix={key: prefix.get(key) for key in ('initial_updates', 'external_seed', 'native_steps', 'issued_count', 'counters')},
        truth=truth, effects=effects, duration=duration,
        raw_global_effects={model: effect(gain(prediction, model, 'global_next', raw=True), truth['global_next']['gain']) for model in GLOBAL_MODELS},
        global_to_fixed300_proxy={model: effect(gain(prediction, model, 'global_next'), truth['fixed300']['gain']) for model in ('base_analytic', 'global58')},
        fixed300_level_errors=fixed_levels, background=snapshots, background_residuals=residuals,
        assignment={key: context.get(key) for key in ('assignment_position', 'assignment_success_index', 'assignment_block_index')},
        coverage=low.get('coverage', {}), source_vs_frozen_nominal=compare_source(low, high, state),
        suffix_training_delta={name: value.get('suffix', {}).get('training_delta') for name, value in (('low', low), ('high', high))},
        execution={name: {key: value.get('execution', {}).get(key) for key in ('requested_hold_seconds', 'executed_hold_seconds')} for name, value in (('low', low), ('high', high))},
        budget={name: value.get('budget', {}) for name, value in (('low', low), ('high', high))})
    return result


def summarize(pairs, endpoint):
    models = GLOBAL_MODELS if endpoint == 'global_next' else MODELS
    valid = [row for row in pairs if row['selection_status'] == 'selected' and row['integrity']['all_pass']]
    common = [row for row in valid if all(row['effects'][endpoint][model] is not None for model in models)]
    return dict(design_n=len(pairs), selected_valid_n=len(valid), missing_truth=sum(row['truth'][endpoint]['gain'] is None for row in valid),
        model_metrics={model: effect_summary([row['effects'][endpoint][model] for row in valid if row['effects'][endpoint][model] is not None]) for model in models},
        common_n=len(common), common_state_ids=[row['state_id'] for row in common],
        common_metrics={model: effect_summary([row['effects'][endpoint][model] for row in common]) for model in models})


def signal_summary(pairs):
    valid = [row for row in pairs if row['selection_status'] == 'selected' and row['integrity']['all_pass'] and row['truth']['fixed300']['gain'] is not None]
    actual = [row['truth']['fixed300']['gain'] for row in valid]
    centered = [value[3] for row in valid for value in row.get('background_residuals', {}).get('fixed300', []) if value is not None]
    level = [value[2] for row in valid for value in row.get('background_residuals', {}).get('fixed300', []) if value is not None]
    return dict(true_gain=moments(actual), baseline_residual_cost=moments(level), saved_background_residual_cost=moments(centered),
        background_rms_over_mean_absolute_gain=(math.sqrt(sum(x*x for x in centered)/len(centered))/(sum(abs(x) for x in actual)/len(actual)))
            if centered and actual and sum(abs(x) for x in actual) else None)



def compact_coverage(pairs):
    """Store runtime coverage/rank scalars with shared column names."""
    lines = ('7X', '102X', '705X')
    count_fields = ('n', 'zero_holding', 'high_holding')
    diag_fields = ('status', 'background_rows', 'first_rows', 'second_rows',
                   'cold_start_first_rows', 'snapshot_scope_valid')
    design_fields = ('rows', 'columns', 'rank', 'active_columns', 'ridge', 'ridge_precision_condition')
    for pair in pairs:
        value = pair.get('coverage')
        if not value:
            continue
        output = {}
        for arm in ('innovation', 'innovation_line'):
            source = value.get(arm, {})
            rows = {}
            for endpoint in ENDPOINTS:
                coverage = source.get('endpoint_coverage', {}).get(endpoint, {})
                diagnostic = source.get('diagnostics', {}).get(endpoint, {})
                if diagnostic.get('first_coverage') is not None:
                    assert diagnostic['first_coverage'] == coverage
                rows[endpoint] = [
                    [[coverage[line].get(key) for key in count_fields] if line in coverage else None for line in lines],
                    [diagnostic.get(key) for key in diag_fields],
                    [diagnostic.get('action_design', {}).get(key) for key in design_fields],
                    [(diagnostic.get('innovation_error_rmse') or {}).get(key) for key in METRICS],
                    [(diagnostic.get('score_error_rmse') or {}).get(key) for key in METRICS]]
            output[arm] = rows
        pair['coverage'] = output
    return dict(line_order=lines, count_order=count_fields, endpoint_order=(
        'line_first_coverage', 'diagnostic_scalars', 'action_design', 'innovation_error_rmse', 'score_error_rmse'),
        diagnostic_order=diag_fields, action_design_order=design_fields, error_metric_order=METRICS)


STATE_FEATURES = ('forward_headway_deviation', 'backward_headway_deviation', 'queue_capacity',
    'onboard_capacity', 'global_issue_cost_rate_per_1000', 'source_nonholding_per_100')


def action_design_audit(states, values, groups):
    """Describe supported action differences without observing or fitting outcomes."""
    lines = ('7X', '102X', '705X')
    vectors = {}
    coverage = {line: dict(selected_cells=0, executed_low_branches=0, executed_high_branches=0,
        source_nominal_low=0, source_nominal_high=0, source_first=0, source_second=0) for line in lines}
    for state, branches in zip(states, values):
        if state.get('status') != 'selected':
            continue
        line = state['cell']['target_line']
        prediction = state.get('prediction', {})
        features = prediction.get('innovation_diag', {}).get('state_values', {})
        if all(finite(features.get(name)) for name in STATE_FEATURES):
            vectors[state['state_id']] = [float(line == candidate) for candidate in lines]+[features[name] for name in STATE_FEATURES]
        row = coverage[line]
        row['selected_cells'] += 1
        row['source_nominal_low'] += abs(state.get('nominal_hold_seconds', -99.)) < 1e-5
        row['source_nominal_high'] += abs(state.get('nominal_hold_seconds', -99.)-7.5) < 1e-5
        position = branches[0].get('context', {}).get('assignment_position')
        row['source_first'] += position == 0
        row['source_second'] += position == 1
        row['executed_low_branches'] += branches[0].get('execution', {}).get('executed_hold_seconds') == 0.
        row['executed_high_branches'] += branches[1].get('execution', {}).get('executed_hold_seconds') == 7.5
    def describe(ids):
        matrix = np.asarray([vectors[key] for key in ids if key in vectors], dtype=float).reshape(-1, 9)
        return dict(n=len(matrix), columns=9, rank=int(np.linalg.matrix_rank(matrix)) if len(matrix) else 0,
            active_columns=int(np.any(matrix != 0, axis=0).sum()) if len(matrix) else 0,
            missing_feature_states=[key for key in ids if key not in vectors])
    all_ids = [state['state_id'] for state in states if state.get('status') == 'selected']
    unique_ids = [group['state_ids'][0] for group in groups]
    unique_counts = {line: dict(selected_cells=0, source_nominal_low=0, source_nominal_high=0, source_first=0, source_second=0) for line in lines}
    for state, branches in zip(states, values):
        if state['state_id'] not in unique_ids:
            continue
        row = unique_counts[state['cell']['target_line']]
        row['selected_cells'] += 1
        hold = state.get('nominal_hold_seconds', -99.)
        row['source_nominal_low'] += abs(hold) < 1e-5
        row['source_nominal_high'] += abs(hold-7.5) < 1e-5
        position = branches[0].get('context', {}).get('assignment_position')
        row['source_first'] += position == 0
        row['source_second'] += position == 1
    return dict(feature_order=[line+'_holding' for line in lines]+list(STATE_FEATURES),
        low_minus_high_multiplier=-2., state_vectors=vectors,
        design_cells=describe(all_ids), unique_physical_prefixes=describe(unique_ids),
        branch_and_source_coverage=coverage, unique_prefix_source_coverage=unique_counts, interpretation='No outcome fit. Rank and action coverage do not establish control benefit.')


def compact_pairs(pairs):
    """Share repeated field names; preserve failures, unsupported values and zeros."""
    counter_order = list(dict.fromkeys(key for pair in pairs for key in pair.get('prefix', {}).get('counters', {})))
    budget_order = ('prefix_sim_seconds', 'suffix_sim_seconds', 'replay_target_labels', 'replay_policy_updates', 'truth_labels_enter_training')
    for pair in pairs:
        for endpoint, models in pair['effects'].items():
            pair['effects'][endpoint] = [models[model] for model in (GLOBAL_MODELS if endpoint == 'global_next' else MODELS)]
        if pair.get('selection_status') != 'selected':
            pair['budget'] = [[pair['budget'].get(action, {}).get(key) for key in budget_order] for action in ('low', 'high')]
            continue
        pair['raw_global_effects'] = [pair['raw_global_effects'][model] for model in GLOBAL_MODELS]
        pair['global_to_fixed300_proxy'] = [pair['global_to_fixed300_proxy'][model] for model in ('base_analytic', 'global58')]
        pair['fixed300_level_errors'] = [pair['fixed300_level_errors'][model] for model in MODELS]
        counters = pair['prefix']['counters']
        pair['prefix']['counters'] = [counters.get(key) for key in counter_order]
        pair['suffix_training_delta'] = [[pair['suffix_training_delta'].get(action, {}).get(key) for key in counter_order] for action in ('low', 'high')]
        pair['budget'] = [[pair['budget'].get(action, {}).get(key) for key in budget_order] for action in ('low', 'high')]
        pair['execution'] = [[pair['execution'][action][key] for key in ('requested_hold_seconds', 'executed_hold_seconds')] for action in ('low', 'high')]
        for endpoint, row in pair['background'].items():
            row.pop('issued_at')  # Already equal to selected.started_at under the recorded integrity check.
            mean = row['physical_mean']
            pair['background'][endpoint] = [row[key] for key in ('status', 'fit_rows', 'max_available_at', 'max_sequence')]+[[mean[key] for key in METRICS] if mean is not None else None]
    return dict(counter_order=counter_order, budget_order=budget_order,
        action_order=('low', 'high'), execution_order=('requested_hold_seconds', 'executed_hold_seconds'),
        background_order=('status', 'fit_rows', 'max_available_at', 'max_sequence', 'physical_mean'))

def prefix_groups(states, values, pairs):
    """Direct equality, no digest: pre-shock scene labels do not create new states."""
    groups = []
    for state, branches, pair in zip(states, values, pairs):
        if state.get('status') != 'selected':
            continue
        low = branches[0]
        key = dict(selected=state['selected'], prefix=low.get('prefix'))
        found = next((group for group in groups if group['key'] == key), None)
        if found is None:
            found = dict(key=key, state_ids=[], pairs=[])
            groups.append(found)
        found['state_ids'].append(state['state_id'])
        found['pairs'].append(pair)
    checks = []
    pre = [(state, branches, pair) for state, branches, pair in zip(states, values, pairs)
           if state['cell']['window_name'] == 'pre_shock' and state.get('status') == 'selected']
    for seed, line in sorted({(state['cell']['seed'], state['cell']['target_line']) for state, _, _ in pre}):
        peers = [(state, branches, pair) for state, branches, pair in pre
                 if (state['cell']['seed'], state['cell']['target_line']) == (seed, line)]
        if len(peers) < 2:
            continue
        state0, branch0, pair0 = peers[0]
        exact = all(state['selected'] == state0['selected']
            and branches[0].get('prefix') == branch0[0].get('prefix')
            and state.get('prediction') == state0.get('prediction')
            and pair['truth'] == pair0['truth'] and pair['effects'] == pair0['effects']
            for state, branches, pair in peers[1:])
        checks.append(dict(state_ids=[state['state_id'] for state, _, _ in peers],
            prediction_and_truth_exact=exact))
    return groups, checks


def analyze(root, manifest, *, preflight=False):
    states = manifest['states']
    values = [tuple(load_result(branch_path(root, state, action, preflight=preflight))
        for action in ('low', 'high')) for state in states]
    pairs = [analyze_pair(state, *branches) for state, branches in zip(states, values)]
    terminal = all(value.get('status') != 'missing' for branches in values for value in branches)
    groups, duplicate_checks = prefix_groups(states, values, pairs)
    selected_pairs = [pair for pair in pairs if pair['selection_status'] == 'selected']
    checks = dict(all_branches_terminal=terminal,
        all_selected_pairs_integrity=all(pair['integrity']['all_pass'] for pair in selected_pairs),
        missing_selections_retained=all(pair['integrity']['all_pass'] for pair in pairs if pair['selection_status'] != 'selected'),
        duplicate_pre_shock_truth_exact=all(row['prediction_and_truth_exact'] for row in duplicate_checks))
    review = dict(all_states_selected=all(pair['selection_status'] == 'selected' for pair in pairs),
        all_fixed300_truth_complete=all(pair['truth']['fixed300']['gain'] is not None for pair in pairs),
        all_fixed300_models_supported=all(all(value is not None for value in pair['effects']['fixed300'].values()) for pair in pairs))
    if preflight:
        checks.update(preflight_state_selected=review['all_states_selected'], preflight_fixed300_truth=review['all_fixed300_truth_complete'])
    budget_keys = ('prefix_sim_seconds', 'suffix_sim_seconds', 'replay_target_labels', 'replay_policy_updates', 'truth_labels_enter_training')
    budget = {key: sum(value.get('budget', {}).get(key, 0) for branches in values for value in branches) for key in budget_keys}
    budget.update(expected_branch_count=2*len(states), expected_suffix_seconds=2*len(states)*300,
        existing_full_days_recomputed=0, new_training_labels=0, paired_diagnostic_labels_enter_fit=0,
        future_training_slots_per_pair=2, repeated_prefix_labels_are_replay=True)
    checks['diagnostic_truth_not_fitted'] = budget['truth_labels_enter_training'] == 0
    output = dict(status='complete' if terminal else 'incomplete', mode='preflight' if preflight else 'training_period_diagnostic',
        integrity=dict(all_pass=all(checks.values()), checks=checks), budget=budget,
        missing=dict(selection=sum(pair['selection_status'] != 'selected' for pair in pairs),
            branch_statuses=dict(Counter(value.get('status') for branches in values for value in branches)),
            truth={endpoint: sum(pair['truth'][endpoint]['gain'] is None for pair in pairs) for endpoint in ENDPOINTS}),
        state_counts=dict(design_cells=len(states), selected_cells=len(selected_pairs), unique_physical_prefixes=len(groups)),
        duplicate_prefixes=duplicate_checks,
        decision=dict(differential_learning='not_submitted', independent_confirmation='not_submitted',
            review_ready=all(checks.values()) and all(review.values()), checks=review,
            next='Review training-period difference signal and route/action coverage before any budget reallocation.'),
        pairs=pairs)
    if preflight:
        output['pairs'] = [{key: row[key] for key in ('state_id', 'selection_status', 'branch_status', 'status', 'integrity', 'budget')} for row in pairs]
        return compact(output)
    output['schema'] = dict(metric_order=METRICS, endpoint_model_order=MODELS, global_model_order=GLOBAL_MODELS,
        effect_order=EFFECT_FIELDS, aggregate_order=SUMMARY_FIELDS,
        background_residual_order=('physical_baseline_cost', 'saved_background_residual_cost', 'truth_minus_baseline', 'truth_minus_baseline_minus_saved_background'),
        background_action_order=('low', 'high'), fixed300_level_error='prediction minus frozen native truth in low/high action order',
        gain_sign='low cost minus high cost; positive chooses7.5 seconds; ties choose0', numeric_significant_digits=8)
    output['overall'] = {endpoint: summarize(pairs, endpoint) for endpoint in ENDPOINTS}
    unique = [group['pairs'][0] for group in groups]
    output['unique_prefix_overall'] = {endpoint: summarize(unique, endpoint) for endpoint in ENDPOINTS}
    output['signal'] = dict(design_cells=signal_summary(pairs), unique_prefixes=signal_summary(unique))
    output['action_design'] = action_design_audit(states, values, groups)
    strata = []
    for keys in (('scenario',), ('target_line',), ('window_name',), ('scenario', 'target_line')):
        cohorts = sorted({tuple(row[key] for key in keys) for row in pairs})
        for cohort in cohorts:
            subset = [row for row in pairs if tuple(row[key] for key in keys) == cohort]
            strata.append(dict(group=dict(zip(keys, cohort)), fixed300=summarize(subset, 'fixed300'), signal=signal_summary(subset)))
    output['strata'] = strata
    output['duration_groups'] = {name: dict(n=sum(pair['duration'] == name for pair in pairs),
        global_next=summarize([pair for pair in pairs if pair['duration'] == name], 'global_next'),
        raw_global_model_metrics={model: effect_summary([pair['raw_global_effects'][model] for pair in pairs
            if pair['duration'] == name and pair['selection_status'] == 'selected'
            and pair['integrity']['all_pass'] and pair['raw_global_effects'][model] is not None]) for model in GLOBAL_MODELS})
        for name in sorted({pair['duration'] for pair in pairs})}
    output['interpretation'] = dict(
        scope='Twelve design cells reuse three pre-shock physical prefixes across the scene labels; no independent confirmation or deployment claim.',
        source_truth='Original single-action reference continues policy/model learning and future calibration; frozen branch suffixes disable both. Their chosen-action differences are descriptive, not failed exact reproduction.',
        signal='Paired300-second gains compare common-prefix actions. Saved background residuals diagnose level variation on the same frozen suffix; true training-period contrasts were not fitted.',
        matching='Per-model metrics have their own supported states. Only common_metrics compare every listed model on the same state IDs; unsupported is never a zero.',
        proxy='base_analytic and global58 have only global-next endpoint predictions. Against fixed300 they are cross-endpoint direction proxies.',
        next_budget='Any later trained paired outcome consumes one slot per action, two per pair, within a newly declared285-slot allocation. Current diagnostic outcomes enter no fit.')
    # Shared column orders keep every audit while respecting the small-pull budget.
    check_order = list(dict.fromkeys(key for pair in pairs for key in pair['integrity']['checks']))
    output['schema']['pair_integrity_check_order'] = check_order
    output['schema']['coverage'] = compact_coverage(pairs)
    output['schema']['paired_tables'] = compact_pairs(pairs)
    for pair in pairs:
        pair['integrity']['checks'] = [pair['integrity']['checks'].get(key) for key in check_order]
    return compact(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text())
        started = time.monotonic()
        if args.wait:
            paths = [branch_path(args.root, state, action, preflight=args.preflight) for state in manifest['states'] for action in ('low', 'high')]
            while True:
                terminal = sum((path/'summary.json').is_file() or (path/'failure.json').is_file() for path in paths)
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='waiting_training_pairs', current=terminal, total=len(paths), unit='branches', elapsed_seconds=time.monotonic()-started)), flush=True)
                if terminal == len(paths) or time.monotonic()-started >= 72*3600:
                    break
                time.sleep(60)
        value = analyze(args.root, manifest, preflight=args.preflight)
        payload = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'
        if len(payload.encode()) > (PREFLIGHT_MAX_BYTES if args.preflight else MAX_BYTES):
            raise RuntimeError('Reduce the resident training-pair summary before collection')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
        print(json.dumps(dict(status=value['status'], bytes=len(payload.encode()), integrity=value['integrity'])), flush=True)
    except Exception as error:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        (args.out.parent/'summary_failure.json').write_text(json.dumps(dict(type=type(error).__name__, message=str(error)), allow_nan=False)+'\n')
        raise


if __name__ == '__main__':
    main()
