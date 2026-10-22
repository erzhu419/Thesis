"""Small resident audit of a fixed state-response candidate on fresh paired data."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import product
import json
import math
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
LEARNED_MODELS = ('route', 'shared', 'candidate')
MODELS = ('zero',)+LEARNED_MODELS
ENDPOINTS = ('global_next', 'same_vehicle_next', 'fixed300')
METRICS = ('dt', 'wait', 'ride', 'cost')
EFFECT_FIELDS = ('predicted_wait_gain', 'predicted_ride_gain', 'predicted_cost_gain',
    'wait_signed_error', 'ride_signed_error', 'cost_signed_error',
    'direction_correct', 'action_correct', 'regret')
AGGREGATE_FIELDS = ('n', 'cost_gain_mae', 'cost_gain_mse', 'wait_gain_mae', 'ride_gain_mae',
    'direction_correct', 'action_correct', 'predicted_ties', 'true_ties', 'regret_total')
MAX_BYTES = 150000
PREFLIGHT_MAX_BYTES = 20000


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def compact(value):
    if isinstance(value, float):
        return float(format(value, '.8g'))
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [compact(item) for item in value]
    return value


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def load_result(path):
    value = read_json(Path(path)/'summary.json')
    if value is not None:
        return value
    failure = read_json(Path(path)/'failure.json')
    if failure is not None:
        return dict(status='failed', failure={key: failure.get(key) for key in ('type', 'message', 'time')})
    return dict(status='missing')


def settings(protocol, *, preflight=False):
    # Preflight is the first complete pilot cell, not a smaller training run.
    return dict(protocol)


def expected_days(root, protocol, *, preflight=False):
    if preflight:
        p = protocol['preflight']
        return [(dict(seed=p['seed'], scenario=p['scenario']),
                 Path(root)/'runs'/p['scenario']/f"seed{p['seed']}")]
    return [(dict(seed=seed, scenario=scene), Path(root)/'runs'/scene/f'seed{seed}')
        for seed, scene in product(protocol['training_seeds'], protocol['scenarios'])]


def eval_cells(cell, protocol, *, preflight=False):
    p = settings(protocol, preflight=preflight)
    return [dict(**cell, line_id=line, window_name=window['name'],
                 target_id=f"heldout/{line}/{window['name']}")
            for line, window in product(p['target_lines'], p['heldout_windows'])]


def eval_path(root, cell, action, *, preflight=False):
    return Path(root)/'pairs'/cell['scenario']/f"seed{cell['seed']}"/cell['line_id']/cell['window_name']/action


def journal(path, day):
    value = read_json(Path(path)/'acquisitions.json')
    if value is not None:
        return value['records']
    return day.get('acquisitions', [])


def sign(value):
    return 1 if value > 0 else -1 if value < 0 else 0


def truth_endpoint(value, endpoint):
    row = value.get('truth', {}).get(endpoint)
    if not isinstance(row, dict) or row.get('status') != 'complete':
        return None
    return row if all(finite(row.get(key)) for key in METRICS) else None


def prediction_gain(prediction, model):
    if model == 'zero':
        return dict(dt=0., wait=0., ride=0., cost=0.)
    value = prediction.get(model, {})
    row = value.get('gain')
    return row if value.get('status') == 'supported' and isinstance(row, dict) and all(
        finite(row.get(key)) for key in METRICS) else None


def effect(predicted, actual):
    if predicted is None or actual is None:
        return None
    regret = max(0., -actual['cost']) if predicted['cost'] > 0 else max(0., actual['cost'])
    return [predicted['wait'], predicted['ride'], predicted['cost'],
        predicted['wait']-actual['wait'], predicted['ride']-actual['ride'], predicted['cost']-actual['cost'],
        sign(predicted['cost']) == sign(actual['cost']), regret == 0., regret]


def aggregate(values, true_gains):
    n = len(values)
    return [n, sum(abs(row[5]) for row in values)/n if n else None,
        sum(row[5]**2 for row in values)/n if n else None,
        sum(abs(row[3]) for row in values)/n if n else None,
        sum(abs(row[4]) for row in values)/n if n else None,
        sum(row[6] for row in values), sum(row[7] for row in values),
        sum(row[2] == 0 for row in values), sum(value == 0 for value in true_gains),
        sum(row[8] for row in values) if n else None]


def moments(values):
    n = len(values)
    return dict(n=n, mean=sum(values)/n if n else None,
        mean_absolute=sum(abs(x) for x in values)/n if n else None,
        minimum=min(values) if n else None, maximum=max(values) if n else None)


def zero_training(value):
    delta = value.get('suffix', {}).get('training_delta')
    return isinstance(delta, dict) and bool(delta) and all(finite(x) and x == 0 for x in delta.values())


def inspect_pair(cell, source, low, high):
    """Audit one native intervention against its own new common-prefix source."""
    result = dict(**cell, selection_status='selected' if source else 'selection_missing',
        branch_status={arm: value.get('status') for arm, value in (('low', low), ('high', high))})
    if source is None:
        checks = dict(missing_selection_retained=low.get('status') == high.get('status') == 'selection_missing')
        result.update(status='selection_missing', integrity=dict(all_pass=all(checks.values()), checks=checks),
            truth={endpoint: dict(status='missing', low=None, high=None, gain=None) for endpoint in ENDPOINTS},
            effects={model: None for model in MODELS}, duration='missing_selection',
            budget={arm: value.get('budget', {}) for arm, value in (('low', low), ('high', high))})
        return result
    prefix = source['prefix']
    prediction = source['prediction']
    both = (low, high)
    checks = dict(branches_complete=all(value.get('status') == 'complete' for value in both),
        branch_integrity=all(value.get('integrity', {}).get('all_pass') is True for value in both),
        common_source_state=all(value.get('selected') == source['selected'] for value in both),
        common_source_prefix=all(value.get('prefix') == prefix for value in both),
        common_source_prediction=all(value.get('prediction') == prediction for value in both),
        external_seed=prefix.get('external_seed') is not None and all(value.get('prefix', {}).get('external_seed') == prefix['external_seed'] for value in both),
        suffix_zero_training=all(zero_training(value) for value in both),
        suffix300=all(value.get('suffix', {}).get('seconds') == 300. for value in both),
        actual_commands=all(value.get('execution', {}).get('requested_hold_seconds') == seconds
            and value.get('execution', {}).get('executed_hold_seconds') == seconds
            for value, seconds in ((low, 0.), (high, 7.5))),
        same_other_batch_commands=low.get('native', {}).get('other_batch_actions') is not None
            and low.get('native', {}).get('other_batch_actions') == high.get('native', {}).get('other_batch_actions'),
        current_fork_truth_not_trained=all(value.get('budget', {}).get('truth_labels_enter_training') == 0 for value in both),
        source_clock_exact=all(value.get('budget', {}).get('prefix_sim_seconds') == source['selected']['started_at'] for value in both))
    outcome = {}
    for endpoint in ENDPOINTS:
        branches = {arm: truth_endpoint(value, endpoint) for arm, value in (('low', low), ('high', high))}
        ready = all(value is not None for value in branches.values())
        difference = {key: branches['low'][key]-branches['high'][key] for key in METRICS} if ready else None
        outcome[endpoint] = dict(status='complete' if ready else 'missing',
            low=[branches['low'][key] for key in METRICS] if branches['low'] is not None else None,
            high=[branches['high'][key] for key in METRICS] if branches['high'] is not None else None,
            gain=[difference[key] for key in METRICS] if difference is not None else None,
            missing_reasons={arm: value.get('truth', {}).get(endpoint, {}).get('reason') for arm, value in (('low', low), ('high', high)) if branches[arm] is None})
    checks['observed_fixed300_clock'] = all(row is None or row[0] == 300. for row in (
        outcome['fixed300']['low'], outcome['fixed300']['high']))
    checks['predicted_fixed300_clock'] = all(prediction.get(model, {}).get('fixed300_clock') == dict(low=300., high=300.) for model in LEARNED_MODELS)
    predicted = {model: prediction_gain(prediction, model) for model in MODELS}
    checks['predicted_cost_reconstruction'] = all(value is None or abs(value['cost']-2*value['wait']-value['ride']) <= 1e-6 for value in predicted.values())
    checks['predicted_clock_difference_zero'] = all(value is None or value['dt'] == 0. for value in predicted.values())
    actual = outcome['fixed300']['gain']
    actual = dict(zip(METRICS, actual)) if actual is not None else None
    checks['true_cost_reconstruction'] = actual is None or abs(actual['cost']-2*actual['wait']-actual['ride']) <= 1e-6
    gl = outcome['global_next']
    duration = 'missing_global' if gl['low'] is None or gl['high'] is None else 'zero_dt' if gl['low'][0] == gl['high'][0] == 0. else 'positive_dt' if gl['low'][0] > 0 and gl['high'][0] > 0 else 'mixed_dt'
    result.update(status='complete' if all(value.get('status') == 'complete' for value in both) else 'missing_or_failed',
        integrity=dict(all_pass=all(checks.values()), checks=checks),
        selected={key: source['selected'][key] for key in ('sequence', 'started_at', 'line_id', 'vehicle_id', 'station_id')},
        prediction_support={model: prediction.get(model, {}).get('support_pairs') for model in LEARNED_MODELS},
        features=source.get('features4', source.get('features')),
        prefix={key: prefix.get(key) for key in ('initial_updates', 'external_seed', 'native_steps', 'issued_count', 'counters')},
        truth=outcome, effects={model: effect(value, actual) for model, value in predicted.items()}, duration=duration,
        suffix_training_delta={arm: value.get('suffix', {}).get('training_delta') for arm, value in (('low', low), ('high', high))},
        execution={arm: {key: value.get('execution', {}).get(key) for key in ('requested_hold_seconds', 'executed_hold_seconds')} for arm, value in (('low', low), ('high', high))},
        budget={arm: value.get('budget', {}) for arm, value in (('low', low), ('high', high))})
    return result


def summarize(pairs, *, state_ids=True):
    valid = [row for row in pairs if row['selection_status'] == 'selected' and row['integrity']['all_pass']]
    common = [row for row in valid if all(row['effects'][model] is not None for model in MODELS)]
    result = dict(design_n=len(pairs), selected_valid_n=len(valid),
        missing_truth=sum(row['truth']['fixed300']['gain'] is None for row in valid),
        supported={model: sum(row['effects'][model] is not None for row in valid) for model in MODELS},
        model_metrics={model: aggregate([row['effects'][model] for row in valid if row['effects'][model] is not None],
            [row['truth']['fixed300']['gain'][3] for row in valid if row['effects'][model] is not None]) for model in MODELS},
        common_n=len(common), common_state_ids=[f"{row['scenario']}/seed{row['seed']}/{row['line_id']}/{row['window_name']}" for row in common],
        common_metrics={model: aggregate([row['effects'][model] for row in common],
            [row['truth']['fixed300']['gain'][3] for row in common]) for model in MODELS},
        true_cost_gain=moments([row['truth']['fixed300']['gain'][3] for row in valid if row['truth']['fixed300']['gain'] is not None]))
    if not state_ids:
        del result['common_state_ids']
    return result


def inspect_budget(day, records, caches, p):
    """Detect hidden extra labels, early paired feedback, and a third factual row."""
    receipts = day.get('receipts', [])
    budget = day.get('budget', {})
    selected_ids = [sequence for row in receipts for sequence in row.get('sequence_ids', [])]
    ordinary_ids = [sequence for row in receipts for sequence in row.get('ordinary_sequence_ids', [])]
    outcome_ids = [sequence for row in receipts for sequence in row.get('pair_outcome_ids', [])]
    pair_ids = [value for row in receipts for value in row.get('pair_ids', [])]
    sources = {int(row['selected']['sequence']) for row in records if row.get('role') == 'train'}
    cache_by_id = {row['id']: row for row in caches}
    admission = {}
    exact_pair_identities = True
    pair_clock = True
    for receipt in receipts:
        for pair_id in receipt.get('pair_ids', []):
            pair = cache_by_id.get(pair_id)
            if pair is None:
                exact_pair_identities = pair_clock = False
                continue
            labels = [-(2*int(pair['index'])+1), -(2*int(pair['index'])+2)]
            actual_labels = [int(pair['outcomes'][arm]['transition']['sequence']) for arm in ('low', 'high')]
            exact_pair_identities &= actual_labels == labels and receipt.get('pair_outcome_ids') == labels
            earliest = max(float(pair['issued_at'])+300., *(float(pair['outcomes'][arm]['transition']['completed_at']) for arm in ('low', 'high'))) + float(p['feedback_delay_seconds'])
            pair_clock &= pair.get('available_at') == earliest and all(pair['outcomes'][arm]['transition']['available_at'] == earliest for arm in ('low', 'high')) \
                and earliest <= receipt['deadline'] <= receipt['processed_at']
            admission[pair_id] = dict(deadline=receipt['deadline'], processed_at=receipt['processed_at'],
                available_at=pair.get('available_at'), outcome_ids=actual_labels)
    checks = dict(original_slot_budget=len(selected_ids) == p['target_record_budget'] == budget.get('target_labels'),
        packet_count=len(receipts) == (p['feedback_last_deadline']-p['feedback_first_deadline'])//p['feedback_interval_seconds']+1,
        unique_label_ids=len(set(selected_ids)) == len(selected_ids),
        ordinary_ids_native=all(isinstance(x, int) and x >= 0 for x in ordinary_ids),
        paired_ids_counterfactual=all(isinstance(x, int) and x < 0 for x in outcome_ids),
        two_slots_replace_ordinary=all(row.get('records') == 5 and len(row.get('sequence_ids', [])) == 5
            and row.get('paired_records') == len(row.get('pair_outcome_ids', [])) == 2*len(row.get('pair_ids', []))
            and len(row.get('pair_ids', [])) <= 1
            and row.get('ordinary_records') == len(row.get('ordinary_sequence_ids', [])) == 5-row.get('paired_records', -99)
            and set(row.get('sequence_ids', [])) == set(row.get('ordinary_sequence_ids', [])+row.get('pair_outcome_ids', [])) for row in receipts),
        source_factual_not_third_label=not (set(ordinary_ids) & sources),
        pair_admitted_once=len(pair_ids) == len(set(pair_ids)),
        pair_labels_exact=exact_pair_identities,
        paired_feedback_after_native300_delay=pair_clock,
        feedback_not_early=all(row.get('max_available_at', float('inf')) <= row.get('deadline', -1)
            and row['deadline'] <= row['processed_at'] for row in receipts),
        zero_global_not_fit=all(row.get('learned_global_records') == row.get('positive_global_records')
            and row.get('learned_global_records', 99)+row.get('zero_global_records', 99) == 5 for row in receipts),
        readouts_admitted_pairs_only=all(day.get('readouts', {}).get(model, {}).get('pair_ids') == pair_ids for model in LEARNED_MODELS),
        readouts_pair_counts=all(day.get('readouts', {}).get(model, {}).get('pairs') == len(pair_ids) for model in LEARNED_MODELS),
        admitted_pair_roles=all(cache_by_id[pair_id].get('role') == 'train' for pair_id in pair_ids if pair_id in cache_by_id),
        heldout_not_fit=not (set(pair_ids) & {row['id'] for row in records if row.get('role') == 'heldout'}),
        label_partition=sum(row.get('ordinary_records', 0) for row in receipts) == budget.get('ordinary_labels')
            and len(outcome_ids) == budget.get('paired_action_labels') == 2*len(pair_ids)
            and len(ordinary_ids)+len(outcome_ids) == budget.get('target_labels'),
        selected_ids_match_summary=set(selected_ids) == set(budget.get('selected_sequence_ids', [])),
        pair_ids_match_summary=pair_ids == budget.get('admitted_pair_ids'))
    return dict(all_pass=all(checks.values()), checks=checks,
        actual=dict(slots=len(selected_ids), packets=len(receipts), ordinary_labels=len(ordinary_ids),
            paired_outcome_labels=len(outcome_ids), admitted_pairs=len(pair_ids), reserved_source_issues=len(sources),
            factual_sources_present_in_ordinary=sorted(set(ordinary_ids) & sources),
            zero_global_labels=sum(row.get('zero_global_records', 0) for row in receipts)), admission=admission)


def main_day_paths(root, protocol, *, preflight=False):
    return expected_days(root, protocol, preflight=preflight)


def design_coverage(records, admission, p):
    training = [row for row in records if row['role'] == 'train']
    heldout = [row for row in records if row['role'] == 'heldout']
    expected_training = len(p['target_lines'])*len(p['training_windows'])
    expected_heldout = len(p['target_lines'])*len(p['heldout_windows'])
    by_line = {}
    for line in p['target_lines']:
        selected = {row['window_name'] for row in heldout if row['line_id'] == line}
        by_line[line] = dict(training_expected=len(p['training_windows']),
            training_selected=sum(row['line_id'] == line for row in training),
            training_admitted=sum(row['line_id'] == line and row['id'] in admission for row in training),
            heldout_expected=len(p['heldout_windows']), heldout_selected=len(selected),
            missing_heldout_windows=[window['name'] for window in p['heldout_windows']
                                    if window['name'] not in selected])
    return dict(training_expected=expected_training, training_selected=len(training),
        training_admitted=len(admission), heldout_expected=expected_heldout,
        heldout_selected=len(heldout), by_line=by_line)


def paired_label_age(source, records, admission):
    """Ordinary world labels do not refresh the paired-difference readouts."""
    now = source['selected']['started_at']
    ready = [row for row in records if row.get('role') == 'train' and row['id'] in admission
             and admission[row['id']]['processed_at'] <= now]
    same_line = [row for row in ready if row['line_id'] == source['line_id']]
    latest = lambda rows: max(rows, key=lambda row: (
        admission[row['id']]['processed_at'], row['selected']['started_at'], row['id'])) if rows else None
    line, shared = latest(same_line), latest(ready)
    result = dict(line_support_pairs=len(same_line), shared_support_pairs=len(ready))
    for name, row in (('line', line), ('shared', shared)):
        issued = row['selected']['started_at'] if row else None
        record = admission[row['id']] if row else {}
        admitted = record.get('processed_at')
        result.update({name+'_last_pair_id': row['id'] if row else None,
            name+'_last_issue': issued, name+'_last_available': record.get('available_at'),
            name+'_last_admitted': admitted,
            name+'_issue_age_seconds': now-issued if issued is not None else None,
            name+'_admitted_age_seconds': now-admitted if admitted is not None else None})
    return result


def load_range(records):
    values=[row['features'][4] for row in records]
    return dict(pairs=len(values),minimum=min(values) if values else None,
        maximum=max(values) if values else None)


def issue_load_coverage(source, records, admission):
    """Report same-line feature range separately from the original count support."""
    now=source['selected']['started_at']
    ready=[row for row in records if row.get('role') == 'train' and row['line_id'] == source['line_id']
        and row['id'] in admission and admission[row['id']]['processed_at'] <= now]
    result=load_range(ready)
    observed=source['features'][4]
    result.update(observed=observed,in_line_range=result['minimum'] <= observed <= result['maximum'] if ready else None)
    return result


def summarize_label_age(rows):
    selected = [row for row in rows if row['selection_status'] == 'selected']
    return {key: moments([row.get('paired_label_age', {}).get(key) for row in selected
                         if finite(row.get('paired_label_age', {}).get(key))])
            for key in ('line_issue_age_seconds', 'line_admitted_age_seconds',
                        'shared_admitted_age_seconds')}


def inspect_day(cell, path, value, protocol, *, preflight=False):
    p = settings(protocol, preflight=preflight)
    records = journal(path, value)
    train_records = [row for row in records if row.get('role') == 'train']
    caches = [read_json(Path(path)/'pair_cache'/f"{row['index']}.json") for row in train_records]
    caches = [row for row in caches if row is not None]
    budget_audit = inspect_budget(value, records, caches, p)
    training = []
    cache_matches_source = cache_matches_children = True
    for source in train_records:
        children = [load_result(Path(path)/'training_branches'/str(source['index'])/arm) for arm in ('low', 'high')]
        cache = next((row for row in caches if row['id'] == source['id']), None)
        cache_matches_source &= cache is not None and all(cache.get(key) == wanted for key, wanted in (
            ('role', 'train'), ('index', source['index']), ('line_id', source['line_id']),
            ('source_sequence', source['selected']['sequence']), ('issued_at', source['selected']['started_at']),
            ('features', source.get('features4', source.get('features')))))
        for arm, child in zip(('low', 'high'), children):
            cached = cache.get('outcomes', {}).get(arm, {}) if cache is not None else {}
            cache_matches_children &= cached.get('truth') == child.get('truth', {}).get('fixed300') and bool(cached)
            original = child.get('global_transition')
            cache_matches_children &= isinstance(original, dict) and {key: item for key, item in cached.get('transition', {}).items()
                if key not in ('sequence', 'available_at')} == {key: item for key, item in (original or {}).items()
                if key not in ('sequence', 'available_at')}
        pair_cell = dict(**cell, line_id=source['line_id'], window_name=source['window_name'], target_id=source['id'])
        row = inspect_pair(pair_cell, source, *children)
        row['role'] = 'train'
        row['admission'] = budget_audit['admission'].get(source['id'])
        training.append(row)
    budget = value.get('budget', {})
    initializer = value.get('initializer', {})
    checks = dict(native_day_complete=value.get('status') == 'complete',
        native_day_integrity=value.get('integrity', {}).get('all_pass') is True,
        actual_budget_audit=budget_audit['all_pass'],
        training_children_exact=all(row['integrity']['all_pass'] for row in training),
        all_training_pair_caches=len(caches) == len(train_records),
        training_cache_matches_new_source=cache_matches_source,
        training_cache_matches_native_truth=cache_matches_children,
        initializer_unchanged=initializer.get('initial_updates') == p['initializer_updates'],
        fixed_source_history_budget=initializer.get('history_rows') == p['history_rows'] and initializer.get('source_rows') == p['source_rows'],
        native_duration=budget.get('native_seconds') == p['horizon_seconds'],
        policy_update_budget=budget.get('policy_updates') == p['policy_update_budget'],
        trust_update_budget=budget.get('trust_updates') == p['policy_update_budget'],
        world_update_budget=budget.get('world_updates') == 2*len(value.get('receipts', []))*p['model_refit_updates'])
    selected = {row['id']: row for row in records if row.get('role') == 'heldout'}
    plans = {f"{role}/{line}/{window['name']}": (role, line, window) for role, windows in (
        ('train', p['training_windows']), ('heldout', p['heldout_windows']))
        for line, window in product(p['target_lines'], windows)}
    checks['predefined_state_selection'] = len({row['id'] for row in records}) == len(records) and all(
        row['id'] in plans and row['role'] == plans[row['id']][0] and row['line_id'] == plans[row['id']][1]
        and plans[row['id']][2]['start'] <= row['selected']['started_at'] < plans[row['id']][2]['end'] for row in records)
    checks['heldout_after_final_training'] = all(row['selected']['started_at'] > budget.get('training_stopped_at', float('inf')) for row in selected.values())
    train_sequences = {row['selected']['sequence'] for row in train_records}
    checks['training_heldout_states_disjoint'] = not (train_sequences & {row['selected']['sequence'] for row in selected.values()})
    coverage = design_coverage(records, budget_audit['admission'], p)
    if preflight:
        checks['preflight_training_plan_selected'] = coverage['training_selected'] == coverage['training_expected']
        checks['preflight_training_plan_admitted'] = coverage['training_admitted'] == coverage['training_expected']
        checks['preflight_all_lines_trained'] = all(row['training_admitted'] == row['training_expected']
                                                  for row in coverage['by_line'].values())
        checks['preflight_all_lines_two_windows'] = all(row['heldout_selected'] == row['heldout_expected']
                                                       for row in coverage['by_line'].values())
    day = dict(**cell, status=value.get('status'), integrity=dict(all_pass=all(checks.values()), checks=checks),
        budget={key: item for key, item in budget.items() if key not in ('selected_sequence_ids', 'reserved_source_sequences', 'excluded_factual_sequences')},
        budget_audit=budget_audit, initializer=initializer, readouts=value.get('readouts', {}),
        selected_training_pairs=len(train_records), heldout_anchors=len(selected), design_coverage=coverage,
        issue_load_training_by_line={line:load_range([row for row in train_records
            if row['line_id'] == line and row['id'] in budget_audit['admission']]) for line in p['target_lines']},
        missing_cells=value.get('missing_cells', []), cost=value.get('cost'), failure=value.get('failure'))
    return day, records, training


def add_compute(rows):
    keys = ('prefix_sim_seconds', 'suffix_sim_seconds', 'replay_target_labels', 'replay_policy_updates', 'truth_labels_enter_training')
    return {key: sum(row.get('budget', {}).get(arm, {}).get(key, 0) for row in rows for arm in ('low', 'high')) for key in keys}


def stability(pairs):
    groups = []
    stable = [row for row in pairs if row['scenario'] == 'stable']
    for line in sorted({row['line_id'] for row in stable}):
        values = [row for row in stable if row['line_id'] == line]
        summary = summarize(values)
        metrics = summary['common_metrics']
        comparisons = {}
        for model in ('shared','candidate'):
            comparisons[model] = {}
            for reference in ('route', 'zero'):
                comparisons[model][reference] = dict(common_n=summary['common_n'],
                    cost_gain_mae_change=metrics[model][1]-metrics[reference][1] if summary['common_n'] else None,
                    direction_correct_change=metrics[model][5]-metrics[reference][5],
                    selected_action_regret_change=metrics[model][9]-metrics[reference][9] if summary['common_n'] else None)
        groups.append(dict(line_id=line, comparison=comparisons, heldout=summary))
    return groups


def compress_pairs(rows):
    counter_order = list(dict.fromkeys(key for row in rows for key in row.get('prefix', {}).get('counters', {})))
    check_order = list(dict.fromkeys(key for row in rows for key in row['integrity']['checks']))
    budget_order = ('prefix_sim_seconds', 'suffix_sim_seconds', 'replay_target_labels', 'replay_policy_updates', 'truth_labels_enter_training')
    selected_order=('sequence','started_at','vehicle_id','station_id')
    for row in rows:
        row['integrity']['checks'] = [row['integrity']['checks'].get(key) for key in check_order]
        row['effects'] = [row['effects'][model] for model in MODELS]
        row['truth'] = [[row['truth'][endpoint][key] for key in ('low', 'high', 'gain', 'missing_reasons')]
            if 'missing_reasons' in row['truth'][endpoint] else
            [row['truth'][endpoint][key] for key in ('low', 'high', 'gain')]+[{}] for endpoint in ENDPOINTS]
        row['budget'] = [[row['budget'].get(arm, {}).get(key) for key in budget_order] for arm in ('low', 'high')]
        if row['selection_status'] != 'selected':
            continue
        row['selected']=[row['selected'][key] for key in selected_order]
        counters = row['prefix'].get('counters', {})
        row['prefix']['counters'] = [counters.get(key) for key in counter_order]
        row['suffix_training_delta'] = [[row['suffix_training_delta'].get(arm, {}).get(key) for key in counter_order] for arm in ('low', 'high')]
        row['execution'] = [[row['execution'][arm].get(key) for key in ('requested_hold_seconds', 'executed_hold_seconds')] for arm in ('low', 'high')]
    return dict(counter_order=counter_order, pair_check_order=check_order, budget_order=budget_order,
        selected_order=selected_order,
        action_order=('low', 'high'), execution_order=('requested_hold_seconds', 'executed_hold_seconds'),
        truth_endpoint_order=ENDPOINTS, truth_fields=('low', 'high', 'gain', 'missing_reasons'))


def check_summary(value, order=None):
    """Keep failures explicit while avoiding repeated success-name maps."""
    raw=value.get('checks',{})
    items=list(zip(order,raw)) if isinstance(raw,list) else list(raw.items())
    return dict(all_pass=value.get('all_pass'),check_count=sum(passed is not None for _,passed in items),
        failed_checks=[name for name,passed in items if passed is False])


def condense_days(days):
    for day in days:
        day['integrity']=check_summary(day['integrity'])
        day['budget_audit']['checks']=check_summary(day['budget_audit'])
        day['budget'].pop('semantics',None)
        for readout in day.get('readouts',{}).values():
            for key in ('semantics','pair_ids','model'):
                readout.pop(key,None)


def condense_report(value):
    """Preserve outcome data and totals; omit repeated per-row bookkeeping."""
    condense_days(value['days'])
    order=value['schema']['pairs']['pair_check_order']
    for row in value['training_pairs']+value['heldout_pairs']:
        row['integrity']=check_summary(row['integrity'],order)
    omitted=('budget','suffix_training_delta','execution','branch_status','prefix')
    for row in value['training_pairs']:
        for key in omitted:row.pop(key,None)
    age_order=list(dict.fromkeys(key for row in value['heldout_pairs'] for key in row.get('paired_label_age',{})))
    for row in value['heldout_pairs']:
        if 'paired_label_age' in row:
            row['paired_label_age']=[row['paired_label_age'].get(key) for key in age_order]
    for name in ('heldout_overall','training_at_issue_diagnostic'):
        value[name].pop('common_state_ids',None)
    summaries=[row['heldout'] for row in value['heldout_strata']]
    summaries += [row['heldout'] for row in value['stable_selected_action_comparison']]
    summaries += [row['fixed300'] for row in value['heldout_duration_groups'].values()]
    for summary in summaries:
        summary.pop('common_state_ids',None)
        summary.pop('model_metrics',None)
    value['schema']['training_row_omissions']=omitted
    value['schema']['paired_label_age_order']=age_order
    value['schema']['training_prefix_units']='Exact groups compare full selected state, issue features and source prefix before export; raw prefix counters remain resident and are audited before omission.'
    value['schema']['stratum_metrics']='common_metrics compare all four models on identical states; per-model support is retained, and overall model_metrics retain separate support cohorts.'
    value['schema']['pair_integrity']='all_pass, count of applicable checks and failed names; full raw branch records remain resident.'
    value.pop('interpretation',None)
    return value


def analyze(root, protocol=None, *, preflight=False):
    p = read_json(HERE/'protocol.json') if protocol is None else protocol
    day_paths = expected_days(root, p, preflight=preflight)
    days, heldout, training, day_statuses = [], [], [], []
    training_prefix_groups=[]
    for cell, path in day_paths:
        value = load_result(path)
        day_statuses.append(value.get('status'))
        day, records, train_rows = inspect_day(cell, path, value, p, preflight=preflight)
        days.append(day)
        training.extend(train_rows)
        for source in records:
            if source.get('role') != 'train':continue
            identity=[source['selected'],source.get('features4',source.get('features')),source['prefix']]
            reference=dict(**cell,target_id=source['id'])
            group=next((row for row in training_prefix_groups if row[0] == identity),None)
            if group is None:training_prefix_groups.append([identity,[reference]])
            else:group[1].append(reference)
        sources = {row['id']: row for row in records if row.get('role') == 'heldout'}
        for state in eval_cells(cell, p, preflight=preflight):
            low, high = (load_result(eval_path(root, state, arm, preflight=preflight)) for arm in ('low', 'high'))
            source = sources.get(state['target_id'])
            row = inspect_pair(state, source, low, high)
            if source is not None:
                row['paired_label_age'] = paired_label_age(source, records, day['budget_audit']['admission'])
                row['issue_load_coverage']=issue_load_coverage(source,records,day['budget_audit']['admission'])
                row['integrity']['checks']['paired_support_exact'] = all(
                    source['prediction'].get(model, {}).get('support_pairs') == row['paired_label_age']['line_support_pairs']
                    for model in LEARNED_MODELS)
                row['integrity']['all_pass'] = all(row['integrity']['checks'].values())
            heldout.append(row)
    terminal = all(status != 'missing' for status in day_statuses) and all(
        status != 'missing' for row in heldout for status in row['branch_status'].values())
    checks = dict(all_days_terminal=all(status != 'missing' for status in day_statuses),
        all_days_integrity=all(day['integrity']['all_pass'] for day in days),
        all_heldout_terminal=all(status != 'missing' for row in heldout for status in row['branch_status'].values()),
        all_heldout_integrity=all(row['integrity']['all_pass'] for row in heldout),
        all_child_suffix_training_zero=all(row['integrity']['checks'].get('suffix_zero_training', row['selection_status'] != 'selected') for row in training+heldout),
        evaluation_truth_not_fit=add_compute(heldout)['truth_labels_enter_training'] == 0)
    review = dict(heldout_all_selected=all(row['selection_status'] == 'selected' for row in heldout),
        heldout_fixed300_truth_complete=all(row['truth']['fixed300']['gain'] is not None for row in heldout),
        all_readouts_supported=all(all(row['effects'][model] is not None for model in MODELS) for row in heldout),
        training_full_design_admitted=all(day['design_coverage']['training_selected'] == day['design_coverage']['training_expected']
            == day['design_coverage']['training_admitted'] for day in days),
        three_lines_two_windows=all(all(row['heldout_selected'] == row['heldout_expected']
            for row in day['design_coverage']['by_line'].values()) for day in days))
    if preflight:
        checks.update(preflight_heldout_selected=review['heldout_all_selected'],
            preflight_heldout_truth=review['heldout_fixed300_truth_complete'],
            preflight_readouts_supported=review['all_readouts_supported'])
    slot_count = sum(day['budget_audit']['actual']['slots'] for day in days)
    budget = dict(main_training_slots=slot_count, expected_main_training_slots=len(days)*settings(p, preflight=preflight)['target_record_budget'],
        main_paired_outcomes=sum(day['budget_audit']['actual']['paired_outcome_labels'] for day in days),
        main_ordinary_outcomes=sum(day['budget_audit']['actual']['ordinary_labels'] for day in days),
        main_training_pairs=sum(day['budget_audit']['actual']['admitted_pairs'] for day in days),
        main_native_seconds=sum(day['budget'].get('native_seconds', 0) for day in days),
        training_branch_compute=add_compute(training), heldout_branch_compute=add_compute(heldout),
        training_pair_slots_per_pair=2, repeated_prefix_labels_are_replay=True)
    value = dict(status='complete' if terminal else 'incomplete', mode='reused_full_budget_preflight' if preflight else 'state_response_development',
        integrity=dict(all_pass=all(checks.values()), checks=checks), budget=budget,
        missing=dict(day_statuses=dict(Counter(day_statuses)),
            heldout_branch_statuses=dict(Counter(status for row in heldout for status in row['branch_status'].values())),
            heldout_selection=sum(row['selection_status'] != 'selected' for row in heldout),
            heldout_truth={endpoint: sum(row['truth'][endpoint]['gain'] is None for row in heldout) for endpoint in ENDPOINTS}),
        decision=dict(independent_confirmation='not_submitted', deployment='not_enabled',
            review_ready=all(checks.values()) and all(review.values()),
            checks=review, next='Review new heldout direction, regret and stable selected-action costs before control integration or independent confirmation.'),
        days=days)
    if preflight:
        # This gate reuses the pilot's first day and twelve heldout tails.
        # Detailed coefficients/endpoint arrays remain resident until main review.
        for day in days:
            day.pop('readouts', None)
            day.pop('cost', None)
        def gate_pair(row):
            item = {key: row[key] for key in ('target_id', 'selection_status', 'branch_status')}
            item['integrity'] = dict(all_pass=row['integrity']['all_pass'],
                failed_checks=[key for key, passed in row['integrity']['checks'].items() if not passed])
            if 'paired_label_age' in row:
                item['paired_label_age'] = row['paired_label_age']
            if 'issue_load_coverage' in row:
                item['issue_load_coverage']=row['issue_load_coverage']
            return item
        value['training_pairs'] = [gate_pair(row) for row in training]
        value['heldout_pairs'] = [gate_pair(row) for row in heldout]
        value['paired_label_age'] = summarize_label_age(heldout)
        value['reuse'] = dict(first_pilot_day=True, extra_main_days=0, extra_heldout_branches=0,
            reason='This full-budget day and its twelve heldout branches are counted once in the four-day pilot.')
        condense_days(days)
        return compact(value)
    value['heldout_overall'] = summarize(heldout)
    value['heldout_label_age'] = summarize_label_age(heldout)
    value['training_at_issue_diagnostic'] = summarize(training)
    value['stable_selected_action_comparison'] = stability(heldout)
    strata = []
    for keys in (('scenario',), ('seed',), ('line_id',), ('window_name',), ('scenario', 'line_id'), ('seed', 'scenario')):
        for cohort in sorted({tuple(row[key] for key in keys) for row in heldout}):
            subset = [row for row in heldout if tuple(row[key] for key in keys) == cohort]
            strata.append(dict(group=dict(zip(keys, cohort)), heldout=summarize(subset, state_ids=False),
                               paired_label_age=summarize_label_age(subset)))
    value['heldout_strata'] = strata
    value['heldout_duration_groups'] = {name: dict(design_n=sum(row['duration'] == name for row in heldout),
        fixed300=summarize([row for row in heldout if row['duration'] == name]))
        for name in sorted({row['duration'] for row in heldout})}
    value['schema'] = dict(metric_order=METRICS, model_order=MODELS, effect_order=EFFECT_FIELDS, aggregate_order=AGGREGATE_FIELDS,
        gain_sign='low cost minus high cost; positive chooses7.5 seconds; ties choose0', numeric_precision_significant_digits=8,
        training_truth='New paired labels may fit only after both outcomes become available and consume two replacement slots; at-issue prediction errors are descriptive training diagnostics.',
        heldout_truth='Both native300-second tails are frozen. No heldout target enters any fit. Action regret is a passive decision diagnostic, not deployed day cost.')
    value['schema']['pairs'] = compress_pairs(training+heldout)
    value['training_prefix_units']=dict(selected=len(training),distinct_exact=len(training_prefix_groups),
        duplicate_groups=[references for _,references in training_prefix_groups if len(references)>1])
    value['training_pairs'] = training
    value['heldout_pairs'] = heldout
    value['interpretation'] = dict(
        comparator='The original shared4/route3 readouts and one fixed state-response candidate use exactly the same admitted pairs on one new controller flow.',
        matching='Only common_metrics compare every model on identical heldout states; per-model metrics retain their actual support.',
        clock='Both response targets have analytic300-second clocks. Global zero-duration rows can consume label slots and supervise fixed300 response, but do not enter world/trust transition fitting.',
        budget='Each pair replaces two ordinary labels in a five-slot packet. Replayed child prefixes are computation, not additional independent training observations.',
        freshness='Pair issue, release and actual admission ages are recorded separately; ordinary target labels and repeated refits do not refresh paired-difference supervision.',
        preflight_reuse='The complete stable first-seed day and its twelve heldout branches are the first pilot cell; no duplicated labels or tails are counted.',
        scope='New training prefixes and heldout states replace prior diagnostic points. All models are passive; main day cost is common and cannot establish controller superiority.')
    return compact(condense_report(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    try:
        p = read_json(HERE/'protocol.json')
        started = time.monotonic()
        if args.wait:
            paths = [path for _, path in expected_days(args.root, p, preflight=args.preflight)]
            paths += [eval_path(args.root, state, arm, preflight=args.preflight)
                for cell, _ in expected_days(args.root, p, preflight=args.preflight)
                for state in eval_cells(cell, p, preflight=args.preflight) for arm in ('low', 'high')]
            while True:
                terminal = sum((path/'summary.json').is_file() or (path/'failure.json').is_file() for path in paths)
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='waiting_budgeted_difference', current=terminal,
                    total=len(paths), unit='runs', elapsed_seconds=time.monotonic()-started)), flush=True)
                if terminal == len(paths) or time.monotonic()-started >= 72*3600:
                    break
                time.sleep(60)
        value = analyze(args.root, p, preflight=args.preflight)
        payload = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'
        if len(payload.encode()) > (PREFLIGHT_MAX_BYTES if args.preflight else MAX_BYTES):
            raise RuntimeError('Reduce the resident difference-learning summary before collection')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
        print(json.dumps(dict(status=value['status'], bytes=len(payload.encode()), integrity=value['integrity'])), flush=True)
    except Exception as error:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        (args.out.parent/'summary_failure.json').write_text(json.dumps(dict(type=type(error).__name__, message=str(error)))+'\n')
        raise


if __name__ == '__main__':
    main()
