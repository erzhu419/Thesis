"""Training coverage gate, then fresh service validation; never starts response B."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.mcwm_closed_loop_service_shift_v21_20261022 import analyze as old
from experiments.mcwm_closed_loop_stratified_pairs_v23_20261023.shift_model import ServiceShiftReadout

LINES, KINDS = old.LINES, old.KINDS
ENDPOINTS = tuple(product(KINDS, range(3)))
MODELS = ('shared', 'history')
read, load, audit, finite = old.read, old.load, old.audit, old.finite


def flatten(value):
    return [value.get(kind, [None]*3)[i] for kind, i in ENDPOINTS]


def replay(training, issue, model):
    """Replay only the saved admission prefix, including its within-clock order."""
    count = issue['prefix']['counters']['service_'+model+'_pairs']
    if not 0 <= count <= len(training):
        raise ValueError('Invalid admitted service prefix')
    past, now = training[:count], issue['selected']['started_at']
    for row in past:
        if row['id'] == issue['id'] or row['issued_at'] >= now:
            raise ValueError('Current or future truth entered the issue forecast')
    estimator = ServiceShiftReadout(use_history=model == 'history')
    estimator.accept(past, now)
    estimator.refit()
    return estimator.predict(issue['features'], issue['service_history']), [r['id'] for r in past]


def metrics(rows):
    result = {}
    for model in MODELS:
        state_errors, correct, negative, hits, zero = [], 0, 0, 0, 0
        for row in rows:
            valid = [i for i, truth in enumerate(row['truth']) if finite(truth) and
                     all(finite(row['prediction'][m][i]) for m in MODELS)]
            if valid:
                state_errors.append(sum(abs(row['prediction'][model][i]-row['truth'][i]) for i in valid)/len(valid))
            for i in valid:
                truth, estimate = row['truth'][i], row['prediction'][model][i]
                correct += (estimate > 0)-(estimate < 0) == (truth > 0)-(truth < 0)
                negative += truth < 0
                hits += truth < 0 and estimate < 0
                zero += truth == 0
        result[model] = dict(states=len(state_errors), mae_seconds=sum(state_errors)/len(state_errors)
            if state_errors else None, sign_correct=correct, negative_truth_endpoints=negative,
            negative_hits=hits if negative else None, zero_truth_endpoints=zero)
    return result


def duration(branches):
    clocks = [branches[arm].get('truth', {}).get('global_next', {}).get('dt') for arm in ('low', 'high')]
    if not all(finite(value) for value in clocks):
        return 'missing_global'
    return 'both_zero' if all(value == 0 for value in clocks) else 'one_zero' if any(value == 0 for value in clocks) else 'positive'


def parallel_audit(receipts, expected_ids):
    """Positive native wall-time overlap is evidence, launching code alone is not."""
    rows, checks = [], dict(pair_ids=len(receipts) == len(expected_ids) and
        {r.get('pair_id') for r in receipts} == set(expected_ids))
    for value in receipts:
        arms = value.get('native_wall', {})
        a, b = arms.get('low', {}), arms.get('high', {})
        times = [arm.get(k) for arm in (a, b) for k in ('native_started_wall', 'native_finished_wall')]
        timed = all(finite(v) for v in times)
        overlap = min(a['native_finished_wall'], b['native_finished_wall'])-max(a['native_started_wall'], b['native_started_wall']) if timed else None
        passed = timed and overlap > 0 and a.get('pid') != b.get('pid') and all(
            isinstance(arm.get('pid'), int) for arm in (a, b)) and value.get('all_pass') is True and (
            finite(value.get('parent_native_before')) and value['parent_native_before'] == value.get('parent_native_after'))
        checks[str(value.get('pair_id'))] = passed
        rows.append(dict(id=value.get('pair_id'), passed=passed, overlap_seconds=overlap))
    return dict(**audit(checks), pairs=rows)


def _paths(root, protocol):
    stage = root/'stage_a'
    return stage, stage/'runs'/'stable'/('seed'+str(protocol['service_seed']))


def native_day(day):
    b, init = day.get('budget', {}), day.get('initializer', {})
    n = b.get('training_pairs')
    return audit(dict(complete=day.get('status') == 'complete',
        runner=day.get('integrity', {}).get('all_pass') is True,
        initializer=init.get('initial_updates') == 50000 and init.get('history_rows') == 32768 and init.get('source_rows') == 4096,
        label_accounting=isinstance(n, int) and 0 <= n <= 9 and b.get('target_labels') == 285 and
            b.get('paired_action_labels') == 2*n and b.get('ordinary_labels') == 285-2*n and b.get('paired_outcomes_per_pair') == 2,
        clock=b.get('native_seconds') == 24000,
        updates=b.get('policy_updates') == 456 and b.get('trust_updates') == 456,
        no_response_candidate='candidate' not in day.get('readouts', {})))


def training_gate(root):
    root = Path(root)
    p = read(root/'protocol.json')
    _, source = _paths(root, p)
    day = load(source)
    if day.get('status') != 'complete':
        return dict(status='incomplete' if day['status'] == 'pending' else day['status'],
            stage='train', integrity=dict(all_pass=False), scientific_gate=dict(passed=False), training_gate=dict(passed=False), service_validation_eligible=False,
            decision=dict(service_validation_eligible=False, stage_b='closed'))
    journal = [r for r in read(source/'acquisitions.json', {}).get('records', day.get('acquisitions', []))
               if r['role'] == 'train']
    ordered_ids = day.get('budget', {}).get('admitted_pair_ids', [])
    cache = {r['id']: r for r in (read(path) for path in (source/'pair_cache').glob('*.json'))}
    metadata = {r['id']: r for r in day.get('training_pairs', [])}
    training, rows, pair_cases, missing = [], [], [], Counter()
    journal_ids = [r['id'] for r in journal]
    sample_checks = day.get('sampling_summary', {}).get('checks', {})
    checks = dict(native_day=native_day(day)['all_pass'],
        admitted_ids=len(ordered_ids) == len(set(ordered_ids)) and set(ordered_ids) <= set(journal_ids),
        selected_ids_unique=len(journal_ids) == len(set(journal_ids)),
        selected_cache_metadata=all(k in cache and k in metadata for k in journal_ids),
        sampling=all(sample_checks.get(k) is True for k in ('exactly_nine_cells', 'selected_once', 'selected_only_planned')))
    if not all(checks[k] for k in ('admitted_ids', 'selected_ids_unique', 'selected_cache_metadata')):
        return dict(status='integrity_failed', stage='train', integrity=audit(checks),
            scientific_gate=dict(passed=False), training_gate=dict(passed=False), service_validation_eligible=False,
            decision=dict(service_validation_eligible=False, stage_b='closed'), missing=dict(training_cache_or_admission=1))
    for identifier in ordered_ids:
        row, meta = deepcopy(cache[identifier]), metadata[identifier]
        if row.get('role') != 'train' or meta.get('admitted') is not True:
            raise ValueError('Service fitting accepts this day\'s admitted training pairs only')
        row.update(selected_for_training=True, admitted_at=meta['admitted_at'])
        training.append(row)
    max_error, finite_count, none_count = 0., 0, 0
    for issue in journal:
        identifier = issue['id']
        record = cache[identifier]
        history = old.history_coverage(issue.get('service_history', {}), issue['selected']['started_at'])
        checks[identifier+'/history'] = history['supported'] and history['causal']
        checks[identifier+'/cache'] = all(issue[k] == record[k] for k in ('features', 'service_history'))
        prediction, prefix_ids = {}, {}
        for model in MODELS:
            forecast, prefix_ids[model] = replay(training, issue, model)
            prediction[model] = flatten(forecast)
            for value, expected in zip(prediction[model], flatten(issue['service_prediction'][model])):
                if expected is None:
                    key = identifier+'/'+model+'/unsupported'
                    checks[key] = checks.get(key, True) and value is None
                    none_count += 1
                else:
                    if not finite(value):
                        raise ValueError('Supported saved forecast became unsupported')
                    max_error = max(max_error, abs(value-expected))
                    finite_count += 1
        checks[identifier+'/admission_order'] = prefix_ids['shared'] == prefix_ids['history']
        admitted = identifier in set(ordered_ids)
        truth = [None if value is None else value+7.5 for value in flatten(record['service_targets'])] if admitted else [None]*6
        valid = [i for i in range(6) if finite(truth[i]) and all(finite(prediction[m][i]) for m in MODELS)]
        past = training[:len(prefix_ids['shared'])]
        same_line = [r for r in past if r['line_id'] == issue['line_id']]
        mean = issue['service_history'].get('mean_residual_scaled')
        reference = [r['service_history']['mean_residual_scaled'] for r in same_line]
        rows.append(dict(id=identifier, admitted=admitted, line_id=issue['line_id'], window=issue['window_name'],
            issue_at=issue['selected']['started_at'], admitted_ids=prefix_ids['shared'],
            history=history, history_mean_seconds=mean*60 if finite(mean) else None,
            history_extrapolation=not min(reference) <= mean <= max(reference) if reference and finite(mean) else None,
            truth=truth, prediction=prediction, scored_endpoints=len(valid),
            scored_reversal_endpoints=sum(truth[i] < 0 for i in valid),
            unscored_reversal_endpoints=sum(finite(truth[i]) and truth[i] < 0 and i not in valid for i in range(6))))
        branches = {arm: load(source/'training_branches'/str(record['index'])/arm) for arm in ('low', 'high')}
        case = old.pair_evidence(issue, branches['low'], branches['high'])
        rows[-1]['global_next_duration'] = duration(branches)
        checks[identifier+'/native_pair'] = case['integrity']['all_pass']
        pair_cases.append(dict(id=identifier, status=case['status'], integrity=case['integrity']))
        missing.update((case.get('service') or {}).get('missing', {}))
    checks['saved_forecast_replay'] = max_error <= 1e-10
    parallel = parallel_audit(day.get('parallel_pairs', []), journal_ids)
    checks['same_pair_actual_parallel'] = parallel['all_pass']
    integrity = audit(checks)
    macro = metrics(rows)
    by_line = {line: metrics([r for r in rows if r['line_id'] == line]) for line in LINES}
    scored = [r for r in rows if r['scored_endpoints']]
    reversal_states = [r['id'] for r in scored if r['scored_reversal_endpoints']]
    gate_checks = dict(native_integrity=integrity['all_pass'], nine_budgeted_pairs=len(ordered_ids) == 9,
        three_pairs_each_line=all(sum(r['line_id'] == line and r['admitted'] for r in rows) == 3 for line in LINES),
        all_selected_admitted=set(ordered_ids) == set(journal_ids),
        issue_state_diversity=all(day.get('sampling_summary', {}).get('by_line', {}).get(line, {}).get('unique_joint_strata_count', 0) >= 2 for line in LINES),
        supported_two_states_each_line=all(by_line[line]['history']['states'] >= 2 for line in LINES),
        scored_reversal_state=bool(reversal_states))
    gate = dict(passed=all(gate_checks.values()), checks=gate_checks,
        rule='Training coverage and scoreable response support only; no model selection or truth-directed resampling.')
    budget = dict(main_native_seconds=day['budget'].get('native_seconds'),
        training_labels=day['budget'].get('target_labels'), training_pairs=len(ordered_ids),
        paired_labels=day['budget'].get('paired_action_labels'), ordinary_labels=day['budget'].get('ordinary_labels'),
        zero_global_admitted=day['budget'].get('zero_global_admitted'),
        training_branches=len(day.get('training_branches', [])), **day.get('training_branch_budget', {}),
        heldout_truth_fitted=0, v20_truth_fitted=0)
    return dict(status='complete' if integrity['all_pass'] else 'integrity_failed', stage='train', priority='high',
        integrity=integrity, scientific_gate=gate, training_gate=gate, service_validation_eligible=gate['passed'], sampling=day.get('sampling_summary', {}), parallel=parallel,
        prediction_replay=dict(max_absolute_error=max_error, finite_values=finite_count, unsupported_values=none_count),
        macro=macro, by_line=by_line, rows=rows, native_pairs=pair_cases,
        coverage=dict(training_states=len(rows), admitted_states=len(ordered_ids), scored_states=len(scored),
            missing_selected_ids=day.get('sampling_summary', {}).get('missing_ids', []),
            unadmitted_ids=[r['id'] for r in rows if not r['admitted']],
            cold_start_or_unsupported=[r['id'] for r in rows if r['admitted'] and not r['scored_endpoints']],
            scored_reversal_states=reversal_states, scored_reversal_endpoints=sum(r['scored_reversal_endpoints'] for r in rows),
            unscored_reversal_endpoints=sum(r['unscored_reversal_endpoints'] for r in rows),
            missing_truth_endpoints=sum(not finite(v) for r in rows if r['admitted'] for v in r['truth']),
            extrapolated_scored_states=sum(r['history_extrapolation'] is True for r in scored)),
        missing=dict(missing), global_next_duration=dict(Counter(r['global_next_duration'] for r in rows)), budget=budget,
        decision=dict(service_validation_eligible=gate['passed'], stage_b='closed', independent_confirmation='not_submitted'),
        scope='This new day training rows only, in original admission order. No old heldout read or fit. Cold-start truth cannot establish scoreable reversal coverage. Correlated endpoints are summarized within states.')


def service_validation(root):
    root = Path(root)
    p = read(root/'protocol.json')
    stage, source = _paths(root, p)
    train = training_gate(root)
    if train['status'] != 'complete' or not train['scientific_gate']['passed']:
        return dict(status='blocked_by_train_gate', stage='service', integrity=train['integrity'],
            scientific_gate=dict(passed=False), decision=dict(stage_b='closed', independent_confirmation='not_submitted'))
    day, cases, outcomes, missing = load(source), [], [], Counter()
    journal = read(source/'acquisitions.json', {}).get('records', day.get('acquisitions', []))
    for line, window in product(p['target_lines'], p['heldout_windows']):
        identifier = 'heldout/'+line+'/'+window['name']
        issue = next((r for r in journal if r['id'] == identifier), None)
        pair_path = stage/'pairs'/'stable'/('seed'+str(p['service_seed']))/line/window['name']
        branches = {arm: load(pair_path/arm) for arm in ('low', 'high')}
        receipt, pair_failure = read(pair_path/'parallel_execution.json'), read(pair_path/'failure.json')
        if issue is not None:
            outcomes.extend(branches.values())
        case = dict(id=identifier, line_id=line, window=window['name'], **old.pair_evidence(issue, branches['low'], branches['high']))
        walls = {arm: branches[arm].get('execution_wall', {}) for arm in ('low', 'high')}
        intervals = [walls[arm].get(key) for arm in ('low', 'high') for key in ('native_started_wall', 'native_finished_wall')]
        overlap = min(walls[a]['native_finished_wall'] for a in walls)-max(walls[a]['native_started_wall'] for a in walls) if all(finite(v) for v in intervals) else None
        case['native_overlap_seconds'] = overlap
        case['launcher_status'] = 'not_selected' if issue is None else 'failed' if pair_failure else 'pending' if receipt is None else 'complete' if receipt.get('integrity', {}).get('all_pass') is True else 'integrity_failed'
        if pair_failure:
            case['launcher_failure'] = pair_failure
        case['global_next_duration'] = duration(branches)
        if issue is not None:
            reference = [r['history_mean_seconds'] for r in train['rows'] if r['line_id'] == line and finite(r['history_mean_seconds'])]
            mean = issue.get('service_history', {}).get('mean_residual_scaled')
            case['history_extrapolation'] = not min(reference) <= mean*60 <= max(reference) if reference and finite(mean) else None
        if issue is not None and all(v['status'] == 'complete' for v in branches.values()):
            parallel = finite(overlap) and overlap > 0 and walls['low'].get('pid') != walls['high'].get('pid')
            case['integrity']['check_count'] += 1
            if not parallel:
                case['integrity']['failed_checks'].append('same_pair_actual_parallel')
                case['integrity']['all_pass'] = False
                case['status'] = 'integrity_failed'
        if case.get('service'):
            missing.update(case['service']['missing'])
        cases.append(case)
    checks = dict(training_gate=train['scientific_gate']['passed'], native_pairs=all(c['integrity']['all_pass'] for c in cases),
        pair_launchers=all(c['launcher_status'] in ('complete', 'not_selected') for c in cases))
    integrity = audit(checks)
    pending = any(v['status'] == 'pending' for v in outcomes) or any(c['launcher_status'] == 'pending' for c in cases)
    failed = any(v['status'] == 'failed' for v in outcomes) or any(c['launcher_status'] == 'failed' for c in cases)
    terminal_bad = any((c['status'] == 'integrity_failed' and 'pending' not in c['branch_status'].values()) or c['launcher_status'] == 'integrity_failed' for c in cases)
    status = 'failed' if failed else 'integrity_failed' if terminal_bad else 'incomplete' if pending else 'complete' if integrity['all_pass'] else 'integrity_failed'
    gate = old.scientific_gate(cases, integrity)
    reversal_states, negative, hits = [], 0, 0
    for case in cases:
        score = case.get('service') or {}
        valid = [r for r in score.get('rows', []) if finite(r[2]) and finite(r[4])]
        if any(r[2] < 0 for r in valid):
            reversal_states.append(case['id'])
        negative += sum(r[2] < 0 for r in valid)
        hits += sum(r[2] < 0 and r[4] < 0 for r in valid)
    gate['checks'].update(fresh_reversal_state=bool(reversal_states), history_reversal_hit=hits > 0)
    gate['passed'] = all(gate['checks'].values())
    gate['reversal'] = dict(states=reversal_states, endpoints=negative, history_hits=hits if negative else None)
    return dict(status=status, stage='service', priority='high', integrity=integrity,
        scientific_gate=gate, cases=cases, missing=dict(missing),
        coverage=dict(history_supported_states=sum(c['history']['supported'] for c in cases),
            history_causal_states=sum(c['history']['causal'] for c in cases),
            history_extrapolated_states=sum(c.get('history_extrapolation') is True for c in cases),
            missing_selections=sum(c['status'] == 'selection_missing' for c in cases),
            global_next_duration=dict(Counter(c['global_next_duration'] for c in cases))),
        budget=dict(training_labels=train['budget']['training_labels'], training_pairs=9,
            heldout_branches_planned=12, heldout_branches_completed=sum(v['status'] == 'complete' for v in outcomes),
            heldout_suffix_seconds=sum(v.get('budget', {}).get('suffix_sim_seconds', 0) for v in outcomes),
            heldout_prefix_seconds=sum(v.get('budget', {}).get('prefix_sim_seconds', 0) for v in outcomes),
            replayed_labels=sum(v.get('budget', {}).get('replay_target_labels', 0) for v in outcomes),
            heldout_labels_enter_training=sum(v.get('budget', {}).get('truth_labels_enter_training', 0) for v in outcomes),
            failed_branches=sum(v['status'] == 'failed' for v in outcomes),
            failed_branch_saved_accounting=[v.get('failure', {}) for v in outcomes if v['status'] == 'failed'],
            accounting_complete=not pending and not failed),
        decision=dict(stage_b='closed_pending_manual_review', independent_confirmation='not_submitted'),
        scope='Fresh six-state stable service validation. Service gate is macro MAE improvement, no line regression, at least one fresh reversal state and a correct negative history forecast. No response-B submission; no control effect claim.')


def analyze(root, stage):
    return training_gate(root) if stage == 'train' else service_validation(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('train', 'service'), required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    while True:
        value = analyze(args.root, args.stage)
        if value['status'] != 'incomplete' or not args.wait:
            break
        time.sleep(20)
    text = json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n'
    if len(text.encode()) > 42000:
        raise RuntimeError('Reduce resident summary before export')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        stream.write(text)
    print(json.dumps(dict(status=value['status'], scientific_gate=value['scientific_gate']['passed'], bytes=len(text.encode()))))


if __name__ == '__main__':
    main()
