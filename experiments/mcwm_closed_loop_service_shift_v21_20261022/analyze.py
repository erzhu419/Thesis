"""Resident service-history gate, followed by separate fresh response evidence."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import product
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.mcwm_closed_loop_state_response_v18_20261022.analyze import (
    aggregate, effect, prediction_gain, truth_endpoint,
)

LINES = ('7X', '102X', '705X')
KINDS = ('arrival', 'departure')
MODELS = ('zero', 'route', 'shared', 'candidate')
CELLS = {
    'service': [(6201, 'stable')],
    'response': [(6202, 'stable'), (6201, 'local_speed_drop'), (6202, 'local_speed_drop')],
}
PREFIX_CHECKS = ('full_native_prefix_exact', 'acquisition_predictions_exact',
    'selected_evidence_exact', 'source_receipts_exact', 'source_counters_exact',
    'suffix_zero_training', 'native_suffix_exact', 'actual_action_exact',
    'fixed300_truth', 'no_future_calibration', 'global_transition_observed')


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def compact(value):
    if isinstance(value, float):
        return float(format(value, '.8g'))
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [compact(v) for v in value]
    return value


def audit(checks):
    return dict(all_pass=all(checks.values()), check_count=len(checks),
        failed_checks=[k for k, value in checks.items() if not value])


def load(path):
    value = read(Path(path)/'summary.json')
    if value is not None:
        return value
    failure = read(Path(path)/'failure.json')
    return dict(status='failed', failure=failure) if failure else dict(status='pending')


def history_coverage(snapshot, issued_at):
    segments = snapshot.get('segments', [])
    latest = [r.get('latest') for r in segments]
    causal = len(latest) == 3 and all(isinstance(r, dict) and
        finite(r.get('completed_at')) and finite(r.get('observed_at')) and
        r['completed_at'] < issued_at and r['observed_at'] < issued_at for r in latest)
    return dict(supported=snapshot.get('status') == 'supported' and
        finite(snapshot.get('mean_residual_scaled')), causal=causal,
        supported_segments=sum(isinstance(r, dict) for r in latest),
        ages=[r.get('age_seconds', r.get('age')) if isinstance(r, dict) else None for r in latest])


def service_score(source, low, high):
    """Compare the two frozen forecasts on exactly the same observed endpoints."""
    observed = {a: v.get('service_shift', {}) for a, v in (('low', low), ('high', high))}
    prediction = source.get('service_prediction', {})
    errors = {m: [] for m in ('shared', 'history')}
    rows, missing = [], Counter()
    station_match = observed['low'].get('stations') == observed['high'].get('stations') and (
        len(observed['low'].get('stations', [])) == 3)
    complete_predictions = True
    for kind, index in product(KINDS, range(3)):
        values = {a: (v.get(kind, [])[index] if len(v.get(kind, [])) > index else None)
            for a, v in observed.items()}
        for arm, value in values.items():
            if value is None:
                statuses = observed[arm].get('status', {}).get(kind, [])
                reason = statuses[index] if len(statuses) > index else 'missing_field'
                missing[arm+'/'+kind+'/'+reason] += 1
        actual = values['high']-values['low'] if all(finite(x) for x in values.values()) else None
        pred = {m: (prediction.get(m, {}).get(kind, [])[index]
            if len(prediction.get(m, {}).get(kind, [])) > index else None) for m in errors}
        if actual is not None:
            complete_predictions &= all(finite(x) for x in pred.values())
            if all(finite(x) for x in pred.values()):
                for model in errors:
                    errors[model].append(abs(pred[model]-actual))
        rows.append([kind, index+1, actual, pred['shared'], pred['history']])
    counts = dict(Counter(r[0] for r in rows if r[2] is not None))
    downstream = any(r[1] >= 2 and r[2] is not None for r in rows)
    usable = station_match and complete_predictions and downstream and all(counts.get(k, 0) > 0 for k in KINDS)
    return dict(status='scored' if usable else 'unscored', stations_match=station_match,
        observed_counts=counts, downstream_observed=downstream, missing=dict(missing), rows=rows,
        mae={m: sum(e)/len(e) if usable and e else None for m, e in errors.items()},
        forecast_status={m: prediction.get(m, {}).get('status', 'missing') for m in errors})


def pair_evidence(source, low, high):
    if source is None:
        return dict(status='selection_missing', integrity=audit({'missing_retained': True}),
            branch_status={}, history=dict(supported=False, causal=False), service=None)
    checks = {}
    statuses = {a: v.get('status') for a, v in (('low', low), ('high', high))}
    for arm, value, hold in (('low', low, 0.), ('high', high, 7.5)):
        checks[arm+'/complete'] = value.get('status') == 'complete'
        checks[arm+'/runner'] = value.get('integrity', {}).get('all_pass') is True
        checks[arm+'/source'] = all(value.get(k) == source.get(k)
            for k in ('selected', 'prefix', 'prediction', 'service_history', 'service_prediction'))
        checks[arm+'/prefix_checks'] = all(value.get('integrity', {}).get('checks', {}).get(k) is True
            for k in PREFIX_CHECKS)
        delta = value.get('suffix', {}).get('training_delta')
        checks[arm+'/zero_updates'] = isinstance(delta, dict) and bool(delta) and all(x == 0 for x in delta.values())
        checks[arm+'/actual_action'] = value.get('execution') == dict(requested_hold_seconds=hold, executed_hold_seconds=hold)
        checks[arm+'/budget'] = (value.get('budget', {}).get('truth_labels_enter_training') == 0 and
            value.get('budget', {}).get('prefix_sim_seconds') == source['selected']['started_at'] and
            value.get('suffix', {}).get('seconds') == 300)
    checks['sibling_commands'] = low.get('native', {}).get('other_batch_actions') is not None and (
        low['native']['other_batch_actions'] == high.get('native', {}).get('other_batch_actions'))
    checks['external_seed'] = source.get('prefix', {}).get('external_seed') is not None
    integrity = audit(checks)
    return dict(status='complete' if integrity['all_pass'] else 'integrity_failed',
        integrity=integrity, branch_status=statuses,
        history=history_coverage(source.get('service_history', {}), source['selected']['started_at']),
        service=service_score(source, low, high) if integrity['all_pass'] else None)


def native_day(day, stage):
    b = day.get('budget', {})
    init = day.get('initializer', {})
    checks = dict(complete=day.get('status') == 'complete',
        runner=day.get('integrity', {}).get('all_pass') is True,
        initializer=init.get('initial_updates') == 50000 and init.get('history_rows') == 32768
            and init.get('source_rows') == 4096,
        labels=b.get('target_labels') == 285 and b.get('ordinary_labels') == 267
            and b.get('paired_action_labels') == 18 and b.get('training_pairs') == 9
            and b.get('paired_outcomes_per_pair') == 2,
        clock=b.get('native_seconds') == 24000,
        updates=b.get('policy_updates') == 456 and b.get('trust_updates') == 456,
        pair_coverage=day.get('design_coverage', {}).get('all_pass') is True,
        phase_readouts=('candidate' not in day.get('readouts', {})) if stage == 'service'
            else 'candidate' in day.get('readouts', {}))
    return audit(checks)


def scientific_gate(cases, integrity):
    coverage = len(cases) == 6 and all(c['history']['supported'] and c['history']['causal'] for c in cases)
    scored = [c for c in cases if (c.get('service') or {}).get('status') == 'scored']
    means = {m: sum(c['service']['mae'][m] for c in scored)/len(scored) if scored else None
        for m in ('shared', 'history')}
    by_line = {}
    for line in LINES:
        rows = [c for c in scored if c['line_id'] == line]
        by_line[line] = {m: sum(c['service']['mae'][m] for c in rows)/len(rows) if rows else None
            for m in ('shared', 'history')}
        by_line[line]['states'] = len(rows)
    checks = dict(native_integrity=integrity['all_pass'], history_all_six=coverage,
        three_lines_two_windows=len(cases) == 6 and
            all(len([c for c in cases if c['line_id'] == line]) == 2 for line in LINES),
        all_pairs_scored=len(scored) == 6,
        macro_mae_improved=len(scored) == 6 and means['history'] < means['shared'],
        no_line_regression=all(x['states'] == 2 and x['history'] <= x['shared'] for x in by_line.values()))
    return dict(passed=all(checks.values()), checks=checks, macro_mae_seconds=means,
        by_line=by_line, independent_states=len(scored),
        rule='Mean endpoint MAE within each state, then equal state weights; correlated future stops are not independent samples.')


def response_row(source, low, high):
    truth = {endpoint: None for endpoint in ('global_next', 'same_vehicle_next', 'fixed300')}
    for endpoint in truth:
        a, b = truth_endpoint(low, endpoint), truth_endpoint(high, endpoint)
        if a is not None and b is not None:
            truth[endpoint] = {k: a[k]-b[k] for k in ('dt', 'wait', 'ride', 'cost')}
    return dict(truth=truth, effects={m: effect(prediction_gain(source.get('prediction', {}), m), truth['fixed300']) for m in MODELS},
        duration='missing_global' if truth['global_next'] is None else 'zero_dt' if
            low['truth']['global_next']['dt'] == high['truth']['global_next']['dt'] == 0 else 'positive_dt')


def response_summary(cases):
    def summarize(rows):
        return {m: aggregate([c['response']['effects'][m] for c in rows if c.get('response') and c['response']['effects'][m] is not None],
            [c['response']['truth']['fixed300']['cost'] for c in rows if c.get('response') and c['response']['effects'][m] is not None]) for m in MODELS}
    stable = [c for c in cases if c['scenario'] == 'stable']
    comparisons = {}
    for base in ('zero', 'route', 'shared'):
        rows = [c for c in stable if c.get('response') and all(c['response']['effects'][m] is not None for m in ('candidate', base))]
        differences = [c['response']['effects']['candidate'][8]-c['response']['effects'][base][8] for c in rows]
        comparisons[base] = dict(states=len(rows), candidate_minus_baseline_regret=sum(differences) if rows else None,
            harmful_states=sum(x > 0 for x in differences), improved_states=sum(x < 0 for x in differences))
    return dict(overall=summarize(cases), strata=[dict(scenario=scene, line_id=line, window=window,
        metrics=summarize([c for c in cases if c['scenario'] == scene and c['line_id'] == line and c['window'] == window]))
        for scene, line, window in product(('stable', 'local_speed_drop'), LINES, ('heldout1', 'heldout2'))],
        stable_comparison= comparisons,
        unsupported={m: sum(prediction_gain(c.get('prediction', {}), m) is None for c in cases) for m in MODELS},
        duration=dict(Counter(c['response']['duration'] for c in cases if c.get('response'))))


def analyze(root, stage):
    root = Path(root)
    p = read(root/'protocol.json')
    stage_path = root/('stage_a' if stage == 'service' else 'stage_b')
    cases, days, outcomes, missing = [], [], [], Counter()
    for seed, scenario in CELLS[stage]:
        path = stage_path/'runs'/scenario/f'seed{seed}'
        day = load(path)
        day_integrity = native_day(day, stage)
        days.append(dict(seed=seed, scenario=scenario, status=day['status'], integrity=day_integrity,
            budget={k: day.get('budget', {}).get(k) for k in ('target_labels', 'ordinary_labels', 'paired_action_labels',
                'training_pairs', 'paired_outcomes_per_pair', 'native_seconds', 'policy_updates', 'trust_updates')},
            training_branch_budget=day.get('training_branch_budget', {}),
            training_branch_count=len(day.get('training_branches', []))))
        journal = read(path/'acquisitions.json', {}).get('records', day.get('acquisitions', []))
        for line, window in product(p['target_lines'], p['heldout_windows']):
            target = f"heldout/{line}/{window['name']}"
            source = next((r for r in journal if r['id'] == target), None)
            branches = {a: load(stage_path/'pairs'/scenario/f'seed{seed}'/line/window['name']/a) for a in ('low', 'high')}
            if source is not None:
                outcomes.extend(branches.values())
            ev = pair_evidence(source, branches['low'], branches['high'])
            if source is not None:
                ev['integrity']['check_count'] += 1
                if source.get('prefix', {}).get('external_seed') != seed+(79000 if stage == 'service' else 80000):
                    ev['integrity']['failed_checks'].append('external_seed_contract')
                    ev['integrity']['all_pass'] = False
                    ev['status'] = 'integrity_failed'
                    ev['service'] = None
            case = dict(seed=seed, scenario=scenario, line_id=line, window=window['name'], target_id=target, **ev)
            if source is None:
                missing['selection'] += 1
            if ev['service']:
                missing.update({'service/'+k: v for k, v in ev['service']['missing'].items()})
            if stage == 'response':
                case['prediction'] = source.get('prediction', {}) if source else {}
                case['response'] = response_row(source, branches['low'], branches['high']) if ev['integrity']['all_pass'] and source else None
                if case['response']:
                    missing.update({'endpoint/'+k: 1 for k, value in case['response']['truth'].items() if value is None})
            cases.append(case)
    pending = any(d['status'] == 'pending' for d in days) or any(v['status'] == 'pending' for v in outcomes)
    failed = any(d['status'] == 'failed' for d in days) or any(v['status'] == 'failed' for v in outcomes)
    terminal_bad = any(d['status'] != 'pending' and not d['integrity']['all_pass'] for d in days) or any(
        c['status'] == 'integrity_failed' and 'pending' not in c['branch_status'].values() for c in cases)
    checks = dict(days=all(d['integrity']['all_pass'] for d in days),
        paired_native=all(c['integrity']['all_pass'] for c in cases if c['status'] != 'selection_missing'))
    integrity = audit(checks)
    status = 'failed' if failed else 'integrity_failed' if terminal_bad else 'incomplete' if pending else 'complete' if integrity['all_pass'] else 'integrity_failed'
    budget = dict(main_days_planned=len(CELLS[stage]), main_days_completed=sum(d['status'] == 'complete' for d in days),
        training_labels_planned=285*len(CELLS[stage]),
        training_labels_observed=sum(d['budget'].get('target_labels') or 0 for d in days),
        training_pairs_observed=sum(d['budget'].get('training_pairs') or 0 for d in days),
        training_branch_count=sum(d['training_branch_count'] for d in days),
        heldout_branch_planned=12*len(CELLS[stage]), heldout_branch_completed=sum(v['status'] == 'complete' for v in outcomes),
        heldout_suffix_seconds=sum(v.get('budget', {}).get('suffix_sim_seconds', 0) for v in outcomes),
        training_branch_suffix_seconds=sum(d['training_branch_budget'].get('suffix_sim_seconds', 0) for d in days),
        branch_prefix_replay_seconds=sum(d['training_branch_budget'].get('prefix_sim_seconds', 0) for d in days)+sum(v.get('budget', {}).get('prefix_sim_seconds', 0) for v in outcomes),
        replayed_labels=sum(d['training_branch_budget'].get('replay_target_labels', 0) for d in days)+sum(v.get('budget', {}).get('replay_target_labels', 0) for v in outcomes),
        replayed_policy_updates=sum(d['training_branch_budget'].get('replay_policy_updates', 0) for d in days)+sum(v.get('budget', {}).get('replay_policy_updates', 0) for v in outcomes),
        heldout_labels_enter_training=sum(v.get('budget', {}).get('truth_labels_enter_training', 0) for v in outcomes),
        failed_branches=sum(v['status'] == 'failed' for v in outcomes),
        failed_branch_saved_accounting=[v.get('failure', {}) for v in outcomes if v['status'] == 'failed'],
        accounting_complete=not failed and not pending,
        note='Replay labels are historical computation, not additional target labels. Failed prefixes have only saved accounting; never impute unknown work as zero.')
    result = dict(status=status, stage=stage, integrity=integrity, days=days, cases=cases,
        missing=dict(missing), budget=budget,
        decision=dict(independent_confirmation='not_submitted', v20_truth_fit=False,
            next='Review service gate before response tasks.' if stage == 'service' else 'Review direction, regret and stable degradation before another experiment.'),
        schema=dict(service_rows=['kind', 'next_stop_number', 'actual_high_minus_low_seconds', 'shared_prediction', 'history_prediction'],
            effects=['predicted_wait_gain', 'predicted_ride_gain', 'predicted_cost_gain', 'wait_error', 'ride_error', 'cost_error', 'direction_correct', 'action_correct', 'regret'],
            aggregate=['n', 'cost_mae', 'cost_mse', 'wait_mae', 'ride_mae', 'direction_correct', 'action_correct', 'predicted_ties', 'true_ties', 'regret_total']),
        limitations='Service gate uses six stable states; future stops within a state are correlated. Response stage has one fresh stable day and two shock days, no independent confirmation. All readouts are passive: selected-action regret does not establish candidate full-day closed-loop cost. Censored service clocks are absent observations, never zero times.')
    if stage == 'service':
        result['scientific_gate'] = scientific_gate(cases, integrity)
    else:
        result['response'] = response_summary(cases)
    return compact(result)


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    while True:
        value = analyze(args.root, args.stage)
        if value['status'] != 'incomplete' or not args.wait:
            break
        time.sleep(20)
    data = json.dumps(value, separators=(',', ':'), ensure_ascii=False, allow_nan=False)+'\n'
    limit = 20000 if args.stage == 'service' else 50000
    if len(data.encode()) > limit:
        raise RuntimeError('Reduce resident summary before export')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        f.write(data)
    print(json.dumps(dict(status=value['status'], integrity=value['integrity'],
        scientific_gate=value.get('scientific_gate', {}).get('passed'), bytes=len(data.encode()))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('service', 'response'), required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        failure = args.root/'analysis_failure.json'
        if not failure.exists():
            with failure.open('x') as f:
                json.dump(dict(status='failed', stage=args.stage, output=str(args.out), type=type(error).__name__, message=str(error)), f)
                f.write('\n')
        raise


if __name__ == '__main__':
    main()
