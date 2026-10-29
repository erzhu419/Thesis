"""Small resident comparison of fixed native/tail suffix interventions."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_state_response_v18_20261022.analyze import (
    ENDPOINTS, METRICS, MODELS, compact, effect, inspect_pair, prediction_gain, sign,
)

LINES = ('7X', '102X', '705X')
ARMS = ('low', 'high')
MAX_BYTES = 25000
PREFIX_CHECKS = ('full_native_prefix_exact', 'acquisition_predictions_exact',
    'selected_evidence_exact', 'source_receipts_exact', 'source_counters_exact')


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else default


def load_branch(path):
    value = read(Path(path)/'summary.json')
    if value is not None:
        return value
    failure = read(Path(path)/'failure.json')
    return dict(status='failed', failure={key: failure.get(key) for key in ('type', 'message')}) if failure else dict(status='missing')


def audit(checks):
    return dict(all_pass=all(checks.values()), check_count=len(checks),
        failed_checks=[key for key, ok in checks.items() if not ok])


def source_day(source):
    path = Path(source)/'runs/stable/seed6201'
    day = read(path/'summary.json', {})
    records = read(path/'acquisitions.json', {}).get('records', [])
    selected = {row['id']: row for row in records}
    budget, initializer = day.get('budget', {}), day.get('initializer', {})
    checks = dict(source_day_complete=day.get('status') == 'complete',
        source_integrity=day.get('integrity', {}).get('all_pass') is True,
        training_285=budget.get('target_labels') == 285,
        pair_two_slots=budget.get('paired_outcomes_per_pair') == 2,
        source_partition=budget.get('ordinary_labels') == 267 and budget.get('paired_action_labels') == 18,
        source_nine_pairs=budget.get('training_pairs') == 9,
        source_packets=budget.get('packets') == 57,
        source_duration=budget.get('native_seconds') == 24000.,
        source_initialization=initializer.get('initial_updates') == 50000,
        source_history=initializer.get('history_rows') == 32768 and initializer.get('source_rows') == 4096,
        source_policy_trust=budget.get('policy_updates') == budget.get('trust_updates') == 456,
        source_world_updates=budget.get('world_updates') == 1140,
        source_physical_fixed=day.get('integrity', {}).get('checks', {}).get('physical_source_fixed') is True,
        source_trust_rule=day.get('integrity', {}).get('checks', {}).get('original_trust_rule') is True)
    return path, selected, dict(integrity=audit(checks),
        budget={key: budget.get(key) for key in ('target_labels', 'ordinary_labels', 'paired_action_labels',
            'training_pairs', 'paired_outcomes_per_pair', 'packets', 'native_seconds',
            'policy_updates', 'trust_updates', 'world_updates')}, initializer=initializer)


def branch_checks(anchor, native, tail):
    checks = dict(native_complete=native.get('status') == 'complete', tail_complete=tail.get('status') == 'complete',
        native_integrity=native.get('integrity', {}).get('all_pass') is True,
        tail_integrity=tail.get('integrity', {}).get('all_pass') is True,
        selected_state_exact=native.get('selected') == tail.get('selected') == anchor['selected'],
        source_prefix_exact=native.get('prefix') == tail.get('prefix') == anchor['prefix'],
        source_prediction_exact=native.get('prediction') == tail.get('prediction') == anchor['prediction'],
        fork_batch_other_controls_exact=native.get('native', {}).get('other_batch_actions') is not None
            and native.get('native', {}).get('other_batch_actions') == tail.get('native', {}).get('other_batch_actions'),
        execution_exact=native.get('execution') == tail.get('execution'),
        no_new_training_labels=tail.get('budget', {}).get('truth_labels_enter_training') == 0,
        prefix_clock_exact=tail.get('budget', {}).get('prefix_sim_seconds') == anchor['selected']['started_at'])
    for name in PREFIX_CHECKS:
        checks[name] = all(v.get('integrity', {}).get('checks', {}).get(name) is True for v in (native, tail))
    phase = tail.get('phase_intervention', {})
    checks.update(tail_mapping=phase.get('mapping') == 'tail_after_fork_batch',
        actor_real_clock=phase.get('actor_real_clock') is True,
        fork_batch_native=phase.get('fork_batch_native') is True,
        tail_formula_exact=phase.get('tail_formula_exact') is True)
    return checks


def paths(root, day, anchor, line, arm):
    root = Path(root)
    if line == '7X' and arm == 'low':
        relative = read(root/'protocol.json')['intervention']['first_branch']['output_relative']
        tail_path = root/relative
    else:
        tail_path = root/'branches'/line/arm
    return (day/'training_branches'/str(anchor['index'])/arm,
        tail_path)


def one_arm(root, day, anchor, line, arm):
    native_path, tail_path = paths(root, day, anchor, line, arm)
    native, tail = load_branch(native_path), load_branch(tail_path)
    return native, tail, branch_checks(anchor, native, tail)


def model_effects(prediction, native_gain, tail_gain):
    result = {}
    for model in MODELS:
        predicted = prediction_gain(prediction, model)
        old = effect(predicted, native_gain)
        new = effect(predicted, tail_gain)
        result[model] = dict(prediction=predicted, native=old, tail=new,
            regret_change=new[-1]-old[-1] if new is not None and old is not None else None)
    return result


def analyze(source, root, *, preflight=False):
    day, records, source_info = source_day(source)
    lines = LINES[:1] if preflight else LINES
    rows, tail_budgets, branch_statuses = [], [], []
    for line in lines:
        target_id = f'train/{line}/late'
        anchor = records.get(target_id)
        if anchor is None:
            rows.append(dict(line_id=line, target_id=target_id, status='selection_missing',
                integrity=dict(all_pass=False, failed_checks=['fixed_anchor_selected'])))
            continue
        branches, checks = {}, {}
        for arm in ARMS[:1] if preflight else ARMS:
            native, tail, compared = one_arm(root, day, anchor, line, arm)
            branches[arm] = (native, tail)
            checks.update({arm+'/'+key: ok for key, ok in compared.items()})
            branch_statuses += [native.get('status'), tail.get('status')]
            tail_budgets.append(tail.get('budget', {}))
        selected = {key: anchor['selected'][key] for key in ('sequence', 'started_at', 'vehicle_id', 'station_id')}
        row = dict(line_id=line, target_id=target_id, selected=selected,
            external_seed=anchor['prefix']['external_seed'], actor_updates=anchor['prefix']['counters']['policy'],
            status='complete' if all(status != 'missing' for pair in branches.values() for status in [pair[0].get('status'), pair[1].get('status')]) else 'incomplete',
            branch_status={arm: dict(native=pair[0].get('status'), tail=pair[1].get('status')) for arm, pair in branches.items()},
            phase={arm: pair[1].get('phase_intervention', {}) for arm, pair in branches.items()})
        if preflight:
            # One original low arm is reused, not a seventh suffix.
            checks['low/actual_zero'] = all(v.get('execution') == dict(requested_hold_seconds=0., executed_hold_seconds=0.) for v in branches['low'])
            for policy, value in zip(('native', 'tail'), branches['low']):
                delta = value.get('suffix', {}).get('training_delta')
                checks[policy+'/suffix_frozen'] = isinstance(delta, dict) and bool(delta) and all(x == 0 for x in delta.values())
                checks[policy+'/suffix300'] = value.get('suffix', {}).get('seconds') == 300
                fixed = value.get('truth', {}).get('fixed300', {})
                checks[policy+'/fixed300_truth'] = fixed.get('status') == 'complete' and fixed.get('dt') == 300
            row['integrity'] = audit(checks)
            rows.append(row)
            continue
        cell = dict(seed=6201, scenario='stable', line_id=line, window_name='late')
        pairs = {policy: inspect_pair(cell, anchor, branches['low'][index], branches['high'][index])
            for index, policy in enumerate(('native', 'tail'))}
        checks.update({policy+'/'+key: ok for policy, pair in pairs.items() for key, ok in pair['integrity']['checks'].items()})
        gains = {policy: pair['truth']['fixed300']['gain'] for policy, pair in pairs.items()}
        ready = all(value is not None for value in gains.values())
        gain_by_policy = {policy: dict(zip(METRICS, gain)) if gain is not None else None
            for policy, gain in gains.items()}
        native_gain, tail_gain = gain_by_policy['native'], gain_by_policy['tail']
        row.update(integrity=audit(checks), duration={policy: pair['duration'] for policy, pair in pairs.items()},
            truth={policy: pair['truth'] for policy, pair in pairs.items()},
            delta_F_gain={key: tail_gain[key]-native_gain[key] for key in METRICS} if ready else None,
            true_sign_changed=sign(tail_gain['cost']) != sign(native_gain['cost']) if ready else None,
            model_effects=model_effects(anchor['prediction'], native_gain, tail_gain))
        rows.append(row)
    expected = 1 if preflight else 6
    failed_prefix = read(Path(root)/'protocol.json', {}).get('diagnostic_compute_budget', {}).get('failed_preflight_prefix', {})
    terminal = len(rows) == len(lines) and all(row['status'] != 'incomplete' for row in rows)
    checks = dict(source_contract=source_info['integrity']['all_pass'],
        fixed_anchors_selected=all(row['status'] != 'selection_missing' for row in rows),
        all_branches_terminal=terminal,
        all_intervention_checks=all(row['integrity']['all_pass'] for row in rows),
        diagnostic_truth_not_fit=all(b.get('truth_labels_enter_training') == 0 for b in tail_budgets),
        suffix_budget=len(tail_budgets) == expected and all(b.get('suffix_sim_seconds') == 300 for b in tail_budgets))
    result = dict(status='complete' if terminal else 'incomplete',
        mode='reused_first_low_preflight' if preflight else 'fixed_execution_phase_diagnosis',
        integrity=dict(all_pass=all(checks.values()), checks=checks), source=source_info,
        budget=dict(new_main_days=0, new_training_labels=0, diagnostic_action_outcomes=expected,
            completed_tail_outcomes=sum(pair['tail'] == 'complete' for row in rows
                for pair in row.get('branch_status', {}).values()),
            completed_tail_suffix_seconds=sum(b.get('suffix_sim_seconds', 0) for b in tail_budgets),
            suffix_seconds_planned=300*expected, reused_native_action_outcomes=expected,
            prefix_replay_seconds=sum(b.get('prefix_sim_seconds', 0) for b in tail_budgets),
            replayed_target_labels=sum(b.get('replay_target_labels', 0) for b in tail_budgets),
            replayed_policy_updates=sum(b.get('replay_policy_updates', 0) for b in tail_budgets),
            failed_prefix_replay=failed_prefix,
            total_prefix_replay_seconds=sum(b.get('prefix_sim_seconds', 0) for b in tail_budgets)+failed_prefix.get('native_seconds_replayed', 0),
            total_replayed_target_labels=sum(b.get('replay_target_labels', 0) for b in tail_budgets)+failed_prefix.get('replayed_target_labels', 0),
            total_replayed_policy_updates=sum(b.get('replay_policy_updates', 0) for b in tail_budgets)+failed_prefix.get('replayed_policy_updates', 0),
            truth_labels_enter_training=sum(b.get('truth_labels_enter_training', 0) for b in tail_budgets),
            preflight_is_first_of_six=True),
        branch_statuses=dict(Counter(branch_statuses)), cases=rows,
        decision=dict(independent_confirmation='not_submitted', deployment='not_enabled',
            next='Review local effect of the execution rule; three fixed states cannot establish the sole cause of v18 response errors.'),
        schema=dict(metric_order=METRICS, model_order=MODELS,
            effect_order=('predicted_wait_gain', 'predicted_ride_gain', 'predicted_cost_gain', 'wait_signed_error',
                'ride_signed_error', 'cost_signed_error', 'direction_correct', 'action_correct', 'regret'),
            gain_sign='G=C(0)-C(7.5), C=2*wait+ride; positive chooses7.5, ties0.',
            delta_F='tail gain minus native gain, same prefix, actor and genuine clock; predictions are never refitted.',
            clock='Global/same-car endpoints retain native event clocks; fixed300 has the same exact300-second clock.'))
    if not preflight:
        selected = [row for row in rows if 'truth' in row]
        result['missing'] = dict(selection=len(rows)-len(selected),
            endpoints={policy: {endpoint: sum(row['truth'][policy][endpoint]['gain'] is None
                for row in selected) for endpoint in ENDPOINTS} for policy in ('native', 'tail')})
        result['unsupported'] = {model: sum(row['model_effects'][model]['prediction'] is None
            for row in selected) for model in MODELS}
        result['duration_counts'] = {policy: dict(Counter(row['duration'][policy]
            for row in selected)) for policy in ('native', 'tail')}
        result['model_metrics'] = {}
        for model in MODELS:
            # Compare the same supported states with both complete fixed-clock targets.
            common = [row['model_effects'][model] for row in selected
                if row['model_effects'][model]['native'] is not None
                and row['model_effects'][model]['tail'] is not None]
            metrics = {}
            for policy in ('native', 'tail'):
                effects = [item[policy] for item in common]
                metrics[policy] = dict(n=len(effects), direction_correct=sum(e[6] for e in effects),
                    action_correct=sum(e[7] for e in effects),
                    regret_total=sum(e[8] for e in effects) if effects else None,
                    cost_gain_mae=sum(abs(e[5]) for e in effects)/len(effects) if effects else None)
            metrics['regret_change'] = sum(item['regret_change'] for item in common) if common else None
            result['model_metrics'][model] = metrics
    return compact(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    previous = None
    while True:
        value = analyze(args.source, args.root, preflight=args.preflight)
        state = (value['status'], value['branch_statuses'])
        if value['status'] == 'complete' or not args.wait:
            break
        if state != previous:
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='waiting_execution_phase_branches', **value['branch_statuses'])), flush=True)
            previous = state
        time.sleep(20)
    data = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'
    if len(data.encode()) > MAX_BYTES:
        raise RuntimeError('Reduce the resident summary before collection')
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(data)
    print(json.dumps(dict(status=value['status'], integrity=value['integrity'], bytes=len(data.encode())), ensure_ascii=False))


if __name__ == '__main__':
    main()
