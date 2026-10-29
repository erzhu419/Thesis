"""Resident v38 batch analysis; only the bounded summary is transferred."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_multievent_credit_v38_20261024.analyze_chain import (
    MODELS, analyze_pair,
)


SCENARIOS = ('stable', 'local_speed_drop')
LINES = ('7X', '102X', '705X')
WINDOWS = ('early', 'late')
MAX_BYTES = 24000
METRIC_COLUMNS = ('states', 'nonzero_truth', 'zero_truth', 'q_direction_correct', 'q_regret',
    'first_target_direction_correct', 'first_target_regret', 'stitched_scored',
    'stitched_direction_correct_descriptive', 'stitched_regret_descriptive',
    'equal_endpoint_clocks', 'bootstrap_opposes_reward', 'bootstrap_dominates_reward')
MODEL_COLUMNS = ('q_advantage', 'q_direction_correct', 'q_regret', 'first_target_advantage',
    'first_target_direction_correct', 'first_target_regret', 'endpoint_reward_delta',
    'endpoint_bootstrap_delta', 'endpoint_stitched_delta', 'endpoint_continuation_gap_delta',
    'endpoint_native_td_delta', 'endpoint_target_lag_delta', 'endpoint_mixed_behavior_gap_delta')


def read(path):
    return json.loads(Path(path).read_text())


def optional(path):
    return read(path) if Path(path).is_file() else None


def jsonlines(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as output:
        json.dump(value, output, separators=(',', ':'), allow_nan=False)
        output.write('\n')


def _onset(result, selected):
    first = result['semantic_alignment']['first_differences']
    issue = selected['started_at']
    reward = first.get('r')
    return dict(fixed_cost=result['fixed_clock_cost']['first_differences'].get('cost'),
        fixed_wait=result['fixed_clock_cost']['first_differences'].get('wait'),
        fixed_ride=result['fixed_clock_cost']['first_differences'].get('ride'),
        event_reward=(dict(relative_completion=[clock-issue for clock in reward['completion_clocks']],
            same_issue_clock=reward['equal_issue_clock'],
            same_completion_clock=reward['equal_completion_clock'],
            high_minus_low=reward['high_minus_low']) if reward else None),
        bootstrap={model: (dict(relative_completion=[clock-issue for clock in
            first[model+'/bootstrap_contribution']['completion_clocks']],
            same_completion_clock=first[model+'/bootstrap_contribution']['equal_completion_clock'],
            high_minus_low=first[model+'/bootstrap_contribution']['high_minus_low'])
            if model+'/bootstrap_contribution' in first else None) for model in MODELS})


def _model_compact(point):
    q, td = point['initial_q'], point['first_step_target']
    endpoint = point['endpoint']
    difference = endpoint['high_minus_low'] if endpoint is not None else {}
    return [q['advantage_high_minus_low'], q['direction_correct'], q['regret'],
        td['advantage_high_minus_low'], td['direction_correct'], td['regret'],
        *[difference.get(key) for key in ('reward_scaled_sum', 'final_bootstrap', 'stitched_return',
            'continuation_gap_sum', 'native_td_min_sum', 'target_lag_gap_sum',
            'mixed_vs_recorded_target_q_gap_sum')]]


def state_analysis(case_root, out, scenario, line, window):
    slot = line+'/'+window
    native = case_root/'native'/'states'/slot.replace('/', '_')
    state_path = native/'state_summary.json'
    record = optional(state_path)
    source_paths = dict(state=str(state_path),
        branch={arm: str(native/arm/'summary.json') for arm in ('low', 'high')})
    full = dict(scenario=scenario, line_id=line, window=window, slot=slot, sources=source_paths)
    missing = []
    required = [state_path]
    for arm in ('low', 'high'):
        required.extend((native/arm/'summary.json', native/arm/'ticks.jsonl',
            case_root/'scored'/'states'/slot.replace('/', '_')/arm/'events.jsonl'))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        full.update(status='missing', missing=missing, integrity=dict(all_pass=False))
        write(out/scenario/(slot.replace('/', '_')+'.json'), full)
        return dict(slot=slot, status='missing', missing_files=len(missing)), None
    try:
        summaries = {arm: read(native/arm/'summary.json') for arm in ('low', 'high')}
        rows = {arm: jsonlines(case_root/'scored'/'states'/slot.replace('/', '_')/arm/'events.jsonl')
            for arm in ('low', 'high')}
        ticks = {arm: jsonlines(native/arm/'ticks.jsonl') for arm in ('low', 'high')}
        result = analyze_pair(rows['low'], rows['high'], ticks['low'], ticks['high'],
            selected=record['selected'],
            exact300_by_arm={arm: value['truth']['fixed300'] for arm, value in summaries.items()},
            endpoint_by_arm={arm: value['real_endpoint'] for arm, value in summaries.items()})
        receipt = record['pair_receipt']
        checks = dict(chain_accounting=result['integrity']['all_pass'],
            two_suffix_integrity=all(value['integrity']['all_pass'] for value in summaries.values()),
            common_issue=all(value['selected'] == record['selected'] for value in summaries.values()),
            same_external_seed=summaries['low']['external_seed'] == summaries['high']['external_seed'] == 106237,
            actual_parallel=receipt['all_pass'] is True and receipt['overlap_seconds'] > 0.,
            native_feasible=record['native_feasibility']['feasible'] is True,
            actual_0_1875=all(abs(summaries[arm]['execution']['native_applied_hold_seconds']-hold) <= 1e-5
                and abs(summaries[arm]['execution']['executed_hold_seconds']-hold) <= 1e-5
                for arm, hold in (('low', 0.), ('high', 1.875))),
            zero_suffix_training=all(value['integrity']['checks']['suffix_zero_training'] is True
                and value['budget']['new_training_labels'] == value['budget']['new_optimizer_updates'] == 0
                for value in summaries.values()),
            original_early_exact=window != 'early' or record['old_early_replay']['all_pass'] is True,
            late_native_window=window != 'late' or 19200. <= record['selected']['started_at'] < 19500.)
        all_pass = all(checks.values())
        full.update(status=result['status'] if all_pass else 'integrity_failed',
            integrity=dict(all_pass=all_pass, checks=checks), analysis=result)
        compact = dict(slot=slot, status=full['status'], integrity=all_pass,
            issue=record['selected']['started_at'], overlap=receipt['overlap_seconds'],
            benefit300=result.get('fixed300', {}).get('benefit_low_minus_high'),
            fixed300_high_minus_low=result.get('fixed300', {}).get('high_minus_low'),
            missing_endpoints=sum(value['real_endpoint']['status'] != 'observed' for value in summaries.values()),
            zero_duration=[value['chain']['zero_duration_count'] for value in summaries.values()],
            overrun=[value['budget']['overrun_native_seconds'] for value in summaries.values()],
            endpoint_clocks=[value['real_endpoint']['clock'] for value in summaries.values()])
        if all_pass:
            compact.update(onset=_onset(result, record['selected']),
                models={model: _model_compact(point) for model, point in result['models'].items()})
            write(out/scenario/(slot.replace('/', '_')+'.json'), full)
            return compact, result
        compact['failed_checks'] = [key for key, value in checks.items() if not value]
        write(out/scenario/(slot.replace('/', '_')+'.json'), full)
        return compact, None
    except (ValueError, KeyError, TypeError, AssertionError) as error:
        full.update(status='analysis_failed', integrity=dict(all_pass=False),
            error=dict(type=type(error).__name__, message=str(error)))
        write(out/scenario/(slot.replace('/', '_')+'.json'), full)
        return dict(slot=slot, status='analysis_failed', error=full['error']), None


def _stats(results):
    results = list(results)
    values = {}
    for model in MODELS:
        counts = Counter()
        for result in results:
            counts['states'] += 1
            zero = result['fixed300']['zero_cost_benefit']
            counts['zero_truth' if zero else 'nonzero_truth'] += 1
            point = result['models'][model]
            for source, prefix in (('initial_q', 'q'), ('first_step_target', 'first_target')):
                counts[prefix+'_direction_correct'] += point[source]['direction_correct'] is True
                counts[prefix+'_regret'] += point[source]['regret']
            endpoint = point['endpoint']
            if endpoint is not None:
                counts['stitched_scored'] += 1
                scores = endpoint['stitched_vs_fixed300_descriptive']
                counts['stitched_direction_correct_descriptive'] += scores['direction_correct'] is True
                counts['stitched_regret_descriptive'] += scores['regret']
                counts['equal_endpoint_clocks'] += endpoint['same_clock']
                difference = endpoint['high_minus_low']
                reward, bootstrap = difference['reward_scaled_sum'], difference['final_bootstrap']
                counts['bootstrap_opposes_reward'] += reward*bootstrap < 0.
                counts['bootstrap_dominates_reward'] += abs(bootstrap) > abs(reward)
        values[model] = [counts[column] for column in METRIC_COLUMNS]
    return values


def _budget(native_summary, native_failure, case_root):
    if native_summary is not None:
        return dict(native_summary['budget'], accounting='completed_native_summary')
    if native_failure is not None and 'budget' in native_failure:
        return dict(native_failure['budget'], accounting='failed_prefix_observed_budget')
    # Missing accounting stays unknown. It is not free simulation or a zero
    # cost replacement for an incomplete native prefix.
    return dict(new_native_prefixes=1 if (case_root/'native').exists() else None,
        prefix_native_seconds=native_failure.get('native_time') if native_failure else None,
        actual_suffix_native_seconds=None, overrun_native_seconds=None,
        new_training_labels=None, new_optimizer_updates=None, accounting='incomplete_accounting')


def analyze(runs, out):
    runs, out = Path(runs), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cases, all_results, all_gates = {}, {}, {}
    for scenario in SCENARIOS:
        root = runs/scenario
        pipeline = optional(root/'summary.json')
        native = optional(root/'native'/'summary.json')
        failure = optional(root/'native'/'failure.json')
        model_gate = optional(root/'model_gate.json')
        scored = optional(root/'scored'/'summary.json')
        model_proof = model_gate if model_gate is not None else scored.get('integrity') if scored else None
        native_checks = native['integrity']['checks'] if native else {}
        budget = _budget(native, failure, root)
        gates = dict(pipeline_finished=pipeline is not None,
            pipeline_integrity=pipeline is not None and pipeline.get('integrity', {}).get('all_pass') is True,
            original_prefix_exact=native_checks.get('full_v37_prefix_exact') is True,
            full24000=native_checks.get('full24000_parent') is True,
            source50000_initializer=model_proof is not None and model_proof.get('source_initializer_updates') == 50000,
            frozen285=model_proof is not None and model_proof.get('source_feedback_slots') == 285,
            frozen456=model_proof is not None and model_proof.get('source_updates') == 456,
            original_predictions_and_rng=model_proof is not None and model_proof.get('all_pass') is True,
            old12_truth_unread=model_proof is not None and model_proof.get('old_twelve_state_truth_read') is False,
            no_new_training=model_proof is not None and model_proof.get('new_updates') == model_proof.get('new_feedback') == 0
                and budget.get('new_training_labels') == budget.get('new_optimizer_updates') == 0,
            scoring_finished=scored is not None and scored.get('status') == 'complete',
            native_all_checks=native is not None and native.get('integrity', {}).get('all_pass') is True)
        states, results = [], {}
        for line in LINES:
            for window in WINDOWS:
                compact, result = state_analysis(root, out, scenario, line, window)
                states.append(compact)
                if result is not None:
                    results[compact['slot']] = result
        gates['six_valid_states'] = len(results) == 6
        gates['all_real_endpoints'] = len(results) == 6 and all(value['status'] == 'complete' for value in results.values())
        groups = dict(all=_stats(results.values()))
        groups.update({'line/'+line: _stats(value for slot, value in results.items() if slot.startswith(line+'/')) for line in LINES})
        groups.update({'window/'+window: _stats(value for slot, value in results.items() if slot.endswith('/'+window)) for window in WINDOWS})
        selection = native.get('selection', {}) if native else {}
        missing = [dict(slot=value['slot'], status=value['status'],
            visible_selection=selection.get(value['slot'])) for value in states if value['status'] != 'complete']
        cases[scenario] = dict(status='complete' if all(gates.values()) else 'partial', gates=gates,
            budget=budget, states=states, metrics=groups, missing_or_partial=missing,
            unsupported_visible_attempts={slot: value['reasons'] for slot, value in selection.items()
                if value.get('reasons')},
            native_failure=failure.get('message') if failure else None)
        all_results[scenario], all_gates[scenario] = results, gates
    complete = all(all(value.values()) for value in all_gates.values())
    stable = cases['stable']['metrics']
    regret_index = METRIC_COLUMNS.index('q_regret')
    stability = {group: {model: values[model][regret_index]-values['zero_memory'][regret_index]
        for model in ('per_head_level', 'shared_level')} for group, values in stable.items()}
    observations = dict(available_states=sum(len(value) for value in all_results.values()),
        expected_states=12, nonzero_fixed300=0, zero_fixed300=0,
        initial_reward_equal_but_fixed300_differs=0, later_event_reward_gap_by300=0,
        reward_gap_only_after300=0, asynchronous_real_endpoints=0)
    for results in all_results.values():
        for result in results.values():
            nonzero = not result['fixed300']['zero_cost_benefit']
            observations['nonzero_fixed300' if nonzero else 'zero_fixed300'] += 1
            first = result['semantic_alignment']['first_differences'].get('r')
            # First-step immediate rewards are available from each arm's
            # unambiguous first real transition, including zero-duration rows.
            equal_reward = (result['arms']['low']['models']['zero_memory']['first_step']['reward_raw_sum'] ==
                result['arms']['high']['models']['zero_memory']['first_step']['reward_raw_sum'])
            observations['initial_reward_equal_but_fixed300_differs'] += equal_reward and nonzero
            # Use the original issue clock, not a guessed event index. Every
            # analyzed arm retains the exact300-to-ledger endpoint clock.
            fixed_horizon = result['arms']['low']['partial_to300']['clock_to']
            if first:
                observations['later_event_reward_gap_by300'] += max(first['completion_clocks']) <= fixed_horizon
                observations['reward_gap_only_after300'] += max(first['completion_clocks']) > fixed_horizon
            endpoint = result['models']['zero_memory']['endpoint']
            observations['asynchronous_real_endpoints'] += endpoint is not None and not endpoint['same_clock']
    summary = dict(status='complete' if complete else 'partial',
        integrity=dict(all_pass=complete, cases=all_gates),
        schema=dict(model_state_columns=list(MODEL_COLUMNS), metric_columns=list(METRIC_COLUMNS),
            benefit='low cost minus high cost; positive favors1.875s',
            q_and_target='high minus low; positive favors1.875s',
            stitched='descriptive common-continuation return at observed true event endpoints'),
        cases=cases, observations=observations,
        stable_q_choice_regret_delta_vs_zero=stability,
        decision=dict(target_change_selected=False, new_training=False, new_native_branches=False,
            next_gate='Review delayed reward onset, CRN bootstrap, continuation and TD gaps before selecting one target change'
                if complete else 'Resolve missing coverage or integrity evidence before choosing a target change'),
        limitations=('Q-choice regret is a300s branch diagnostic, not full-day deployed-policy cost. '
            'Stitched targets use frozen mixed bootstrap under common recorded continuation, with different '
            'end clocks shown explicitly; no synthetic state or bootstrap at300s. Zero-gain truth is '
            'excluded from direction counts. Missing, unsupported and endpoint-censored cases remain separate.'))
    encoded = json.dumps(summary, separators=(',', ':'), allow_nan=False)+'\n'
    if len(encoded.encode()) > MAX_BYTES:
        # Full inputs and all witnesses already exist in per-state resident
        # files. Reduce only floating-point presentation in the small export.
        def compact(value):
            if isinstance(value, float):
                return float(format(value, '.7g'))
            if isinstance(value, dict):
                return {key: compact(item) for key, item in value.items()}
            if isinstance(value, list):
                return [compact(item) for item in value]
            return value
        summary = compact(summary)
        encoded = json.dumps(summary, separators=(',', ':'), allow_nan=False)+'\n'
    if len(encoded.encode()) > MAX_BYTES:
        raise RuntimeError('The bounded v38 summary exceeds24000bytes; full resident state diagnostics are retained')
    (out/'summary.json').write_text(encoded)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze(args.runs, args.out)
    except Exception as error:
        args.out.mkdir(parents=True, exist_ok=True)
        write(args.out/'failure.json', dict(type=type(error).__name__, message=str(error)))
        raise
    print(json.dumps(dict(status=result['status'], integrity=result['integrity']['all_pass'],
        states=result['observations']['available_states'], summary=str(args.out/'summary.json'))), flush=True)


if __name__ == '__main__':
    main()
