"""No models or simulator: attribute existing real-chain targets on the server."""
from pathlib import Path
import json
import math
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, separators=(',', ':'), allow_nan=False)
        handle.write('\n')


def stats(values):
    import numpy as np
    array = np.asarray(values, dtype=float)
    return dict(mean=float(array.mean()), std=float(array.std(ddof=1)),
                min=float(array.min()), max=float(array.max()))


def compact_pair(pair, witness):
    arms = {}
    for arm in ('low', 'high'):
        value = pair[arm]
        algebra = value['algebraic_ablations']['cases']
        attenuation = value['attenuation']
        arms[arm] = dict(clock=value['clock'], attenuation_logs={group:
            [attenuation[group][field] for field in ('link_count', 'gamma', 'mixing', 'c', 'multiplication_rounding')]
            for group in ('all_active_links', 'zero_duration_links', 'positive_duration_links')},
            full={key: value['full'][key] for key in ('cutoff_weight', 'cutoff_bootstrap',
                'intermediate_bootstrap_minus_matching_q', 'cutoff_true_terminal',
                'supported_event_count', 'first_stop')},
            cutoff_perturbations=[[p[key] for key in ('realized_bootstrap_delta',
                'root_target_delta', 'unrounded_algebraic_response')]
                for p in value['cutoff_sensitivity']['perturbations']],
            algebraic_upper_weights={key: row['cutoff_weight'] for key, row in algebra.items()})
        links = witness[arm+'_links'] if witness is not None else None
        if links is not None:
            arms[arm]['first_common_cost_weight'] = dict(links=links,
                value=1. if links == 0 else value['links'][links-1]['root_reach_weight'])
    rows = []
    for prefix, lo, hi in zip(pair['fixed_event_prefix_differences'],
                              pair['low']['prefixes'], pair['high']['prefixes']):
        diff, parts = prefix['high_minus_low'], prefix['component_difference']
        rows.append([lo['event_count'], lo['clock']['physical_seconds'], hi['clock']['physical_seconds'],
            lo['cutoff_weight'], hi['cutoff_weight'], diff['trace_target'], parts['reward'],
            parts['bootstrap_minus_matching_q'], diff['cutoff_bootstrap'],
            diff['intermediate_bootstrap_minus_matching_q'], parts['rounding'],
            prefix['same_physical_endpoint']])
    full = pair['full_difference']
    return dict(arms=arms, prefix_rows=rows,
        first_ordinal_reward_difference=pair['first_paired_reward_difference_event'],
        full_difference=dict(high_minus_low=full['high_minus_low'],
            components=full['component_difference'], closure_residual=full['difference_closure_residual']))


def main():
    from experiments.mcwm_closed_loop_trace_target_v44_20261024.probe import _source
    from experiments.mcwm_closed_loop_trace_target_v44_20261024.trace import decompose_trace
    from experiments.mcwm_closed_loop_trace_target_v44_20261024.summary import paired_decomposition
    from experiments.mcwm_closed_loop_v43_review_20261024.reward_clock import (
        summarize_pair, _clock_costs,
    )
    from experiments.mcwm_closed_loop_credit_reach_v48_20261025.metrics import paired_diagnosis
    protocol, started = read(HERE/'protocol.json'), time.monotonic()
    out = HERE/'detail'
    out.mkdir(exist_ok=False)
    cases, checks, counts = {}, {}, dict(mean8_chains=0, source_lambda0_rows=0)
    for case in protocol['scenarios']:
        root = HERE.parent/protocol['source_version']/'runs'/case
        _, pilot, updates, snapshot, ordinary, all_rows, chains, credit = _source(root)
        native = read(HERE.parent/protocol['native_target_arrays']/'runs'/case/'probe_detail.json')
        meanroot = HERE.parent/protocol['mean8_arrays']/'runs'/case
        if not native['integrity']['all_pass']:
            raise RuntimeError('Existing native target audit failed')
        indexed = {(c['line_id'], c['arm']): c for c in chains}
        clock_pairs = {}
        for line in protocol['lines']:
            lo, hi = [indexed[(line, arm)] for arm in ('low', 'high')]
            clock = summarize_pair(lo['records'], hi['records'],
                root/'branches'/line/'low/suffix_trace.jsonl',
                root/'branches'/line/'high/suffix_trace.jsonl',
                root/'learning/credit_packets.jsonl', line_id=line)
            costs = [_clock_costs(c['records']) for c in (lo, hi)]
            common = sorted(costs[0].keys() & costs[1].keys())
            curve = [[t-float(lo['records'][0]['started_at']),
                      costs[1][t]['cost']-costs[0][t]['cost']] for t in common]
            nonzero = [(t, 1 if value > 0 else -1) for t, value in curve if abs(value) > 1e-7]
            crossings = [right[0] for left, right in zip(nonzero, nonzero[1:]) if left[1] != right[1]]
            physical = {}
            for arm, source in zip(('low', 'high'), (lo, hi)):
                value = read(source['branch']['summary_path'])
                physical[arm] = {key: value['physical'][key] for key in
                    ('departure', 'next_same_vehicle_arrival', 'coverage')}
            clock['common_clock_cost_sign_crossings_seconds'] = crossings
            clock['physical'] = physical
            for key in ('first_difference_credit', 'last_common_credit_at456', 'credit_columns'):
                clock.pop(key)
            write(out/(case+'_'+line+'_common_clocks.json'), curve)
            clock_pairs[line] = clock
        checkpoints = {}
        for checkpoint in protocol['checkpoints']:
            old = native['checkpoints'][checkpoint]
            source_bank = {(c['line_id'], c['arm']): c
                for c in old['seeds']['336202']['chains']}
            for chain in source_bank.values():
                if not decompose_trace(chain['rows'], chain['c'], 0.)['lambda_zero_native_exact']:
                    raise RuntimeError('Existing native lambda0 parity failed')
                counts['source_lambda0_rows'] += len(chain['rows'])
            for row in old['ordinary_rows_first_seed']:
                if not decompose_trace([row], [], 0.)['lambda_zero_native_exact']:
                    raise RuntimeError('Ordinary native lambda0 parity failed')
                counts['source_lambda0_rows'] += 1
            rows_by_line = {}
            for line in protocol['lines']:
                first, block_pairs = None, []
                for block in range(protocol['blocks']):
                    bank = read(meanroot/checkpoint/'detail'/f'{line}_block{block:02d}.json')
                    sourcechains, traces = {}, {}
                    for arm in ('low', 'high'):
                        cached = source_bank[(line, arm)]
                        raw = indexed[(line, arm)]['records']
                        rows = bank['raw_scores'][arm]['mean8']
                        if len(rows) != 45 or any(
                            a['sequence'] != b['sequence'] or a['started_at'] != b['started_at']
                            or a['completed_at'] != b['completed_at'] or a['dt'] != b['dt']
                            or a['reward_scaled'] != c['reward_scaled']
                            for a, b, c in zip(rows, raw, cached['rows'])):
                            raise RuntimeError('Cached M8 rows do not describe the original real chain')
                        c = cached['c']
                        logged = credit[24 if checkpoint == 'actual_post24' else 456][(line, arm)]
                        if c != [r['c'] for r in logged['rows']]:
                            raise RuntimeError('Executed correction coefficients differ from source logs')
                        sourcechains[arm] = dict(rows=rows, c=c)
                        traces[arm] = decompose_trace(rows, c, 1.)
                        zero = decompose_trace(rows, c, 0.)
                        if not zero['lambda_zero_native_exact']:
                            raise RuntimeError('M8 lambda0 target identity failed')
                        expected = bank['decomposition'][arm]['mean8']['trace_target']
                        if traces[arm]['root']['trace_target'] != expected:
                            raise RuntimeError('M8 full45 target no longer matches existing arrays')
                        counts['mean8_chains'] += 1
                    pair = paired_decomposition(traces['low'], traces['high'])
                    block_pairs.append(pair)
                    if block == 0:
                        first = paired_diagnosis(sourcechains['low'], sourcechains['high'])
                        for arm in ('low', 'high'):
                            if (not first[arm]['lambda_zero']['native_exact'] or
                                    max(abs(x['closure_residual']) for x in first[arm]['all_prefixes']) > 1e-8):
                                raise RuntimeError('Real-prefix native target accounting does not close')
                write(out/f'{case}_{checkpoint}_{line}.json', dict(first_block=first, blocks=block_pairs))
                fields = dict(target=[x['trace_difference'] for x in block_pairs],
                    reward=[x['components']['reward'] for x in block_pairs],
                    bootstrap_minus_q=[x['bootstrap_minus_matching_Q'] for x in block_pairs],
                    terminal_bootstrap=[x['terminal_bootstrap_difference'] for x in block_pairs],
                    common_weight_reward=[x['reward_value_vs_weight']['common_weight_value_difference'] for x in block_pairs],
                    reward_weight_change=[x['reward_value_vs_weight']['common_value_weight_difference'] for x in block_pairs])
                rows_by_line[line] = dict(block_stats={key: stats(values) for key, values in fields.items()},
                    first_block=compact_pair(first, clock_pairs[line]['first_common_cost_difference']))
            checkpoints[checkpoint] = rows_by_line
        checks[case] = dict(original_source_complete=True, real_state_memory_links=True,
            original285_unique_feedback=True, original456_updates=True,
            F_ratios_exact=True, cached_mean8_targets_exact=True, lambda0_exact=True,
            reward_ledger_consistent=all(check['reward_ledger_consistent']
                for pair in clock_pairs.values() for check in pair['ledger_checks'].values()),
            no_artificial_state_or_terminal=True)
        cases[case] = dict(clock_pairs=clock_pairs, checkpoints=checkpoints)
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(case=case, status='complete')), flush=True)
    # All real-prefix and link arrays stay resident. Only fixed summaries cross the network.
    passed = all(all(x.values()) for x in checks.values())
    result = dict(status='complete' if passed else 'failed', integrity=dict(all_pass=passed,
        checks=checks), protocol='protocol.json', counts=counts, cases=cases,
        schema=dict(prefix_columns=['event_count', 'low_seconds', 'high_seconds', 'low_cutoff_weight',
            'high_cutoff_weight', 'target_delta', 'reward_delta', 'bootstrap_minus_Q_delta',
            'cutoff_bootstrap_delta', 'intermediate_bootstrap_minus_Q_delta', 'rounding_delta',
            'same_actual_closure_clock'], differences='1.875_seconds_minus0_seconds',
            targets='scaled_critic_target_units; event-indexed, not300s_cost_or_matched_clock_return',
            clocks='raw_real_passenger_cost_units on shared actual closure clocks',
            estimator='v46 frozen v43 checkpoint M8 arrays;16 blocks are not new physical states',
            attenuation_log_columns=['link_count', 'log_gamma', 'log_mixing', 'log_c', 'log_float32_rounding'],
            cutoff_perturbation_columns=['realized_bootstrap_delta', 'actual_float32_root_delta', 'algebraic_root_response'],
            algebraic_upper_weights='c=1/mixing=1 accounting only; not proposed legal targets'),
        budget=dict(new_training_updates=0, new_feedback_slots=0, new_native_branches=0,
                    new_native_seconds=0, new_target_draws=0, checkpoints_loaded=0),
        elapsed_seconds=time.monotonic()-started)
    write(HERE/'resident_summary.json', result)
    from experiments.mcwm_closed_loop_trace_target_v44_20261024.summary import compact
    encoded = json.dumps(compact(result), separators=(',', ':'), allow_nan=False)+'\n'
    if len(encoded.encode()) > protocol['summary_max_bytes']:
        raise RuntimeError('Reduce the resident summary before pulling')
    with (HERE/'summary.json').open('x') as handle:
        handle.write(encoded)
    print(json.dumps(dict(status=result['status'], integrity=result['integrity'],
        bytes=len(encoded.encode()), seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
