"""Read-only case/checkpoint shard using resident real v43 rows and v44 arrays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, max_bytes=None):
    encoded = json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n'
    if max_bytes is not None and len(encoded.encode()) > max_bytes:
        raise RuntimeError('The summary exceeds the declared small-pull budget')
    with Path(path).open('x') as handle:
        handle.write(encoded)
    return len(encoded.encode())


def run(scenario, checkpoint):
    protocol = read(HERE/'protocol.json')
    if scenario not in protocol['scenarios'] or checkpoint not in protocol['checkpoints']:
        raise ValueError('Undeclared scenario/checkpoint shard')
    out = HERE/'runs'/scenario/checkpoint
    out.mkdir(parents=True, exist_ok=False)
    (out/'detail').mkdir()
    started = time.time()
    try:
        import numpy as np
        import torch
        from experiments.mcwm_closed_loop_positive_mean_v46_20261024.analysis import analyze_bank, _stats
        from experiments.mcwm_closed_loop_positive_mean_v46_20261024.estimator import score_row_block
        from experiments.mcwm_closed_loop_positive_mean_v46_20261024.summary import compact_result
        from experiments.mcwm_closed_loop_trace_components_v45_20261024.components import (
            decompose_components, compare_pair,
        )
        from experiments.mcwm_closed_loop_trace_target_v44_20261024.probe import (
            _loaded, jsonlines, optimizer_versions, replay_parity,
        )
        from experiments.mcwm_closed_loop_trace_target_v44_20261024.trace import decompose_trace
        from experiments.mcwm_closed_loop_multievent_credit_v38_20261024.frozen import frozen_state, unchanged
        from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _rng, _equal
        torch.set_num_threads(1)
        arms, lines = ('low', 'high'), protocol['lines']
        arrays_root = HERE.parent/protocol['array_source_version']/'runs'/scenario
        model_root = HERE.parent/protocol['model_source_version']/'runs'/scenario
        source, source_small = read(arrays_root/'probe_detail.json'), read(arrays_root/'summary.json')
        if not (source['integrity']['all_pass'] and source_small['integrity']['all_pass']
                and source['source_budget']['feedback_slots'] == 285
                and source['source_budget']['updates'] == 456
                and source['source_contract']['initialization_seed'] == protocol['training_seed']
                and source['source_contract']['native_external_seed'] == protocol['native_external_seed']):
            raise RuntimeError('The completed resident v44 source contract differs')
        old = source['checkpoints'][checkpoint]
        old_seeds = protocol['old_parity_seeds']
        bank = {seed: {(row['line_id'], row['arm']): row
                       for row in old['seeds'][str(seed)]['chains']} for seed in old_seeds}
        receipt = read(model_root/'response_receipt.json')
        if receipt['processed_at'] != 9630. or not receipt['all270_admitted_before_updates']:
            raise RuntimeError('The original v43 response admission boundary differs')
        ordinary = [row for index in (1, 2, 3)
                    for row in read(model_root/f'warmup_packet_{index}.json')['rows']]
        indexed = {row['line_id']: row for row in receipt['pairs']}
        if len(ordinary) != 15 or set(indexed) != set(lines):
            raise RuntimeError('The original15 warmup rows or three root pairs are missing')
        actual_rows = {}
        for line in lines:
            for arm in arms:
                rows = [item['transition'] for item in jsonlines(indexed[line]['branches'][arm]['events_path'])]
                if (len(rows) != 45 or rows[0]['sequence'] != indexed[line]['selected']['sequence']
                        or any(row['available_at'] > 9630. for row in rows)):
                    raise RuntimeError('A fixed real45-event chain is incomplete, late or misaligned')
                actual_rows[(line, arm)] = rows
        response = [actual_rows[(line, arm)][event]
                    for event in range(45) for line in lines for arm in arms]
        all_rows = ordinary+response
        if (len(all_rows) != 285 or len({row['feedback_id'] for row in all_rows}) != 285
                or [row['feedback_id'] for row in response] != receipt['feedback_ids']):
            raise RuntimeError('Original285 feedback slots or admitted response ordering differ')
        snapshot = read(model_root/'behavior_snapshot.json')
        path = snapshot['resident_checkpoint'] if checkpoint == 'actual_post24' else model_root/'learning/post456_model.pt'
        learner, payload = _loaded(path)
        update, slots = (24, 15) if checkpoint == 'actual_post24' else (456, 285)
        if learner.total_updates != 50000+update or learner.replay_buffer.target_received != slots:
            raise RuntimeError('The frozen source checkpoint no longer has its original admission/update budget')
        if checkpoint == 'actual_post24':
            if (payload['scenario'] != scenario or payload['seed'] != protocol['training_seed']
                    or payload['total_updates'] != 50024 or snapshot['total_updates'] != 50024):
                raise RuntimeError('The actual post24 behavior checkpoint identity differs')
        elif payload['updates'] != 456 or payload['source'] != 'new_v43_native_feedback':
            raise RuntimeError('The completed post456 checkpoint identity differs')
        before, rng, adam = frozen_state(learner), _rng(), optimizer_versions(learner)
        parity = replay_parity(learner, all_rows[:slots])
        counts = dict(old_native_row_reproductions=0, old_arm_trace_reproductions=0,
                      ordinary_native_rows=0, m1_native_first_draw_rows=0,
                      m1_lambda0_rows=0, m8_lambda0_rows=0,
                      fixed_qbar_comparisons=0, raw_positive_draws=0)
        maxima = dict(component_closure=0.)

        def lambda_zero(rows, c, estimator):
            result = decompose_trace(rows, c, 0.)
            original = np.asarray([row['native_target'] for row in rows], dtype=np.float32)
            actual = np.asarray(result['targets'], dtype=np.float32)
            if not result['lambda_zero_native_exact'] or not np.array_equal(original.view(np.uint32), actual.view(np.uint32)):
                raise RuntimeError('Lambda-zero must copy the corresponding M1 or M8 one-step targets exactly')
            counts[estimator+'_lambda0_rows'] += len(rows)

        ordinary_scores = []
        for index, row in enumerate(ordinary):
            scored = score_row_block(learner, row, None,
                draw_seeds=protocol['blocks'][0]['draw_seeds'], cached_m1=old['ordinary_rows_first_seed'][index])
            lambda_zero([scored['m1']], [], 'm1')
            lambda_zero([scored['m8']], [], 'm8')
            counts['ordinary_native_rows'] += 1
            counts['raw_positive_draws'] += 8
            ordinary_scores.append(scored)
        write(out/'detail/ordinary_first_block.json', ordinary_scores)
        results = {}
        for line in lines:
            analyzed_blocks = []
            root_targets = {arm: dict(m1=[], m8=[]) for arm in arms}
            for block in protocol['blocks']:
                block_id, seeds = block['block_id'], block['draw_seeds']
                rows_by_arm = {arm: dict(single=[[] for _ in seeds], mean8=[], draws=[]) for arm in arms}
                for event in range(45):
                    for arm in arms:
                        records = actual_rows[(line, arm)]
                        cached = bank[seeds[0]][(line, arm)]['rows'][event] if seeds[0] in bank else None
                        value = score_row_block(learner, records[event], records[event+1] if event < 44 else None,
                            draw_seeds=seeds, cached_m1=cached)
                        if not value['integrity']['all_pass'] or not value['first_m1_native_exact']:
                            raise RuntimeError('A frozen target score failed native parity or mutation checks')
                        counts['m1_native_first_draw_rows'] += 1
                        counts['raw_positive_draws'] += 8
                        if cached is not None:
                            counts['old_native_row_reproductions'] += 1
                        reference_qbar = bank[old_seeds[0]][(line, arm)]['rows'][event]['recorded_next_target_q']
                        if any(row['recorded_next_target_q'] != reference_qbar
                               for row in value['singles']+[value['m8']]):
                            raise RuntimeError('Matching executed-action Qbar changed across estimator blocks')
                        counts['fixed_qbar_comparisons'] += 9
                        for index, row in enumerate(value['singles']):
                            rows_by_arm[arm]['single'][index].append(row)
                        rows_by_arm[arm]['mean8'].append(value['m8'])
                        rows_by_arm[arm]['draws'].append(value['draw_details'])
                decomposition = {}
                for arm in arms:
                    c = bank[old_seeds[0]][(line, arm)]['c']
                    if any(bank[seed][(line, arm)]['c'] != c for seed in old_seeds):
                        raise RuntimeError('Actual executed credit coefficients must remain independent of scoring seeds')
                    for rows in rows_by_arm[arm]['single']:
                        lambda_zero(rows, c, 'm1')
                    lambda_zero(rows_by_arm[arm]['mean8'], c, 'm8')
                    singles = [decompose_components(rows, c) for rows in rows_by_arm[arm]['single']]
                    mean8 = decompose_components(rows_by_arm[arm]['mean8'], c)
                    for scored in singles+[mean8]:
                        maxima['component_closure'] = max(maxima['component_closure'], abs(scored['reconstruction_residual']))
                    if maxima['component_closure'] >= 1e-8:
                        raise RuntimeError('Trace component accounting does not close')
                    decomposition[arm] = dict(single=singles, mean8=mean8)
                    root_targets[arm]['m1'].extend(row['trace_target'] for row in singles)
                    root_targets[arm]['m8'].append(mean8['trace_target'])
                    if seeds[0] in bank:
                        previous = decompose_components(bank[seeds[0]][(line, arm)]['rows'], c)
                        if singles[0]['trace_target'] != previous['trace_target']:
                            raise RuntimeError('An old seed no longer reproduces its complete45-event root trace')
                        counts['old_arm_trace_reproductions'] += 1
                single_pairs = [compare_pair(decomposition['low']['single'][index],
                    decomposition['high']['single'][index]) for index in range(8)]
                mean8_pair = compare_pair(decomposition['low']['mean8'], decomposition['high']['mean8'])
                analyzed_blocks.append(dict(block_id=block_id, single_pairs=single_pairs, mean8_pair=mean8_pair))
                write(out/'detail'/f'{line}_block{block_id:02d}.json', dict(block=block,
                    raw_scores=rows_by_arm, decomposition=decomposition,
                    single_pairs=single_pairs, mean8_pair=mean8_pair))
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='v46_frozen_joint_mean',
                    scenario=scenario, checkpoint=checkpoint, line=line, block=block_id,
                    blocks=protocol['block_count'], events_per_arm=45)), flush=True)
            results[line] = dict(**analyze_bank(analyzed_blocks), arm_stats={arm: {
                name: _stats(values) for name, values in by_estimator.items()}
                for arm, by_estimator in root_targets.items()})
        frozen = unchanged(learner, before, rng)
        if not _equal(adam, optimizer_versions(learner)):
            raise RuntimeError('Frozen scoring changed an Adam tensor')
        if (counts['ordinary_native_rows'] != 15 or counts['old_native_row_reproductions'] != 810
                or counts['old_arm_trace_reproductions'] != 18
                or counts['m1_native_first_draw_rows'] != 4320):
            raise RuntimeError('The predeclared scoring/parity scope is incomplete')
        proof = dict(all_pass=True, source_v44_audits_pass=True, original285_slots456_updates=True,
            admitted_rows=slots, source_updates=update, replay_values_and_memory_exact=parity['all_pass'],
            model_adam_rng_unchanged=frozen['all_pass'], old_three_seed_rows_and_full_traces_exact=True,
            all285_first_seed_native_exact=True, lambda0_m1_native_exact=True,
            lambda0_m8_candidate_exact=True, F_gate_zero_branch_min2_c_Qbar_unchanged=True,
            first_draw_native_subset_order=True, root_intervention_excluded=True,
            fixed45_real_events=True, collector_cutoff_is_not_terminal=True,
            old_full_prefix_audits_reused=True, current_snapshot_future_rows_not_admitted=True,
            original_SGD_target_draws_replayed=False, counts=counts, maxima=maxima)
        value = dict(status='complete', scenario=scenario, checkpoint=checkpoint,
            lines=results, proof=proof, priority='high', wall_seconds=time.time()-started,
            source_budget=source['source_budget'], source_contract=source['source_contract'],
            diagnostic_budget=dict(new_training_updates=0, new_feedback_slots=0,
                new_native_branches=0, new_native_seconds=0, checkpoints_downloaded=0,
                resident_checkpoints_loaded=1, distinct_declared_scoring_seeds=128,
                physical_issue_states=3, fixed_real_events_per_arm=45),
            decision=dict(training_started=False, target_change_installed=False,
                next_gate='Review estimator variance, sign mixtures and surviving bootstrap-minus-Q means before any shadow learning.'))
        write(out/'proof.json', proof)
        write(out/'result_stats.json', value)
        small = compact_result(value)
        size = write(out/'summary.json', small, protocol['summary_max_bytes_per_shard'])
        print(json.dumps(dict(status='complete', scenario=scenario, checkpoint=checkpoint,
            bytes=size, wall_seconds=value['wall_seconds'], proof=proof)), flush=True)
        return value
    except Exception as error:
        write(out/'failure.json', dict(status='failed', type=type(error).__name__, message=str(error),
            new_training_updates=0, new_native_branches=0, elapsed_seconds=time.time()-started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    run(args.scenario, args.checkpoint)
