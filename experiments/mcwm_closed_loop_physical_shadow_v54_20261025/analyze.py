"""Resident same-state contrasts; only the bounded final summary leaves server."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments.mcwm_closed_loop_joint_shadow_v47_20261024.summary import (
    compact, encode_bounded, signed_stats,
)
from experiments.mcwm_closed_loop_physical_shadow_v54_20261025.score import LINES, FIXED_HOLDS_SECONDS


ARMS = ('one_step', 'trace')
METRICS = ('first_actual_candidate_q_level', 'requested_adv1.875', 'requested_adv3.75', 'requested_adv7.5',
           'actor_proposal_seconds', 'deterministic_F_holding_seconds',
           'expected_F_holding_seconds')
STAT_COLUMNS = ['line', 'stage', 'metric', 'n', 'mean_signed', 'mean_absolute',
    'max_absolute', 'positive', 'negative', 'zero', 'nonzero_sign_changes',
    'longest_same_nonzero_sign_run', 'cancellation_fraction']
POINT_COLUMNS = ['line', 'updates', 'first_actual_q_one_step', 'first_actual_q_trace',
    'requested_adv1.875_one_step', 'requested_adv1.875_trace', 'requested_adv3.75_one_step', 'requested_adv3.75_trace',
    'requested_adv7.5_one_step', 'requested_adv7.5_trace', 'proposal_one_step', 'proposal_trace',
    'det_Fhold_one_step', 'det_Fhold_trace', 'E_Fhold_one_step', 'E_Fhold_trace',
    'E_Fhold_signed', 'E_Fhold_absolute', 'E_Fhold_CRN_block_sd',
    'best_hold_one_step', 'best_hold_trace', 'best_executed_disagrees',
    'candidate_F_aliases', 'actor_F_alias']


def _rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def contrast(left, right):
    """Trace minus synchronized M8 one-step on identical physical candidates."""
    equal = ('bank_id', 'line_id', 'sequence', 'issue_clock', 'memory_observed_at',
        'packet', 'cumulative_updates', 'snapshot_clock', 'candidate_labels',
        'candidate_proposal_actions', 'candidate_executed_actions',
        'candidate_executed_seconds', 'candidate_alias_groups',
        'fixed_requested_executed_seconds', 'fixed_executed_supported')
    if any(left[name] != right[name] for name in equal):
        raise ValueError('Both arms must score the same issue and the same F candidate union')
    if not left['shared_actor_candidates_complete'] or not right['shared_actor_candidates_complete']:
        raise ValueError('Formal comparisons require both arms actor candidates at every packet')
    a, b = left['actor'], right['actor']
    advantages = [high-low for low, high in zip(left['advantage_min'], right['advantage_min'])]
    supported = left['fixed_executed_supported']
    requested_observed = [supported[0] and supported[index] for index in range(4)]
    proposal = b['proposal_seconds']-a['proposal_seconds']
    deterministic = b['executed_seconds']-a['executed_seconds']
    expected = b['expected_holding_seconds']-a['expected_holding_seconds']
    differences = [high-low for low, high in zip(a['expected_holding_block_means'], b['expected_holding_block_means'])]
    mean = math.fsum(differences)/len(differences)
    block_sd = math.sqrt(math.fsum((value-mean)**2 for value in differences)/(len(differences)-1))
    ai, bi = left['best_candidate'], right['best_candidate']
    best_different = left['candidate_executed_actions'][ai] != right['candidate_executed_actions'][bi]
    alias = a['proposal_action'] != b['proposal_action'] and a['executed_action'] == b['executed_action']
    initial_exact = (left['q_heads'] == right['q_heads'] and a == b)
    return dict(line=left['line_id'], updates=left['cumulative_updates'],
        packet=left['packet'], initial_exact=initial_exact,
        metrics=[right['q_min'][0]-left['q_min'][0],
                 *[advantages[index] if requested_observed[index] else None for index in (1, 2, 3)],
                 proposal, deterministic, expected],
        best_executed_disagrees=best_different, actor_F_alias=alias,
        actor_candidates_coincide=a['proposal_action'] == b['proposal_action'],
        actor_executed_coincide=a['executed_action'] == b['executed_action'],
        fixed_candidates_aliased=len(set(left['candidate_alias_groups'][:4])) < 4,
        point=[left['q_min'][0], right['q_min'][0],
            *[value for index in (1, 2, 3) for value in
                ((left['advantage_min'][index], right['advantage_min'][index])
                 if requested_observed[index] else (None, None))],
            a['proposal_seconds'], b['proposal_seconds'], a['executed_seconds'], b['executed_seconds'],
            a['expected_holding_seconds'], b['expected_holding_seconds'],
            expected, abs(expected), block_sd,
            left['candidate_executed_seconds'][ai], right['candidate_executed_seconds'][bi],
            int(best_different), len(left['candidate_alias_groups'])-left['unique_executed_candidates'], int(alias)])


def summarize_scores(one_step_records, trace_records, protocol):
    expected = protocol.get('score_after_updates', list(range(24, 457, 8)))
    for records in (one_step_records, trace_records):
        if [record['cumulative_updates'] for record in records] != expected:
            raise ValueError('The initial post24 and all54 response packet scores are required')
    series, points, metadata = {line: [] for line in LINES}, [], None
    export = protocol.get('export_after_updates', [24, 32, 240, 456])
    initial_exact = True
    for first, second in zip(one_step_records, trace_records):
        if any(first[name] != second[name] for name in ('packet', 'cumulative_updates', 'processed_at')):
            raise ValueError('Packet identity and physical admission clock must agree')
        left, right = ({row['line_id']: row for row in record['rows']} for record in (first, second))
        if set(left) != set(LINES) or set(right) != set(LINES) or len(first['rows']) != 3 or len(second['rows']) != 3:
            raise ValueError('The same three physical line roots must remain present')
        current_metadata = []
        for line_id, line in enumerate(LINES):
            a, b = left[line], right[line]
            result = contrast(a, b)
            initial_exact &= result['updates'] != 24 or result['initial_exact']
            series[line].append(result)
            current_metadata.append([line_id, a['bank_id'], a['sequence'], a['issue_clock'],
                a['memory_observed_at'], a['candidate_executed_seconds'][:4], a['candidate_alias_groups'][:4],
                a['fixed_requested_executed_seconds'], a['fixed_executed_supported']])
            if result['updates'] in export:
                points.append([line_id, result['updates'], *result['point']])
        if metadata is None:
            metadata = current_metadata
        elif metadata != current_metadata:
            raise ValueError('The fixed factual bank or fixed candidate F mapping changed')
    statistics, persistence = [], []
    for line_id, line in enumerate(LINES):
        active = [row for row in series[line] if row['updates'] > 24]
        stages = (active, [row for row in active if row['updates'] <= 240],
                  [row for row in active if row['updates'] > 240])
        for stage_id, rows in enumerate(stages):
            for metric_id in range(len(METRICS)):
                observed = [row['metrics'][metric_id] for row in rows if row['metrics'][metric_id] is not None]
                stats = signed_stats(observed) if observed else [0, None, None, None, 0, 0, 0, 0, 0]
                cancelled = (1.-abs(stats[1])/stats[2] if stats[2] else 0.) if observed else None
                statistics.append([line_id, stage_id, metric_id, *stats, cancelled])
        persistence.append([line_id, len(active),
            sum(row['best_executed_disagrees'] for row in active),
            sum(row['actor_candidates_coincide'] for row in active),
            sum(row['actor_executed_coincide'] for row in active),
            sum(row['actor_F_alias'] for row in active),
            sum(row['fixed_candidates_aliased'] for row in active)])
    return dict(points=points, statistics=statistics, persistence=persistence, bank=metadata,
        requested_executed_grid_coverage={LINES[row[0]]: dict(requested_seconds=row[7],
            actual_seconds=row[5], supported=row[8], unsupported_requests=[hold for hold, supported
                in zip(row[7], row[8]) if not supported]) for row in metadata},
        checks=dict(initial_post24_scores_exact=bool(initial_exact), same55_checkpoints=True,
            same_three_factual_issues=True, same_shared_actor_candidate_union=True,
            fixed_bank_and_F_preserved=True),
        direction='trace_minus_synchronized_M8_one_step')


def aggregate(workers, score_records, protocol):
    scenarios = protocol.get('scenarios', ['stable', 'local_speed_drop'])
    failed = [f'{case}/{arm}' for case in scenarios for arm in ARMS
        if workers[case][arm].get('status') != 'complete'
        or not workers[case][arm].get('proof', {}).get('all_pass', False)]
    if failed:
        return dict(status='failed', priority='high', integrity=dict(all_pass=False, failed_workers=failed),
            workers=workers, score_aggregation_skipped=True,
            missing_spend='Read worker failure receipts; no missing worker is counted as zero')
    cases, checks = {}, {}
    for case in scenarios:
        pair = summarize_scores(score_records[case]['one_step'], score_records[case]['trace'], protocol)
        one, trace = workers[case]['one_step'], workers[case]['trace']
        # Sum actual packet update intersections, excluding inter-worker waits
        # and the already retained one-step preflight. Envelopes alone can
        # overlap even when optimizer work is entirely serial.
        overlap = math.fsum(max(0., min(a[1], b[1])-max(a[0], b[0]))
            for a in one['wall']['update_intervals'] for b in trace['wall']['update_intervals'])
        budget_exact = all(worker['budget']['feedback_slots'] == 285
            and worker['budget']['warmup_updates_reused'] == 24
            and worker['budget']['response_updates'] == 432
            and worker['budget']['optimizer_updates'] == 456
            and worker['budget']['retained_preflight_response_updates'] == (8 if arm == 'one_step' else 0)
            and worker['budget']['actual_updates_this_process'] == (424 if arm == 'one_step' else 432)
            for arm, worker in (('one_step', one), ('trace', trace)))
        checks[case] = dict(**pair['checks'], workers_all_pass=True,
            budget285_and456_exact=budget_exact, actual_training_overlap=overlap > 0.)
        # These are compact worker receipts, not parameter or diagnostic arrays.
        cases[case] = dict(**pair, actual_training_overlap_seconds=max(0., overlap), workers=dict(one_step=one, trace=trace))
    all_pass = all(all(values.values()) for values in checks.values())
    return dict(status='complete' if all_pass else 'failed', priority='high',
        integrity=dict(all_pass=all_pass, checks=checks), cases=cases,
        labels=dict(lines=LINES, metrics=METRICS,
            stages=['all_response32_456', 'response32_240', 'response248_456'],
            candidate_requested_executed_seconds=FIXED_HOLDS_SECONDS),
        columns=dict(points=POINT_COLUMNS, statistics=STAT_COLUMNS,
            persistence=['line', 'packets', 'best_executed_disagreement_packets',
                'actor_candidates_coincide_packets', 'actor_executed_coincide_packets',
                'actor_F_alias_packets', 'fixed_candidate_alias_packets'],
            bank=['line', 'bank_id', 'sequence', 'issue_clock', 'memory_observed_at',
                  'fixed_candidate_executed_seconds', 'fixed_candidate_alias_groups',
                  'fixed_requested_executed_seconds', 'fixed_executed_supported']),
        preserved_contract=dict(initializer_updates=50000, slots_per_arm=285,
            reused_warmup_updates=24, response_updates_per_arm=432, total_updates_per_arm=456,
            horizon_seconds=24000, F_unchanged=True, trust_unchanged=True),
        new_budget=dict(new_feedback_slots=0, new_native_branches=0, new_native_seconds=0,
            response_shadow_updates=4*432, separate_lambda_zero_reference_updates=2*8,
            checkpoints_downloaded=0),
        interpretation=dict(advantages='Each candidate min-Q minus its own first actual candidate value; requested0 can be unreachable under completion floor',
            requested_grid='Try proposal h then8h through unchangedF; unsupported requested durations retain actualF actions and are not counted as successful physical coverage',
            holding='Analytic hurdle gate with128 fixed positive Gaussian CRNs; signed and absolute differences are both retained',
            cancellation='1-abs(mean signed)/mean absolute; zero when all differences are zero',
            dependence='Repeated snapshots of3 fixed issues per scenario are descriptive, not independent held-out states',
            scope='No counterfactual truth,300s regret,stable no-harm or passenger-cost improvement established'),
        decision=dict(automatic_native_branches=False,
            next_gate='Inspect supported reward-credit contribution separately from Q level and terminal bootstrap; advance to independent physical validation only if common-action advantages and F holding show persistent material separation without isolated line or state failure.'))


def _worker_brief(worker):
    if worker.get('status') != 'complete':
        return worker
    proof = worker['proof']
    checks = proof.get('checks', {})
    wall = {key: value for key, value in worker['wall'].items() if key != 'update_intervals'}
    wall['actual_update_intervals'] = len(worker['wall']['update_intervals'])
    return dict(status=worker['status'], budget=worker['budget'], wall=wall,
        proof=dict(all_pass=proof['all_pass'], checks_passed=sum(bool(v) for v in checks.values()),
            checks_total=len(checks), failed_checks=[name for name, value in checks.items() if not value]),
        diagnostics={key: worker[key] for key in ('target_diagnostics', 'extra_compute',
            'target_path', 'lambda_zero_preflight') if key in worker})


def main():
    protocol = json.loads((HERE/'protocol.json').read_text())
    workers, scores = {}, {}
    for case in protocol['scenarios']:
        workers[case], scores[case] = {}, {}
        for arm in ARMS:
            root = HERE/'runs'/case/arm
            receipt = root/'summary.json'
            workers[case][arm] = json.loads((receipt if receipt.exists() else root/'failure.json').read_text())
            if workers[case][arm].get('status') == 'complete':
                scores[case][arm] = _rows(root/'scores.jsonl')
    result = aggregate(workers, scores, protocol)
    output = HERE/'analysis'
    output.mkdir(exist_ok=True)
    with (output/'result_stats.json').open('x') as handle:
        json.dump(result, handle, separators=(',', ':'), allow_nan=False)
        handle.write('\n')
    bounded = compact(result)
    # Keep packet intervals resident. Their exact intersections above are the
    # parallel evidence; the downloadable receipt only needs their counts.
    for case in result.get('cases', {}):
        for arm in ARMS:
            brief = _worker_brief(workers[case][arm])
            bounded['cases'][case]['workers'][arm] = compact(brief)
            bounded['cases'][case]['workers'][arm]['wall'] = brief['wall']
    encoded = encode_bounded(bounded, protocol.get('summary_max_bytes', 24000))
    with (output/'summary.json').open('x') as handle:
        handle.write(encoded)
    print(json.dumps(dict(status=result['status'], bytes=len(encoded.encode()), integrity=result['integrity'])), flush=True)
    return result


if __name__ == '__main__':
    main()
