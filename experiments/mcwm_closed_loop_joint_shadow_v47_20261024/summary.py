"""Small worker receipts and descriptive, matched fixed-state score contrasts."""
from __future__ import annotations

import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
HOLDS = (0., .9375, 1.875, 3.75, 7.5, 15.)
LINES = ('7X', '102X', '705X')
STAGES = ('all_updates', 'warmup8_24', 'response32_456')
METRICS = ('q0_level', 'proposal_seconds', 'F_holding_seconds',
           'adv0.9375', 'adv1.875', 'adv3.75', 'adv7.5', 'adv15')
STAT_COLUMNS = ['line', 'stage', 'metric', 'n', 'mean_signed', 'mean_absolute',
    'max_absolute', 'positive', 'negative', 'zero', 'nonzero_sign_changes',
    'longest_same_nonzero_sign_run']
POINT_COLUMNS = ['line', 'updates', 'snapshot_clock', 'q0_M1', 'q0_M8',
    'proposal_M1', 'proposal_M8', 'holding_M1', 'holding_M8',
    'proposal_signed', 'proposal_absolute', 'holding_signed', 'holding_absolute',
    'best_index_M1', 'best_index_M8', 'best_executed_disagrees', 'actor_F_alias',
    *[f'adv_M1_{hold:g}' for hold in HOLDS[1:]],
    *[f'adv_M8_{hold:g}' for hold in HOLDS[1:]],
    *[f'adv_difference_{hold:g}' for hold in HOLDS[1:]]]


def compact(value):
    if isinstance(value, float):
        return float(f'{value:.9g}')
    if isinstance(value, (tuple, list)):
        return [compact(item) for item in value]
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()}
    return value


def encode_bounded(value, maximum):
    encoded = json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n'
    size = len(encoded.encode())
    if size > maximum:
        raise RuntimeError(f'Resident summary {size} bytes exceeds declared limit {maximum}')
    return encoded


def compact_worker(result):
    """Bound the runner's receipt; all 58 full score records remain resident."""
    fields = ('status', 'scenario', 'arm', 'phase', 'priority', 'proof', 'budget',
              'extra_compute', 'actual_target_path', 'wall', 'paths')
    value = compact({name: result[name] for name in fields})
    # Nine significant digits are insufficient for absolute epoch clocks.
    # Keep the recorded precision used by the actual-overlap calculation.
    value['wall'] = result['wall']
    protocol = json.loads((HERE/'protocol.json').read_text())
    encode_bounded(value, protocol['summary_max_bytes_per_worker'])
    return value


def signed_stats(values):
    values = list(values)
    signs = [(value > 0)-(value < 0) for value in values]
    nonzero = [sign for sign in signs if sign]
    changes = sum(left != right for left, right in zip(nonzero, nonzero[1:]))
    longest = run = 0
    previous = None
    for sign in signs:
        run = run+1 if sign and sign == previous else int(bool(sign))
        longest = max(longest, run)
        previous = sign
    return [len(values), math.fsum(values)/len(values),
        math.fsum(abs(value) for value in values)/len(values), max(map(abs, values)),
        signs.count(1), signs.count(-1), signs.count(0), changes, longest]


def contrast(left, right):
    """M8 minus M1 on the exact same executed grid and saved issue context."""
    equal_fields = ('bank_id', 'line_id', 'root_arm', 'state_alias_group', 'sequence',
        'issue_clock', 'canonical_clock', 'memory_observed_at',
        'candidate_proposals_seconds', 'candidate_executed_actions',
        'candidate_executed_seconds', 'candidate_alias_groups')
    if any(left[name] != right[name] for name in equal_fields):
        raise ValueError('The paired learners did not score the same saved root and executed grid')
    proposal = right['actor_proposal_seconds']-left['actor_proposal_seconds']
    holding = right['actor_executed_seconds']-left['actor_executed_seconds']
    advantages = [high-low for low, high in zip(left['advantage_min'], right['advantage_min'])]
    lbest, rbest = left['best_fixed_candidate'], right['best_fixed_candidate']
    different_best = (left['candidate_executed_actions'][lbest] !=
                      right['candidate_executed_actions'][rbest])
    actor_alias = (left['actor_proposal_action'] != right['actor_proposal_action'] and
                   left['actor_executed_action'] == right['actor_executed_action'])
    return dict(line=left['line_id'], q0_level=right['q_min'][0]-left['q_min'][0],
        proposal_seconds=proposal, F_holding_seconds=holding, advantages=advantages,
        metrics=[right['q_min'][0]-left['q_min'][0], proposal, holding, *advantages[1:]],
        best_executed_disagrees=bool(different_best), actor_F_alias=bool(actor_alias),
        initial_score_exact=(left['q_heads'] == right['q_heads'] and
            left['actor_proposal_action'] == right['actor_proposal_action'] and
            left['actor_executed_action'] == right['actor_executed_action']),
        point=[left['q_min'][0], right['q_min'][0],
            left['actor_proposal_seconds'], right['actor_proposal_seconds'],
            left['actor_executed_seconds'], right['actor_executed_seconds'],
            proposal, abs(proposal), holding, abs(holding), lbest, rbest,
            int(different_best), int(actor_alias),
            *left['advantage_min'][1:], *right['advantage_min'][1:], *advantages[1:]])


def summarize_scores(m1_records, m8_records, protocol):
    """Use all snapshots for persistence; export only declared checkpoints."""
    expected = protocol['score_after_updates']
    if ([record['cumulative_updates'] for record in m1_records] != expected or
            [record['cumulative_updates'] for record in m8_records] != expected):
        raise ValueError('Initial and all57 cumulative score checkpoints are required')
    series = {line: [] for line in LINES}
    points, bank_metadata, backups = [], None, None
    initial_exact = True
    for first, second in zip(m1_records, m8_records):
        updates = first['cumulative_updates']
        if any(first[name] != second[name] for name in ('packet', 'cumulative_updates', 'processed_at')):
            raise ValueError('Paired score clocks or packet indices differ')
        left = {row['bank_id']: row for row in first['rows']}
        right = {row['bank_id']: row for row in second['rows']}
        expected_bank = {f'{line}/{arm}' for line in LINES for arm in ('low', 'high')}
        if set(left) != expected_bank or set(right) != expected_bank:
            raise ValueError('The six source root records must be preserved')
        current_backups = {name: left['7X/low'][name] for name in ('online_backup', 'offline_objective')}
        if any(any(row[name] != current_backups[name] for name in current_backups)
               for row in [*left.values(), *right.values()]):
            raise ValueError('Native source backup partitions differ between score records')
        if backups is None:
            backups = current_backups
        elif backups != current_backups:
            raise ValueError('Native source backup partitions changed during replay')
        current_metadata = []
        for line_index, line in enumerate(LINES):
            # Both labeled response arms start at the same physical issue.
            # Compare them explicitly, then retain one representative in means.
            for bank in (left, right):
                low, high = bank[f'{line}/low'], bank[f'{line}/high']
                if any(low[name] != high[name] for name in ('state_alias_group',
                    'issue_clock', 'memory_observed_at', 'q_heads', 'actor_proposal_action',
                    'actor_executed_action', 'candidate_executed_actions')):
                    raise ValueError('The paired source roots no longer represent the same physical issue')
            low, high = left[f'{line}/low'], right[f'{line}/low']
            result = contrast(low, high)
            initial_exact &= updates != 0 or result['initial_score_exact']
            series[line].append(dict(updates=updates, **result))
            current_metadata.append([line_index, low['issue_clock'], low['canonical_clock'],
                low['memory_observed_at'], low['state_alias_group'],
                low['candidate_executed_seconds'], low['candidate_alias_groups']])
            if updates in protocol['export_after_updates']:
                points.append([line_index, updates, first['processed_at'], *result['point']])
        if bank_metadata is None:
            bank_metadata = current_metadata
        elif bank_metadata != current_metadata:
            raise ValueError('The fixed scoring bank or F changed between updates')
    stats, persistence = [], []
    for line_index, line in enumerate(LINES):
        active = [row for row in series[line] if row['updates'] > 0]
        stages = [active, [row for row in active if row['updates'] <= 24],
                  [row for row in active if row['updates'] >= 32]]
        for stage_index, rows in enumerate(stages):
            for metric_index in range(len(METRICS)):
                stats.append([line_index, stage_index, metric_index,
                              *signed_stats(row['metrics'][metric_index] for row in rows)])
        persistence.append([line_index, len(active),
            sum(row['best_executed_disagrees'] for row in active),
            sum(row['actor_F_alias'] for row in active),
            sum(row['F_holding_seconds'] != 0. for row in active)])
    return dict(points=points, statistics=stats, persistence=persistence,
        bank=bank_metadata, checks=dict(initial_scores_exact=bool(initial_exact),
            same58_checkpoints=True, common_executed_grid=True, six_roots_three_physical_issues=True,
            fixed_bank_and_F_preserved=True),
        configured_learning_modes=backups,
        estimator_scope='Only positive joint SAC estimation changes on actual target calls; configured offline objective does not identify online replay partitions',
        scope='M8_minus_M1_conditional_scores_no300s_truth_or_policy_cost')
