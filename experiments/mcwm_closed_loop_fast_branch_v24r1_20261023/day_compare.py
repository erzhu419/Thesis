"""Compare a complete live-fork execution day to its unchanged v23 science."""
from __future__ import annotations

import argparse
from itertools import zip_longest
import json
from pathlib import Path
import time

if __package__:
    from .compare import ARMS, compare_fields, complete, differences, native_overlap, read, zero_training
else:
    from compare import ARMS, compare_fields, complete, differences, native_overlap, read, zero_training


MAIN_FIELDS = ('initializer', 'budget', 'receipts', 'readouts', 'service_readouts',
    'clock_audit', 'design_coverage', 'freshness', 'cost', 'anchors', 'training_pairs',
    'history_coverage', 'sampling_summary', 'missing_cells', 'scenario')


def trace_comparison(expected_path, observed_path):
    """Read each resident row exactly once; retain no trace bodies or digests."""
    counts, different, first = [0, 0], 0, []
    with Path(expected_path).open() as expected, Path(observed_path).open() as observed:
        for index, (left, right) in enumerate(zip_longest(expected, observed), 1):
            counts[0] += left is not None
            counts[1] += right is not None
            if left is None or right is None:
                different += 1
                if len(first) < 3:
                    first.append(dict(line=index, reason='missing_row',
                        expected_present=left is not None, observed_present=right is not None))
                continue
            lvalue, rvalue = json.loads(left), json.loads(right)
            if lvalue != rvalue:
                different += 1
                if len(first) < 3:
                    first.append(dict(line=index, differences=differences(lvalue, rvalue, limit=3)))
    return dict(all_pass=counts[0] == counts[1] and counts[0] > 0 and different == 0,
        expected_rows=counts[0], observed_rows=counts[1], differing_rows=different, first_differences=first)


def field_comparison(expected, observed, fields):
    checks, first = {}, []
    for name in fields:
        if name not in expected or name not in observed:
            checks[name] = False
            if len(first) < 8:
                first.append(dict(path=name, reason='required_field_missing'))
        else:
            changed = differences(expected[name], observed[name], name, limit=2)
            checks[name] = not changed
            first.extend(changed[:max(0, 8-len(first))])
    return dict(all_pass=all(checks.values()), failed_fields=[k for k, value in checks.items() if not value],
        first_differences=first)


def brief_replay(result):
    """Bound diagnostics even when many correlated values drift together."""
    failed = [name for name, passed in result['checks'].items() if not passed]
    first = []
    for name in failed:
        first.extend(result['mismatches'][name][:max(0, 2-len(first))])
        if len(first) == 2:
            break
    return dict(all_pass=result['all_pass'], failed_fields=failed, first_differences=first)


def parallel_metrics(children):
    """Measured child occupancy; the ratio is not a serial-counterfactual speedup."""
    clocks = {arm:children[arm].get('execution_wall', {}) for arm in ARMS}
    overlap = native_overlap(clocks)
    if overlap is None:
        return None
    durations = {arm:clocks[arm]['native_finished_wall']-clocks[arm]['native_started_wall']
        for arm in ARMS}
    span = max(value['native_finished_wall'] for value in clocks.values())-min(
        value['native_started_wall'] for value in clocks.values())
    return dict(arm_native_seconds=durations, arm_native_seconds_sum=sum(durations.values()),
        pair_span_seconds=span, overlap_seconds=max(0., overlap),
        overlap_factor=sum(durations.values())/span,
        arm_wall_seconds={arm:children[arm].get('wall_seconds') for arm in ARMS})


def aggregate_parallel(pairs, name):
    values = [row['parallel'][name] for row in pairs]
    if any(value is None for value in values):
        return None
    arms = sum(value['arm_native_seconds_sum'] for value in values)
    spans = sum(value['pair_span_seconds'] for value in values)
    walls = [seconds for value in values for seconds in value['arm_wall_seconds'].values()]
    return dict(arm_native_seconds_sum=arms, pair_span_seconds_sum=spans,
        overlap_seconds_sum=sum(value['overlap_seconds'] for value in values),
        weighted_overlap_factor=arms/spans if spans > 0 else None,
        arm_wall_seconds_sum=sum(walls) if all(isinstance(value, (int, float)) for value in walls) else None)


def compare(fast, source):
    fast, source = Path(fast), Path(source)
    failures = [path for path in [fast/'failure.json', source/'failure.json',
        *fast.glob('training_branches/*/*/failure.json')] if path.is_file()]
    if failures:
        return dict(status='native_failed', integrity=dict(all_pass=False),
            failure_files=[str(path) for path in failures], stage_b='closed')
    required = [root/name for root in (fast, source)
        for name in ('summary.json', 'acquisitions.json', 'prefix_trace.jsonl')]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return dict(status='pending', missing=missing, stage_b='closed')
    old, new = read(source/'summary.json'), read(fast/'summary.json')
    records = {name:read(root/'acquisitions.json')['records'] for name, root in (('old', source), ('new', fast))}
    pairs = [row for row in records['old'] if row['role'] == 'train']
    expected_files = [root/'training_branches'/str(row['index'])/arm/'summary.json'
        for root in (source, fast) for row in pairs for arm in ARMS]
    expected_files += [fast/'training_branches'/str(row['index'])/'live_fork.json' for row in pairs]
    missing = [str(path) for path in expected_files if not path.is_file()]
    if missing:
        return dict(status='pending', missing=missing, stage_b='closed')
    main = field_comparison(old, new, MAIN_FIELDS)
    acquisitions = field_comparison(dict(records=records['old']), dict(records=records['new']), ('records',))
    traces = trace_comparison(source/'prefix_trace.jsonl', fast/'prefix_trace.jsonl')
    pair_results, branches = [], []
    for row in pairs:
        base = fast/'training_branches'/str(row['index'])
        live = read(base/'live_fork.json')
        routes = live.get('route_streams', {})
        old_children = {arm:read(source/'training_branches'/str(row['index'])/arm/'summary.json') for arm in ARMS}
        children = {arm:read(base/arm/'summary.json') for arm in ARMS}
        replay = {arm:brief_replay(compare_fields(old_children[arm], children[arm])) for arm in ARMS}
        issue = row['selected']['started_at']
        overlap = native_overlap(live.get('children', {}))
        child_overlap = native_overlap({arm:children[arm].get('execution_wall', {}) for arm in ARMS})
        checks = dict(
            original_native_integrity=all(complete(value) for value in old_children.values()),
            new_native_integrity=all(complete(value) for value in children.values()),
            exact_science=all(value['all_pass'] for value in replay.values()),
            parent_native_clock_unchanged=live.get('native_before') == live.get('native_after') == issue,
            parent_counters_unchanged=bool(live.get('counters_before')) and live.get('counters_before') == live.get('counters_after'),
            parent_rng_unchanged=live.get('parent_rng_unchanged') is True,
            parent_trace_cookie_unchanged=live.get('parent_trace_cookie_unchanged') is True,
            parent_route_offsets_unchanged=routes.get('parent_offsets_unchanged') is True
                and routes.get('parent_offsets_before') == routes.get('parent_offsets_after'),
            route_stream_count_recorded=type(routes.get('count')) is int and routes['count'] >= 0
                and isinstance(routes.get('parent_offsets_before'), dict)
                and len(routes['parent_offsets_before']) == routes['count'],
            actual_parallel=overlap is not None and overlap > 0 and child_overlap is not None and child_overlap > 0,
            native_wall_matches=all(live.get('children', {}).get(arm) == children[arm].get('execution_wall') for arm in ARMS),
            recorded_overlap_matches=overlap is not None and live.get('overlap_seconds') == overlap,
            actual_actions=all(children[arm].get('execution', {}).get('requested_hold_seconds') == hold
                and children[arm].get('execution', {}).get('executed_hold_seconds') == hold
                for arm, hold in (('low', 0.), ('high', 7.5))),
            zero_suffix_training=all(zero_training(value) for value in children.values()),
            no_truth_refit=all(value.get('budget', {}).get('truth_labels_enter_training') == 0 for value in children.values()),
            zero_child_prefix_replay=all(value.get('budget', {}).get('prefix_sim_seconds') == 0
                and value.get('budget', {}).get('replay_target_labels') == 0
                and value.get('budget', {}).get('replay_policy_updates') == 0 for value in children.values()),
            suffix_exact_300=all(value.get('suffix', {}).get('seconds') == 300
                and value.get('budget', {}).get('suffix_sim_seconds') == 300 for value in children.values()),
            inherited_prefix_declared=all(value.get('fork_reuse', {}).get('prefix_execution') == 'inherited_not_independently_replayed'
                and value.get('fork_reuse', {}).get('parent_verified_prefix_seconds') == issue
                and value.get('fork_reuse', {}).get('parent_recorded_issues') == value['prefix']['issued_count']
                and value.get('fork_reuse', {}).get('inherited_target_labels') == value['prefix']['counters']['target_labels']
                for value in children.values()),
        )
        pair_results.append(dict(id=row['id'], index=row['index'], issue_seconds=issue,
            all_pass=all(checks.values()), failed_checks=[key for key, value in checks.items() if not value],
            native_overlap_seconds=overlap, route_stream_count=routes.get('count'),
            parallel=dict(original_replay=parallel_metrics(old_children), fast_suffix=parallel_metrics(children)),
            replay=replay))
        branches.extend(children.values())
    aggregate = new.get('training_branch_budget', {})
    budget = new.get('budget', {})
    checks = dict(
        source_main_complete=complete(old), fast_main_complete=complete(new),
        main_science_exact=main['all_pass'], complete_trace_exact=traces['all_pass'],
        acquisitions_exact=acquisitions['all_pass'],
        acquisition_files_match_main=all(records[name] == value.get('acquisitions') for name, value in (('old', old), ('new', new))),
        nine_distinct_training_pairs=len(pairs) == 9 and len({row['id'] for row in pairs}) == 9
            and len({row['index'] for row in pairs}) == 9,
        eighteen_suffixes=len(branches) == 18,
        all_pairs_equivalent=all(value['all_pass'] for value in pair_results),
        route_stream_detachment_exercised=any(type(value['route_stream_count']) is int
            and value['route_stream_count'] > 0 for value in pair_results),
        original_initializer= new.get('initializer', {}).get('initial_updates') == 50000,
        budget_285=budget.get('target_labels') == 285 and budget.get('ordinary_labels') == 267
            and budget.get('paired_action_labels') == 18 and budget.get('training_pairs') == 9
            and budget.get('paired_outcomes_per_pair') == 2 and budget.get('packets') == 57,
        updates_unchanged=budget.get('policy_updates') == 456 and budget.get('trust_updates') == 456,
        main_native_24000=budget.get('native_seconds') == 24000,
        branch_compute_unchanged_endpoints=aggregate.get('suffix_sim_seconds') == 5400
            and sum(value['budget']['suffix_sim_seconds'] for value in branches) == 5400,
        repeated_prefix_training_eliminated=all(aggregate.get(key) == 0
            for key in ('prefix_sim_seconds', 'replay_target_labels', 'replay_policy_updates')),
    )
    passed = all(checks.values())
    old_wall, new_wall = old.get('wall_seconds'), new.get('wall_seconds')
    speedup = old_wall/new_wall if isinstance(old_wall, (int, float)) and isinstance(new_wall, (int, float)) and old_wall > 0 and new_wall > 0 else None
    result = dict(status='complete' if passed else 'equivalence_failed',
        integrity=dict(all_pass=passed, checks=checks), main_science=main,
        acquisitions=acquisitions, full_native_trace=traces, paired_suffixes=pair_results,
        compute=dict(main_native_seconds=budget.get('native_seconds'), training_labels=budget.get('target_labels'),
            paired_suffix_count=len(branches), suffix_sim_seconds=aggregate.get('suffix_sim_seconds'),
            child_prefix_replay_seconds=aggregate.get('prefix_sim_seconds'),
            child_training_replays=aggregate.get('replay_target_labels'),
            inherited_prefix_seconds=sum(row['selected']['started_at']*2 for row in pairs),
            total_native_seconds=budget.get('native_seconds', 0)+aggregate.get('suffix_sim_seconds', 0),
            original_day_wall_seconds=old_wall, fast_day_wall_seconds=new_wall, observed_wall_speedup=speedup,
            parallel=dict(original_replay=aggregate_parallel(pair_results, 'original_replay'),
                fast_suffix=aggregate_parallel(pair_results, 'fast_suffix'),
                interpretation='Overlap factor is measured arm occupancy divided by pair span, not a measured serial-counterfactual speedup. Original arms include prefix replay; fast arms execute only suffixes.')),
        decision=dict(execution_only=True, scientific_candidate_changed=False,
            ready_for_next_experiment=passed, independent_confirmation='not_submitted', stage_b='closed',
            scope='One same-seed execution benchmark; no independent scientific confirmation.'),
        evidence=dict(fast=str(fast), source=str(source)))
    if len(json.dumps(result, ensure_ascii=False).encode()) > 20000:
        raise RuntimeError('Day gate exceeded20KB; reduce diagnostics on the server')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fast', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    while True:
        result = compare(args.fast, args.source)
        if result['status'] != 'pending' or not args.wait:
            break
        time.sleep(15)
    if result['status'] == 'pending':
        print(json.dumps(result, ensure_ascii=False))
        raise SystemExit(2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as output:
        json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps(dict(status=result['status'], all_pass=result['integrity']['all_pass'], out=str(args.out))))
    if not result['integrity']['all_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
