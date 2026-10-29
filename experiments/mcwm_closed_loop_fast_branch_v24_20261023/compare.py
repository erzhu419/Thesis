"""Small resident gate for a native fork equivalence probe; never starts a run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

FIELDS = ('selected', 'prefix', 'prediction', 'service_prediction', 'service_history',
    'execution', 'truth', 'service_shift', 'global_transition', 'suffix', 'native', 'clock_audit')
ARMS = ('low', 'high')


def read(path):
    return json.loads(Path(path).read_text())


def differences(expected, observed, path='', limit=8):
    """Exact recursive comparison; diagnostics never include large source values."""
    result = []

    def visit(left, right, name):
        if len(result) >= limit:
            return
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(left.keys() | right.keys()):
                child = name+'.'+key if name else key
                if key not in left or key not in right:
                    result.append(dict(path=child, reason='missing_key',
                        expected_present=key in left, observed_present=key in right))
                else:
                    visit(left[key], right[key], child)
                if len(result) >= limit:
                    break
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                result.append(dict(path=name, reason='length', expected=len(left), observed=len(right)))
            for index, (lvalue, rvalue) in enumerate(zip(left, right)):
                visit(lvalue, rvalue, f'{name}[{index}]')
        elif left != right:
            result.append(dict(path=name, reason='value', expected=repr(left)[:100],
                observed=repr(right)[:100]))

    visit(expected, observed, path)
    return result


def compare_fields(expected, observed):
    checks, mismatch = {}, {}
    for field in FIELDS:
        if field not in expected or field not in observed:
            checks[field] = False
            mismatch[field] = [dict(path=field, reason='required_field_missing')]
            continue
        mismatch[field] = differences(expected[field], observed[field], field)
        checks[field] = not mismatch[field]
    return dict(all_pass=all(checks.values()), checks=checks,
        mismatches={key: value for key, value in mismatch.items() if value})


def zero_training(row):
    delta = row.get('suffix', {}).get('training_delta', {})
    return bool(delta) and all(value == 0 for value in delta.values())


def complete(row):
    return row.get('status') == 'complete' and row.get('integrity', {}).get('all_pass') is True


def native_overlap(children):
    try:
        pids = [children[arm]['pid'] for arm in ARMS]
        starts = [children[arm]['native_started_wall'] for arm in ARMS]
        ends = [children[arm]['native_finished_wall'] for arm in ARMS]
        finite = all(isinstance(v, (int, float)) and math.isfinite(v) for v in starts+ends)
        if not finite or pids[0] == pids[1] or any(end <= start for start, end in zip(starts, ends)):
            return None
        return min(ends)-max(starts)
    except (KeyError, TypeError):
        return None


def compare(probe, source, target_id):
    probe, source = Path(probe), Path(source)
    required = [probe/'summary.json', probe/'live_fork.json', source/'acquisitions.json']
    required += [probe/arm/'summary.json' for arm in ARMS]
    failures = [path for path in [probe/'failure.json', *(probe/arm/'failure.json' for arm in ARMS)]
        if path.is_file()]
    if failures:
        return dict(status='native_failed', integrity=dict(all_pass=False),
            failure_files=[str(path) for path in failures], stage_b='closed')
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return dict(status='pending', missing=missing, stage_b='closed')
    records = read(source/'acquisitions.json')['records']
    selected = [row for row in records if row['id'] == target_id]
    if len(selected) != 1 or selected[0].get('role') != 'train':
        raise ValueError('Target must identify exactly one saved training acquisition')
    expected = selected[0]
    index = expected['index']
    original_paths = {arm:source/'training_branches'/str(index)/arm/'summary.json' for arm in ARMS}
    missing = [str(path) for path in original_paths.values() if not path.is_file()]
    if missing:
        return dict(status='pending', missing=missing, stage_b='closed')
    old = {arm:read(path) for arm, path in original_paths.items()}
    children = {arm:read(probe/arm/'summary.json') for arm in ARMS}
    parent, live = read(probe/'summary.json'), read(probe/'live_fork.json')
    replay = {arm:compare_fields(old[arm], children[arm]) for arm in ARMS}
    parent_replay = compare_fields(children['low'], parent)
    overlap = native_overlap(live.get('children', {}))
    summary_overlap = native_overlap({arm:children[arm].get('execution_wall', {}) for arm in ARMS})
    expected_clock = expected['selected']['started_at']
    checks = dict(
        old_branches_complete=all(complete(value) for value in old.values()),
        probe_branches_complete=all(complete(value) for value in children.values()),
        parent_complete=complete(parent),
        saved_target_matches=all(old[arm].get('id') == target_id and old[arm].get('selected') == expected['selected']
            for arm in ARMS),
        old_branches_exact=all(value['all_pass'] for value in replay.values()),
        parent_continuation_exact=parent_replay['all_pass'],
        parent_clock_fixed=live.get('native_before') == live.get('native_after') == expected_clock,
        parent_counters_fixed=bool(live.get('counters_before')) and live.get('counters_before') == live.get('counters_after'),
        parent_rng_fixed=live.get('parent_rng_unchanged') is True,
        parent_trace_cookie_fixed=live.get('parent_trace_cookie_unchanged') is True,
        children_actually_parallel=overlap is not None and overlap > 0 and summary_overlap is not None and summary_overlap > 0,
        child_native_wall_matches=all(live.get('children', {}).get(arm) == children[arm].get('execution_wall') for arm in ARMS),
        recorded_overlap_matches=overlap is not None and live.get('overlap_seconds') == overlap,
        actual_actions=all(children[arm].get('execution', {}).get('requested_hold_seconds') == hold
            and children[arm].get('execution', {}).get('executed_hold_seconds') == hold for arm, hold in (('low', 0.), ('high', 7.5))),
        all_suffixes_zero_training=all(zero_training(value) for value in [parent, *children.values()]),
        all_suffixes_300=all(value.get('suffix', {}).get('seconds') == 300
            and value.get('budget', {}).get('suffix_sim_seconds') == 300 for value in [parent, *children.values()]),
        children_no_prefix_replay=all(value.get('budget', {}).get('prefix_sim_seconds') == 0 for value in children.values()),
        parent_one_prefix_replay=parent.get('budget', {}).get('prefix_sim_seconds') == expected_clock,
        diagnostic_truth_not_fitted=all(value.get('budget', {}).get('truth_labels_enter_training') == 0
            for value in [parent, *children.values()]),
    )
    passed = all(checks.values())
    result = dict(status='complete' if passed else 'equivalence_failed',
        integrity=dict(all_pass=passed, checks=checks),
        target=dict(id=target_id, index=index, issue_seconds=expected_clock),
        exact_replay=dict(children_to_original=replay, parent_to_low_child=parent_replay),
        execution=dict(overlap_seconds=overlap,
            original_wall_seconds={arm:old[arm].get('wall_seconds') for arm in ARMS},
            fork_child_wall_seconds={arm:children[arm].get('wall_seconds') for arm in ARMS},
            parent_wall_seconds=parent.get('wall_seconds')),
        diagnostic_budget=dict(prefix_sim_seconds=expected_clock, suffixes=3,
            suffix_sim_seconds=900, new_training_labels=0, new_training_updates=0),
        decision=dict(native_fork_equivalence='passed' if passed else 'failed',
            full_day_acceleration_deployed=False, stage_b='closed',
            next='Review native equivalence before a separate full-day acceleration validation.'),
        evidence=dict(probe=str(probe), source=str(source), original_branches={arm:str(path) for arm, path in original_paths.items()}))
    if len(json.dumps(result, ensure_ascii=False).encode()) > 20000:
        raise RuntimeError('Gate exceeded small-summary budget')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target-id', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    while True:
        result = compare(args.probe, args.source, args.target_id)
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
    print(json.dumps(dict(status=result['status'], all_pass=result['integrity']['all_pass'],
        out=str(args.out)), ensure_ascii=False))
    if not result['integrity']['all_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
