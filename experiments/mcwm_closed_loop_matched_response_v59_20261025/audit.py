"""Resident tensor comparison and measured overlap for the two matched workers."""
from __future__ import annotations

import json
from pathlib import Path


def overlap(left, right):
    i = j = 0
    total = 0.
    while i < len(left) and j < len(right):
        a, b = left[i], right[j]
        total += max(0., min(a[1], b[1])-max(a[0], b[0]))
        if a[1] <= b[1]:
            i += 1
        else:
            j += 1
    return total


def records(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_case(root):
    import torch
    from experiments.mcwm_closed_loop_paired_coverage_v37_20261024.bootstrap import formal
    from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal

    root = Path(root)
    initial = {arm: torch.load(root/arm/'initial_state.pt', map_location='cpu', weights_only=False)
               for arm in ('fresh','no_fresh')}
    rows = {arm: records(root/arm/'packets.jsonl') for arm in initial}
    summaries = {arm: json.loads((root/arm/'summary.json').read_text()) for arm in initial}
    checks = dict(same_initializer_parameters_Adam_and_all_RNG=_equal(initial['fresh'], initial['no_fresh']),
        both_workers_complete=all(value['integrity']['all_pass'] for value in summaries.values()),
        both57packets=all(len(value) == 57 for value in rows.values()),
        no_fresh_input_file=summaries['no_fresh']['source_file'] == 'covariates_dataset.json',
        no_fresh_storage=summaries['no_fresh']['replay']['stored_fields'] == ['a','memory_s','s'],
        same_prefix=summaries['fresh']['source_prefix'] == summaries['no_fresh']['source_prefix'])
    fields = ('packet', 'updates', 'optimizer_steps', 'sampling', 'rng', 'extra_rng')
    for field in fields:
        checks['all_packets_same_'+field] = len(rows['fresh']) == len(rows['no_fresh']) and all(
            left[field] == right[field] for left, right in zip(rows['fresh'], rows['no_fresh']))
    # The outcome-free export deliberately removes positive/zero-dt receipt
    # statistics. Compare its retained scheduling/identity fields, not outcomes.
    checks['all_packets_same_receipt'] = len(rows['fresh']) == len(rows['no_fresh']) and all(
        all(key in left['receipt'] and left['receipt'][key] == value
            for key,value in right['receipt'].items())
        for left,right in zip(rows['fresh'], rows['no_fresh']))
    intervals = {arm: [(value['update_started_wall'], value['update_finished_wall'])
                      for value in group] for arm,group in rows.items()}
    actual_overlap = overlap(intervals['fresh'], intervals['no_fresh'])
    checks['workers_actually_parallel'] = actual_overlap > 0.
    return dict(all_pass=all(checks.values()), checks=checks,
        optimizer_overlap_seconds=actual_overlap,
        first_update_wall={arm: values[0][0] for arm,values in intervals.items()},
        final_update_wall={arm: values[-1][1] for arm,values in intervals.items()},
        new_optimizer_updates=0, checkpoints_downloaded=0)


def write_audit_if_complete(root):
    root = Path(root)
    path = root/'matched_audit.json'
    if path.exists():
        return json.loads(path.read_text())
    if not all((root/arm/name).exists() for arm in ('fresh','no_fresh')
               for name in ('summary.json','initial_state.pt','packets.jsonl')):
        return None
    result = audit_case(root)
    with path.open('x') as handle:
        json.dump(result, handle, separators=(',', ':'), allow_nan=False)
        handle.write('\n')
    return result
