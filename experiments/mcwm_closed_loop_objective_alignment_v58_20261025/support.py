"""Read-only support audit of the event chains actually written by v38-v55.

An issue trace and a passenger ledger are different artifacts from a completed
Bellman transition. Missing transitions never acquire an inferred successor.
"""
from __future__ import annotations

import json
from pathlib import Path


VERSIONS = {
    'v38': 'mcwm_closed_loop_multievent_credit_v38_20261024',
    'v43': 'mcwm_closed_loop_sync_pilot_v43_20261024',
    'v44': 'mcwm_closed_loop_trace_target_v44_20261024',
    'v46': 'mcwm_closed_loop_positive_mean_v46_20261024',
    'v48': 'mcwm_closed_loop_credit_reach_v48_20261025',
    'v53': 'mcwm_closed_loop_sync_collection_v53_20261025',
    'v55': 'mcwm_closed_loop_physical_truth_v55_20261025',
}
REQUIRED = ('s', 'a', 'ns', 'r', 'dt', 'started_at', 'completed_at',
            'memory_s', 'memory_ns', 'terminal', 'sequence')


def read_rows(path):
    """Read one declared resident chain, never discover or concatenate files."""
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _transition(record):
    return record.get('transition', record)


def _zero_training(counters):
    if not counters or 'before' not in counters or 'after' not in counters:
        return dict(status='missing', verified=False)
    before, after = counters['before'], counters['after']
    exact = before == after and all(value == 0 for value in counters.get('delta', {}).values())
    return dict(status='verified' if exact else 'changed', verified=exact)


def validate_chain(records, *, expected_root=None, counters=None,
                   writer_expected=True, source_kind='events', horizon_seconds=300.):
    """Audit JSON TDs and their real endpoints; zero-duration events remain legal.

    ``source_kind='issue_trace'`` is an explicit unsupported input, even when
    its fields happen to resemble state/action observations.
    """
    training = _zero_training(counters)
    if source_kind == 'issue_trace':
        return dict(status='unsupported_artifact', legal=False, rows=0,
                    reason='issue traces do not contain completed TD feedback',
                    zero_training=training)
    if records is None:
        return dict(status='missing' if writer_expected else 'not_expected', legal=False,
                    rows=0, reason='no event-chain writer' if not writer_expected else 'declared chain absent',
                    zero_training=training)
    if not records:
        return dict(status='empty', legal=False, rows=0, zero_training=training)
    rows = [_transition(record) for record in records]
    absent = [{'row': i, 'fields': [key for key in REQUIRED if key not in row]}
              for i, row in enumerate(rows) if any(key not in row for key in REQUIRED)]
    if absent:
        memory_only = all(set(item['fields']) <= {'memory_s', 'memory_ns'} for item in absent)
        raw_complete = all(all(key in row and row[key] is not None for key in
            ('s', 'a', 'ns', 'r', 'dt', 'started_at', 'completed_at')) for row in rows)
        raw_valid = raw_complete and all(row['completed_at']-row['started_at'] == row['dt']
            and row['dt'] >= 0 for row in rows)
        return dict(status='memory_not_saved' if source_kind == 'summary_first_transition'
                    and memory_only else 'invalid_schema', legal=False, rows=len(rows),
                    missing_fields=absent, raw_physical_transition_available=raw_valid,
                    memory_writer_expected=source_kind != 'summary_first_transition',
                    zero_training=training)
    checks = dict(
        complete_state_action_endpoint=all(row[key] is not None for row in rows
            for key in ('s', 'a', 'ns', 'memory_s', 'memory_ns')),
        actual_clock_dt=all(row['completed_at']-row['started_at'] == row['dt']
            and row['dt'] >= 0 for row in rows),
        causal_memory=all(isinstance(row['memory_s'], dict) and isinstance(row['memory_ns'], dict)
            and row['memory_s'].get('observed_at') == row['started_at']
            and row['memory_ns'].get('observed_at') == row['completed_at'] for row in rows),
        continuous_sequence=all(left['sequence']+1 == right['sequence']
            for left, right in zip(rows, rows[1:])),
        exact_clock_links=all(left['completed_at'] == right['started_at']
            for left, right in zip(rows, rows[1:])),
        exact_state_links=all(left['ns'] == right['s'] for left, right in zip(rows, rows[1:])),
        exact_memory_links=all(left['memory_ns'] == right['memory_s']
            for left, right in zip(rows, rows[1:])),
        no_continuation_after_terminal=all(not row['terminal'] for row in rows[:-1]),
    )
    if expected_root is not None:
        checks['selected_root_exact'] = all(rows[0].get(key) == expected_root[key]
            for key in ('sequence', 'started_at', 's', 'memory_s') if key in expected_root)
    has_ledgers = all('ledger_s' in record and 'ledger_ns' in record for record in records)
    ledger = dict(status='not_expected', verified=None)
    if has_ledgers:
        matched = all(row['r'] == -(2*(record['ledger_ns']['accrued_wait_seconds']-
            record['ledger_s']['accrued_wait_seconds'])+
            record['ledger_ns']['accrued_in_vehicle_seconds']-
            record['ledger_s']['accrued_in_vehicle_seconds'])
            and record['ledger_s']['time_seconds'] == row['started_at']
            and record['ledger_ns']['time_seconds'] == row['completed_at']
            for record, row in zip(records, rows))
        checks['reward_ledger_exact'] = matched
        ledger = dict(status='verified' if matched else 'mismatch', verified=matched)
    start, end = rows[0]['started_at'], rows[-1]['completed_at']
    endpoints = [dict(sequence=row['sequence']+1, clock=row['completed_at'],
                      span=row['completed_at']-start, terminal=row['terminal']) for row in rows]
    before = [point for point in endpoints if point['span'] <= horizon_seconds]
    after = next((point for point in endpoints if point['span'] >= horizon_seconds), None)
    exact = next((point for point in endpoints if point['span'] == horizon_seconds), None)
    legal = all(checks.values())
    return dict(status='legal' if legal else 'invalid', legal=legal, rows=len(rows), checks=checks,
                zero_duration_rows=sum(row['dt'] == 0 for row in rows),
                root_clock=start, final_clock=end, span_seconds=end-start,
                endpoint_at_300=exact, first_endpoint_at_or_after_300=after,
                last_endpoint_at_or_before_300=before[-1] if before else None,
                reaches_300=legal and after is not None,
                actual_300_state=legal and exact is not None,
                terminal_at_end=bool(rows[-1]['terminal']),
                ledger=ledger, zero_training=training)


def validate_pair(low_records, high_records, low_meta, high_meta, *, horizon_seconds=300.):
    """Distinguish legal same-state action feedback from common-clock ledgers."""
    low = validate_chain(low_records, expected_root=low_meta.get('selected'),
                         counters=low_meta.get('counters'), horizon_seconds=horizon_seconds)
    high = validate_chain(high_records, expected_root=high_meta.get('selected'),
                          counters=high_meta.get('counters'), horizon_seconds=horizon_seconds)
    left = _transition(low_records[0]) if low_records else {}
    right = _transition(high_records[0]) if high_records else {}
    checks = dict(
        legal_low=low['legal'], legal_high=high['legal'],
        root_clock_exact=bool(left and right) and left.get('started_at') == right.get('started_at'),
        root_sequence_exact=bool(left and right) and left.get('sequence') == right.get('sequence'),
        root_state_exact=bool(left and right) and left.get('s') == right.get('s'),
        root_memory_exact=bool(left and right) and left.get('memory_s') == right.get('memory_s'),
        selected_context_exact=low_meta.get('selected') is not None
            and low_meta.get('selected') == high_meta.get('selected'),
        exogenous_seed_exact=low_meta.get('external_seed') is not None
            and low_meta.get('external_seed') == high_meta.get('external_seed'),
        suffix_zero_training=low['zero_training']['verified'] and high['zero_training']['verified'],
    )
    other_low = low_meta.get('native', {}).get('other_batch_actions')
    other_high = high_meta.get('native', {}).get('other_batch_actions')
    checks['other_batch_controls_exact'] = other_low is not None and other_low == other_high
    paired = all(checks.values())
    exact_clocks = paired and low['final_clock'] == high['final_clock']
    both300 = paired and low.get('reaches_300', False) and high.get('reaches_300', False)
    distinct = bool(left and right) and left.get('a') != right.get('a')
    return dict(status='supported' if paired else 'unsupported', paired_root_supported=paired,
                same_state_action_difference=paired and distinct, action_distinct=distinct,
                checks=checks, low=low, high=high,
                both_chains_reach_300=both300,
                both_exact_300_states=paired and low.get('actual_300_state', False)
                    and high.get('actual_300_state', False),
                final_clock_equal=exact_clocks,
                matched_long_horizon_soft_Q_return=False,
                boundary='Real paired chains permit their recorded return decomposition; '
                    'finite rewards need continuation entropy and genuine terminal bootstrap '
                    'before comparison with long-horizon soft Q.')


def inventory(experiments_root, versions=None, scenarios=('stable', 'local_speed_drop')):
    """Stat declared artifact names only; no recursive search and no model load."""
    experiments_root = Path(experiments_root)
    requested = tuple(versions or VERSIONS)
    if any(version not in VERSIONS for version in requested):
        raise ValueError('Only declared v38/v43/v44/v46/v48/v53/v55 writers are supported')
    patterns = {
        'v38': [('native/states/*/low/events.jsonl', 'completed_events', True),
                ('native/states/*/high/events.jsonl', 'completed_events', True),
                ('native/states/*/low/summary.json', 'paired_summary_and_real_endpoint', True),
                ('native/states/*/high/summary.json', 'paired_summary_and_real_endpoint', True)],
        'v43': [('branches/*/low/events.jsonl', 'completed_events', True),
                ('branches/*/high/events.jsonl', 'completed_events', True),
                ('response_receipt.json', 'paired_root_manifest', True)],
        'v44': [('probe_detail.json', 'scored_v43_arrays_not_new_endpoints', True)],
        'v46': [('*/detail/*_block*.json', 'scored_v43_arrays_not_new_endpoints', True)],
        'v48': [('*', 'cached_attribution_only_no_new_endpoints', False)],
        'v53': [('continuous_feedback.jsonl', 'one_factual_chain_no_action_pair', True),
                ('flow_coverage.json', 'real_endpoint_coverage', True)],
        'v55': [('states/*/state_summary.json', 'paired_root_manifest', True),
                ('states/*/low/summary.json', 'one_TD_and_300_ledger', True),
                ('states/*/high/summary.json', 'one_TD_and_300_ledger', True),
                ('states/*/low/events.jsonl', 'unwritten_event_chain', False),
                ('states/*/high/events.jsonl', 'unwritten_event_chain', False)],
    }
    output = {}
    for version in requested:
        base = experiments_root/VERSIONS[version]
        per_case = {}
        for scenario in scenarios:
            case = base/'detail' if version == 'v48' else base/'runs'/scenario
            artifacts = []
            for pattern, role, expected in patterns[version]:
                if version == 'v48':
                    pattern = scenario+'_*.json'
                found = sorted(case.glob(pattern))
                artifacts.append(dict(pattern=pattern, role=role, writer_expected=expected,
                    status='present' if found else 'missing' if expected else 'not_expected',
                    paths=[str(path) for path in found], count=len(found),
                    bytes=sum(path.stat().st_size for path in found if path.is_file())))
            per_case[scenario] = dict(case_exists=case.is_dir(), artifacts=artifacts,
                prefix_trace_is_completed_feedback=False)
        output[version] = per_case
    return output
