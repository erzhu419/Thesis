"""A fixed prospective 285-slot ledger, with no native collection or SGD.

Slot names describe *new* feedback rights.  They never re-authorize an old
physical row.  Six 45-event branches replace the old 9-pair/ordinary design;
fixed issue windows make all required feedback available before its first
planned packet, provided every branch actually closes all 45 native events.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


ARMS = ('low', 'high')
LINES = ('7X', '102X', '705X')
ISSUE_WINDOWS = ((1200., 1500.), (6600., 6900.), (12000., 12300.))
EVENTS_PER_ARM = 45
BRANCH_CAP_SECONDS = 300.
DELAY_SECONDS = 30.


def build_plan():
    """Allocate exactly 57 five-row packets and eight updates per packet."""
    old_pairs = [f'old_pair:{-i}' for i in range(1, 19)]
    retained = {line: {arm: f'old_pair:{-(6*i+j+1)}' for j, arm in enumerate(ARMS)}
                for i, line in enumerate(LINES)}
    retained_ids = {value for item in retained.values() for value in item.values()}
    continuation_funding = ([f'old_ordinary:{i}' for i in range(15, 267)] +
                            [value for value in old_pairs if value not in retained_ids])
    funding = iter(continuation_funding)
    slots = [dict(index=i, kind='ordinary', new_id=f'ordinary:{i}',
                  replaces=f'old_ordinary:{i}') for i in range(15)]
    pairs = []
    for pair_index, (line, window) in enumerate(zip(LINES, ISSUE_WINDOWS)):
        pair_id = f'train/{line}/fixed'
        first_packet = 4 + 18*pair_index
        first_deadline = 1200. + (first_packet-1)*300.
        latest_available = window[1] + BRANCH_CAP_SECONDS + DELAY_SECONDS
        pair = dict(id=pair_id, line_id=line, issue_window=list(window),
                    selection='first_issue_visible_jointly_executable_state_in_window',
                    root_holds_seconds={'low': 0., 'high': 1.875},
                    events_per_arm=EVENTS_PER_ARM, branch_cap_seconds=BRANCH_CAP_SECONDS,
                    first_packet=first_packet, final_packet=first_packet+17,
                    first_deadline=first_deadline,
                    final_deadline=first_deadline+17*300.,
                    final_packet_label_age_upper_bound_seconds=first_deadline+17*300.-window[0],
                    release_upper_bound_exclusive=latest_available,
                    release_margin_seconds=first_deadline-latest_available)
        pairs.append(pair)
        # Alternation keeps every admitted arm prefix contiguous; a root may
        # use only successors already assigned distinct slots and admitted.
        for event in range(EVENTS_PER_ARM):
            for arm in ARMS:
                slots.append(dict(index=len(slots), kind='branch', pair_id=pair_id,
                    arm=arm, event=event, new_id=f'{pair_id}/{arm}/{event}',
                    replaces=retained[line][arm] if event == 0 else next(funding)))
    packets = []
    for offset in range(57):
        rows = slots[offset*5:(offset+1)*5]
        packets.append(dict(index=offset+1, deadline=1200.+offset*300.,
            updates=8, slots=rows))
    counts = Counter(row['kind'] for row in slots)
    replacements = [row['replaces'] for row in slots]
    expected = {f'old_ordinary:{i}' for i in range(267)} | set(old_pairs)
    checks = dict(
        slots_285=len(slots) == 285,
        ordinary_15_branch_270=counts == {'ordinary': 15, 'branch': 270},
        six_roots_264_continuations=sum(row.get('event') == 0 for row in slots) == 6
            and sum(row.get('event', 0) > 0 for row in slots) == 264,
        each_old_abstract_entitlement_used_once=len(set(replacements)) == 285
            and set(replacements) == expected,
        fixed_57x5=all(len(packet['slots']) == 5 for packet in packets),
        fixed_57x8=sum(packet['updates'] for packet in packets) == 456,
        fixed_last_deadline_18000=packets[-1]['deadline'] == 18000.,
        all_branch_release_bounds_precede_first_packet=all(
            pair['release_upper_bound_exclusive'] < pair['first_deadline'] for pair in pairs),
        each_arm_prefix_admitted_in_order=all(
            [row['event'] for row in slots if row.get('pair_id') == pair['id']
             and row.get('arm') == arm] == list(range(45))
            for pair in pairs for arm in ARMS))
    return dict(status='prospective_budget_design', pairs=pairs, packets=packets,
        slots=slots, checks=checks, all_pass=all(checks.values()),
        financing=dict(old_ordinary=267, ordinary_retained=15, ordinary_replaced=252,
            old_paired_roots=18, paired_root_entitlements_retained=6,
            paired_root_entitlements_cancelled=12, continuation_slots=264),
        current_compute=dict(new_native_prefixes=0, new_native_branches=0,
            new_training_labels=0, new_optimizer_updates=0),
        interpretation=dict(
            actual_new_prefix_available=False, actual_branch_feasibility_proven=False,
            diagnostics_authorize_training=False, labels_are_new_physical_rows=True,
            zero_duration_events_consume_one_slot=True,
            physical_availability_is_not_budget_admission=True,
            trace_uses_only_already_admitted_contiguous_prefix=True,
            final45_available_only_at_last_arm_admission=True,
            short_missing_or_late_branch='fail_without_ordinary_fill_reselection_or_synthetic_state',
            stop_rule='first45_true_closures_or_issue_plus300_whichever_first_not_reward_dependent',
            ordinary_selection='five_latest_positive_dt_nonterminal_rows_released_in_each_of_first3_packet_windows_excluding_reserved_pair_source_issues',
            ordinary_and_branch_source_exclusion='each_root_factual_issue_reserved_before_feedback_ingest_no_third_outcome_label',
            limitations=['one_issue_window_per_line_confounds_line_and_window',
                'feedback_queue_spans18_packets_and_can_be_old_when_admitted',
                'no_three_line_two_window_coverage_or_control_improvement_claim']))


def admitted_prefixes(plan, packet_index):
    """Return only event counts whose distinct slots were admitted by this packet."""
    if not 0 <= packet_index <= 57:
        raise ValueError('packet_index must lie in 0..57')
    counts = {f"{pair['id']}/{arm}": 0 for pair in plan['pairs'] for arm in ARMS}
    for row in plan['slots'][:packet_index*5]:
        if row['kind'] == 'branch':
            key = f"{row['pair_id']}/{row['arm']}"
            if row['event'] != counts[key]:
                raise ValueError('Admitted branch rows are not a contiguous prefix')
            counts[key] += 1
    return counts


def audit_real_branch(rows, *, issue, first_deadline, cap_seconds=300., delay=30.):
    """Audit future REAL rows, including zero-dt rows, without admitting any.

    A diagnostic/heldout chain may be passed for a geometry check; this function
    never grants training rights.  Caller must retain prefix/branch/arm origin.
    """
    rows = [value.get('transition', value) for value in rows]
    reasons = []
    if len(rows) != EVENTS_PER_ARM:
        reasons.append('need_exactly45_real_closed_events')
    for index, row in enumerate(rows):
        if row['completed_at']-row['started_at'] != row['dt'] or row['dt'] < 0:
            reasons.append(f'event{index}:physical_clock_mismatch')
        if row['completed_at'] > issue+cap_seconds:
            reasons.append(f'event{index}:past_fixed_native_cap')
        if row['available_at'] < row['completed_at']+delay:
            reasons.append(f'event{index}:feedback_precedes_release')
        if row['available_at'] > first_deadline:
            reasons.append(f'event{index}:not_ready_for_first_fixed_packet')
        if row.get('terminal', False) and index != len(rows)-1:
            reasons.append(f'event{index}:trace_continues_past_native_terminal')
        if index == 0 and row['started_at'] != issue:
            reasons.append('root_not_at_declared_issue')
        if index:
            previous = rows[index-1]
            if (row['sequence'] != previous['sequence']+1
                    or row['started_at'] != previous['completed_at']):
                reasons.append(f'event{index}:native_event_was_skipped')
            for current, preceding in (('s', 'ns'), ('raw_s', 'raw_ns'),
                                       ('memory_s', 'memory_ns')):
                if row[current] != previous[preceding]:
                    reasons.append(f'event{index}:{current}_continuity_missing')
    return dict(all_pass=not reasons, reasons=reasons, real_events=len(rows),
        zero_duration_events=sum(row['dt'] == 0 for row in rows),
        physical_span_seconds=(rows[-1]['completed_at']-issue if rows else None),
        physically_available_at=max((row['available_at'] for row in rows), default=None),
        training_authorized=False, feedback_added=0)


def compact_plan(plan):
    """Complete slot/packet mapping in a small JSON rather than a large ledger."""
    def token(row):
        identity = (f"O{row['index']}" if row['kind'] == 'ordinary' else
            f"P{next(i for i, pair in enumerate(plan['pairs']) if pair['id'] == row['pair_id'])}"
            f"/{row['arm'][0]}/{row['event']}")
        return identity+'<='+row['replaces'].replace('old_ordinary:', 'O').replace('old_pair:', 'P')
    return {key: value for key, value in plan.items() if key not in ('slots', 'packets')} | dict(
        token_legend='O ordinary; P pair index in pairs; l/h arm; /N zero-based true-event ordinal; <= abstract old entitlement, never old physical row',
        packets=[dict(index=p['index'], deadline=p['deadline'], updates=p['updates'],
                      slots=[token(row) for row in p['slots']],
                      admitted_prefixes=admitted_prefixes(plan, p['index']))
                 for p in plan['packets']])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = compact_plan(build_plan())
    encoded = json.dumps(result, ensure_ascii=False, separators=(',', ':'))+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(encoded)
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()
