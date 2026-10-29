"""Final v40 same-window design: admit all response rows before response SGD.

This prospective manifest does not collect data, train, or grant an old
diagnostic transition training rights.  The superseded budget_plan calendar
is deliberately left intact as the submitted capacity draft.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


LINES = ('7X', '102X', '705X')
ARMS = ('low', 'high')
ISSUE_WINDOW = (9000., 9300.)
RELEASE_CLOCK = 9630.
EVENTS_PER_ARM = 45


def build_release_plan():
    old_pairs = [f'old_pair:{-i}' for i in range(1, 19)]
    roots = {line: {arm: f'old_pair:{-(6*i+j+1)}' for j, arm in enumerate(ARMS)}
             for i, line in enumerate(LINES)}
    retained = {value for pair in roots.values() for value in pair.values()}
    funding = iter([f'old_ordinary:{i}' for i in range(15, 267)] +
                   [value for value in old_pairs if value not in retained])
    slots = [dict(index=i, kind='ordinary', new_id=f'ordinary:{i}',
                  replaces=f'old_ordinary:{i}') for i in range(15)]
    pairs = [dict(id=f'train/{line}/common_window', line_id=line,
        issue_window=list(ISSUE_WINDOW),
        selection='first_issue_visible_jointly_executable_state_in_window',
        root_holds_seconds={'low': 0., 'high': 1.875},
        events_per_arm=45, native_branch_cap_seconds=300.,
        feedback_delay_seconds=30.,
        stop_rule='first45_true_closures_or_issue_plus300_whichever_first',
        release_upper_bound_exclusive=RELEASE_CLOCK,
        label_age_upper_bound_seconds=630.) for line in LINES]
    # Interleaving is fixed before data.  Every response row is admitted before
    # the first response gradient, so each line has the same 432 opportunities.
    for event in range(EVENTS_PER_ARM):
        for pair in pairs:
            for arm in ARMS:
                slots.append(dict(index=len(slots), kind='branch', pair_id=pair['id'],
                    line_id=pair['line_id'], arm=arm, event=event,
                    new_id=f"{pair['id']}/{arm}/{event}",
                    replaces=roots[pair['line_id']][arm] if event == 0 else next(funding)))
    packets = []
    for index in range(57):
        warmup = index < 3
        packets.append(dict(index=index+1,
            native_clock=(1200.+index*300. if warmup else RELEASE_CLOCK),
            phase='warmup' if warmup else 'response_admission',
            updates_after_packet=8 if warmup else 0,
            slots=slots[5*index:5*(index+1)]))
    phases = [dict(id=f'warmup_{i}_update', native_clock=1200.+(i-1)*300.,
                   admit_packet=i, slots_after_admission=5*i, updates=8,
                   updates_before=8*(i-1), updates_after=8*i)
              for i in range(1, 4)]
    phases += [dict(id='response_admit', native_clock=RELEASE_CLOCK,
                   packets=list(range(4, 58)), new_slots=270, updates=0,
                   require_all270_physically_released=True),
               dict(id='response_update', native_clock=RELEASE_CLOCK,
                    require_admitted_slots=285, updates_before=24,
                    updates=432, updates_after=456),
               dict(id='frozen', native_clock_interval=[RELEASE_CLOCK, 24000.],
                    updates_before=456, updates=0, updates_after=456)]
    replacements = [row['replaces'] for row in slots]
    expected_old = {f'old_ordinary:{i}' for i in range(267)} | set(old_pairs)
    checks = dict(
        slots285=len(slots) == 285,
        ordinary15_response270=Counter(row['kind'] for row in slots) ==
            {'ordinary': 15, 'branch': 270},
        distinct_new_slots=len({row['new_id'] for row in slots}) == 285,
        financing_exact=len(set(replacements)) == 285 and set(replacements) == expected_old,
        packets57x5=len(packets) == 57 and all(len(packet['slots']) == 5 for packet in packets),
        updates24_plus432=sum(phase['updates'] for phase in phases) == 456,
        no_response_sgd_during_admission=all(packet['updates_after_packet'] == 0
            for packet in packets[3:]),
        three_lines_same_window=all(pair['issue_window'] == list(ISSUE_WINDOW) for pair in pairs),
        release_covers_all_real_closures=ISSUE_WINDOW[1]+300.+30. == RELEASE_CLOCK)
    return dict(status='final_prospective_release_design', pairs=pairs, packets=packets,
        slots=slots, phases=phases, checks=checks, all_pass=all(checks.values()),
        source_contract=dict(initializer_updates=50000, horizon_seconds=24000,
            feedback_slots=285, optimizer_updates=456,
            frozen_behavior_reference='immutable_original50000_complete_stochastic_Hurdle_then_original_F',
            warmup_learner_does_not_modify_behavior_reference=True,
            F_unchanged=True, trust_unchanged=True,
            comparison='future_trace_and_same_flow_one_step_control_share_slots_calendar_sampling_and_update_order'),
        financing=dict(ordinary_retained=15, ordinary_replaced=252,
            root_entitlements_retained=6, root_entitlements_cancelled=12,
            response_roots=6, continuation_slots=264),
        rights=dict(old_rows_reused_as_training=False,
            diagnostic_truth_admitted=False, new_physical_prefix_required=True,
            every_true_event_including_zero_dt_occupies_one_slot=True,
            all45_actual_closures_required_before_response_admission=True,
            no_reward_dependent_stop_or_replacement=True,
            no_synthetic300_state=True,
            parent_native_clock_must_reach9630_without_branch_wall_time_advance=True,
            missing_short_or_late_branch='fail_without_ordinary_fill_or_reselection'),
        scientific_scope=dict(
            superseded_draft='budget_plan.py staggered_window_long_age_calendar_is_rejected',
            all_lines_have_same_window_and432_response_update_opportunities=True,
            target_policy_updates_stop_at9630=True,
            original_uniform_release_calendar_is_changed=True,
            control_must_share_new_calendar_and_freeze_clock=True,
            no_policy_improvement_claim_from_capacity_or_support_precheck=True),
        current_compute=dict(new_native_prefixes=0, new_native_branches=0,
            new_training_labels=0, new_optimizer_updates=0))


def require_phase(phase, *, native_clock, admitted_slot_ids, completed_updates,
                  available_response_slot_ids=(), latest_response_available_at=None):
    """Return the licensed update count, or reject a premature/partial phase.

    Call once at the boundary of the declared phase.  Actual TD/memory/behavior
    checks remain separate; the identities here are its recorded slot receipts.
    This function grants no rights to v38/v39 diagnostic rows.
    """
    plan = build_release_plan()
    expected = [row['new_id'] for row in plan['slots']]
    admitted = list(admitted_slot_ids)
    if len(set(admitted)) != len(admitted):
        raise ValueError('One physical feedback slot was admitted more than once')
    if not set(admitted) <= set(expected):
        raise ValueError('Unexpected or old diagnostic feedback slot')
    if phase.startswith('warmup_'):
        declared = next((item for item in plan['phases'] if item['id'] == phase), None)
        if declared is None:
            raise ValueError('Unknown warmup phase')
        if native_clock != declared['native_clock']:
            raise ValueError('Warmup must use its declared native release clock')
        if set(admitted) != set(expected[:declared['slots_after_admission']]):
            raise ValueError('Warmup feedback is incomplete or response feedback entered early')
        if completed_updates != declared['updates_before']:
            raise ValueError('Warmup update order/count changed')
        return dict(phase=phase, allowed_updates=8, updates_after=declared['updates_after'])
    if phase == 'frozen':
        if not RELEASE_CLOCK <= native_clock <= 24000.:
            raise ValueError('Frozen period lies outside the native target horizon')
        if set(admitted) != set(expected) or completed_updates != 456:
            raise ValueError('Cannot declare the final freeze before285/456 is complete')
        return dict(phase=phase, allowed_updates=0, updates_after=456)
    if phase not in ('response_admit', 'response_update'):
        raise ValueError('Unknown phase')
    if native_clock != RELEASE_CLOCK:
        raise ValueError('Response phase must run at native9630, without advancing the clock for SGD')
    if completed_updates != 24:
        raise ValueError('Response phase requires exactly24 warmup updates and no response SGD yet')
    available = list(available_response_slot_ids)
    if (len(available) != 270 or len(set(available)) != 270
            or set(available) != set(expected[15:])):
        raise ValueError('All270 real response rows must be physically available')
    if latest_response_available_at is None or latest_response_available_at > RELEASE_CLOCK:
        raise ValueError('Response rows are missing their valid native release time')
    if phase == 'response_admit':
        if (len(admitted) < 15 or len(admitted) % 5
                or set(admitted) != set(expected[:len(admitted)])):
            raise ValueError('Response admission must follow the complete fixed five-slot packet order')
        return dict(phase=phase, allowed_updates=0, updates_after=24,
                    remaining_admission_packets=(285-len(admitted))//5)
    if set(admitted) != set(expected):
        raise ValueError('Cannot start432 response updates before all285 slots, including270 response rows, are admitted')
    return dict(phase=phase, allowed_updates=432, updates_after=456)


def compact_release_plan(plan):
    def token(row):
        identity = (f"O{row['index']}" if row['kind'] == 'ordinary' else
                    f"{row['line_id']}/{row['arm'][0]}/{row['event']}")
        funding = row['replaces'].replace('old_ordinary:', 'O').replace('old_pair:', 'P')
        return identity+'<='+funding
    return {key: value for key, value in plan.items() if key not in ('slots', 'packets')} | dict(
        slot_token_legend='O ordinary, line/l|h/event is zero-based real branch event; <= old abstract entitlement, never old physical feedback',
        packets=[{key: value for key, value in packet.items() if key != 'slots'} |
                 dict(slots=[token(row) for row in packet['slots']])
                 for packet in plan['packets']])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = json.dumps(compact_release_plan(build_release_plan()),
                      ensure_ascii=False, separators=(',', ':'))+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(data)
    else:
        print(data, end='')


if __name__ == '__main__':
    main()
