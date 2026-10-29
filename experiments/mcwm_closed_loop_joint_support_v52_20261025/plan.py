"""Fixed 285-slot design and outcome-blind old-flow capacity projection."""
from __future__ import annotations

from experiments.mcwm_closed_loop_physical_feedback_v49_20261025.plan import (
    LINES, _behavior_complete,
)

RESPONSE_SLOTS = 271


def build_plan():
    return dict(
        status='prospective_design_only', scheduler_priority='high',
        budget=dict(ordinary_slots=14, continuous_native_td_slots=RESPONSE_SLOTS,
            total_slots=285, warmup_container_sizes=[5, 5, 4],
            warmup_deadlines=[1200, 1500, 1800], warmup_updates=24,
            response_admission_container_sizes=[5]*53+[6],
            response_update_packets=54, updates_per_packet=8,
            response_updates=432, total_updates=456, response_release_clock=9630),
        replacement=dict(reference='v49 prospective entitlements, never old physical labels',
            retained_ordinary=[f'ordinary:{i}' for i in range(14)],
            cancelled='ordinary:14', replacement='continuous:270',
            retained_continuous_range=[0, 269],
            all285_feedback_identities_must_be_new=True),
        collection=dict(single_factual_stream=True, required_issue_endpoints=272,
            start='first_real_issue_with_9000_le_clock_lt_9300',
            native_transition_count=RESPONSE_SLOTS, latest_real_endpoint=9600,
            feedback_delay_seconds=30, horizon_seconds=24000,
            zero_duration_consumes_one_slot=True,
            roots='first_issue_per_line_in_fixed_9000_9300_window',
            required_physical_span_seconds=300, shared_suffixes_not_recounted=True,
            stop='missing_short_late_or_under300_fails_without_reselection_or_fill'),
        synchronization=dict(
            snapshot='new_post24_from14', save_after_actual_warmup_updates=24,
            save_strictly_before_clock=9000,
            save_contents='complete actor plus actual clock, model and Adam continuation',
            behavior='complete frozen stochastic new_post24_from14 through unchanged F',
            SAC_evaluator_actor='same immutable issue-pre new_post24_from14',
            learner_actor_alpha_V_Q_Adam='continue normal 24+432 updates; never substituted for immutable evaluator',
            lambda_zero='bitwise own synchronized Y, not original current-actor Y',
            current_actor_RN_is_one=False,
            interpretation='SAC component evaluated under fixed mu_star; whole mixed backup retains changing native V,alpha,target-Q and .1V/.9SAC weights'),
        target_design=dict(capacity=RESPONSE_SLOTS, eta=9/(RESPONSE_SLOTS-1),
            physical_retention_seconds=300, maximum_future_residual_mass=9,
            status='design_only_not_installed',
            correction='actual F-pushforward clipped RN of immutable evaluator over identical issue behavior; current actor drift logged separately'),
        release=dict(all271_admitted_before_response_updates=True,
            container_size_is_not_update_count=True,
            no_response_or_suffix_updates_before9630=True,
            no_updates_after456=True,
            matched_controls_share_new14_labels_and_271_flow_calendar=True),
        unchanged=['initializer6202_50000', '456_updates', '24000_seconds', 'F', 'trust'],
        current_compute=dict(new_native_branches=0, new_training_labels=0, optimizer_updates=0))


def precheck_issues(issue_records, behavior_records):
    """Project fixed 271-event design onto old real issues, without reading outcomes."""
    issues = list(issue_records)
    offset = next((i for i, row in enumerate(issues) if 9000 <= row['started_at'] < 9300), None)
    selected = issues[offset:offset+RESPONSE_SLOTS+1] if offset is not None else []
    # The final observed issue has no closed TD unless its successor is present.
    sources = selected[:-1]
    endpoint = selected[-1] if len(selected) == RESPONSE_SLOTS+1 else None
    behavior = {row['sequence']: row for row in behavior_records}
    contexts = [behavior.get(row['sequence']) for row in sources]
    complete = all(value is not None and _behavior_complete(value.get('behavior_s')) for value in contexts)
    checks = dict(fixed_first_issue_found=offset is not None,
        exact271_native_td=endpoint is not None,
        adjacent_native_sequences=all(b['sequence'] == a['sequence']+1 for a, b in zip(selected, selected[1:])),
        one_native_episode=bool(selected) and len({row['episode_id'] for row in selected}) == 1,
        chronological_real_clocks=all(b['started_at'] >= a['started_at'] for a, b in zip(selected, selected[1:])),
        real_endpoint_by9600=endpoint is not None and endpoint['started_at'] <= 9600,
        released_by9630=endpoint is not None and endpoint['started_at']+30 <= 9630,
        behavior_complete=complete,
        unique_behavior_identity=len(behavior) == len(behavior_records),
        behavior_identity=all(value is not None and value['started_at'] == row['started_at']
            and value['line_id'] == row['line_id'] for row, value in zip(sources, contexts)))
    checks['old_mu24_without_intervention'] = complete and all(
        value['behavior_s']['actor_snapshot'] == 'actual_post24' and value['behavior_s']['intervention'] is False
        for value in contexts)
    checks['actual_actor_F_inputs_equal'] = complete and all(
        value['behavior_s']['policy_input'] == value['behavior_s']['execution_input'] for value in contexts)
    checks['nominal_equals_actual_execution'] = complete and all(
        value['behavior_s']['nominal_executed_action'] == value['behavior_s']['executed_action'] for value in contexts)
    draws = [value['behavior_s']['action_rng'] for value in contexts] if complete else []
    checks['continuous_action_rng'] = complete and all(
        b['stream_seed'] == a['stream_seed'] and b['draw_ordinal'] == a['draw_ordinal']+1
        for a, b in zip(draws, draws[1:]))
    roots = {}
    for line in LINES:
        index = next((i for i, row in enumerate(sources)
            if row['line_id'] == line and 9000 <= row['started_at'] < 9300), None)
        root = sources[index] if index is not None else None
        end_index = next((j for j in range(index+1, len(selected))
            if selected[j]['started_at'] >= root['started_at']+300), None) if root else None
        roots[line] = dict(root_index=index, root_sequence=root['sequence'] if root else None,
            root_clock=root['started_at'] if root else None,
            available_suffix_td_slots=len(sources)-index if root else 0,
            physical_span_to_final_endpoint=endpoint['started_at']-root['started_at'] if endpoint and root else None,
            requested_300_clock=root['started_at']+300 if root else None,
            first_real_300_endpoint_index=end_index,
            first_real_300_endpoint_clock=selected[end_index]['started_at'] if end_index is not None else None,
            native_transitions_to_300=end_index-index if end_index is not None else None)
    checks['three_lines_have_roots'] = all(value['root_index'] is not None for value in roots.values())
    checks['all_roots_reach300'] = all(value['native_transitions_to_300'] is not None for value in roots.values())
    coverage = {}
    for line in (*LINES, 'all'):
        values = [value['behavior_s'] for row, value in zip(sources, contexts)
            if value is not None and _behavior_complete(value.get('behavior_s'))
            and (line == 'all' or row['line_id'] == line)]
        coverage[line] = dict(rows=len(values),
            executed_zero_strict=sum(value['executed_action'][0] == -1 for value in values),
            executed_positive=sum(value['executed_action'][0] > -1 for value in values))
    durations = [b['started_at']-a['started_at'] for a, b in zip(selected, selected[1:])]
    return dict(status='old_geometry_behavior_pass' if all(checks.values()) else 'old_projection_failed',
        all_pass=all(checks.values()), checks=checks,
        failure_reasons=[key for key, ok in checks.items() if not ok],
        selection=dict(input_start_offset=offset, start_sequence=selected[0]['sequence'] if selected else None,
            start_clock=selected[0]['started_at'] if selected else None,
            endpoint_sequence=endpoint['sequence'] if endpoint else None,
            endpoint_clock=endpoint['started_at'] if endpoint else None,
            physically_available_at=endpoint['started_at']+30 if endpoint else None,
            reselection_attempts=0),
        budget=dict(unique_native_td_slots=len(durations), zero_duration_slots=durations.count(0),
            prospective_total_slots=14+len(durations),
            root_suffix_references=sum(value['available_suffix_td_slots'] for value in roots.values()),
            shared_references_are_not_extra_slots=True),
        root_coverage=roots, native_action_coverage=coverage,
        missing=dict(native_closures=max(0, RESPONSE_SLOTS-len(durations)),
            behavior_rows=sum(value is None for value in contexts),
            roots=[line for line, value in roots.items() if value['root_index'] is None]),
        scope='old15-label-mu24 geometry and recorded behavior only; future14-label snapshot and flow are not certified',
        future_snapshot_available=False, future_joint_support_pass=False,
        outcomes_read=0, new_feedback_slots=0, new_optimizer_updates=0)
