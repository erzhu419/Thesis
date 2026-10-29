"""Outcome-blind physical coverage of one prospective 270-TD factual stream."""
from __future__ import annotations

import math

LINES = ('7X', '102X', '705X')
ISSUE_FIELDS = ('sequence', 'started_at', 'line_id', 'episode_id')
BEHAVIOR_FIELDS = ('actor_snapshot', 'intervention', 'policy_input', 'execution_input',
                   'parameters', 'proposal', 'nominal_executed_action', 'executed_action', 'action_rng')
PARAMETER_FIELDS = ('positive_probability', 'zero_probability', 'mean', 'std', 'gate', 'log_std')


def build_plan():
    return dict(
        status='prospective_design_only', scheduler_priority='high',
        budget=dict(ordinary_slots=15, continuous_native_td_slots=270, total_slots=285,
                    warmup_updates=24, response_updates=432, total_updates=456,
                    warmup_deadlines=[1200, 1500, 1800], response_release_clock=9630,
                    all270_admitted_before_response_updates=True),
        collection=dict(single_factual_stream=True, required_issue_endpoints=271,
                        start='first_real_issue_with_9000_le_clock_lt_9300',
                        native_transition_count=270, latest_real_endpoint=9600,
                        feedback_delay_seconds=30, horizon_seconds=24000,
                        zero_duration_consumes_one_slot=True,
                        forced_internal_actions=False, behavior_snapshot='actual_post24'),
        roots=dict(lines=list(LINES), selection='first_issue_per_line_in_fixed_9000_9300_window',
                   required_physical_span_seconds=300, suffixes_shared_not_recounted=True),
        accounting=dict(slot_unit='one_real_adjacent_native_TD_including_dt0',
                        unique_response_slots=270, roots_are_existing_rows_not_extra_labels=True,
                        intermediate_events_not_thinned_or_aggregated=True,
                        old_heldout_admission=False, synthetic_states=0),
        correction=dict(continuation='full_stochastic_mu24_then_unchanged_F',
                        actual_rn='pi_F_over_recorded_mu_F_at_each_real_successor',
                        ideal_bound='0.9**trace_links_at_lambda1_gamma1_c1_only',
                        ideal_bound_is_measured_policy_drift=False,
                        forward_F_parity='resident_runner_reuses_audited_native_transform'),
        schema=dict(issue_fields=list(ISSUE_FIELDS),
                    behavior_record_fields=['sequence', 'started_at', 'line_id', 'behavior_s'],
                    behavior_fields=list(BEHAVIOR_FIELDS), parameter_fields=list(PARAMETER_FIELDS),
                    input_dimensions=229, action_dimensions=1,
                    action_rng_fields=['stream_seed', 'draw_ordinal']),
        stop='missing_short_late_or_under300_fails_without_reselection_or_old_row_fill',
        unchanged=['initializer50000', '456_updates', '24000_seconds', 'F', 'trust'],
        current_compute=dict(new_native_branches=0, new_training_labels=0, optimizer_updates=0))


def _vector(value, size):
    return isinstance(value, (list, tuple)) and len(value) == size and all(
        isinstance(x, (int, float)) and math.isfinite(x) for x in value)


def _behavior_complete(value):
    if not isinstance(value, dict) or any(key not in value for key in BEHAVIOR_FIELDS):
        return False
    parameters, rng = value['parameters'], value['action_rng']
    return (isinstance(parameters, dict) and all(key in parameters and
            isinstance(parameters[key], (int, float)) and math.isfinite(parameters[key])
            for key in PARAMETER_FIELDS) and parameters['std'] > 0
        and 0 < parameters['positive_probability'] < 1
        and 0 < parameters['zero_probability'] < 1
        and abs(parameters['positive_probability']+parameters['zero_probability']-1) < 1e-6
        and all(_vector(value[key], 229) for key in ('policy_input', 'execution_input'))
        and all(_vector(value[key], 1) and -1 <= value[key][0] <= 1
                for key in ('proposal', 'nominal_executed_action', 'executed_action'))
        and isinstance(rng, dict) and all(isinstance(rng.get(key), int)
                                        for key in ('stream_seed', 'draw_ordinal')))


def precheck_issues(issue_records, behavior_records):
    """Read chronology/behavior only; never choose a start using rewards or costs."""
    issues = list(issue_records)
    starts = [i for i, row in enumerate(issues) if 9000 <= row['started_at'] < 9300]
    offset = starts[0] if starts else None
    selected = issues[offset:offset+271] if offset is not None else []
    sources, endpoint = selected[:270], selected[-1] if len(selected) == 271 else None
    behavior = {}
    duplicates = set()
    for row in behavior_records:
        sequence = row['sequence']
        if sequence in behavior:
            duplicates.add(sequence)
        behavior[sequence] = row
    contexts = [behavior.get(row['sequence']) for row in sources]
    complete = all(value is not None and _behavior_complete(value.get('behavior_s')) for value in contexts)
    checks = dict(fixed_first_issue_found=offset is not None,
        exact270_native_td=len(selected) == 271,
        adjacent_native_sequences=all(right['sequence'] == left['sequence']+1
                                    for left, right in zip(selected, selected[1:])),
        one_native_episode=bool(selected) and len({row['episode_id'] for row in selected}) == 1,
        chronological_real_clocks=all(right['started_at'] >= left['started_at']
                                     for left, right in zip(selected, selected[1:])),
        real_endpoint_by9600=endpoint is not None and endpoint['started_at'] <= 9600,
        released_by9630=endpoint is not None and endpoint['started_at']+30 <= 9630,
        behavior_complete=complete,
        unique_behavior_identity=all(row['sequence'] not in duplicates for row in sources),
        behavior_identity=all(value is not None and value['started_at'] == row['started_at']
                              and value['line_id'] == row['line_id'] for row, value in zip(sources, contexts)))
    checks['mu24_without_intervention'] = complete and all(
        value['behavior_s']['actor_snapshot'] == 'actual_post24' and value['behavior_s']['intervention'] is False
        for value in contexts)
    checks['actual_actor_F_inputs_equal'] = complete and all(
        value['behavior_s']['policy_input'] == value['behavior_s']['execution_input'] for value in contexts)
    checks['nominal_equals_actual_execution'] = complete and all(
        value['behavior_s']['nominal_executed_action'] == value['behavior_s']['executed_action'] for value in contexts)
    draws = [value['behavior_s']['action_rng'] for value in contexts] if complete else []
    checks['continuous_action_rng'] = complete and all(
        right['stream_seed'] == left['stream_seed'] and right['draw_ordinal'] == left['draw_ordinal']+1
        for left, right in zip(draws, draws[1:]))
    roots = {}
    for line in LINES:
        index = next((i for i, row in enumerate(sources)
                      if row['line_id'] == line and 9000 <= row['started_at'] < 9300), None)
        root = sources[index] if index is not None else None
        target = root['started_at']+300 if root else None
        end_index = next((j for j in range(index+1, len(selected))
                          if selected[j]['started_at'] >= target), None) if root else None
        closures = end_index-index if end_index is not None else None
        links = closures-1 if closures is not None else None
        roots[line] = dict(root_index=index, root_sequence=root['sequence'] if root else None,
            root_clock=root['started_at'] if root else None,
            available_suffix_td_slots=len(sources)-index if root else 0,
            physical_span_to_final_endpoint=endpoint['started_at']-root['started_at'] if endpoint and root else None,
            requested_300_clock=target, first_real_300_endpoint_index=end_index,
            first_real_300_endpoint_clock=selected[end_index]['started_at'] if end_index is not None else None,
            native_transitions_to_300=closures, trace_links_to_300=links,
            ideal_c1_mixed_weight_to_300=.9**links if links is not None else None,
            trace_links_to_final_endpoint=len(sources)-index-1 if root and endpoint else None,
            ideal_c1_mixed_weight_to_final_endpoint=(.9**(len(sources)-index-1)
                if root and endpoint else None))
    checks['three_lines_have_roots'] = all(value['root_index'] is not None for value in roots.values())
    checks['all_roots_reach300'] = all(value['native_transitions_to_300'] is not None for value in roots.values())
    coverage = {}
    for line in (*LINES, 'all'):
        values = [value['behavior_s'] for row, value in zip(sources, contexts)
                  if value is not None and _behavior_complete(value.get('behavior_s'))
                  and (line == 'all' or row['line_id'] == line)]
        coverage[line] = dict(rows=len(values), proposal_zero=sum(x['proposal'][0] == -1 for x in values),
            proposal_positive=sum(x['proposal'][0] > -1 for x in values),
            executed_zero_strict=sum(x['executed_action'][0] == -1 for x in values),
            executed_positive=sum(x['executed_action'][0] > -1 for x in values),
            proposal_min=min((x['proposal'][0] for x in values), default=None),
            proposal_max=max((x['proposal'][0] for x in values), default=None))
    durations = [right['started_at']-left['started_at'] for left, right in zip(selected, selected[1:])]
    return dict(status='geometry_behavior_pass' if all(checks.values()) else 'precheck_failed',
        all_pass=all(checks.values()), checks=checks,
        failure_reasons=[key for key, ok in checks.items() if not ok],
        selection=dict(input_start_offset=offset, start_sequence=selected[0]['sequence'] if selected else None,
                       start_clock=selected[0]['started_at'] if selected else None,
                       endpoint_sequence=endpoint['sequence'] if endpoint else None,
                       endpoint_clock=endpoint['started_at'] if endpoint else None,
                       physically_available_at=endpoint['started_at']+30 if endpoint else None,
                       reselection_attempts=0),
        budget=dict(unique_native_td_slots=len(durations), zero_duration_slots=durations.count(0),
                    final_prospective_total_slots=15+len(durations),
                    root_suffix_references=sum(value['available_suffix_td_slots'] for value in roots.values()),
                    root_suffix_references_are_not_new_slots=True),
        root_coverage=roots, native_action_coverage=coverage,
        ideal_credit_scope='gamma1_lambda1_c1_algebraic_upper_bound_not_measured_pi_drift',
        unverified=['native_forward_F_parity', 'actual_pi_F_mu_F_drift', 'future_native_reachability'],
        schema=build_plan()['schema'], outcomes_read=0, new_feedback_slots=0, new_optimizer_updates=0)
