"""Admit the first reused native packet with independently selected data pools."""
from experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026.preflight import (
    native_preflight as original_preflight)
from experiments.mcwm_closed_loop_decoupled_feedback_v73_20261026.selection import (
    selection_checks, summarize_selection)


def native_preflight(learner, replay, contract, initializer, packet, receipt,
                     clock_audit, action_audit, *, seed, scenario, arm,
                     runtime_contract, prefix_before_first_update, outcome_packet):
    # The old first_feedback receipt describes policy covariates; retain that
    # exact meaning even when the independently selected outcome ids differ.
    original_receipt = dict(receipt, sequence_ids=receipt['covariate_sequence_ids'],
        max_available_at=receipt['covariate_max_available_at'])
    result = original_preflight(learner, replay, contract, initializer, packet, original_receipt,
        clock_audit, action_audit, seed=seed, scenario=scenario, arm=arm,
        runtime_contract=runtime_contract)
    result['checks'].update({'selection/'+key: value for key, value in
        selection_checks([receipt], expected_packets=1, expected_records=5).items()})
    result['checks'].update(
        pre_update_prefix_recorded=bool(prefix_before_first_update['rows'] > 0
            and prefix_before_first_update['clock'] == receipt['processed_at']
            and prefix_before_first_update['initial_updates'] == 50000),
        exact_covariate_selection=[row.sequence for row in packet] == receipt['covariate_sequence_ids'],
        exact_outcome_selection=[row.sequence for row in outcome_packet] == receipt['outcome_sequence_ids'],
        covariates_only_sa=all(set(row) == {'s', 'a'} for row in replay.recent),
        separate_fresh_outcomes=len(replay.fresh_outcomes) == (5 if arm == 'fresh' else 0),
        no_fresh_distinct_rows=arm == 'fresh' or replay.summary()['distinct_target_records_used'] == 5)
    result['first_feedback']['covariate_sequence_ids'] = receipt['covariate_sequence_ids']
    result['first_outcome_feedback'] = dict(processed_at=receipt['processed_at'],
        sequence_ids=receipt['outcome_sequence_ids'],
        admitted_complete_outcomes=5 if arm == 'fresh' else 0)
    result['capacity_selection'] = summarize_selection([receipt])
    result['prefix_before_first_update'] = prefix_before_first_update
    result['technical_gate_pass'] = all(result['checks'].values())
    return result
