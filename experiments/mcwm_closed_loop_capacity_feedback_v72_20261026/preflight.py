"""Admit the reused full-day first packet without changing v70 training."""
from experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026.preflight import (
    native_preflight as original_preflight)
from experiments.mcwm_closed_loop_capacity_feedback_v72_20261026.selection import (
    selection_checks, summarize_selection)


def native_preflight(learner, replay, contract, initializer, packet, receipt,
                     clock_audit, action_audit, *, seed, scenario, arm,
                     runtime_contract, prefix_before_first_update):
    result = original_preflight(learner, replay, contract, initializer, packet, receipt,
        clock_audit, action_audit, seed=seed, scenario=scenario, arm=arm,
        runtime_contract=runtime_contract)
    result['checks'].update({'selection/'+key: value for key, value in
        selection_checks([receipt], expected_packets=1, expected_records=5).items()})
    result['checks']['pre_update_prefix_recorded'] = bool(
        prefix_before_first_update['rows'] > 0
        and prefix_before_first_update['clock'] == receipt['processed_at']
        and prefix_before_first_update['initial_updates'] == 50000)
    result['capacity_selection'] = summarize_selection([receipt])
    result['prefix_before_first_update'] = prefix_before_first_update
    result['technical_gate_pass'] = all(result['checks'].values())
    return result
