"""Reuse v68 native admission and add only the fixed recent30 contract."""
from experiments.mcwm_closed_loop_executed_cql_v68_20261026.preflight import (
    native_preflight as original_preflight)
from experiments.mcwm_closed_loop_recent30_cql_v69_20261026.learning import recency_checks


def native_preflight(learner, replay, contract, initializer, packet, receipt,
                     clock_audit, action_audit, *, seed, scenario, arm, runtime_contract):
    value = original_preflight(learner,replay,contract,initializer,packet,receipt,
        clock_audit,action_audit,seed=seed,scenario=scenario,arm=arm,runtime_contract=runtime_contract)
    value['checks'].update({'recency/'+key: result for key,result in
        recency_checks(replay.summary(),arm,8,seed).items()})
    value['technical_gate_pass'] = all(value['checks'].values())
    value['recency'] = replay.summary()
    return value
