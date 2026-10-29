"""First original-schedule mixed-target packet and its executable-action CQL receipt."""
from experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026.learning import (
    SCHEDULE, cql_checks, recency_checks, target_checks)


def native_preflight(learner, replay, contract, initializer, packet, receipt,
                     clock_audit, action_audit, *, seed, scenario, arm, runtime_contract):
    schedule, targets = learner.actor_schedule_audit(), learner.target_audit()
    cql = learner.cql_audit()
    fresh = arm == 'fresh'
    actual = [schedule[key+'_updates'] for key in ('critic', 'value', 'actor', 'alpha')]
    checks = dict(first_packet_five_slots=replay.target_received == len(packet) == 5,
        exact_optimizer_counts=actual == [8, 8, 8, 8],
        original_optimizer_order=schedule['optimizer_order_exact'],
        one_packet=schedule['packets'] == 1,
        fixed_native_schedule=schedule['schedule'] == SCHEDULE,
        no_training_kl=schedule['kl_penalty'] is False,
        diagnostic_only_snapshot=schedule['reference_for_diagnostics_only'],
        no_added_main_sampling=schedule['additional_random_draws'] == 0,
        no_fresh_outcome_isolation=fresh or all(set(row) == {'s', 'a'} for row in replay.recent),
        real_actor_clock=clock_audit['actor_real_clock'], shared_F=action_audit['shared_rule_exact'])
    checks.update({'target/'+key: flag for key,flag in
        target_checks(targets, arm, calls=8).items()})
    checks.update({'recency/'+key: flag for key,flag in
        recency_checks(replay.summary(),arm,8,seed).items()})
    checks.update({'cql/'+key: value for key, value in
        cql_checks(cql, expected_calls=8, rows_per_call=128).items()})
    return dict(technical_gate_pass=all(checks.values()), checks=checks, seed=seed,
        scenario=scenario, arm=arm,
        initializer=dict(path=str(initializer), initial_updates=learner.total_updates-8,
            history_rows=len(replay.history['s']), source_rows=len(replay.source['s'])),
        first_feedback=dict(processed_at=receipt['processed_at'], sequence_ids=receipt['sequence_ids']),
        execution_contract=runtime_contract, feedback_slots=replay.target_received,
        policy_updates=schedule['actor_updates'],
        **{key+'_updates':schedule[key+'_updates'] for key in ('critic', 'value', 'actor', 'alpha')},
        packet_schedule=actual, actor_schedule=schedule, target_audit=targets,
        cql_audit=cql, recency=replay.summary(), reused_full_native_day=True)
