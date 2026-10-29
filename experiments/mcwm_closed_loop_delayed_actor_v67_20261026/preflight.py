"""First native packet verifies eight critic/value then one actor/alpha step."""
from experiments.mcwm_closed_loop_delayed_actor_v67_20261026.learning import SCHEDULE


def native_preflight(learner, replay, contract, initializer, packet, receipt,
                     clock_audit, action_audit, *, seed, scenario, arm, runtime_contract):
    schedule, targets = learner.actor_schedule_audit(), learner.target_audit()
    fresh = arm == 'fresh'
    actual = [schedule[key+'_updates'] for key in ('critic', 'value', 'actor', 'alpha')]
    checks = dict(first_packet_five_slots=replay.target_received == len(packet) == 5,
        exact_optimizer_counts=actual == [8, 8, 1, 1],
        actor_after_eighth_critic=schedule['actor_update_after_critic_steps'] == [8]
            and schedule['optimizer_order_exact'],
        one_packet=schedule['packets'] == 1,
        fixed_native_schedule=schedule['schedule'] == SCHEDULE,
        no_training_kl=schedule['kl_penalty'] is False,
        diagnostic_only_snapshot=schedule['reference_for_diagnostics_only'],
        no_extra_sampling=schedule['additional_random_draws'] == 0,
        original_target_draws=targets['parent_mixed_target_calls'] == 8
            and targets['parent_mixed_target_rows'] == 8*256,
        original_fresh_target=targets['fresh_value_replacement_calls'] == (8 if fresh else 0)
            and targets['fresh_value_replacement_rows'] == (8*64 if fresh else 0)
            and targets['no_fresh_unchanged_returns'] == (0 if fresh else 8),
        no_fresh_outcome_isolation=fresh or all(set(row) == {'s', 'a'} for row in replay.recent),
        real_actor_clock=clock_audit['actor_real_clock'], shared_F=action_audit['shared_rule_exact'])
    return dict(technical_gate_pass=all(checks.values()), checks=checks, seed=seed,
        scenario=scenario, arm=arm,
        initializer=dict(path=str(initializer), initial_updates=learner.total_updates-8,
            history_rows=len(replay.history['s']), source_rows=len(replay.source['s'])),
        first_feedback=dict(processed_at=receipt['processed_at'], sequence_ids=receipt['sequence_ids']),
        execution_contract=runtime_contract, feedback_slots=replay.target_received,
        policy_updates=schedule['actor_updates'],
        **{key+'_updates':schedule[key+'_updates'] for key in ('critic', 'value', 'actor', 'alpha')},
        packet_schedule=actual, actor_schedule=schedule, target_audit=targets,
        reused_full_native_day=True)
