"""Catch stale inherited285-record and four-day claims in generated run metadata."""
from experiments.mcwm_closed_loop_decoupled_feedback_v73_20261026.control import protocol


def test_generated_protocol_exposes_split_information_and_five_day_compute_contract():
    p = protocol()
    info, permissions, compute = p['information_contract'], p['data_permissions'], p['compute_budget']
    assert info['changed'] is True
    assert info['covariate_slots_per_day'] == permissions['covariate_slots_per_day'] == 285
    assert info['complete_outcome_slots_per_fresh_day'] == permissions['complete_outcome_slots_per_fresh_day'] == 285
    assert info['complete_outcome_slots_per_no_fresh_day'] == permissions['complete_outcome_slots_per_no_fresh_day'] == 0
    assert permissions['maximum_unique_learner_target_records_per_fresh_day'] == 342
    assert permissions['maximum_unique_learner_target_records_per_no_fresh_day'] == 285
    assert p['target_record_budget'] == 285  # Unchanged runtime covariate counter.
    assert 'covariate' in p['target_record_budget_scope']
    assert 'at most342' in p['data_rights'] and 'zero complete' in p['data_rights']
    assert compute['new_main_days'] == compute['fresh_main_days']+compute['no_fresh_regression_days'] == 5
    assert compute['cached_v70_no_fresh_comparison_days'] == 4
    assert compute['new_main_sim_seconds'] == 5*24000
    assert compute['covariate_slots'] == 5*285
    assert compute['fresh_supervised_outcomes'] == 4*285
    assert compute['updates_per_optimizer'] == 5*456
    assert 'pilot_main_days' not in compute
    assert compute['counterfactual_branches'] == compute['initializer_retraining_updates'] == 0
    assert 'One paired initialization seed' not in p['interpretation']
    assert '6201/6203' in p['interpretation']
    assert 'regression pass' in p['stages']['remaining']
    assert p['policy_update_budget'] == p['critic_update_budget'] == p['value_update_budget'] == 456
    assert p['feedback_first_deadline'] == 1200 and p['feedback_last_deadline'] == 18000
    assert p['runtime_contract']['optimizer_schedule'] == 'resident actor-before-critic including IQL V and Adam state'
