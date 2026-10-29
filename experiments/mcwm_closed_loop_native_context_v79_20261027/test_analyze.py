import json

from analyze import GROUPS, RUNTIME_CHECKS, SOURCE_CONTRACT, analyze


def setup(tmp_path):
    protocol = dict(seeds=[6208, 6207], external_seed_offset=100000,
        horizon_seconds=24000, native_stop_seconds=16200, maximum_native_seconds=32400,
        observation_window=[12000, 16200], new_initialization_updates=0,
        new_training_updates=0, new_feedback_slots=0,
        source_contract=dict(covariates=285, fresh_complete_outcomes=285, updates_per_family=456),
        source_budget_consumed=dict(covariates=0, complete_outcomes=0, updates_per_family=0))
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    for seed in (6208, 6207):
        directory = tmp_path/'runs'/f'seed{seed}'
        directory.mkdir(parents=True)
        summary = dict(status='complete', cell=dict(seed=seed, scenario='stable', external_seed=seed+100000),
            integrity=dict(all_pass=True, checks={key: True for key in RUNTIME_CHECKS}),
            budget=dict(actual_simulated_seconds=16200, training_updates=0, feedback_slots=0,
                covariate_slots=0, complete_outcome_slots=0, initializer_updates=0, counterfactual_branches=0),
            initializer=dict(updates=50000), original_learning_contract=SOURCE_CONTRACT,
            endpoint=dict(time_seconds=16200),
            control=dict(actor_real_clock=True, shared_execution_rule_exact=True),
            execution_wall=dict(native_started_wall=10, native_finished_wall=20),
            native_observation=dict(all_pass=True, checks=dict(order=True, rng=True, apply=True),
                window=[12000, 16200], right_open=True,
                captures=dict(total=100, supported=100, observed=100, unsupported=0, terminal=3),
                coverage=dict(overall=dict(total=100, observed=100), by_line={'7X': {}},
                    subgroups={key: dict(total=4, supported=4, observed=4, unsupported=0) for key in GROUPS},
                    membership_unknown={key: 0 for key in GROUPS})))
        (directory/'summary.json').write_text(json.dumps(summary))


def change(tmp_path, seed, mutate):
    path = tmp_path/'runs'/f'seed{seed}'/'summary.json'
    summary = json.loads(path.read_text())
    mutate(summary)
    path.write_text(json.dumps(summary))


def test_full_native_support_permits_design_only(tmp_path):
    setup(tmp_path)
    out = analyze(tmp_path)
    assert out['status'] == 'complete'
    assert out['budget']['actual_simulated_seconds'] == 32400
    assert out['runtime_integrity']['actual_parallel_overlap_seconds'] == 10
    assert out['decision']['may_design_budget_matched_collection'] is True
    assert out['decision']['training_authorized'] is False
    assert out['decision']['new_collection_started'] is False


def test_majority_empty_queues_cannot_hide_unsupported_nonempty_group(tmp_path):
    setup(tmp_path)
    def mutate(summary):
        obs = summary['native_observation']
        obs['captures'].update(observed=99, unsupported=1)
        obs['coverage']['subgroups']['nonempty_queue'].update(supported=3, observed=3, unsupported=1)
    change(tmp_path, 6208, mutate)
    out = analyze(tmp_path)
    assert out['status'] == 'complete'
    assert out['runtime_integrity']['all_pass'] is True
    assert out['decision']['native_joint_coverage_verified'] is False
    assert out['budget']['failed_work'] == []


def test_absent_release_group_and_unknown_membership_block_native_gate(tmp_path):
    setup(tmp_path)
    def mutate(summary):
        coverage = summary['native_observation']['coverage']
        coverage['subgroups']['release_positive'].update(total=0, supported=0, observed=0)
        coverage['membership_unknown']['release_positive'] = 2
    change(tmp_path, 6207, mutate)
    out = analyze(tmp_path)
    assert out['status'] == 'complete'
    assert out['decision']['may_design_budget_matched_collection'] is False
    native = out['cells'][1]['native_coverage']
    assert native['required_group_checks']['release_positive'] is False
    assert native['unknown_memberships_clear'] is False


def test_missing_run_preserves_unknown_budget_and_failed_timing_is_runtime_failure(tmp_path):
    setup(tmp_path)
    path = tmp_path/'runs/seed6207/summary.json'
    saved = path.read_text()
    path.unlink()
    out = analyze(tmp_path)
    assert out['status'] == 'incomplete'
    assert out['budget']['actual_simulated_seconds'] is None
    assert out['budget']['documented_simulated_seconds'] == 16200
    assert out['cells'][1]['native_coverage'] is None
    path.write_text(saved)
    def mutate(summary):
        summary['native_observation']['checks']['apply'] = False
    change(tmp_path, 6207, mutate)
    out = analyze(tmp_path)
    assert out['status'] == 'integrity_failed'
    assert out['decision']['native_joint_coverage_verified'] is False
