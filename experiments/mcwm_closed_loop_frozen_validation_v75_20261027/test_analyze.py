"""Test the new validation unit, primary gate, budgets and bounded export."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.mcwm_closed_loop_frozen_validation_v75_20261027 import analyze as A

PROTOCOL = dict(development_seeds=[6206, 6207, 6208], scenarios=['stable', 'local_speed_drop'],
    external_seed_offset=100000, recipe='latest5_covariates_capacity_outcomes', full_day_departed_passengers=15079)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def populate(tmp_path, monkeypatch, *, deltas=(-2., -2., -2., -2., -2., -2.)):
    root = tmp_path/'v75'
    write(root/'protocol.json', PROTOCOL)
    reference = tmp_path/'mcwm_closed_loop_decoupled_feedback_v73_20261026/runs/executed_cql/development/seed6201/stable/fresh'
    write(reference/'summary.json', dict(runtime_contract=dict(F='frozen')))
    write(reference/'protocol.json', dict(policy_config=dict(ensemble_size=2, alpha=.2)))
    for seed in PROTOCOL['development_seeds']:
        initializer = str(root/f'inputs/h2oplus_seed{seed}.pt')
        write(root/f'inputs/seed{seed}/initializer_gate.json', dict(status='complete', all_pass=True,
            seed=seed, offline_updates=50000, online_updates=0, recipe_exact=True, initializer_path=initializer))
    for index, (seed, scene) in enumerate((seed, scene) for seed in PROTOCOL['development_seeds']
            for scene in PROTOCOL['scenarios']):
        for arm in A.ARMS:
            directory = A.run_dir(root, seed, scene, arm)
            cost = 100.+(deltas[index] if arm == 'fresh' else 0.)
            n = 15079
            config = dict(policy_config=dict(ensemble_size=2, alpha=.2))
            budget = dict(native_seconds=24000, packets=57, policy_updates=456, critic_updates=456,
                value_updates=456, actor_updates=456, alpha_updates=456, world_updates=0, trust_updates=0,
                counterfactual_queries=0, covariate_slots=285, complete_outcome_slots=285 if arm == 'fresh' else 0)
            summary = dict(status='complete', cell=dict(seed=seed, scenario=scene, arm=arm,
                candidate='executed_cql', recipe=PROTOCOL['recipe'], external_seed=seed+100000),
                runtime_contract=dict(F='frozen'), initializer=dict(path=str(root/f'inputs/h2oplus_seed{seed}.pt'), initial_updates=50000),
                clock_audit=dict(first_feedback_applied_at=1205.), execution_wall=dict(native_started_wall=100., native_finished_wall=200.),
                budget=budget, replay=dict(arm=arm, covariate_records_received=285,
                    outcome_labels_received=285 if arm == 'fresh' else 0,
                    distinct_target_records_used=306 if arm == 'fresh' else 285),
                cost=dict(departed_passengers=n, accrued_wait_seconds=10*n,
                    accrued_in_vehicle_seconds=(cost-20)*n, generalized_passenger_seconds=cost*n,
                    generalized_seconds_per_departed_passenger=cost), wall_seconds=101.)
            write(directory/'summary.json', summary)
            write(directory/'protocol.json', config)
            write(directory/'preflight.json', dict(technical_gate_pass=True, initializer=dict(updates=50000),
                first_feedback=dict(ids=[1, 2, 3, 4, 5]), execution_contract=dict(F='frozen')))
            jsonl(directory/'prefix_trace.jsonl', [dict(started_at=1100., physical='same'), dict(started_at=1205., arm=arm)])
            jsonl(directory/'metrics.jsonl', [dict(time_seconds=t, accrued_wait_seconds=summary['cost']['accrued_wait_seconds']*f,
                accrued_in_vehicle_seconds=summary['cost']['accrued_in_vehicle_seconds']*f)
                for t, f in ((9000,.3), (12000,.45), (18300,.8), (24000,1.))])
    monkeypatch.setattr(A, 'native_checks', lambda *args, **kwargs: dict(native_contract=True))
    return root


def modify(path, update):
    value = json.loads(path.read_text())
    update(value)
    write(path, value)


def test_twelve_new_days_and_initialization_budget_are_separate(tmp_path, monkeypatch):
    root = populate(tmp_path, monkeypatch)
    main, evidence, pools, training = A.analyze(root)
    assert main['integrity']['all_pass'], main['integrity']['failed_checks']
    assert main['decision']['scientific_gate_pass']
    assert len(main['cells']) == 6 and len(evidence['pairs']) == 6
    assert len(pools['cells']) == len(training['cells']) == 12
    budget = main['budget']
    assert budget['initialization']['actual_offline_updates'] == 150000
    assert budget['response']['actual_native_seconds'] == 288000
    assert budget['response']['actual_covariate_slots'] == 3420
    assert budget['response']['actual_complete_fresh_results'] == 1710
    assert budget['response']['actual_updates_per_optimizer'] == dict(critic=5472, value=5472, actor=5472, alpha=5472)
    assert budget['response']['cached_cost_baseline_days'] == 0


def test_three_seed_unit_averages_two_scenarios_before_summary(tmp_path, monkeypatch):
    root = populate(tmp_path, monkeypatch, deltas=(-8., 2., -2., -2., 4., -6.))
    value = A.analyze(root)[0]['independent_seed_readout']
    assert value['complete_seeds'] == 3
    assert [row['periods']['full_day']['paired_scenario_mean'] for row in value['rows']] == [-3., -2., -1.]
    assert value['mean_paired_difference'] == -2.
    assert value['improved_seeds'] == 3


def test_positive_stable_tail_is_reported_without_reopening_old_v73_gate(tmp_path, monkeypatch):
    root = populate(tmp_path, monkeypatch)
    for seed in PROTOCOL['development_seeds']:
        directory = A.run_dir(root, seed, 'stable', 'fresh')
        points = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
        for point in points:
            if point['time_seconds'] == 18300:
                point['accrued_in_vehicle_seconds'] -= 10*15079
        jsonl(directory/'metrics.jsonl', points)
    main = A.analyze(root)[0]
    assert main['decision']['scientific_gate_pass']
    assert main['aggregates']['post_budget']['by_scenario']['stable']['mean_cost_difference'] > 0
    assert main['decision']['old_v73_gate_reopened'] is False


def test_a_failed_primary_scenario_mean_is_not_rescued_by_overall_mean(tmp_path, monkeypatch):
    root = populate(tmp_path, monkeypatch, deltas=(1., -5., 1., -5., 1., -5.))
    main = A.analyze(root)[0]
    assert main['integrity']['all_pass']
    assert main['aggregates']['full_day']['overall']['mean_cost_difference'] < 0
    assert not main['decision']['scientific_gate_pass']


@pytest.mark.parametrize('change', ['missing_arm', 'no_overlap', 'wrong_initializer', 'missing_tail', 'runtime_changed'])
def test_incomplete_or_shifted_contract_cannot_pass(tmp_path, monkeypatch, change):
    root = populate(tmp_path, monkeypatch)
    directory = A.run_dir(root, 6207, 'stable', 'no_fresh')
    if change == 'missing_arm':
        (directory/'summary.json').unlink()
    elif change == 'no_overlap':
        modify(directory/'summary.json', lambda row: row.update(execution_wall=dict(native_started_wall=201., native_finished_wall=300.)))
    elif change == 'wrong_initializer':
        modify(directory/'summary.json', lambda row: row['initializer'].update(path='other.pt'))
    elif change == 'missing_tail':
        points = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
        jsonl(directory/'metrics.jsonl', [x for x in points if x['time_seconds'] != 18300])
    else:
        modify(directory/'summary.json', lambda row: row.update(runtime_contract=dict(F='changed')))
    main = A.analyze(root)[0]
    assert not main['integrity']['all_pass']
    assert not main['decision']['scientific_gate_pass']
    if change == 'missing_arm':
        assert main['budget']['response']['actual_covariate_slots'] is None
        assert main['independent_seed_readout']['complete_seeds'] == 2


def test_incomplete_initialization_is_not_counted_as_fifty_thousand(tmp_path, monkeypatch):
    root = populate(tmp_path, monkeypatch)
    modify(root/'inputs/seed6208/initializer_gate.json', lambda row: row.update(offline_updates=40000, all_pass=False))
    main = A.analyze(root)[0]
    assert not main['integrity']['all_pass']
    assert main['budget']['initialization']['actual_offline_updates'] is None
    assert not main['decision']['scientific_gate_pass']


def test_fixed_four_export_files_stay_small(tmp_path, monkeypatch):
    root = populate(tmp_path, monkeypatch)
    out = root/'results/validation.json'
    result = A.export(root, out)
    assert set(result['files']) == {'validation.json', 'evidence.json', 'pools.json', 'training.json'}
    assert all(size <= 48000 for size in result['files'].values())
