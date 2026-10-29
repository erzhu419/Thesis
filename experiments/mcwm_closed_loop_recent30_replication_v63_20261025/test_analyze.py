"""Do not let old gains, missing cells or native failures pass replication."""
import json

import pytest

from experiments.mcwm_closed_loop_recent30_replication_v63_20261025 import analyze as a
from experiments.mcwm_closed_loop_control_campaign_v60_20261025 import control_analyze as native
from experiments.mcwm_closed_loop_control_campaign_v60_20261025.test_control_analyze import (
    fixture_run, write)


def populate(tmp_path, *, new_deltas=None, old_deltas=None):
    root, development = tmp_path/'replication', tmp_path/'development'
    protocol = dict(development_seeds=[6201, 6202], confirmation_seeds=[6203, 6204, 6205],
        scenarios=list(a.SCENARIOS))
    for directory in (root, development):
        write(directory/'protocol.json', protocol)
    for directory, stage, seeds, deltas in (
            (development, 'development', a.DEVELOPMENT_SEEDS,
             old_deltas if old_deltas is not None else [-10., 1., -10., 1.]),
            (root, 'confirmation', a.INDEPENDENT_SEEDS,
             new_deltas if new_deltas is not None else [-2.]*6)):
        for index, seed in enumerate(seeds):
            for offset, scenario in enumerate(a.SCENARIOS):
                delta = deltas[index*2+offset]
                for arm in ('fresh', 'no_fresh'):
                    fixture_run(directory, 'recent30', stage, seed, scenario, arm,
                        cost=100.+delta if arm == 'fresh' else 100.)
    reference = development/'results/development.json'
    report = native.analyze(development, 'recent30', 'development')
    assert report['integrity']['all_pass'] and not report['decision']['scientific_gate_pass']
    write(reference, report)
    return root, reference


def test_both_fixed_gates_can_pass_without_relabeling_failed_development(tmp_path):
    root, reference = populate(tmp_path)
    before = reference.read_text()
    report = a.analyze(root, development_path=reference)
    assert report['status'] == 'complete' and report['integrity']['all_pass']
    assert report['independent3']['gate']['all_pass']
    assert report['retained5']['gate']['all_pass']
    assert report['decision']['confirmed_control_improvement']
    assert report['decision']['ordinary_dripping_goal_achieved'] is False
    assert report['decision']['further_seed_expansion_allowed'] is False
    assert report['development_reference']['decision']['scientific_gate_pass'] is False
    assert report['decision']['original_development_gate_failed'] is True
    assert reference.read_text() == before
    assert report['independent3']['seeds'] == [6203, 6204, 6205]
    assert report['retained5']['seeds'] == [6201, 6202, 6203, 6204, 6205]
    assert report['expected_cells'] == report['available_cells'] == len(report['cells']) == 10
    assert len(json.dumps(report, separators=(',', ':'), allow_nan=False).encode()) < 48000


def test_retained_gain_cannot_hide_failed_independent_result(tmp_path):
    root, reference = populate(tmp_path, new_deltas=[1., -1.]*3)
    report = a.analyze(root, development_path=reference)
    assert report['retained5']['gate']['all_pass']
    assert not report['independent3']['gate']['all_pass']
    assert report['independent3']['overall']['mean_cost_difference'] == 0.
    assert not report['decision']['confirmed_control_improvement']
    assert report['decision']['reason'] == 'independent_cost_gate_failed'


def test_independent_gain_cannot_erase_retained_shock_loss(tmp_path):
    root, reference = populate(tmp_path, old_deltas=[-50., 40., -50., 40.])
    report = a.analyze(root, development_path=reference)
    assert report['independent3']['gate']['all_pass']
    assert report['retained5']['overall']['mean_cost_difference'] < 0
    assert report['retained5']['by_scenario']['local_speed_drop']['mean_cost_difference'] > 0
    assert not report['retained5']['gate']['all_pass']
    assert not report['decision']['confirmed_control_improvement']


def test_win_loss_rates_and_magnitudes_keep_losing_seed(tmp_path):
    root, reference = populate(tmp_path, new_deltas=[-5., -5., -5., -5., 1., 1.])
    report = a.analyze(root, development_path=reference)
    independent = report['independent3']['overall']
    assert independent['wins'] == 4 and independent['losses'] == 2 and independent['ties'] == 0
    assert independent['win_percent_of_complete'] == pytest.approx(200./3)
    assert independent['loss_percent_of_complete'] == pytest.approx(100./3)
    assert independent['winning_mean_cost_reduction'] == 5.
    assert independent['losing_mean_cost_increase'] == 1.
    assert report['independent3']['by_seed']['6205']['mean_cost_difference'] == 1.
    assert report['decision']['confirmed_control_improvement']


@pytest.mark.parametrize('new_deltas', [[0., 0.]*3, [-2., 0.]*3, [1., -4.]*3])
def test_original_strict_shock_and_overall_and_stable_conditions_are_unchanged(tmp_path, new_deltas):
    root, reference = populate(tmp_path, new_deltas=new_deltas)
    report = a.analyze(root, development_path=reference)
    assert not report['independent3']['gate']['all_pass']
    assert not report['decision']['confirmed_control_improvement']


def test_missing_and_failure_keep_null_cell_and_consumed_work(tmp_path):
    root, reference = populate(tmp_path)
    path = root/'runs/recent30/confirmation/seed6205/local_speed_drop/fresh'
    (path/'summary.json').unlink()
    report = a.analyze(root, development_path=reference)
    assert report['status'] == 'pending'
    assert len(report['cells']) == 10 and report['available_cells'] == 9
    missing = next(row for row in report['cells'] if row['seed'] == 6205 and row['scenario'] == 'local_speed_drop')
    assert missing['cost_difference'] is None and missing['candidate_cost'] is None
    assert report['independent3']['overall']['complete_cells'] == 5
    assert report['independent3']['overall']['missing_or_invalid_cells'] == 1
    assert not report['decision']['confirmed_control_improvement']
    consumed = dict(feedback_slots=50, policy_updates=80, packets=10)
    write(path/'failure.json', dict(type='NativeError', native_time=3999.,
        consumed_budget=consumed, wall_seconds=12.))
    report = a.analyze(root, development_path=reference)
    assert report['status'] == 'integrity_failed'
    assert report['independent_audit']['failed_run_work'][0]['consumed_budget'] == consumed
    assert report['independent_audit']['failed_run_work'][0]['native_time'] == 3999.
    assert not report['decision']['confirmed_control_improvement']


def test_no_fresh_leakage_blocks_even_both_negative_cost_gates(tmp_path):
    root, reference = populate(tmp_path)
    path = root/'runs/recent30/confirmation/seed6203/stable/no_fresh/summary.json'
    value = json.loads(path.read_text())
    value['replay']['fresh_outcomes_stored'] = 1
    write(path, value)
    report = a.analyze(root, development_path=reference)
    assert report['independent3']['overall']['mean_cost_difference'] < 0
    assert not report['independent_audit']['integrity']['all_pass']
    work = report['independent_audit']['failed_summary_work']
    assert len(work) == 2
    assert all(row['consumed_budget']['target_feedback_slots'] == 285 for row in work)
    assert all(row['consumed_budget']['policy_updates'] == 456 for row in work)
    assert not report['decision']['confirmed_control_improvement']
    assert report['decision']['ordinary_dripping_goal_achieved'] is False


def test_missing_development_reference_does_not_become_four_zero_effects(tmp_path):
    root, reference = populate(tmp_path)
    reference.unlink()
    report = a.analyze(root, development_path=reference)
    assert report['status'] == 'pending' and report['development_reference']['missing']
    assert report['independent3']['gate']['all_pass']
    assert report['retained5']['overall']['complete_cells'] == 6
    assert report['retained5']['overall']['missing_or_invalid_cells'] == 4
    assert all(row['cost_difference'] is None for row in report['cells'][:4])
    assert not report['decision']['confirmed_control_improvement']


def test_development_decision_cannot_be_rewritten_as_a_pass(tmp_path):
    root, reference = populate(tmp_path)
    old = json.loads(reference.read_text())
    old['decision']['scientific_gate_pass'] = True
    write(reference, old)
    report = a.analyze(root, development_path=reference)
    assert not report['integrity']['source_development_failed_gate_preserved']
    assert not report['decision']['confirmed_control_improvement']


def test_all_fixed_seeds_required_no_duplicate_or_substitute(tmp_path):
    root, reference = populate(tmp_path)
    old = json.loads(reference.read_text())
    old['cells'][2] = old['cells'][0]
    write(reference, old)
    report = a.analyze(root, development_path=reference)
    assert report['development_reference']['membership']['duplicate_cells']
    assert not report['development_reference']['membership']['exact_fixed_membership']
    assert not report['decision']['confirmed_control_improvement']
