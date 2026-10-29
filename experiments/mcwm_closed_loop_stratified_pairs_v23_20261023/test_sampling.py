"""Selection must remain causal and preserve the fixed nine-pair contract."""
from copy import deepcopy
import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location('v23_sampling', Path(__file__).with_name('sampling.py'))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
StratifiedPairSampler, visible_stratum = MODULE.StratifiedPairSampler, MODULE.visible_stratum


def plans():
    windows = [('bootstrap', 1200, 2700), ('response', 3000, 6600), ('late', 15000, 17670)]
    return [dict(id=f'train/{line}/{name}', role='train', line_id=line, window_name=name,
        window=dict(name=name, start=start, end=end))
        for line in ('7X', '102X', '705X') for name, start, end in windows]


def history(now, residuals=(-2., 1., 3.)):
    return dict(status='supported', segments=[dict(available=True,
        latest=dict(completed_at=now-5., observed_at=now-1., residual=value)) for value in residuals])


def select(sampler, plan, now, headway=1., residuals=(-2., 1., 3.)):
    return sampler.consider(plan, issued_at=now, line_id=plan['line_id'],
        features=[1., 0., 0., headway], history=history(now, residuals))


def test_stratum_uses_original_headway_and_third_history_with_fixed_hold_boundaries():
    for third, expected in ((-7.5001, 'below_minus_hold'), (-7.5, 'within_hold'),
            (7.5, 'within_hold'), (7.5001, 'above_hold')):
        value = visible_stratum(issued_at=100., features=[1., 0., 0., -.25],
            history=history(100., (100., -100., third)))
        assert value['headway_sign'] == -1
        assert value['third_residual_bin'] == expected
        assert value['third_residual_seconds'] == third
        assert value['history_spread_seconds'] == 200.
    assert visible_stratum(issued_at=100., features=[1., 0., 0., 0.],
        history=history(100.))['headway_sign'] == 0


def test_equal_or_future_history_is_never_selected_and_inputs_are_unchanged():
    plan = plans()[0]
    for clock_key in ('completed_at', 'observed_at'):
        sampler = StratifiedPairSampler(plans(), ['7X', '102X', '705X'])
        snapshot = history(1300.)
        snapshot['segments'][2]['latest'][clock_key] = 1300.
        before = deepcopy(snapshot)
        assert sampler.consider(plan, issued_at=1300., line_id='7X',
            features=[1., 0., 0., .2], history=snapshot) is None
        assert snapshot == before
        assert sampler.report()['history_unsupported_issues'][plan['id']] == 1
        assert select(sampler, plan, 1400.)['selection_reason'] == 'bootstrap_first_eligible'


def test_unseen_joint_selection_then_one_way_last_third_fallback():
    sampler = StratifiedPairSampler(plans(), ['7X', '102X', '705X'])
    bootstrap, response, late = plans()[:3]
    first = select(sampler, bootstrap, 1300.)
    assert first['target_met'] and not first['fallback']
    assert select(sampler, response, 3100.) is None
    second = select(sampler, response, 3300., residuals=(-2., 1., 8.))
    assert second['selection_reason'] == 'first_unseen_joint_stratum'
    assert select(sampler, late, 16779.) is None
    third = select(sampler, late, 16780.)  # Exact two-thirds boundary.
    assert third['fallback'] and not third['target_met']
    assert select(sampler, late, 16800., headway=-1.) is None
    third['stratum']['headway_sign'] = 99  # Returned metadata cannot mutate state.
    report = sampler.report()
    assert report['by_line']['7X']['unique_joint_strata_count'] == 2
    assert report['selected'][-1]['stratum']['headway_sign'] == 1


def test_nine_cells_and_scientific_diversity_are_separate_from_structure():
    sampler = StratifiedPairSampler(plans(), ['7X', '102X', '705X'])
    for plan in plans():
        clock = {'bootstrap': 1300., 'response': 5400., 'late': 16780.}[plan['window_name']]
        assert select(sampler, plan, clock) is not None
        assert select(sampler, plan, clock+1.) is None
    report = sampler.report()
    assert report['all_pass']
    assert len(report['selected']) == 9 and not report['missing_ids']
    assert all(row['selected_pairs'] == 3 and row['unique_joint_strata_count'] == 1
        for row in report['by_line'].values())


def test_expired_window_remains_missing_instead_of_retrospective_selection():
    sampler = StratifiedPairSampler(plans(), ['7X', '102X', '705X'])
    bootstrap, response = plans()[:2]
    assert select(sampler, bootstrap, 1300.)
    assert select(sampler, response, 3100.) is None
    assert select(sampler, response, 6600., headway=-1.) is None
    report = sampler.report()
    assert response['id'] in report['missing_ids']
    assert not report['all_pass']
