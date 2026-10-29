from copy import deepcopy
import json
import unittest

from experiments.mcwm_closed_loop_fresh_response_v86_20261027.analysis import analyze


def artifact():
    arms = {}
    for name, source, seed in (('stale', 'v84_stale', 106207), ('fresh', 'v85_fresh', 216207)):
        arms[name] = dict(gate=dict(all_pass=True), fits=2,
            models=dict(line=dict(design_rank=3), state=dict(design_rank=4)),
            training_scope=dict(source_scope=source, pairs=6, physical_outcome_labels=12,
                source_seed=6207, external_seed=seed, label_available_at=24030,
                cases=[scene+'/'+line for scene in ('stable', 'local_speed_drop')
                       for line in ('7X', '102X', '705X')], heldout_labels_used=0))
    return dict(gate=dict(all_pass=True), fits=4, ridge=1., denominator=15079., hold_scale=7.5,
        arms=arms, action_support=dict(reference_min=3., reference_max_exclusive=3.75,
            candidate=7.5, candidate_tolerance=1e-5),
        legacy_action_support=dict(reference_min=3., reference_max_inclusive=3.6,
            candidate=7.5, candidate_tolerance=1e-5),
        actor_updates=0, critic_updates=0, value_updates=0, B_updates=0, world_updates=0, trust_updates=0)


def set_prediction(pair, name, estimate, supported=True, reason=None):
    pair['prediction']['models'][name] = dict(supported=supported, reason=reason,
        predicted_delta=estimate, choice='candidate' if estimate is not None and estimate < 0 else 'reference')


def set_delta(pair, value):
    pair['remaining']['candidate']['ride'] = pair['remaining']['reference']['ride']+value
    pair['remaining']['candidate']['cost'] = pair['remaining']['reference']['cost']+value
    pair['delta'] = dict(wait=0., ride=value, cost=value)


def scenes():
    result = []
    for scenario, targets in (('stable', (-30., 20., -10.)), ('local_speed_drop', (-5., 10., -2.))):
        pairs, selection = [], {}
        for index, (line, delta) in enumerate(zip(('7X', '102X', '705X'), targets)):
            root = dict(started_at=12005.+index, sequence=100+index, trip_id=line+'_9', station_id=line+'15')
            reference = dict(wait=100., ride=800., cost=1000., dt=24000-root['started_at'],
                observed_at=24000, available_at=24030)
            candidate = dict(reference, ride=800.+delta, cost=1000.+delta)
            pair = dict(status='complete', all_pass=True, checks=dict(native_fork=True), line_id=line,
                root=root, reference_hold_seconds=3.3, candidate_hold_seconds=7.5,
                prediction=dict(supported=True, reason=None, z=.2, models={}),
                prediction_recorded_at=root['started_at'], before_suffixes=True, overlap_seconds=20.,
                remaining=dict(reference=reference, candidate=candidate), delta=dict(wait=0., ride=delta, cost=delta),
                physical=dict(reference=dict(departure_clock=12006., next_arrival_clock=12090., missing={}),
                    candidate=dict(departure_clock=12011., next_arrival_clock=12092., missing={})),
                reference_full_endpoint=dict(wait=10000., ride=30000., cost=50000.))
            set_prediction(pair, 'stale', 1.)
            set_prediction(pair, 'legacy_narrow', 1.)
            set_prediction(pair, 'fresh', -1. if delta < 0 else 1.)
            pairs.append(pair)
            selection[line] = dict(selected=root, attempted=1, reasons={}, status='selected')
        old = dict(source='v75', source_arm='fresh', seed=6207, scenario=scenario,
            packets=37, covariates=185, complete_outcomes=185,
            counts=dict(critic=296, value=296, actor=296, alpha=296), data_deadline=12000,
            max_available_at=12000., published_clock=12001., deterministic=True, actor_frozen=True)
        result.append(dict(status='complete', scenario=scenario, seed=6207, external_seed=316207,
            prefix_id='v86/'+scenario, departed_passengers=15079, integrity=dict(all_pass=True, checks={}),
            reference_snapshot=dict(prior_run_reference=True, available_before_heldout_episode=True,
                source_snapshot_metadata=old),
            budget=dict(native_prefix_seconds=12020., suffix_native_seconds=71950.,
                total_native_seconds=83970., prefixes=1, suffixes=6, labels_observed=6,
                new_initialization_updates=0, prefix_training_updates=0, suffix_updates=0, labels_fed_back=0),
            pairs=pairs, selection=selection))
    return result


class AnalysisTest(unittest.TestCase):
    def test_primary_uses_native_single_intervention_full_day_cost(self):
        report = analyze(scenes(), artifact())
        self.assertTrue(report['integrity']['all_pass'])
        self.assertTrue(report['promotion']['passed'])
        self.assertEqual(report['overall']['models']['fresh']['sum_selected_minus_reference']['cost'], -47.)
        primary = report['primary']
        self.assertEqual(primary['models']['fresh']['interventions'], 2)
        self.assertAlmostEqual(primary['mean_comparisons_per_passenger']['fresh_minus_stale']['cost'], -17.5/15079)
        self.assertEqual(primary['by_scenario']['stable']['models']['fresh']['full_selected_endpoint']['cost'], 49970.)
        self.assertLess(len(json.dumps(report).encode()), 64000)
        self.assertNotIn('line', report['overall']['models'])

    def test_later_query_gains_cannot_hide_first_intervention_loss(self):
        values = scenes()
        set_delta(values[0]['pairs'][0], 15.)
        set_delta(values[0]['pairs'][2], -100.)
        report = analyze(values, artifact())
        self.assertLess(report['by_scenario']['stable']['models']['fresh']['sum_selected_minus_reference']['cost'], 0.)
        chosen = report['primary']['by_scenario']['stable']['models']['fresh']
        self.assertEqual(chosen['selected_root']['line_id'], '7X')
        self.assertEqual(chosen['selected_minus_reference']['cost'], 15.)
        self.assertEqual(chosen['single_opportunity_oracle_regret'], 115.)
        self.assertFalse(report['promotion']['checks']['stable_fresh_not_worse_than_reference'])
        self.assertFalse(report['promotion']['passed'])

    def test_fresh_must_beat_stale_not_only_reference(self):
        values = scenes()
        for scene in values:
            for pair in scene['pairs']:
                pair['prediction']['models']['stale'] = deepcopy(pair['prediction']['models']['fresh'])
                pair['prediction']['models']['legacy_narrow'] = deepcopy(pair['prediction']['models']['fresh'])
        report = analyze(values, artifact())
        self.assertTrue(report['integrity']['all_pass'])
        self.assertFalse(report['promotion']['checks']['fresh_mean_beats_stale'])
        self.assertFalse(report['promotion']['passed'])

    def test_guard_release_is_separate_from_fresh_label_gain(self):
        values = scenes()
        first = values[0]['pairs'][0]
        first['reference_hold_seconds'] = 3.7
        set_prediction(first, 'stale', -1.)
        set_prediction(first, 'legacy_narrow', None, False, 'reference_holding_outside_3_to_3_6')
        report = analyze(values, artifact())
        self.assertTrue(report['integrity']['all_pass'])
        primary = report['primary']['by_scenario']['stable']
        self.assertEqual(primary['comparisons']['stale_minus_legacy_narrow']['cost'], -30.)
        self.assertEqual(primary['comparisons']['fresh_minus_stale']['cost'], 0.)
        self.assertEqual(report['overall']['models']['legacy_narrow']['prediction_count'], 5)
        self.assertEqual(report['overall']['models']['stale']['prediction_count'], 6)

    def test_ood_kept_as_none_reference_and_skips_to_next_supported_root(self):
        values = scenes()
        first = values[0]['pairs'][0]
        first['reference_hold_seconds'] = 3.75
        for name in ('stale', 'fresh', 'legacy_narrow'):
            set_prediction(first, name, None, False, 'reference_holding_outside_domain')
        report = analyze(values, artifact())
        self.assertTrue(report['integrity']['all_pass'])
        model = report['pairs'][0]['models']['fresh']
        self.assertIsNone(model['predicted_delta'])
        self.assertEqual(model['selected_minus_reference']['cost'], 0.)
        self.assertEqual(model['two_action_regret'], 30.)
        self.assertEqual(report['primary']['by_scenario']['stable']['models']['fresh']['selected_root']['line_id'], '705X')
        set_prediction(first, 'fresh', 0., False, 'reference_holding_outside_domain')
        self.assertFalse(analyze(values, artifact())['integrity']['all_pass'])

    def test_no_intervention_is_baseline_but_not_success(self):
        values = scenes()
        for scene in values:
            for pair in scene['pairs']:
                set_prediction(pair, 'fresh', 1.)
        report = analyze(values, artifact())
        self.assertTrue(report['integrity']['all_pass'])
        self.assertEqual(report['primary']['models']['fresh']['interventions'], 0)
        self.assertEqual(report['primary']['by_scenario']['stable']['models']['fresh']['full_selected_endpoint']['cost'], 50000.)
        self.assertFalse(report['promotion']['passed'])

    def test_each_scene_must_not_worsen_even_if_average_is_good(self):
        values = scenes()
        set_delta(values[0]['pairs'][0], -100.)
        set_delta(values[1]['pairs'][0], 5.)
        report = analyze(values, artifact())
        self.assertTrue(report['promotion']['checks']['fresh_mean_beats_reference'])
        self.assertFalse(report['promotion']['checks']['local_speed_drop_fresh_not_worse_than_reference'])
        self.assertFalse(report['promotion']['passed'])

    def test_missing_or_zero_duration_remains_incomplete_not_zero(self):
        values = scenes()
        values[1]['pairs'].pop()
        report = analyze(values, artifact())
        self.assertEqual(len(report['exceptions']['missing']), 1)
        self.assertIsNone(report['primary']['models']['fresh']['mean_full_day_cost_per_passenger'])
        self.assertIsNone(report['primary']['by_scenario']['local_speed_drop']['models']['fresh']['selected_minus_reference'])
        self.assertFalse(report['promotion']['passed'])
        values = scenes()
        for endpoint in values[0]['pairs'][0]['remaining'].values():
            endpoint['dt'] = 0.
        report = analyze(values, artifact())
        self.assertEqual(len(report['exceptions']['zero_duration']), 1)
        self.assertIsNone(report['primary']['by_scenario']['stable']['models']['fresh']['full_selected_endpoint'])

    def test_wrong_data_scope_budget_or_future_fit_blocks_promotion(self):
        fitted = artifact()
        fitted['arms']['fresh']['training_scope']['heldout_labels_used'] = 1
        self.assertFalse(analyze(scenes(), fitted)['training_checks']['fresh_no_heldout_truth'])
        fitted = artifact()
        fitted['arms']['fresh']['training_scope']['external_seed'] = 316207
        self.assertFalse(analyze(scenes(), fitted)['training_checks']['fresh_six_training_pairs'])
        values = scenes()
        values[0]['budget']['total_native_seconds'] += 2000
        values[0]['budget']['native_prefix_seconds'] += 2000
        self.assertFalse(analyze(values, artifact())['integrity']['checks']['fixed_native_budget'])
        values = scenes()
        values[0]['pairs'][0]['before_suffixes'] = False
        self.assertFalse(analyze(values, artifact())['integrity']['all_pass'])

    def test_reference_endpoint_mismatch_and_false_legacy_coefficients_fail(self):
        values = scenes()
        values[0]['pairs'][1]['reference_full_endpoint']['wait'] += 1
        report = analyze(values, artifact())
        self.assertFalse(report['integrity']['checks']['reference_full_day_endpoints_equal'])
        self.assertFalse(report['promotion']['passed'])
        values = scenes()
        set_prediction(values[0]['pairs'][0], 'legacy_narrow', -.5)
        self.assertFalse(analyze(values, artifact())['integrity']['all_pass'])

    def test_tied_issue_clock_uses_sequence_not_declared_line_order(self):
        values = scenes()
        values[0]['pairs'][2]['root']['started_at'] = 12005.
        values[0]['pairs'][2]['prediction_recorded_at'] = 12005.
        values[0]['pairs'][2]['root']['sequence'] = 1
        for endpoint in values[0]['pairs'][2]['remaining'].values():
            endpoint['dt'] = 11995.
        report = analyze(values, artifact())
        self.assertTrue(report['integrity']['all_pass'])
        self.assertEqual(report['primary']['by_scenario']['stable']['models']['fresh']['selected_root']['line_id'], '705X')


if __name__ == '__main__':
    unittest.main()
