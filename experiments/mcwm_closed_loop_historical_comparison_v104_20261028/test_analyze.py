"""Synthetic costs only; no simulator, checkpoint, fit or historical truth."""
import copy
import unittest

from experiments.mcwm_closed_loop_historical_comparison_v104_20261028.analyze import analyze


def fixture():
    protocol = dict(historical_training_seeds=[6201, 6202, 6203, 6204, 6205],
        external_seeds=[106206, 106207, 106208], scenarios=['stable', 'local_speed_drop'],
        external_seed_offset=100000, full_day_departed_passengers=10,
        horizon_seconds=24000, new_native_seconds_cap=720000,
        periods=dict(full_day=[0, 24000], early=[9000, 12000], post_budget=[18300, 24000]))
    factors = dict(full_day=1., early=.2, post_budget=.3)

    def period(cost_per_passenger, name):
        cost = cost_per_passenger*10*factors[name]
        return dict(status='complete', wait=cost/4, ride=cost/2, cost=cost,
            cost_per_passenger=cost/10, departed=10,
            start=protocol['periods'][name][0], end=protocol['periods'][name][1])

    runs, cells = {}, []
    for external_index, external in enumerate(protocol['external_seeds']):
        for scenario_index, scenario in enumerate(protocol['scenarios']):
            cell = 2*external_index+scenario_index
            for actor_index, actor in enumerate(protocol['historical_training_seeds']):
                runs[(actor, external, scenario)] = dict(status='complete',
                    integrity=dict(all_pass=True), actor_seed=actor, external_seed=external,
                    scenario=scenario, passenger_denominator=10,
                    periods={name: period(2+4*actor_index+cell, name) for name in factors},
                    budget=dict(native_seconds=24000, native_seconds_is_lower_bound=False,
                        optimizer_updates=0, target_training_labels=0, world_updates=0, trust_updates=0),
                    action_audit=dict(raw_execution_exact=True, canonical_differences=0))
            periods = {}
            for name, factor in factors.items():
                periods[name] = dict(arms={arm: period(value+cell, name)
                    for arm, value in (('fresh', 18), ('no_fresh', 20))},
                    matched=dict(wait_per_passenger=-.5*factor,
                                 ride_per_passenger=-factor, cost_per_passenger=-2*factor))
            cells.append(dict(identity=external-100000, scenario=scenario, periods=periods))
    cached = dict(status='complete', complete=True, integrity=dict(all_pass=True), cells=cells,
        aggregates=dict(full_day=dict(overall=dict(mean_cost_difference=-2.))),
        independent_seed_readout=dict(independent_unit='training seed', complete_seeds=3),
        decision=dict(scientific_gate_pass=True, old_v73_gate_reopened=False),
        budget=dict(response=dict(actual_native_seconds=288000)))
    return protocol, cached, runs


class AnalysisTest(unittest.TestCase):
    def test_fixed_weights_units_and_cached_attribution(self):
        protocol, cached, runs = fixture()
        original = copy.deepcopy(cached)
        result = analyze(protocol, cached, runs)
        self.assertTrue(result['complete'])
        self.assertEqual(cached, original)
        self.assertEqual(result['cached_v75_attribution']['cells'], original['cells'])
        self.assertEqual(result['cached_v75_attribution']['decision'], original['decision'])
        full = result['aggregates']['full_day']
        self.assertAlmostEqual(full['historical_actor_mean_distribution']['mean']['cost_per_passenger'], 12.5)
        self.assertEqual([row['periods']['full_day']['historical_mean']['cost_per_passenger']
                          for row in result['historical_actor_means']], [4.5, 8.5, 12.5, 16.5, 20.5])
        self.assertAlmostEqual(full['historical_actor_mean_distribution']['sample_sd']['cost_per_passenger'], 40**.5)
        for name, factor in (('full_day', 1), ('early', .2), ('post_budget', .3)):
            difference = result['aggregates'][name]['dynamic_minus_historical']['fresh']
            self.assertAlmostEqual(difference['cost_per_passenger'], 8*factor)
            self.assertAlmostEqual(2*difference['wait_per_passenger']+difference['ride_per_passenger'], 8*factor)
            self.assertAlmostEqual(result['aggregates'][name]['dynamic_minus_historical']['no_fresh']['cost_per_passenger'], 10*factor)
        first = result['environment_cells'][0]['periods']['full_day']['historical_distribution']
        self.assertEqual(first['minimum']['cost_per_passenger'], 2)
        self.assertEqual(first['maximum']['cost_per_passenger'], 18)
        self.assertEqual(result['budget']['new_native_seconds'], 720000)
        self.assertFalse(result['decision']['promotion_gate'])

    def test_missing_day_does_not_reweight_or_fill_zero(self):
        protocol, cached, runs = fixture()
        del runs[(6201, 106206, 'stable')]
        result = analyze(protocol, cached, runs)
        self.assertFalse(result['complete'])
        self.assertEqual(result['decision']['state'], 'HOLD_INCOMPLETE')
        self.assertIsNone(result['aggregates'])
        self.assertIsNone(result['environment_cells'][0]['periods']['full_day']['historical_distribution'])
        self.assertIsNone(result['historical_actor_means'][0]['periods']['full_day']['historical_mean'])
        self.assertEqual(result['budget']['known_native_seconds'], 696000)
        self.assertIsNone(result['budget']['new_native_seconds'])
        self.assertEqual(result['budget']['unknown_spend_slots'], 1)
        self.assertEqual(result['cached_v75_attribution']['cells'], cached['cells'])

    def test_bad_period_units_boundary_and_denominator_are_not_ranked(self):
        changes = [('cost', 1.), ('departed', 9), ('start', 8999), ('ride', None)]
        for field, value in changes:
            with self.subTest(field=field):
                protocol, cached, runs = fixture()
                runs[(6201, 106206, 'stable')]['periods']['early'][field] = value
                result = analyze(protocol, cached, runs)
                self.assertFalse(result['complete'])
                self.assertEqual(len(result['invalid']), 1)
                self.assertIsNone(result['aggregates'])

    def test_wrong_actor_identity_cannot_fill_another_slot(self):
        protocol, cached, runs = fixture()
        runs[(6201, 106206, 'stable')]['actor_seed'] = 6202
        result = analyze(protocol, cached, runs)
        self.assertFalse(result['complete'])
        self.assertIn('summary identity differs from its declared slot', result['invalid'][0]['errors'])

    def test_partial_failure_keeps_consumed_seconds(self):
        protocol, cached, runs = fixture()
        row = runs[(6201, 106206, 'stable')]
        row.update(status='failed', integrity=dict(all_pass=False), periods={})
        row['budget']['native_seconds'] = 123.
        result = analyze(protocol, cached, runs)
        self.assertFalse(result['complete'])
        self.assertEqual(result['budget']['new_native_seconds'], 696123.)
        self.assertIsNone(result['aggregates'])

    def test_incomplete_cached_panel_prevents_system_ranking(self):
        protocol, cached, runs = fixture()
        cached['cells'].pop()
        result = analyze(protocol, cached, runs)
        self.assertFalse(result['complete'])
        self.assertIsNone(result['aggregates'])
        self.assertTrue(result['integrity']['cached_validation_errors'])
        self.assertIsNone(result['environment_cells'][0]['periods']['full_day']['dynamic_minus_historical_mean']['fresh'])


if __name__ == '__main__':
    unittest.main()
