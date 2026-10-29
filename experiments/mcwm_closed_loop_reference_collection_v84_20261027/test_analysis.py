from copy import deepcopy
import json
import unittest

from experiments.mcwm_closed_loop_reference_collection_v84_20261027.analysis import analyze


def scenes():
    result = []
    for scenario, differences in (('stable', (-30.,20.,0.)), ('local_speed_drop',(10.,-5.,2.))):
        pairs, selection = [], {}
        for index, (line, difference) in enumerate(zip(('7X','102X','705X'), differences)):
            root = dict(started_at=12005.+index, sequence=100+index, trip_id=line+'_1', station_id=line+'15')
            ref = dict(wait=10., ride=80., cost=100., dt=24000-root['started_at'], observed_at=24000, available_at=24030)
            candidate = dict(ref, ride=80.+difference, cost=100.+difference)
            pairs.append(dict(line_id=line, root=root, status='complete', all_pass=True,
                checks=dict(native_fork=True), reference_hold_seconds=3., candidate_hold_seconds=7.5,
                overlap_seconds=10., remaining=dict(reference=ref, candidate=candidate),
                physical=dict(reference=dict(departure_clock=12006.,next_arrival_clock=12090.,missing={}),
                              candidate=dict(departure_clock=12011.,next_arrival_clock=12091.,missing={})),
                delta=dict(wait=0.,ride=difference,cost=difference)))
            selection[line] = dict(selected=root, attempted=1, reasons={}, status='selected')
        reference = dict(source='v75',source_arm='fresh',seed=6207,scenario=scenario,
            packets=37,covariates=185,complete_outcomes=185,
            counts=dict(critic=296,value=296,actor=296,alpha=296),data_deadline=12000,
            max_available_at=12000,published_clock=12000,deterministic=True,actor_frozen=True,
            future_training_used=False)
        result.append(dict(scenario=scenario,seed=6207,external_seed=106207,status='complete',
            integrity=dict(all_pass=True,checks={}),reference_snapshot=reference,
            source_prefix_rows=8000,source_update_rows=74,departed_passengers=15079,
            budget=dict(native_prefix_seconds=12020.,suffix_native_seconds=71970.,total_native_seconds=83990.,
                restored_covariates=185,restored_complete_outcomes=185,restored_updates_per_optimizer=296,
                suffix_updates=0,labels_fed_back=0,new_initialization_updates=0,prefixes=1,suffixes=6,labels_observed=6),
            pairs=pairs,selection=selection))
    return result


class AnalysisTest(unittest.TestCase):
    def test_direction_two_action_regret_and_holding_orientation(self):
        values = scenes()
        first = values[0]['pairs'][0]
        first['candidate_hold_seconds'] = 0.
        report = analyze(values)
        self.assertTrue(report['integrity']['all_pass'])
        self.assertEqual(report['overall']['directions'], dict(candidate_better=2,reference_better=3,tie=1))
        self.assertEqual(report['overall']['sum_two_action_regret'], dict(reference=35.,candidate=32.))
        self.assertEqual(report['pairs'][0]['more_holding_minus_less_holding_cost'], 30.)
        self.assertEqual(report['pairs'][0]['state_level_and_response']['absolute_response_percent_of_reference'], 30.)
        self.assertLess(len(json.dumps(report).encode()), 48000)

    def test_missing_state_is_not_filled_with_zero(self):
        values = scenes()
        values[1]['pairs'].pop()
        values[1]['selection']['705X']['selected'] = None
        values[1]['selection']['705X']['status'] = 'missing'
        report = analyze(values)
        self.assertEqual(len(report['pairs']), 5)
        self.assertEqual(len(report['exceptions']['missing']), 1)
        self.assertFalse(report['training_readout']['target_data_ready'])
        self.assertEqual(report['overall']['valid_states'], 5)
        self.assertEqual(report['interpretation']['queried_states_not_independent_replicates'], 5)

    def test_premature_and_zero_duration_labels_fail_without_becoming_zero_costs(self):
        values = scenes()
        values[0]['pairs'][0]['remaining']['candidate']['available_at'] = 24000
        values[0]['pairs'][1]['remaining']['candidate']['dt'] = 0
        report = analyze(values)
        self.assertFalse(report['integrity']['all_pass'])
        self.assertEqual(report['overall']['valid_states'], 4)
        self.assertEqual(len(report['exceptions']['zero_duration']), 1)
        self.assertEqual(len(report['exceptions']['invalid_pair_roots']), 2)

    def test_hard_budget_blocks_training_readiness(self):
        values = deepcopy(scenes())
        values[0]['budget']['native_prefix_seconds'] += 2000
        values[0]['budget']['total_native_seconds'] += 2000
        report = analyze(values)
        self.assertFalse(report['integrity']['checks']['fixed_native_budget'])
        self.assertFalse(report['training_readout']['target_data_ready'])


if __name__ == '__main__':
    unittest.main()
