from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.mcwm_closed_loop_service_response_v88_20261028.test_analysis import collector as old_collector
from experiments.mcwm_closed_loop_service_response_v88_20261028.service import service_state
from experiments.mcwm_closed_loop_stratified_collection_v90_20261028.analysis import CUTS, analyze


def protocol():
    return json.loads(Path(__file__).with_name('protocol.json').read_text())


def audit():
    return dict(all_pass=True, checks=dict(original_sources_exact=True), source_initializer=6207,
        external_seed=716207, reference_roots=12, thresholds=dict(CUTS),
        reference_source_paths={scene: '/resident/v84/'+scene+'/reference_actor.pt'
            for scene in ('stable', 'local_speed_drop')},
        reuse_source_pair_path='/resident/v88/runs/preflight/pairs/low/7X/pair_summary.json',
        reuse_source_root=deepcopy(old_collector()['pairs'][0]['root']), reused_labels=2,
        new_fits=0, new_neural_updates=0, new_suffixes=0, new_labels=0, v87_truth_used=0, native_seconds=0)


def scenes():
    result = []
    for scenario in ('stable', 'local_speed_drop'):
        source = old_collector()
        coverage, selection, pairs = {}, {}, []
        for old_key, row in source['coverage'].items():
            line, old_bin = old_key.split('/')
            if line == '7X' and old_bin == 'high':
                row['context']['counts'].update(onboard=5, release_next=0, compatible_next_queue=5,
                    next_queue=5, seats_after_release=45.)
                row['service'] = service_state(row['context'])
                group = 'lighter'
            elif line == '7X':
                group = 'heavier'
            else:
                group = 'lighter' if old_bin == 'low' else 'heavier'
            key = line+'/'+group
            row['load_bin'] = group
            coverage[key] = row
            selection[key] = dict(selected=row['root'], attempted=1, status='selected', reasons={})
            clock = row['root']['started_at']
            reference = dict(dt=24000-clock, wait=100., ride=800., cost=1000., observed_at=24000, available_at=24030)
            candidate = dict(reference, ride=810., cost=1010.)
            reused = scenario == 'stable' and key == '7X/heavier'
            pair = dict(status='complete', all_pass=True,
                checks=dict(same_root=True, same_reference=True, same_seed=True, executed_f_action=True,
                    actual_parallel=True, real_terminal=True, zero_training=True),
                scenario=scenario, external_seed=716207, line_id=line, load_bin=group,
                root=deepcopy(row['root']), service=deepcopy(row['service']), reused=reused,
                source_pair_path=audit()['reuse_source_pair_path'] if reused else None,
                new_stratum_threshold=CUTS[line], reference_hold_seconds=row['ref_hold'],
                candidate_hold_seconds=row['cand_hold'], sim_step_seconds=1., action_step_seconds=1.,
                overlap_seconds=100., remaining=dict(reference=reference, candidate=candidate),
                delta=dict(wait=0., ride=10., cost=10.),
                physical=dict(reference=dict(departure_clock=clock+4, next_arrival_clock=clock+80, missing={}),
                    candidate=dict(departure_clock=clock+8, next_arrival_clock=clock+84, missing={})))
            if reused:
                pair['source_load_bin'] = 'low'
            pairs.append(pair)
        source.update(coverage=coverage, pairs=pairs, selection=selection, scenario=scenario,
            departed_passengers=15079, prefix_id='v90/'+scenario+'/external716207',
            reuse_checks=dict(root=True, context=True, reference=True) if scenario == 'stable' else {},
            prefix_reproduction=dict(all_pass=True, checks=dict(comparison_exact=True, reference_exhausted=True),
                observed_issues=8559, matched_issues=8559),
            training_started=False, models_fitted=False, predictions_used_for_selection=False, labels_used_for_selection=False)
        source['reference_snapshot']['source_snapshot_metadata']['scenario'] = scenario
        source['reference_snapshot']['source_actor_path'] = audit()['reference_source_paths'][scenario]
        new_pairs = [pair for pair in pairs if not pair['reused']]
        seconds = sum(2*pair['remaining']['reference']['dt'] for pair in new_pairs)
        source['budget'].update(suffix_native_seconds=seconds, total_native_seconds=12600.+seconds,
            suffixes=2*len(new_pairs), labels_observed=2*len(new_pairs), unknown_suffix_costs=0,
            reused_pairs=len(pairs)-len(new_pairs), reused_labels=2*(len(pairs)-len(new_pairs)),
            new_neural_updates=0, v87_truth_used=0)
        result.append(source)
    first = min((pair for pair in result[0]['pairs'] if not pair['reused']),
        key=lambda pair: (pair['root']['started_at'], pair['root']['sequence']))
    gate = dict(status='passed', all_pass=True, checks=deepcopy(first['checks']), scenario='stable',
        slot=first['line_id']+'/'+first['load_bin'], root=deepcopy(first['root']), pair_path='/resident/v90/first_pair',
        labels_used_for_gate=False, cost_sign_used_for_gate=False, new_pair_in_formal_cohort=True)
    for scene in result:
        scene['first_pair_gate'] = deepcopy(gate)
    return result


def set_label(pair, delta):
    reference = pair['remaining']['reference']
    pair['remaining']['candidate'].update(ride=reference['ride']+delta, cost=reference['cost']+delta)
    pair['delta'] = dict(wait=0., ride=delta, cost=delta)


class StratifiedCollectionAnalysisTest(unittest.TestCase):
    def test_full_cohort_reuses_one_old_pair_and_only_counts_new_native_budget(self):
        with patch('numpy.linalg.solve', side_effect=AssertionError('No fits during collection')):
            report = analyze(scenes(), protocol(), audit())
        self.assertTrue(report['collection_gate']['passed'])
        self.assertEqual(len(report['pairs']), 12)
        self.assertEqual(report['potential_design']['rows'], 12)
        self.assertEqual(report['potential_design']['ranks'], dict(line=3, conditional=5))
        budget = report['budget']
        self.assertEqual(budget['current_new']['suffixes'], 22)
        self.assertEqual(budget['current_new']['labels_observed'], 22)
        self.assertEqual(budget['current_new']['reused_labels'], 2)
        self.assertEqual(budget['observed_cohort_pairs'], 12)
        self.assertEqual(budget['observed_new_pairs'], 11)
        self.assertEqual(budget['cumulative_native_seconds'], budget['current_new']['total_native_seconds']+49198)
        self.assertFalse(report['interpretation']['policy_cost_evaluated'])
        self.assertNotIn('promotion', report)
        self.assertLess(len(json.dumps(report).encode()), 64000)

    def test_new_threshold_is_used_instead_of_legacy_service_load_bin(self):
        report = analyze(scenes(), protocol(), audit())
        heavy = report['coverage']['stable']['7X/heavier']
        light = report['coverage']['stable']['7X/lighter']
        self.assertEqual(heavy['legacy_half_capacity_bin'], 'low')
        self.assertEqual(heavy['load_bin'], 'heavier')
        self.assertEqual(light['legacy_half_capacity_bin'], 'low')
        self.assertEqual(light['load_bin'], 'lighter')
        self.assertTrue(heavy['valid'])
        self.assertTrue(light['valid'])

    def test_label_signs_and_true_zero_never_decide_collection_gate(self):
        baseline = analyze(scenes(), protocol(), audit())
        for delta in (-100., 0., 100.):
            values = scenes()
            for scene in values:
                for pair in scene['pairs']:
                    set_label(pair, delta)
            result = analyze(values, protocol(), audit())
            self.assertEqual(result['integrity'], baseline['integrity'])
            self.assertTrue(result['collection_gate']['passed'])
            self.assertEqual(result['label_signals']['overall']['valid_pairs'], 12)
            self.assertEqual(result['label_signals']['overall']['physical_outcome_labels'], 24)

    def test_reuse_root_source_and_pair_counter_cannot_drift(self):
        frozen = audit()
        frozen['reuse_source_root']['sequence'] += 1
        result = analyze(scenes(), protocol(), frozen)
        self.assertFalse(result['scene_integrity']['stable']['checks']['source_root_reuse_exact'])
        values = scenes()
        values[1]['pairs'][0]['reused'] = True
        self.assertFalse(analyze(values, protocol(), audit())['collection_gate']['passed'])
        values = scenes()
        reused = next(pair for pair in values[0]['pairs'] if pair['reused'])
        reused['source_pair_path'] = '/different/source.json'
        self.assertFalse(analyze(values, protocol(), audit())['collection_gate']['passed'])

    def test_missing_scene_or_slot_is_explicit_not_filled_as_a_new_sample(self):
        report = analyze(scenes()[:1], protocol(), audit())
        self.assertEqual(report['exceptions']['missing_scenarios'], ['local_speed_drop'])
        self.assertIsNone(report['budget']['cumulative_native_seconds'])
        self.assertFalse(report['collection_gate']['passed'])
        values = scenes()
        key = '705X/heavier'
        values[1]['coverage'].pop(key)
        values[1]['pairs'] = [pair for pair in values[1]['pairs'] if pair['line_id']+'/'+pair['load_bin'] != key]
        result = analyze(values, protocol(), audit())
        self.assertEqual(result['exceptions']['missing_native_pairs'], [dict(scenario='local_speed_drop', slot=key)])
        self.assertEqual(result['budget']['observed_cohort_pairs'], 11)
        self.assertFalse(result['collection_gate']['passed'])

    def test_first_new_pair_gate_is_same_technical_record_and_ignores_cost_sign(self):
        values = scenes()
        values[1]['first_pair_gate']['cost_sign_used_for_gate'] = True
        self.assertFalse(analyze(values, protocol(), audit())['collection_gate']['passed'])
        values = scenes()
        for scene in values:
            scene['first_pair_gate']['root']['sequence'] += 1
        self.assertFalse(analyze(values, protocol(), audit())['first_pair_checks']['first_new_native_pair'])
        values = scenes()
        for scene in values:
            scene['first_pair_gate']['checks'] = dict(unrelated=True)
        self.assertFalse(analyze(values, protocol(), audit())['first_pair_checks']['first_gate_checks_match_native_pair'])

    def test_noop_zero_duration_and_truncated_label_are_not_valid_labels(self):
        values = scenes()
        pair = values[1]['pairs'][0]
        pair['physical']['candidate']['departure_clock'] = pair['physical']['reference']['departure_clock']
        set_label(pair, 0.)
        result = analyze(values, protocol(), audit())
        invalid = next(row for row in result['pairs'] if row['scenario'] == 'local_speed_drop' and not row['valid'])
        self.assertIsNone(invalid['label_delta'])
        self.assertFalse(result['collection_gate']['passed'])
        values = scenes()
        for outcome in values[1]['pairs'][0]['remaining'].values():
            outcome['dt'] = 0.
        self.assertEqual(len(analyze(values, protocol(), audit())['exceptions']['zero_duration']), 1)
        values = scenes()
        values[1]['pairs'][0]['remaining']['candidate']['observed_at'] = 20000
        self.assertFalse(analyze(values, protocol(), audit())['collection_gate']['passed'])

    def test_future_truth_fit_bad_context_or_inexact_stable_prefix_blocks(self):
        values = scenes()
        values[0]['prefix_reproduction']['checks']['reference_exhausted'] = False
        self.assertFalse(analyze(values, protocol(), audit())['collection_gate']['passed'])
        for key in ('new_fits', 'new_neural_updates', 'v87_truth_used', 'labels_fed_back'):
            values = scenes()
            values[1]['budget'][key] = 1
            self.assertFalse(analyze(values, protocol(), audit())['collection_gate']['passed'], key)
        values = scenes()
        values[1]['coverage']['102X/heavier']['observation_checks']['rng_unchanged'] = False
        self.assertFalse(analyze(values, protocol(), audit())['collection_gate']['passed'])
        altered = protocol()
        altered['thresholds']['7X'] = .2
        self.assertFalse(analyze(scenes(), altered, audit())['collection_gate']['passed'])

    def test_native_budget_not_inflated_with_old_labels_or_unknown_spend(self):
        values = scenes()
        values[0]['budget']['labels_observed'] += 2
        self.assertFalse(analyze(values, protocol(), audit())['collection_gate']['passed'])
        values = scenes()
        values[0]['budget']['total_native_seconds'] += 100
        values[0]['budget']['suffix_native_seconds'] += 100
        result = analyze(values, protocol(), audit())
        self.assertFalse(result['integrity']['checks']['current_new_budget'])
        values = scenes()
        values[0]['budget']['unknown_suffix_costs'] = 1
        result = analyze(values, protocol(), audit())
        self.assertIsNone(result['budget']['cumulative_native_seconds'])
        self.assertFalse(result['collection_gate']['passed'])


if __name__ == '__main__':
    unittest.main()
