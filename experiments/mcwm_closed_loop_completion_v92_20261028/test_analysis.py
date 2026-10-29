from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.mcwm_closed_loop_stratified_collection_v90_20261028.test_analysis import scenes as original_scenes
from experiments.mcwm_closed_loop_service_response_v88_20261028.service import service_state
from experiments.mcwm_closed_loop_completion_v92_20261028.analysis import CUTS, LINES, NEW_CASE, _condition_readout, analyze


def protocol():
    return json.loads(Path(__file__).with_name('protocol.json').read_text())


def condition(now, value):
    return dict(status='supported', reason=None, value=value, issue_clock=now, historical_only=True,
        baseline_before=9000., baseline_counts={line: 3 for line in LINES},
        baseline_medians_seconds={line: 90. for line in LINES},
        baseline_latest_completed_at={line: 8000. for line in LINES},
        baseline_latest_observed_at={line: 8000. for line in LINES},
        latest_completed_at=now-100, latest_observed_at=now-100, latest_age_seconds=100.,
        latest_sources=[dict(line_id='7X', trip_id='strictly_prior', completed_at=now-100, observed_at=now-100,
            nonholding_ready_seconds=90.+60*value, baseline_nonholding_seconds=90., normalized_slowdown=value)])


def compact_condition(now, value):
    result = condition(now, value)
    for key in ('baseline_before', 'baseline_latest_completed_at', 'baseline_latest_observed_at'):
        result.pop(key)
    result['baseline_last_observed_at'] = 8000.
    return result


def inputs():
    source = original_scenes()
    old, new = [], None
    for scene in source:
        scenario = scene['scenario']
        for pair in scene['pairs']:
            key = pair['line_id']+'/'+pair['load_bin']
            value = dict(pair, context=deepcopy(scene['coverage'][key]['context']),
                condition=compact_condition(pair['root']['started_at'], 0. if scenario == 'stable' else 1.))
            path = '/resident/v90/'+scenario+'/'+key+'/pair_summary.json'
            value.update(source_pair_path=path,
                outcome_source_pair_path=pair['source_pair_path'] if pair['reused'] else path)
            if (scenario, key) == NEW_CASE:
                new = value
            else:
                old.append(value)
    audit = dict(all_pass=True, checks=dict(old_roots_and_conditions_exact=True),
        source_initializer=6207, external_seed=716207, reference_roots=12, thresholds=dict(CUTS),
        reference_source_paths={scene['scenario']: scene['reference_snapshot']['source_actor_path'] for scene in source},
        reused_pairs=11, reused_labels=22, historical_native_seconds=314090,
        new_fits=0, new_neural_updates=0, new_suffixes=0, new_labels=0, v87_truth_used=0, native_seconds=0,
        old_pairs=old)
    root = dict(new['root'], started_at=12700., sequence=9000)
    ctx = new['context']
    ctx['clocks'].update(issue=12700., event=12699., obs_start=12700., obs_end=12700.)
    new.update(root=root, service=service_state(ctx), condition=condition(12700., 1.),
        source_pair_path=None, outcome_source_pair_path=None)
    for outcome in new['remaining'].values():
        outcome['dt'] = 11300.
    new['physical'] = dict(reference=dict(departure_clock=12704., next_arrival_clock=12780., missing={}),
        candidate=dict(departure_clock=12708., next_arrival_clock=12784., missing={}))
    budget = dict(prefixes=1, suffixes=2, labels_observed=2, native_prefix_seconds=12900.,
        suffix_native_seconds=22600., total_native_seconds=35500., unknown_suffix_costs=0,
        new_initialization_updates=0, prefix_training_updates=0, suffix_updates=0, labels_fed_back=0,
        new_fits=0, new_neural_updates=0, v87_truth_used=0, reused_pairs=11, reused_labels=22)
    collector = dict(status='complete', integrity=dict(all_pass=True, checks=dict(reproduction=True)),
        seed=6207, external_seed=716207, scenario='local_speed_drop',
        reference_snapshot=source[1]['reference_snapshot'],
        pairs=[new], coverage={NEW_CASE[1]: dict(root=root, context=ctx, service=new['service'], condition=new['condition'])},
        prefix_reproduction=dict(all_pass=True, checks=dict(comparison_exact=True, reference_exhausted=True),
            observed_issues=9001, matched_issues=9001),
        reuse_checks={pair['line_id']+'/'+pair['load_bin']: dict(root=True, condition=True, service=True)
            for pair in old if pair['scenario'] == 'local_speed_drop'}, budget=budget,
        first_pair_gate=dict(all_pass=True, checks=deepcopy(new['checks']), labels_used_for_gate=False,
            cost_sign_used_for_gate=False))
    return collector, audit


def set_label(pair, delta):
    reference = pair['remaining']['reference']
    pair['remaining']['candidate'].update(ride=reference['ride']+delta, cost=reference['cost']+delta)
    pair['delta'] = dict(wait=0., ride=delta, cost=delta)


class CompletionAnalysisTest(unittest.TestCase):
    def test_one_new_query_completes_fixed_old_eleven_and_only_covariate_ranks_are_computed(self):
        collector, audit = inputs()
        with patch('numpy.linalg.solve', side_effect=AssertionError('No fitting')):
            with patch('numpy.linalg.lstsq', side_effect=AssertionError('No target least squares')):
                report = analyze(collector, protocol(), audit)
        self.assertTrue(report['completion_gate']['passed'])
        self.assertEqual(report['covariate_design']['ranks'], dict(line=3, service5=5, service_q6=6))
        self.assertEqual(report['budget']['old_logical_pairs'], 11)
        self.assertEqual(report['budget']['new_pairs'], 1)
        self.assertEqual(report['budget']['new']['labels_observed'], 2)
        self.assertEqual(report['budget']['cumulative_native_seconds'], 349590.)
        self.assertEqual(len(report['pairs']), 12)
        self.assertTrue(report['interpretation']['old_v90_hold_unchanged'])
        self.assertFalse(report['interpretation']['policy_cost_evaluated'])
        self.assertLess(len(json.dumps(report).encode()), 64000)
        self.assertTrue(all(row['condition']['value'] == 0. for row in report['pairs'] if row['scenario'] == 'stable'))

    def test_missing_completion_stays_eleven_rows_and_no_zero_label(self):
        collector, audit = inputs()
        collector.update(status='incomplete', pairs=[], coverage={}, first_pair_gate=dict(all_pass=False))
        collector['budget'].update(suffixes=0, labels_observed=0, suffix_native_seconds=0., total_native_seconds=12900.)
        report = analyze(collector, protocol(), audit)
        self.assertFalse(report['completion_gate']['passed'])
        self.assertEqual(report['exceptions']['missing_cases'], [dict(scenario=NEW_CASE[0], slot=NEW_CASE[1])])
        self.assertEqual(len(report['pairs']), 11)
        self.assertEqual(report['budget']['new_pairs'], 0)
        self.assertEqual(report['budget']['cumulative_native_seconds'], 326990.)

    def test_q_missing_or_same_clock_is_not_imputed_even_when_flag_claims_historical(self):
        collector, audit = inputs()
        audit['old_pairs'][0]['condition'].update(status='unsupported', value=None, reason='history_missing')
        report = analyze(collector, protocol(), audit)
        self.assertIsNone(report['pairs'][0]['condition']['value'])
        self.assertIsNone(report['covariate_design']['ranks']['service_q6'])
        self.assertFalse(report['completion_gate']['passed'])
        collector, audit = inputs()
        q = audit['old_pairs'][0]['condition']
        q['latest_sources'][0].update(completed_at=q['issue_clock'], observed_at=q['issue_clock'])
        q.update(latest_completed_at=q['issue_clock'], latest_observed_at=q['issue_clock'], latest_age_seconds=0.)
        self.assertFalse(analyze(collector, protocol(), audit)['completion_gate']['passed'])

    def test_q_coverage_and_variation_are_distinct_from_full_rank_service(self):
        collector, audit = inputs()
        for pair in audit['old_pairs']:
            pair['condition'] = compact_condition(pair['root']['started_at'], 0.)
        collector['pairs'][0]['condition'] = condition(collector['pairs'][0]['root']['started_at'], 0.)
        collector['coverage'][NEW_CASE[1]]['condition'] = collector['pairs'][0]['condition']
        report = analyze(collector, protocol(), audit)
        self.assertTrue(report['integrity']['checks']['all_twelve_q_supported'])
        self.assertEqual(report['covariate_design']['ranks']['service5'], 5)
        self.assertEqual(report['covariate_design']['ranks']['service_q6'], 5)
        self.assertTrue(all(row['range'] == 0. for row in report['covariate_design']['q_variation_by_line'].values()))
        self.assertFalse(report['completion_gate']['passed'])

    def test_noop_and_truncated_labels_remain_invalid(self):
        collector, audit = inputs()
        pair = collector['pairs'][0]
        pair['physical']['candidate']['departure_clock'] = pair['physical']['reference']['departure_clock']
        set_label(pair, 0.)
        report = analyze(collector, protocol(), audit)
        self.assertIsNone(report['pairs'][-1]['label_delta'])
        self.assertFalse(report['completion_gate']['passed'])
        collector, audit = inputs()
        collector['pairs'][0]['remaining']['candidate']['observed_at'] = 20000
        self.assertFalse(analyze(collector, protocol(), audit)['completion_gate']['passed'])

    def test_label_signs_do_not_decide_completion_or_rank(self):
        original, audit_original = inputs()
        baseline = analyze(original, protocol(), audit_original)
        for delta in (-100., 0., 100.):
            collector, audit = inputs()
            for pair in audit['old_pairs']+collector['pairs']:
                set_label(pair, delta)
            report = analyze(collector, protocol(), audit)
            self.assertEqual(report['integrity'], baseline['integrity'])
            self.assertEqual(report['covariate_design'], baseline['covariate_design'])
            self.assertTrue(report['completion_gate']['passed'])

    def test_old_roots_reproduction_and_new_slot_cannot_be_substituted(self):
        collector, audit = inputs()
        collector['reuse_checks']['7X/heavier']['root'] = False
        self.assertFalse(analyze(collector, protocol(), audit)['completion_gate']['passed'])
        collector, audit = inputs()
        audit['old_pairs'].append(deepcopy(audit['old_pairs'][0]))
        with self.assertRaises(ValueError):
            analyze(collector, protocol(), audit)
        collector, audit = inputs()
        collector['pairs'][0]['scenario'] = 'stable'
        with self.assertRaises(ValueError):
            analyze(collector, protocol(), audit)

    def test_new_budget_fit_or_first_gate_sign_violation_holds(self):
        for key in ('new_fits', 'new_neural_updates', 'v87_truth_used', 'labels_fed_back'):
            collector, audit = inputs()
            collector['budget'][key] = 1
            self.assertFalse(analyze(collector, protocol(), audit)['completion_gate']['passed'])
        collector, audit = inputs()
        collector['first_pair_gate']['cost_sign_used_for_gate'] = True
        self.assertFalse(analyze(collector, protocol(), audit)['completion_gate']['passed'])
        collector, audit = inputs()
        collector['budget'].update(total_native_seconds=37000., suffix_native_seconds=24100.)
        report = analyze(collector, protocol(), audit)
        self.assertFalse(report['integrity']['checks']['new_native_budget'])

    def test_actual_v91_compact_manifest_condition_keeps_its_real_clock_provenance(self):
        # First old row from the resident v92 freeze_audit manifest.
        raw = dict(status='supported', reason=None, value=0.05221744179725647,
            historical_only=True, issue_clock=12001., baseline_counts={'7X': 21, '102X': 10, '705X': 6},
            baseline_medians_seconds={'7X': 73.31773471832275, '102X': 72.83684408664703, '705X': 75.39873117208481},
            latest_completed_at=11916., latest_observed_at=11916., latest_age_seconds=85.,
            baseline_last_observed_at=8932., latest_sources=[dict(line_id='7X', trip_id='7X_30',
                completed_at=11916., observed_at=11916., nonholding_ready_seconds=76.45078122615814,
                baseline_nonholding_seconds=73.31773471832275, normalized_slowdown=0.05221744179725647)])
        value = _condition_readout(raw, 12001., 'old_v90_fixed')
        self.assertTrue(value['valid'])
        self.assertEqual(value['value'], raw['value'])
        self.assertEqual(value['baseline_provenance'], dict(source_schema='v91_fixed_compact', baseline_last_observed_at=8932.))
        self.assertNotIn('baseline_latest_completed_at', value['baseline_provenance'])


if __name__ == '__main__':
    unittest.main()
