from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.mcwm_closed_loop_load_coverage_v89_20261028.analysis import GUARDS, analyze


def protocol():
    return json.loads(Path(__file__).with_name('protocol.json').read_text())


def audit():
    return dict(all_pass=True, checks=dict(original_reference_exact=True), source_initializer=6207,
        external_seed=716207, scenario='stable', reference_roots=6,
        source_actor_path='/resident/v84/stable/reference_actor.pt',
        reference_trace_path='/resident/v88/runs/preflight/prefix_trace.jsonl',
        new_fits=0, new_neural_updates=0, new_suffixes=0, new_labels=0,
        v87_truth_used=0, native_seconds=0)


def collector(inside=(.1, .1, .1, .1, .2, .3, .4), outside=(.5, .8)):
    rows = []
    for index, (in_domain, load) in enumerate([(True, value) for value in inside]
            +[(False, value) for value in outside]):
        rows.append(dict(sequence=8000+index, trip_id='7X_'+str(index), station_id='7X15',
            issue_clock=float(12000+index), event_clock=float(11999+index),
            obs_start=float(12000+index), obs_end=float(12000+index),
            in_domain=in_domain, domain_reason=None if in_domain else 'nonpositive_holding_difference',
            at_stop=True, sim_step_seconds=1., action_step_seconds=1., onboard=round(load*50),
            capacity=50., load=load, observed=True, guard_checks={key: True for key in GUARDS},
            ref_hold=3.4 if in_domain else 4., cand_hold=7.5 if in_domain else 0.))
    old = dict(source='v75', source_arm='fresh', seed=6207, scenario='stable', packets=37,
        covariates=185, complete_outcomes=185, counts=dict(critic=296, value=296, actor=296, alpha=296),
        max_available_at=12000, data_deadline=12000, deterministic=True, actor_frozen=True)
    return dict(status='complete', integrity=dict(all_pass=True, checks=dict(actor_frozen=True)),
        seed=6207, external_seed=716207, scenario='stable', prefix_id='v89/stable/external716207',
        reference_snapshot=dict(prior_run_reference=True, available_before_heldout_episode=True,
            source_snapshot_metadata=old, source_actor_path=audit()['source_actor_path'], heldout_external_seed=716207),
        trace_replay=dict(all_pass=True, comparison_exact=True, reference_exhausted=True,
            observed_issues=8500, matched_issues=8500, reference_trace_path=audit()['reference_trace_path'],
            window_7x_issues=len(rows), window_7x_sequences=[row['sequence'] for row in rows]),
        observation_checks={key: True for key in GUARDS}, rows=rows,
        budget=dict(actual_simulated_seconds=12600, prefixes=1, suffixes=0, labels=0,
            new_initialization_updates=0, new_fits=0, new_neural_updates=0,
            feedback_admission=0, v87_truth_used=0), training_started=False, labels_used_for_selection=False)


class LoadCoverageAnalysisTest(unittest.TestCase):
    def test_distinct_value_midgap_ignores_frequency_and_reports_both_domains(self):
        values = collector()
        before = deepcopy(values)
        with patch('numpy.linalg.solve', side_effect=AssertionError('No fitting in load coverage')):
            result = analyze(values, protocol(), audit())
        self.assertEqual(values, before)
        self.assertTrue(result['coverage_gate']['passed'])
        domain = result['distributions']
        self.assertEqual(domain['inside']['histogram'], {'0.1': 4, '0.2': 1, '0.3': 1, '0.4': 1})
        self.assertEqual(domain['inside']['distinct_values'], [.1, .2, .3, .4])
        self.assertEqual(domain['outside']['min_load'], .5)
        self.assertEqual(domain['outside']['max_load'], .8)
        self.assertEqual(domain['inside']['original_half_capacity_split']['at_or_above_half'], 0)
        self.assertEqual(domain['outside']['original_half_capacity_split']['at_or_above_half'], 2)
        split = result['stratification']
        self.assertEqual(split['boundary'], .25)
        self.assertEqual(split['selected_adjacent_distinct_values'], [.2, .3])
        self.assertEqual(split['sample_counts'], dict(lighter=5, heavier=2))
        self.assertEqual(split['earliest_issue']['lighter']['sequence'], 8000)
        self.assertEqual(split['earliest_issue']['heavier']['sequence'], 8005)
        self.assertTrue(result['interpretation']['old_v88_high_not_filled'])
        self.assertLess(len(json.dumps(result).encode()), 16000)

    def test_odd_distinct_count_uses_declared_upper_middle_gap(self):
        result = analyze(collector(inside=(.1, .4, .8), outside=()), protocol(), audit())
        self.assertTrue(result['coverage_gate']['passed'])
        self.assertEqual(result['stratification']['distinct_gap_index'], 1)
        self.assertEqual(result['stratification']['selected_adjacent_distinct_values'], [.4, .8])
        self.assertAlmostEqual(result['stratification']['boundary'], .6)
        self.assertEqual(result['stratification']['sample_counts'], dict(lighter=2, heavier=1))

    def test_single_inside_load_is_honest_hold_even_if_outside_varies(self):
        result = analyze(collector(inside=(.2, .2), outside=(.1, .8)), protocol(), audit())
        self.assertTrue(result['integrity']['all_pass'])
        self.assertFalse(result['coverage_gate']['passed'])
        self.assertIsNone(result['stratification']['boundary'])
        self.assertEqual(result['stratification']['sample_counts'], dict(lighter=None, heavier=None))
        self.assertIsNone(result['stratification']['earliest_issue']['lighter'])

    def test_no_hardcoded_39_and_trace_cohort_identity_not_only_row_count(self):
        values = collector(inside=(.2, .3), outside=())
        self.assertTrue(analyze(values, protocol(), audit())['coverage_gate']['passed'])
        values['rows'][1]['sequence'] += 1
        result = analyze(values, protocol(), audit())
        self.assertTrue(result['integrity']['checks']['all_window_7x_rows'])
        self.assertFalse(result['integrity']['checks']['all_window_7x_sequences'])
        self.assertIsNone(result['stratification']['boundary'])
        values = collector()
        values['rows'].pop()
        self.assertFalse(analyze(values, protocol(), audit())['integrity']['checks']['all_window_7x_rows'])

    def test_missing_native_load_not_zero_filled_and_prevents_freeze(self):
        values = collector()
        values['rows'][0].update(onboard=None, load=None, observed=False)
        result = analyze(values, protocol(), audit())
        inside = result['distributions']['inside']
        self.assertEqual(inside['issues'], 7)
        self.assertEqual(inside['observed_loads'], 6)
        self.assertEqual(inside['missing_loads'], 1)
        self.assertNotIn('0.0', inside['histogram'])
        self.assertIsNone(result['stratification']['boundary'])
        self.assertFalse(result['coverage_gate']['passed'])

    def test_bad_clock_ledger_rng_or_trace_never_freezes_partial_good_data(self):
        for name in ('clock_unchanged', 'ledger_unchanged', 'rng_unchanged', 'before_action'):
            values = collector()
            values['rows'][0]['guard_checks'][name] = False
            result = analyze(values, protocol(), audit())
            self.assertIsNone(result['stratification']['boundary'], name)
            self.assertFalse(result['coverage_gate']['passed'])
        values = collector()
        values['trace_replay']['matched_issues'] -= 1
        result = analyze(values, protocol(), audit())
        self.assertFalse(result['integrity']['checks']['full_trace_reproduced'])
        self.assertIsNone(result['stratification']['boundary'])
        values = collector()
        values['rows'][0]['obs_end'] += 1
        result = analyze(values, protocol(), audit())
        self.assertIsNone(result['stratification']['boundary'])
        self.assertIn('atomic_observation', result['exceptions']['invalid_observations'][0]['failed_checks'])

    def test_native_load_identity_and_original_f_domain_are_verified(self):
        values = collector()
        values['rows'][0]['load'] = .2
        result = analyze(values, protocol(), audit())
        self.assertFalse(result['integrity']['checks']['complete_native_loads'])
        values = collector()
        values['rows'][0].update(ref_hold=4., cand_hold=0.)
        result = analyze(values, protocol(), audit())
        self.assertFalse(result['integrity']['checks']['all_issue_observations_valid'])
        self.assertIsNone(result['stratification']['boundary'])

    def test_nonstop_issue_is_reported_outside_without_discarding_native_load(self):
        values = collector(inside=(.1, .2), outside=(.8,))
        values['rows'][-1].update(ref_hold=3.4, cand_hold=7.5, at_stop=False,
            domain_reason='native_not_at_stop')
        result = analyze(values, protocol(), audit())
        self.assertTrue(result['integrity']['all_pass'])
        self.assertEqual(result['distributions']['outside']['histogram'], {'0.8': 1})

    def test_suffix_label_fit_or_v87_truth_forbidden_and_no_early_prefix(self):
        for name in ('suffixes', 'labels', 'new_fits', 'new_neural_updates', 'feedback_admission', 'v87_truth_used'):
            values = collector()
            values['budget'][name] = 1
            self.assertFalse(analyze(values, protocol(), audit())['coverage_gate']['passed'], name)
        frozen = audit()
        frozen['new_labels'] = 1
        self.assertFalse(analyze(collector(), protocol(), frozen)['coverage_gate']['passed'])
        values = collector()
        values['budget']['actual_simulated_seconds'] = 12599
        self.assertFalse(analyze(values, protocol(), audit())['coverage_gate']['passed'])
        altered = protocol()
        altered['stratification']['rule'] = 'frequency_median'
        self.assertFalse(analyze(collector(), altered, audit())['coverage_gate']['passed'])


if __name__ == '__main__':
    unittest.main()
