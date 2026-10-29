"""Focused tests for physical cutoff semantics and loss of reward credit."""
import math
import unittest

import numpy as np

from .metrics import diagnose_chain, paired_diagnosis


def row(reward=0., bootstrap=0., *, start=0., end=0., baseline=0.):
    reward, bootstrap = np.float32(reward), np.float32(bootstrap)
    return dict(reward_scaled=float(reward), bootstrap_contribution=float(bootstrap),
        native_target=float(np.float32(reward+bootstrap)), recorded_next_target_q=baseline,
        dt=end-start, started_at=start, completed_at=end, terminal=False, gamma=1.)


class CreditReachTests(unittest.TestCase):
    def test_zero_dt_consumes_links_not_physical_time_and_sources_separate(self):
        chain = dict(rows=[row(), row(), row(3., 7., end=50.)], c=[.5, .5])
        result = diagnose_chain(chain, prefix_counts=(1, 2, 3))
        self.assertTrue(result['lambda_zero']['native_exact'])
        self.assertEqual(result['clock']['physical_seconds'], 50.)
        logs = result['attenuation']['zero_duration_links']
        self.assertEqual(logs['link_count'], 2)
        self.assertAlmostEqual(logs['mixing'], 2*math.log(.9))
        self.assertAlmostEqual(logs['c'], 2*math.log(.5))
        self.assertEqual(logs['gamma'], 0.)
        full = result['full']
        self.assertAlmostEqual(full['cutoff_weight'], float(np.float32(.45))**2)
        self.assertAlmostEqual(full['future_reward'], 3.*full['cutoff_weight'])
        self.assertAlmostEqual(full['cutoff_bootstrap'], 7.*full['cutoff_weight'])
        self.assertEqual(result['first_nonzero_reward_event'], 2)
        self.assertEqual(result['prefixes'][1]['clock']['physical_seconds'], 0.)

    def test_cutoff_retains_bootstrap_and_native_float32_perturbation(self):
        result = diagnose_chain(dict(rows=[row(), row(1., 4., end=1.)], c=[1.]),
                                prefix_counts=(1, 2))
        self.assertEqual(result['prefixes'][1]['cutoff_native_bootstrap'], 4.)
        self.assertFalse(result['prefixes'][1]['cutoff_true_terminal'])
        for probe in result['cutoff_sensitivity']['perturbations']:
            self.assertAlmostEqual(probe['root_target_delta'], .9*probe['realized_bootstrap_delta'], places=6)
        self.assertAlmostEqual(result['full']['trace_target'], 4.5, places=6)

    def test_large_bootstrap_cancellation_and_pair_difference_remain_closed(self):
        low = dict(rows=[row(.1, -10000., baseline=-9999.5),
                         row(.2, -10000.5, end=2.)], c=[.7])
        high = dict(rows=[row(.1, -10000., baseline=-9999.5),
                          row(.3, -10000.25, end=3.)], c=[.7])
        result = paired_diagnosis(low, high, prefix_counts=(1, 2))
        self.assertLess(abs(result['full_difference']['difference_closure_residual']), 1e-10)
        self.assertFalse(result['full_difference']['same_physical_endpoint'])
        for arm in ('low', 'high'):
            full = result[arm]['full']
            self.assertLess(abs(full['closure_residual']), 1e-10)
            self.assertNotEqual(full['components']['rounding'], 0.)
            self.assertAlmostEqual(full['components']['bootstrap_minus_matching_q'],
                full['cutoff_bootstrap']+full['intermediate_bootstrap_minus_matching_q'])

    def test_absent_support_and_ablation_have_explicit_nonlearning_scope(self):
        result = diagnose_chain(dict(rows=[row(), row(10., 4., end=9.)], c=[None]),
                                prefix_counts=(1, 2))
        self.assertEqual(result['full']['trace_target'], 0.)
        self.assertEqual(result['full']['supported_event_count'], 1)
        self.assertEqual(result['attenuation']['first_stop']['reason'],
                         'executed_action_support_unavailable')
        self.assertGreater(result['algebraic_ablations']['cases']['c_one']['trace_target'], 0.)
        self.assertEqual(result['new_training_updates'], 0)
        self.assertEqual(result['synthetic_states'], 0)


if __name__ == '__main__':
    unittest.main()
