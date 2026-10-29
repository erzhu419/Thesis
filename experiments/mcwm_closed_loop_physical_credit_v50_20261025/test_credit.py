"""Tests of the changed credit operator and numerical probe accounting."""
import copy
import math
import unittest

import numpy as np

from .credit import (physical_trace, probe_bootstrap_shift, probe_reward_band,
                     probe_uniform_q_shift)
from experiments.mcwm_closed_loop_trace_target_v44_20261024.trace import decompose_trace


def row(reward=0., bootstrap=0., *, start=0., end=0., baseline=0., terminal=False):
    reward, bootstrap = np.float32(reward), np.float32(bootstrap)
    return dict(reward_scaled=float(reward), bootstrap_contribution=float(bootstrap),
        native_target=float(np.float32(reward+bootstrap)), recorded_next_target_q=baseline,
        dt=end-start, started_at=start, completed_at=end, terminal=terminal, gamma=1.)


class PhysicalCreditTests(unittest.TestCase):
    def test_lambda_zero_preserves_every_native_bit(self):
        rows = [row(.1, -10000., end=1.), row(.2, 4., start=1., end=3.)]
        result = physical_trace(rows, [.7], 0.)
        np.testing.assert_array_equal(np.asarray(result['targets'], dtype=np.float32).view(np.uint32),
            np.asarray([value['native_target'] for value in rows], dtype=np.float32).view(np.uint32))
        self.assertTrue(result['lambda_zero_native_exact'])
        self.assertEqual(result['original_backup_sac_weight'], .9)

    def test_physical_span_weights_keep_zero_duration_link(self):
        rows = [row(end=30.), row(start=30., end=30.), row(5., start=30., end=60.)]
        result = physical_trace(rows, [.5, .5], 1.)
        self.assertEqual(len(result['links']), 2)
        self.assertEqual(result['links'][1]['physical_retention'], 1.)
        self.assertEqual(result['links'][1]['actual_c'], .5)
        self.assertEqual(result['links'][1]['effective_c'], .5)
        self.assertAlmostEqual(result['reach_weights'][-1], math.exp(-30./300.)*.25, places=7)
        self.assertFalse(result['links'][0]['effective_c_is_actual_RN'])

    def test_missing_support_and_matching_q_stop_without_joining(self):
        rows = [row(), row(), row(100.)]
        missing = physical_trace(rows, [None, 1.], 1.)
        self.assertEqual(missing['reach_weights'], [1., 0., 0.])
        self.assertEqual(missing['first_stop']['reason'], 'executed_action_support_unavailable')
        rows[0]['recorded_next_target_q'] = None
        missing = physical_trace(rows, [1., 1.], 1.)
        self.assertEqual(missing['first_stop']['reason'], 'matching_subset_target_q_unavailable')

    def test_zero_time_c_one_can_reverse_uniform_q_offset(self):
        rows, c = [row() for _ in range(45)], [1.]*44
        new = physical_trace(rows, c, 1.)
        self.assertEqual(new['gain']['sum_reach'], 45.)
        self.assertAlmostEqual(new['gain']['uniform_q_offset_gain'], -3.5)
        new_probe = probe_uniform_q_shift(rows, c)
        self.assertAlmostEqual(new_probe['perturbations'][1]['root_change'], -3.5, places=5)
        old_probe = probe_uniform_q_shift(rows, c, decompose_trace)
        self.assertAlmostEqual(old_probe['linear_reference_gain'], .9**45, places=6)
        self.assertGreater(old_probe['perturbations'][1]['root_change'], 0.)

    def test_real_cutoff_bootstrap_retained_and_terminal_not_perturbed(self):
        rows = [row(), row(1., 4., end=1.)]
        result = physical_trace(rows, [1.], 1.)
        self.assertEqual(result['root']['trace_target'], 5.)
        self.assertEqual(result['terminal_dependency']['raw_bootstrap'], 4.)
        self.assertFalse(result['terminal_dependency']['true_terminal'])
        probe = probe_bootstrap_shift(rows, [1.])
        self.assertEqual([value['root_change'] for value in probe['perturbations']], [-1., 1.])
        terminal = [row(3., terminal=True)]
        self.assertEqual(physical_trace(terminal, [], 1.)['root']['trace_target'], 3.)
        self.assertFalse(probe_bootstrap_shift(terminal, [])['applicable'])

    def test_probes_are_readonly_and_fixed_complete_reward_band(self):
        rows = [row(1., end=20.), row(2., start=20., end=35.),
                row(3., start=35., end=45.), row(4., start=45., end=50.)]
        c, before = [.9, .8, .7], copy.deepcopy(rows)
        result = probe_reward_band(rows, c)
        self.assertEqual(result['selected_count'], 1)
        expected = physical_trace(rows, c, 1.)['reach_weights'][1]*-2.
        self.assertAlmostEqual(result['algebraic_change'], expected)
        self.assertAlmostEqual(result['root_change'], expected, places=5)
        probe_bootstrap_shift(rows, c)
        probe_uniform_q_shift(rows, c)
        self.assertEqual(rows, before)
        self.assertEqual(c, [.9, .8, .7])


if __name__ == '__main__':
    unittest.main()
