"""Focused checks for the once-scaled physical credit target."""
import copy
import math
import unittest

import numpy as np

from .credit import (CAPACITY, ETA, bounded_trace, probe_bootstrap_shift,
                     probe_reward_band, probe_uniform_q_shift)
from experiments.mcwm_closed_loop_physical_credit_v50_20261025.credit import physical_trace
from experiments.mcwm_closed_loop_trace_target_v44_20261024.summary import paired_decomposition


def row(reward=0., bootstrap=0., *, start=0., end=0., baseline=0., terminal=False):
    reward, bootstrap = np.float32(reward), np.float32(bootstrap)
    return dict(reward_scaled=float(reward), bootstrap_contribution=float(bootstrap),
        native_target=float(np.float32(reward + bootstrap)), recorded_next_target_q=baseline,
        dt=end-start, started_at=start, completed_at=end, terminal=terminal, gamma=1.)


class BoundedCreditTests(unittest.TestCase):
    def test_lambda_zero_native_bits_and_suffix_targets(self):
        rows = [row(.1, -10000., end=1.), row(.2, 4., start=1., end=3.), row(1., 2., start=3., end=8.)]
        zero = bounded_trace(rows, [.7, .8], 0.)
        np.testing.assert_array_equal(np.asarray(zero['targets'], dtype=np.float32).view(np.uint32),
            np.asarray([value['native_target'] for value in rows], dtype=np.float32).view(np.uint32))
        self.assertTrue(zero['lambda_zero_native_exact'])
        full = bounded_trace(rows, [.7, .8], 1.)
        for i in range(len(rows)):
            self.assertEqual(full['targets'][i], bounded_trace(rows[i:], [.7, .8][i:], 1.)['targets'][0])
        self.assertEqual(full['original_backup_sac_weight'], .9)

    def test_capacity_boundary_and_short_zero_time_chain(self):
        rows = [row() for _ in range(CAPACITY)]
        bound = bounded_trace(rows, [1.]*(CAPACITY-1), 1.)
        self.assertAlmostEqual(bound['gain']['future_sum_reach'], 9., places=6)
        self.assertAlmostEqual(bound['gain']['uniform_q_offset_gain'], 0., places=6)
        self.assertEqual(bound['gain']['ideal_future_sum_reach'], 9.)
        short = bounded_trace(rows[:45], [1.]*44, 1.)
        self.assertAlmostEqual(short['gain']['uniform_q_offset_gain'], .9-.1*44*ETA, places=7)
        self.assertAlmostEqual(short['gain']['uniform_q_offset_gain'], .7527881041, places=7)
        response = probe_uniform_q_shift(rows[:45], [1.]*44, bounded_trace)
        self.assertAlmostEqual(response['perturbations'][1]['root_change'],
                               short['gain']['uniform_q_offset_gain'], places=5)
        with self.assertRaisesRegex(ValueError, '270'):
            bounded_trace(rows+[row()], [1.]*CAPACITY, 1.)

    def test_physical_weights_are_scaled_once_and_zero_time_retained(self):
        rows = [row(end=30.), row(start=30., end=30.), row(5., start=30., end=60.)]
        trace = bounded_trace(rows, [.5, .5], 1.)
        physical = physical_trace(rows, [.5, .5], 1.)
        self.assertEqual(len(trace['links']), 2)
        self.assertEqual(trace['links'][1]['physical_retention'], 1.)
        self.assertAlmostEqual(trace['reach_weights'][2], ETA*math.exp(-30/300)*.25, places=8)
        self.assertAlmostEqual(trace['reach_weights'][2]/physical['reach_weights'][2], ETA, places=8)
        self.assertFalse(trace['links'][1]['outer_scale_applied_per_link'])

    def test_support_stop_and_true_terminal_do_not_join_later_rows(self):
        rows = [row(), row(), row(100.)]
        missing = bounded_trace(rows, [None, 1.], 1.)
        self.assertEqual(missing['reach_weights'], [1., 0., 0.])
        self.assertEqual(missing['first_stop']['reason'], 'executed_action_support_unavailable')
        rows[0]['recorded_next_target_q'] = None
        missing = bounded_trace(rows, [1., 1.], 1.)
        self.assertEqual(missing['first_stop']['reason'], 'matching_subset_target_q_unavailable')
        terminal = bounded_trace([row(1., terminal=True), row(100.)], [1.], 1.)
        self.assertEqual(terminal['reach_weights'], [1., 0.])
        self.assertEqual(terminal['root']['trace_target'], 1.)

    def test_cutoff_bootstrap_and_native_decomposition_remain_accounted(self):
        rows = [row(1., 4., end=20.), row(2., 3., start=20., end=35., baseline=1.),
                row(-1., 8., start=35., end=50.)]
        trace = bounded_trace(rows, [.9, .7], .6)
        self.assertFalse(trace['terminal_dependency']['true_terminal'])
        self.assertEqual(trace['terminal_dependency']['raw_bootstrap'], 8.)
        self.assertAlmostEqual(trace['root']['reconstruction_residual'], 0., places=12)
        accounting = trace['rounding_breakdown']
        self.assertAlmostEqual(accounting['combined_recurrence'],
            accounting['scaled_physical_recurrence']+accounting['root_blend']['total'], places=12)
        pair = paired_decomposition(trace, bounded_trace(rows, [.9, .7], 1.))
        self.assertAlmostEqual(pair['reconstruction_residual'], 0., places=12)
        probe = probe_bootstrap_shift(rows, [.9, .7], bounded_trace)
        for value in probe['perturbations']:
            self.assertAlmostEqual(value['root_change'], value['algebraic_change'], places=5)

    def test_probes_do_not_mutate_input_and_use_fixed_reward_endpoints(self):
        rows = [row(1., end=20.), row(2., start=20., end=35.),
                row(3., start=35., end=45.), row(4., start=45., end=50.)]
        c, before = [.9, .8, .7], copy.deepcopy(rows)
        probe = probe_reward_band(rows, c, bounded_trace)
        self.assertEqual(probe['selected_count'], 1)
        expected = -2*bounded_trace(rows, c, 1.)['reach_weights'][1]
        self.assertAlmostEqual(probe['algebraic_change'], expected, places=12)
        self.assertAlmostEqual(probe['root_change'], expected, places=6)
        probe_bootstrap_shift(rows, c, bounded_trace)
        probe_uniform_q_shift(rows, c, bounded_trace)
        self.assertEqual(rows, before)
        self.assertEqual(c, [.9, .8, .7])


if __name__ == '__main__':
    unittest.main()
