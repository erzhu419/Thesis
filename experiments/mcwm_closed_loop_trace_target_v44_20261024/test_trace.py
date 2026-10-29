"""Target parity, reward attribution and physical-cutoff semantics."""
import math
import unittest

import numpy as np

from experiments.mcwm_closed_loop_trace_target_v44_20261024.trace import decompose_trace


def row(reward, bootstrap, baseline=0., *, terminal=False, dt=1., **extra):
    reward, bootstrap = np.float32(reward), np.float32(bootstrap)
    return dict(native_target=float(np.float32(reward+bootstrap)),
        reward_scaled=float(reward), bootstrap_contribution=float(bootstrap),
        recorded_next_target_q=baseline, terminal=terminal, dt=dt, **extra)


class TraceTests(unittest.TestCase):
    def test_lambda_zero_copies_native_bits_and_never_reads_absent_baseline(self):
        rows = [row(-0., -0., None, dt=0.), row(1., 2**-24, float('nan')),
                row(-117., -2**-16)]
        result = decompose_trace(rows, [None, None], 0.)
        expected = np.asarray([r['native_target'] for r in rows], np.float32)
        actual = np.asarray(result['targets'], np.float32)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        self.assertTrue(result['lambda_zero_native_exact'])
        self.assertEqual(result['reach_weights'], [1., 0., 0.])

    def test_reward_bootstrap_and_matching_q_have_exact_additive_accounting(self):
        rows = [row(.1, -10000., -9999.5), row(.2, -10000.5, -10000.125), row(.3, -10000.25)]
        result = decompose_trace(rows, [.7, .8], 1.)
        self.assertLess(abs(result['root']['reconstruction_residual']), 2e-12)
        self.assertNotEqual(result['root']['components']['native_rounding'], 0.)
        self.assertNotEqual(result['root']['components']['recurrence_rounding'], 0.)
        alpha = np.float32(np.float32(.9)*np.float32(.7))
        beta = np.float32(np.float32(.9)*np.float32(.8))
        last = np.float32(rows[-1]['native_target'])
        middle = np.float32(np.float32(rows[1]['native_target'])+
            np.float32(beta*np.float32(last-np.float32(rows[1]['recorded_next_target_q']))))
        first = np.float32(np.float32(rows[0]['native_target'])+
            np.float32(alpha*np.float32(middle-np.float32(rows[0]['recorded_next_target_q']))))
        self.assertEqual(result['targets'][0], float(first))
        self.assertEqual(result['per_event'][1]['previous_link_matching_target_q'], -9999.5)
        self.assertAlmostEqual(result['terminal_dependency']['weighted_native_minus_incoming_matching_q'],
            result['reach_weights'][-1]*(rows[-1]['native_target']-rows[-2]['recorded_next_target_q']), places=12)

    def test_zero_duration_keeps_mixed_path_attenuation_and_last_bootstrap(self):
        rows = [row(0., 0., dt=0.) for _ in range(44)]+[row(3., 7., dt=50.)]
        result = decompose_trace(rows, [1.]*44, 1.)
        coefficient = float(np.float32(.9))**44
        self.assertAlmostEqual(result['terminal_dependency']['coefficient'], coefficient, places=15)
        self.assertAlmostEqual(result['terminal_dependency']['weighted_bootstrap'], 7.*coefficient, places=15)
        self.assertEqual(result['zero_duration_events'], 44)
        self.assertFalse(result['terminal_dependency']['true_terminal'])
        self.assertEqual(result['event_count'], 45)

    def test_true_terminal_stops_link_but_collector_cutoff_does_not(self):
        result = decompose_trace([row(1., 0., terminal=True), row(1000., 10.)], [1.], 1.)
        self.assertEqual(result['targets'][0], 1.)
        self.assertEqual(result['first_stop']['reason'], 'true_terminal_transition')
        self.assertEqual(result['terminal_dependency']['weighted_bootstrap'], 0.)
        with self.assertRaisesRegex(ValueError, 'true terminal'):
            decompose_trace([row(1., 5., terminal=True)], [], 1.)

    def test_missing_matching_subset_is_explicit_and_not_substituted(self):
        result = decompose_trace([row(1., 0., None), row(1000., 7.)], [1.], 1.)
        self.assertEqual(result['targets'][0], 1.)
        self.assertEqual(result['first_stop']['reason'], 'matching_subset_target_q_unavailable')
        self.assertEqual(result['reached_event_count'], 1)

    def test_bootstrap_split_keeps_native_assembly_rounding(self):
        rows = [row(0., .3, value_contribution=float(np.float32(.1)),
                    sac_contribution=float(np.float32(.2)))]
        result = decompose_trace(rows, [], 1.)
        split = result['root']['bootstrap_split']
        self.assertEqual(math.fsum(split.values()), result['root']['components']['bootstrap'])
        self.assertNotEqual(split['assembly_rounding'], 0.)
        rows[0]['native_target'] += 1.
        with self.assertRaisesRegex(ValueError, 'assemble'):
            decompose_trace(rows, [], 1.)


if __name__ == '__main__':
    unittest.main()
