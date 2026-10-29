"""Check matched-block stability, component cancellation and rounding accounting."""
import math
import unittest

import numpy as np

from experiments.mcwm_closed_loop_positive_mean_v46_20261024.analysis import analyze_bank
from experiments.mcwm_closed_loop_trace_components_v45_20261024.components import COMPONENTS


def pair(response, *, q=None, entropy=0., reward=.25, rounding=0., trace_rounding=0.):
    components = dict.fromkeys(COMPONENTS, 0.)
    components.update(reward=reward, mixed_value=.5,
        sac_q_without_entropy=response-.5-entropy if q is None else q,
        entropy=entropy, sac_split_rounding=rounding)
    trace = math.fsum(components.values())+trace_rounding
    return dict(component_totals=components, native_target_difference=.125,
                trace_target_difference=trace, event_count=45)


class AnalysisTests(unittest.TestCase):
    def test_shared_eight_draw_blocks_distinguish_flip_reduction_from_accuracy(self):
        draws = [-7., -5., -3., -1., 3., 5., 7., 9.]
        blocks = [dict(block_id=block, single_pairs=[pair(value+block) for value in draws],
            mean8_pair=pair(1.+block)) for block in range(16)]
        result = analyze_bank(blocks)
        self.assertEqual(result['m1_score_count'], 128)
        self.assertEqual(result['m8_score_count'], 16)
        self.assertEqual(result['sign_stability']['m1_within_block_flip_count'], 7)
        self.assertFalse(result['sign_stability']['m8_across_blocks_sign_flip'])
        self.assertEqual(result['m8']['metrics']['trace_target_difference']['positive'], 16)
        expected_m1 = np.asarray([value+block+.25 for block in range(16) for value in draws])
        expected_m8 = np.arange(16)+1.25
        comparison = result['matched_estimators']['trace_target_difference']
        self.assertAlmostEqual(comparison['m8_over_m1_sample_std'],
            expected_m8.std(ddof=1)/expected_m1.std(ddof=1))
        self.assertEqual(comparison['m8_minus_m1_mean'], 0.)
        self.assertEqual(comparison['block_mean_assembly_difference_max_abs'], 0.)
        self.assertIn('not_accuracy', result['scope'])
        self.assertTrue(result['no_300_second_truth'])

    def test_joint_q_entropy_covariance_and_persistent_residual_are_not_hidden(self):
        blocks = []
        for block in range(2):
            samples = [pair(2., q=float(draw), entropy=1.5-float(draw)) for draw in range(8)]
            blocks.append(dict(block_id=block, single_pairs=samples,
                mean8_pair=pair(2., q=3.5, entropy=-2.)))
        result = analyze_bank(blocks)
        self.assertEqual(result['m1']['metrics']['bootstrap_minus_q']['mean'], 2.)
        self.assertEqual(result['m8']['metrics']['bootstrap_minus_q']['mean'], 2.)
        covariance = result['m1']['q_entropy_covariance']
        self.assertEqual(covariance['joint_variance'], 0.)
        self.assertAlmostEqual(covariance['correlation'], -1.)
        self.assertAlmostEqual(covariance['variance_identity_residual'], 0.)
        self.assertIsNone(result['matched_estimators']['trace_target_difference']['m8_over_m1_sample_std'])

    def test_actual_mean_target_rounding_is_reported_separately(self):
        blocks = [dict(block_id=block, single_pairs=[pair(1.) for _ in range(8)],
            mean8_pair=pair(1., rounding=1e-5, trace_rounding=2e-6)) for block in range(2)]
        result = analyze_bank(blocks)
        metrics = result['m8']['metrics']
        self.assertAlmostEqual(metrics['bootstrap_minus_q']['mean'], 1.)
        self.assertAlmostEqual(metrics['split_rounding']['mean'], 1e-5)
        self.assertAlmostEqual(metrics['reconstruction_residual']['mean'], 2e-6)
        self.assertAlmostEqual(result['matched_estimators']['trace_target_difference'][
            'block_mean_assembly_difference_mean'], 1.2e-5)
        row = dict(zip(result['per_block']['columns'], result['per_block']['rows'][0]))
        self.assertAlmostEqual(row['m8_trace'], row['m8_reward']+row['m8_residual']+
            row['m8_split_rounding']+row['m8_assembly_rounding']+row['m8_reconstruction_residual'])

    def test_incomplete_declared_blocks_are_rejected(self):
        block = dict(block_id=0, single_pairs=[pair(1.) for _ in range(7)], mean8_pair=pair(1.))
        with self.assertRaisesRegex(ValueError, 'eight'):
            analyze_bank([block, dict(block, block_id=1)])


if __name__ == '__main__':
    unittest.main()
