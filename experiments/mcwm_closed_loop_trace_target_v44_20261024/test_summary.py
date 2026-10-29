"""Detect reversed pair signs and attribution of ratio drift to reward changes."""
import math
import unittest
import numpy as np

from experiments.mcwm_closed_loop_trace_target_v44_20261024.trace import decompose_trace
from experiments.mcwm_closed_loop_trace_target_v44_20261024.summary import paired_decomposition


def chain(rewards, bootstraps):
    return [dict(native_target=float(np.float32(np.float32(r)+np.float32(b))),
        reward_scaled=r, bootstrap_contribution=b, terminal=False, dt=1.,
        recorded_next_target_q=-8., value_contribution=.1*b, sac_contribution=.9*b)
        for r, b in zip(rewards, bootstraps)]


class PairTests(unittest.TestCase):
    def test_high_minus_low_and_components_reconstruct_action_difference(self):
        low = decompose_trace(chain([0., -3., -1.], [-10.]*3), [.9, .8], 1.)
        high = decompose_trace(chain([0., -2., -1.], [-10., -9., -8.]), [.6, .7], 1.)
        value = paired_decomposition(low, high)
        self.assertEqual(value['trace_difference'], high['targets'][0]-low['targets'][0])
        self.assertLess(abs(value['reconstruction_residual']), 1e-12)
        self.assertAlmostEqual(math.fsum(value['reward_value_vs_weight'].values()),
                               value['components']['reward'], places=12)

    def test_identical_rewards_separate_weight_drift_from_new_reward_signal(self):
        rows = chain([-1., -100., -100.], [-5.]*3)
        low, high = [decompose_trace(rows, [c, c], 1.) for c in (1., .3)]
        value = paired_decomposition(low, high)
        split = value['reward_value_vs_weight']
        self.assertEqual(split['common_weight_value_difference'], 0.)
        self.assertNotEqual(split['common_value_weight_difference'], 0.)
        self.assertAlmostEqual(split['common_value_weight_difference'], value['components']['reward'])


if __name__ == '__main__':
    unittest.main()
