"""Check attribution cancellation, terminal semantics and crossed-noise isolation."""
import math
import unittest

import numpy as np

from experiments.mcwm_closed_loop_trace_components_v45_20261024.components import (
    COMPONENTS, compare_pair, crossed_variance, decompose_components, factorial_summary,
)
from experiments.mcwm_closed_loop_trace_target_v44_20261024.trace import decompose_trace


def chain(offset=0.):
    result = []
    for index in range(45):
        reward = np.float32(.1*(index % 3)+offset)
        value = np.float32(-1000.13+index*.31)
        sac_q = np.float32(-9999.71+index*.17)
        entropy = np.float32(.011+index*.0003)
        value_part = np.float32(np.float32(.1)*value)
        sac = np.float32(sac_q+entropy)
        sac_part = np.float32(np.float32(.9)*sac)
        bootstrap = np.float32(np.float32(.1)*value+np.float32(.9)*sac)
        result.append(dict(sequence=index, started_at=float(index),
            completed_at=float(index)+(index % 3 != 0), dt=float(index % 3 != 0),
            terminal=False, gamma=1., native_target=float(np.float32(reward+bootstrap)),
            reward_scaled=float(reward), bootstrap_contribution=float(bootstrap),
            recorded_next_target_q=-9099.72+index*.12,
            value_contribution=float(value_part), sac_contribution=float(sac_part),
            next_value=float(value), sac_q_without_entropy=float(sac_q),
            sac_entropy_bonus=float(np.float32(sac-sac_q))))
    return result


class ComponentsTests(unittest.TestCase):
    def test_split_exactly_accounts_native_trace_despite_large_cancellation(self):
        rows, c = chain(), [.7+(index % 3)*.1 for index in range(44)]
        result = decompose_components(rows, c)
        native = decompose_trace(rows, c, 1.)
        self.assertEqual(result['trace_target'], native['targets'][0])
        self.assertLess(abs(result['reconstruction_residual']), 4e-12)
        self.assertNotEqual(result['component_totals']['sac_split_rounding'], 0.)
        self.assertNotEqual(result['component_totals']['native_rounding'], 0.)
        self.assertNotEqual(result['component_totals']['recurrence_rounding'], 0.)
        self.assertEqual(result['per_event'][1]['raw']['previous_link_recorded_q'],
                         float(np.float32(rows[0]['recorded_next_target_q'])))
        for name in COMPONENTS:
            self.assertAlmostEqual(result['component_totals'][name],
                math.fsum(part['components'][name] for part in result['fixed_bins'].values()), places=10)
        self.assertEqual(result['fixed_bins']['root0']['events'], 1)
        self.assertEqual(result['fixed_bins']['events25_44']['events'], 20)

    def test_pair_difference_and_zero_duration_partition_are_complete(self):
        low_rows, high_rows = chain(), chain(.02)
        high_rows[1]['dt'], high_rows[3]['dt'] = 0., 1.
        low = decompose_components(low_rows, [.8]*44)
        high = decompose_components(high_rows, [.8]*44)
        pair = compare_pair(low, high)
        self.assertLess(abs(pair['reconstruction_residual']), 4e-12)
        self.assertEqual(pair['trace_target_difference'], high['trace_target']-low['trace_target'])
        self.assertEqual(sum(value['events'] for value in pair['zero_duration'].values()), 45)
        self.assertEqual(pair['zero_duration']['low_only_zero']['events'], 1)
        self.assertEqual(pair['zero_duration']['high_only_zero']['events'], 1)
        self.assertEqual(len(pair['absolute_event_order']), 45)
        self.assertEqual({value['event'] for value in pair['absolute_event_order']}, set(range(45)))

    def test_lambda_zero_and_terminal_do_not_create_bootstrap_credit(self):
        rows = chain()
        result = decompose_components(rows, [None]*44, 0.)
        self.assertTrue(result['lambda_zero_native_exact'])
        self.assertEqual(result['trace_target'], rows[0]['native_target'])
        for event in result['per_event'][1:]:
            self.assertEqual(event['total'], 0.)
        rows[0].update(terminal=True, bootstrap_contribution=0., value_contribution=0.,
                       sac_contribution=0., native_target=rows[0]['reward_scaled'])
        terminal = decompose_components(rows, [1.]*44)
        self.assertEqual(terminal['trace_target'], rows[0]['reward_scaled'])
        self.assertEqual(terminal['component_totals']['sac_q_without_entropy'], 0.)
        self.assertEqual(terminal['component_totals']['entropy'], 0.)

    def test_crossed_decomposition_separates_draw_subset_and_interaction(self):
        actor = np.array([-2., 0., 2.])
        subset = np.array([-1., 0., 1.])
        interaction = np.outer([-1., 0., 1.], [-1., 0., 1.])*.25
        matrix = .1+actor[:, None]+subset[None, :]+interaction
        result = crossed_variance(matrix)
        np.testing.assert_allclose(result['actor_main_effect'], actor, atol=1e-15)
        np.testing.assert_allclose(result['target_subset_main_effect'], subset, atol=1e-15)
        np.testing.assert_allclose(result['interaction'], interaction, atol=1e-15)
        self.assertAlmostEqual(result['sums_of_squares']['actor'], 24.)
        self.assertAlmostEqual(result['sums_of_squares']['target_subset'], 6.)
        self.assertAlmostEqual(result['sums_of_squares']['interaction'], .25)
        self.assertLess(abs(result['sums_of_squares_residual']), 1e-14)
        self.assertTrue(all(result['actor_draw_fixed_subset_sign_flip']))
        cells = [dict(actor_seed=a, subset_seed=h, metric=float(matrix[a, h]))
                 for a in range(3) for h in range(3)]
        self.assertEqual(factorial_summary(cells, 'metric')['matrix'], matrix.tolist())
        with self.assertRaisesRegex(ValueError, 'unique'):
            factorial_summary(cells[:-1]+[cells[0]], 'metric')


if __name__ == '__main__':
    unittest.main()
