"""Detect altered one-step rounding, unsupported links, and label-budget leaks."""

import unittest

import numpy as np

from experiments.mcwm_closed_loop_trace_support_v39_20261024.trace import construct_trace


class TraceTests(unittest.TestCase):
    def test_lambda_zero_is_bitwise_native_even_with_missing_baselines(self):
        native = np.asarray([-0., np.nextafter(np.float32(1.), np.float32(2.)), -117.000015], dtype=np.float32)
        value = construct_trace(native, [None, float('nan')], [1., 1.], [True]*3, 0., dt=[0., 2., 3.])
        actual = np.asarray(value['targets'], dtype=np.float32)
        np.testing.assert_array_equal(actual.view(np.uint32), native.view(np.uint32))
        self.assertEqual(value['reach_weights'], [1., 0., 0.])
        self.assertEqual(value['zero_duration_events'], 1)

    def test_recurrence_keeps_mixed_q_coefficient_for_every_actual_event(self):
        value = construct_trace([1., 2., 3.], [.5, 1.], [.5, 1.], [True]*3, 1., dt=[0., 0., 4.])
        last = np.float32(3.)
        middle = np.float32(np.float32(2.)+np.float32(np.float32(.9)*np.float32(last-np.float32(1.))))
        first = np.float32(np.float32(1.)+np.float32(np.float32(np.float32(.9)*np.float32(.5))*np.float32(middle-np.float32(.5))))
        np.testing.assert_array_equal(np.asarray(value['targets'], dtype=np.float32), [first, middle, last])
        self.assertAlmostEqual(value['reach_weights'][2], .9*.5*.9, places=7)
        self.assertEqual(value['reached_event_indices'], [0, 1, 2])
        self.assertEqual(value['zero_duration_events'], 2)

    def test_budget_mask_prevents_an_unbudgeted_reward_entering_any_earlier_target(self):
        raw = construct_trace([1., 2., 1000., 4.], [0.]*3, [1.]*3, [True]*4, 1.)
        budget = construct_trace([1., 2., 1000., 4.], [0.]*3, [1.]*3, [True, True, False, True], 1.)
        self.assertGreater(raw['targets'][0], 800.)
        self.assertAlmostEqual(budget['targets'][0], 2.8, places=6)
        self.assertIsNone(budget['targets'][2])
        self.assertEqual(budget['first_stop'], dict(event=2, reason='successor_feedback_not_eligible'))
        self.assertEqual(budget['reached_event_indices'], [0, 1])

    def test_zero_support_and_missing_matched_baseline_truncate_without_substitution(self):
        for c, baseline, reason in ((0., 99., 'zero_execution_support'),
                                    (None, 99., 'executed_action_support_unavailable'),
                                    (1., None, 'matching_subset_target_q_unavailable')):
            value = construct_trace([1., 200.], [baseline], [c], [True]*2, 1.)
            self.assertEqual(value['targets'], [1., 200.])
            self.assertEqual(value['first_stop']['reason'], reason)
            self.assertEqual(value['reach_weights'], [1., 0.])

    def test_synthetic_endpoint_is_not_converted_into_a_target(self):
        value = construct_trace([1., 200.], [0.], [1.], [True]*2, 1., real_endpoints=[True, False])
        self.assertEqual(value['targets'], [1., None])
        self.assertEqual(value['first_stop']['reason'], 'not_a_real_closed_event')
        self.assertEqual(value['reached_event_indices'], [0])


if __name__ == '__main__':
    unittest.main()
