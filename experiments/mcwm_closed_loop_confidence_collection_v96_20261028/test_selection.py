import unittest

from experiments.mcwm_closed_loop_confidence_collection_v96_20261028.selection import (
    context_reason, frozen_predictions, lock_first)


class SelectionTest(unittest.TestCase):
    def test_unsupported_first_slot_cannot_be_replaced(self):
        slots = {'7X':dict(selected=None, status='missing')}
        self.assertTrue(lock_first(slots, '7X', dict(sequence=1)))
        slots['7X']['status'] = 'unsupported'
        self.assertFalse(lock_first(slots, '7X', dict(sequence=2)))
        self.assertEqual(slots['7X']['selected']['sequence'], 1)

    def test_original_per_line_range_not_global_scale_range(self):
        model = dict(common_support=dict(q_training_ranges={'7X':[.01,.46], '705X':[.01,.50]}))
        condition = dict(status='supported', historical_only=True, latest_completed_at=11999.,
            latest_observed_at=11999., latest_age_seconds=1., value=.49)
        service = dict(observed=True)
        self.assertEqual(context_reason(service, condition, '7X', model, 12000.),
                         'history_outside_frozen_line_range')
        self.assertIsNone(context_reason(service, condition, '705X', model, 12000.))
        condition['latest_observed_at'] = 12000.
        self.assertEqual(context_reason(service, condition, '705X', model, 12000.),
                         'history_availability_or_age')

    def test_forward_only_and_action_scaled_features(self):
        model = dict(hold_scale=7.5, design=dict(line_centers={'7X':[.1,-.5,.2]}),
            models={'line3':dict(coef=[1,2,3]), 'service5':dict(coef=[1,2,3,4,5]),
                    'service_history6':dict(coef=[1,2,3,4,5,6])})
        values = frozen_predictions(model, '7X', dict(features=[.3,-.1]), .4, 3.75, 7.5)
        self.assertAlmostEqual(values['line3'], .5)
        self.assertAlmostEqual(values['service5'], 1.9)
        self.assertAlmostEqual(values['service_history6'], 2.5)


if __name__ == '__main__':
    unittest.main()
