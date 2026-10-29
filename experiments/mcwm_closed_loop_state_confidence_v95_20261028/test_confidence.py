from math import inf
import unittest

import numpy as np

from experiments.mcwm_closed_loop_state_confidence_v95_20261028.confidence import (
    calibrated_margin, fit_bias, predict, seed_score, select_first, upper_bound)


def point(index, q, m=-1., delta=None, **extra):
    return dict(active=True, m=m, q=q, delta=delta,
                issue_clock=12000.+index, sequence=index, **extra)


class ConfidenceTest(unittest.TestCase):
    def setUp(self):
        self.model = fit_bias([
            dict(q=0., prediction=-1., delta=-2.),
            dict(q=1., prediction=-1., delta=2.)], [0., 1.])

    def test_fixed_scaling_and_both_penalized_normal_equations(self):
        np.testing.assert_allclose(self.model['coef'], [2./3., 4./3.])
        np.testing.assert_allclose(self.model['gram_inverse'], np.eye(2)/3.)
        self.assertEqual(self.model['linear_solves'], 1)
        self.assertEqual(self.model['rank'], 2)
        self.assertAlmostEqual(predict(self.model, .5, 0.)['bias'], 2./3.)
        self.assertAlmostEqual(predict(self.model, 0., 0.)['bias'], -2./3.)

    def test_single_q_stops_instead_of_silently_changing_the_model(self):
        with self.assertRaises(ValueError):
            fit_bias([dict(q=.5, prediction=-1., delta=1.)]*3, [0., 1.])

    def test_support_scale_increases_outside_training_geometry(self):
        center = predict(self.model, .5, -1.)['support_scale']
        edge = predict(self.model, 0., -1.)['support_scale']
        outside = predict(self.model, 2., -1.)['support_scale']
        self.assertLess(center, edge)
        self.assertLess(edge, outside)
        # Native support is checked separately; this is a mathematical scale test.

    def test_state_dependent_bound_keeps_one_proposal_and_rejects_another(self):
        low, high = point(0, 0.), point(1, 1.)
        self.assertLess(upper_bound(-1., 0., self.model, .25), 0.)
        self.assertGreater(upper_bound(-1., 1., self.model, .25), 0.)
        self.assertIs(select_first([high, low], self.model, .25), low)

    def test_later_negative_missing_outcome_keeps_entire_seed_at_infinity(self):
        rows = [point(0, 0., delta=-1.), point(1, 1.)]
        self.assertEqual(seed_score(rows, self.model), inf)
        rows[1]['delta'] = 4.
        expected = (4.-predict(self.model, 1., -1.)['center'])/predict(self.model, 1., -1.)['support_scale']
        self.assertAlmostEqual(seed_score(rows, self.model), expected)

    def test_inactive_and_nonnegative_predictions_need_no_label(self):
        self.assertEqual(seed_score([dict(active=False), point(0, 0., m=.1)], self.model), 0.)
        lowered = dict(self.model, coef=[-2., 0.])
        self.assertIsNone(select_first([point(0, 0., m=.1)], lowered, 0.))

    def test_selection_reads_no_outcome_and_obeys_issue_order(self):
        early = point(0, 0., delta=object())
        late = point(1, 0., delta=object())
        self.assertIs(select_first([late, early], self.model, 0.), early)

    def test_zero_boundary_is_rejected_and_all_reference_is_only_no_action(self):
        neutral = dict(self.model, coef=[0., 0.], gram_inverse=[[0., 0.], [0., 0.]])
        self.assertEqual(upper_bound(-.5, 0., neutral, .5), 0.)
        self.assertIsNone(select_first([point(0, 0., m=-.5)], neutral, .5))
        self.assertIsNone(select_first([point(0, 0.)], self.model, inf))

    def test_four_seed_finite_sample_calibration_retains_missing_seed(self):
        self.assertEqual(calibrated_margin([0., .2, .1, .3], .2), .3)
        self.assertEqual(calibrated_margin([0., .2, .1, inf], .2), inf)


if __name__ == '__main__':
    unittest.main()
