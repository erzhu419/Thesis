from math import inf
import unittest

from experiments.mcwm_closed_loop_calibration_design_v94_20261028.calibration import (
    calibrated_margin, seed_score, select_first)


def row(sequence, prediction, delta=None, **extra):
    return dict(active=True, predictions={'history6':prediction}, delta=delta,
                issue_clock=12000.+sequence, sequence=sequence, **extra)


class CalibrationTest(unittest.TestCase):
    def test_finite_sample_four_at_eighty_and_nine_at_ninety_percent(self):
        self.assertEqual(calibrated_margin([1.,2.,3.,4.],.2),4.)
        self.assertEqual(calibrated_margin(list(range(1,10)),.1),9.)
        self.assertEqual(calibrated_margin([1.,2.,3.,4.],.1),inf)
        self.assertEqual(calibrated_margin([], .2),inf)

    def test_correlated_opportunities_are_one_seed_maximum(self):
        rows = [row(index,-1.,float(index)) for index in range(6)]
        score = seed_score(rows,'history6')
        self.assertEqual(score,6.)
        self.assertEqual(calibrated_margin([score],.2),inf)

    def test_all_later_negative_points_must_have_outcomes(self):
        rows = [row(0,-.5,-.1),row(1,-2.)]
        self.assertEqual(seed_score(rows,'history6'),inf)
        rows[1]['delta'] = 3.
        self.assertEqual(seed_score(rows,'history6'),5.)
        self.assertIs(select_first(rows,'history6',1.),rows[1])

    def test_inactive_and_nonnegative_points_do_not_require_a_label(self):
        rows = [dict(active=False),row(0,.2),row(1,0.)]
        self.assertEqual(seed_score(rows,'history6'),0.)
        self.assertIsNone(select_first(rows,'history6',0.))

    def test_outcome_sign_and_physical_noop_do_not_filter_scores(self):
        rows = [row(0,-1.,2.,physical_separation=False),row(1,-5.,-1.)]
        self.assertEqual(seed_score(rows,'history6'),4.)

    def test_missing_label_keeps_seed_and_forces_infinite_margin(self):
        scores = [0.,1.,2.,seed_score([row(0,-1.)],'history6')]
        self.assertEqual(len(scores),4)
        margin = calibrated_margin(scores,.2)
        self.assertEqual(margin,inf)
        self.assertIsNone(select_first([row(0,-1.)],'history6',margin))

    def test_negative_bias_clip_cannot_turn_a_positive_point_into_action(self):
        self.assertEqual(seed_score([row(0,-1.,-3.)],'history6'),0.)
        margin = calibrated_margin([-1.,-2.,-.5,-.7],.2)
        self.assertEqual(margin,0.)
        self.assertIsNone(select_first([row(0,.1)],'history6',margin))

    def test_strict_zero_boundary_and_predeclared_issue_order(self):
        first,second = row(1,-1.),row(2,-1.1)
        self.assertIs(select_first([second,first],'history6',1.),second)
        first['predictions']['history6'] = -1.2
        self.assertIs(select_first([second,first],'history6',1.),first)

    def test_models_have_their_own_scores_and_no_truth_driven_selection(self):
        point = row(0,-1.,2.)
        point['predictions']['line3'] = -2.
        self.assertEqual(seed_score([point],'history6'),3.)
        self.assertEqual(seed_score([point],'line3'),4.)
        point['delta'] = object()
        self.assertIs(select_first([point],'history6',0.),point)

    def test_global_band_can_reject_everything_without_showing_improvement(self):
        proposals = [row(0,-.8),row(1,-1.)]
        self.assertIsNone(select_first(proposals,'history6',3.))
        # None describes no intervention. This API makes no performance claim.
        self.assertIsNone(select_first(proposals,'history6',inf))


if __name__ == '__main__':
    unittest.main()
