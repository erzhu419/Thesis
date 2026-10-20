"""Checks for paired recovery censoring and prequential selection bias."""

import unittest

from analysis import paired_control, prediction_summary


def points(rates):
    value = 0.
    output = [dict(time_seconds=9000., accrued_wait_seconds=0., accrued_in_vehicle_seconds=0.)]
    for index, rate in enumerate(rates):
        value += rate*300.
        output.append(dict(time_seconds=9300.+index*300.,
            accrued_wait_seconds=0., accrued_in_vehicle_seconds=value))
    return output


class AnalysisTests(unittest.TestCase):
    def test_recovery_uses_paired_stable_day_and_three_confirmation_bins(self):
        stable = points([100.]*30)
        shock = points([130., 105., 105., 105.]+[110.]*26)
        result = paired_control(stable, shock)
        self.assertEqual(result["recovery_status"], "recovered")
        self.assertEqual(result["recovery_onset_seconds_after_shock"], 300.)
        self.assertEqual(result["recovery_confirmed_seconds_after_shock"], 1200.)
        self.assertAlmostEqual(result["excess_generalized_passenger_seconds"], 91500.)

    def test_no_initial_degradation_and_right_censoring_are_not_zero_recovery(self):
        stable = points([100.]*30)
        for rates, status in (([100.]*30, "no_detectable_early_degradation"),
                              ([130.]*30, "right_censored")):
            result = paired_control(stable, points(rates))
            self.assertEqual(result["recovery_status"], status)
            self.assertIsNone(result["recovery_confirmed_seconds_after_shock"])
        result = paired_control(stable[:-1], points([100.]*30))
        self.assertEqual(result["recovery_status"], "missing_metrics")
        self.assertIsNone(result["excess_generalized_passenger_seconds"])

    def test_saved_selected_score_never_becomes_after_label_minimum(self):
        row = dict(sequence=1, issued_at=9100., completed_at=9200.,
            available_at=9230., scored_at=9240., selected_score=9.,
            selected_family_at_issue="direct", scores={"direct": 9., "residual": 1.})
        result = prediction_summary([row])
        self.assertEqual(result["early_after"]["mean_score"], 9.)
        del row["selected_score"]
        with self.assertRaises(ValueError):
            prediction_summary([row])

    def test_physical_errors_keep_units_and_missing_observation_counts(self):
        rows = [dict(sequence=index, issued_at=issued, completed_at=issued+10.,
                     available_at=issued+40., scored_at=issued+50., selected_score=1.,
                     **errors)
                for index, (issued, errors) in enumerate((
                    (9100., dict(duration_absolute_error=4., cost_absolute_error=10.)),
                    (9200., dict(duration_absolute_error=8., cost_absolute_error=30.)),
                    (9400., dict(duration_absolute_error=None, cost_absolute_error=50.))))]
        summary = prediction_summary(rows)
        early = summary["early_after"]
        self.assertEqual(early["mean_duration_absolute_error_seconds"], 6.)
        self.assertEqual(early["n_duration_absolute_error_seconds"], 2)
        self.assertEqual(early["mean_cost_absolute_error_generalized_passenger_seconds"], 30.)
        self.assertEqual(early["n_cost_absolute_error_generalized_passenger_seconds"], 3)
        bins = {row["start"]: row for row in summary["by_300_second_issue_bin"]}
        self.assertEqual(bins[9000]["mean_duration_absolute_error_seconds"], 6.)
        self.assertEqual(bins[9000]["mean_cost_absolute_error_generalized_passenger_seconds"], 20.)
        self.assertIsNone(bins[9300]["mean_duration_absolute_error_seconds"])
        self.assertEqual(bins[9300]["n_duration_absolute_error_seconds"], 0)
        self.assertIsNone(summary["before"]["mean_cost_absolute_error_generalized_passenger_seconds"])
        for row in rows:
            row.pop("cost_absolute_error")
        self.assertIsNone(prediction_summary(rows)["early_after"][
            "mean_cost_absolute_error_generalized_passenger_seconds"])


if __name__ == "__main__":
    unittest.main()
