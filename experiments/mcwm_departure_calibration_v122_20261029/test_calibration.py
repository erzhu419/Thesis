"""Causal feedback-timing fixtures; no native runtime or real records."""
from copy import deepcopy
import unittest

from experiments.mcwm_departure_calibration_v122_20261029.core import forecast_scene


def record(sequence, issue, deadline, hold, departure):
    return dict(selection=dict(sequence=sequence, packet_deadline=deadline, window_start=issue),
        action=dict(applied_at=issue, hold_seconds=hold),
        labels=dict(receiver_departure=dict(value=departure, right_censored=departure is None)))


class CalibrationTests(unittest.TestCase):
    def test_equal_deadline_waits_until_next_root_and_packet_updates_once(self):
        records = [record(1, 9010., 9300, 2., 9016.),
                   record(2, 9020., 9300, 2., 9028.),
                   record(3, 9300., 9600, 2., 9310.),
                   record(4, 9301., 9600, 2., 9313.)]
        predictions, updates, _ = forecast_scene(records, [0.], 0., cutoff=9000)
        by_sequence = {row["sequence"]: row for row in predictions}
        same_clock, next_root = by_sequence[3], by_sequence[4]
        self.assertEqual(same_clock["predictions"]["fresh"], 9302.)
        self.assertEqual(same_clock["calibration_records"], 1)
        self.assertNotEqual(same_clock["latest_feedback_deadline"], 9300)
        self.assertEqual(next_root["predictions"]["fresh"], 9307.)
        self.assertEqual(next_root["predictions"]["frozen"], 9303.)
        self.assertEqual(next_root["calibration_records"], 3)
        self.assertEqual(next_root["latest_feedback_deadline"], 9300)
        released = [row for row in updates if row["deadline"] == 9300]
        self.assertEqual(len(released), 1)
        self.assertEqual(released[0]["mature_added"], 2)
        self.assertEqual(released[0]["censored"], 0)
        self.assertEqual(released[0]["median_residual"], 4.)

    def test_censored_slot_ignores_later_full_day_outcome(self):
        censored = record(1, 9010., 9300, 3., None)
        censored["labels"]["receiver_departure"]["later_observed_value"] = 9150.
        censored["full_day_departure"] = 9150.
        records = [censored, record(2, 9020., 9300, 2., 9028.),
                   record(3, 9301., 9600, 2., 9311.)]
        predictions, updates, _ = forecast_scene(records, [2.], 2., cutoff=9000)
        after_packet = next(row for row in predictions if row["sequence"] == 3)
        self.assertEqual(after_packet["predictions"]["fresh"], 9307.)
        self.assertEqual(after_packet["calibration_records"], 2)
        packet = next(row for row in updates if row["deadline"] == 9300)
        self.assertEqual((packet["mature_added"], packet["censored"]), (1, 1))
        self.assertEqual(packet["median_residual"], 4.)

    def test_scene_histories_are_isolated_and_future_labels_cannot_leak_back(self):
        initial = [1., 3.]
        first_scene = [record(1, 9010., 9300, 2., 9022.),
                       record(2, 9301., 9600, 2., 9315.),
                       record(3, 9601., 9900, 2., 9623.)]
        first_predictions, _, _ = forecast_scene(first_scene, initial, 2., cutoff=9000)
        self.assertEqual(initial, [1., 3.])
        second_scene = [record(1, 9010., 9300, 2., 9012.),
                        record(2, 9301., 9600, 2., 9305.)]
        second_predictions, _, _ = forecast_scene(second_scene, initial, 2., cutoff=9000)
        self.assertEqual(initial, [1., 3.])
        self.assertEqual(second_predictions[0]["predictions"]["fresh"], 9014.)
        self.assertEqual(second_predictions[0]["calibration_records"], 2)
        self.assertEqual(second_predictions[1]["predictions"]["fresh"], 9304.)
        changed = deepcopy(first_scene)
        changed[1]["labels"]["receiver_departure"]["value"] = 9303.
        changed_predictions, _, _ = forecast_scene(changed, initial, 2., cutoff=9000)
        for original, modified in zip(first_predictions[:2], changed_predictions[:2]):
            self.assertEqual(original["predictions"], modified["predictions"])
            self.assertEqual(original["calibration_records"], modified["calibration_records"])
            self.assertEqual(original["latest_feedback_deadline"], modified["latest_feedback_deadline"])
        self.assertNotEqual(first_predictions[2]["predictions"]["fresh"],
                            changed_predictions[2]["predictions"]["fresh"])
        self.assertEqual(initial, [1., 3.])


if __name__ == "__main__":
    unittest.main()
