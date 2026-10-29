import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("v116_replay", Path(__file__).with_name("replay.py"))
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def action(sequence, clock):
    return dict(agent="7X_1", vehicle_id="fleet1", line_id="7X", station_id="stop1",
        trip_id="7X_1", event=[clock-1, sequence], terminal=False, s=[clock/10, sequence],
        raw_s=[clock, sequence], a=[-.5], proposed_a=[-.75], started_at=clock,
        sequence=sequence, episode_id=0, domain="target", visit_index=2)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, rows):
        path = self.root/name
        path.write_text("".join(json.dumps(row)+"\n" for row in rows))
        return path

    def reader(self, *, actions=None, ticks=None):
        actions = [action(0, 2), action(1, 2), action(2, 3)] if actions is None else actions
        ticks = [dict(start=0, end=1, waiting=2, in_vehicle=1),
                 dict(start=1, end=2, waiting=3, in_vehicle=4),
                 dict(start=2, end=3, waiting=7, in_vehicle=8)] if ticks is None else ticks
        result = replay.ReplayReference(
            [self.write("parent_actions", actions[:1]), self.write("suffix_actions", actions[1:])],
            [self.write("parent_ticks", ticks[:2]), self.write("suffix_ticks", ticks[2:])], 2, 3)
        self.addCleanup(result.close)
        return result

    def check(self, reference, row, **overrides):
        expected = dict(time_seconds=row["started_at"], waiting_passengers=3,
                        in_vehicle_passengers=4, accrued_wait_seconds=5., accrued_in_vehicle_seconds=5.)
        if row["started_at"] == 3:
            expected.update(waiting_passengers=7, in_vehicle_passengers=8,
                            accrued_wait_seconds=12., accrued_in_vehicle_seconds=13.)
        expected.update(overrides)
        self.assertEqual(reference.next_action(row["s"]), row["proposed_a"])
        return reference.check_event({k:v for k,v in row.items() if k != "visit_index"}, expected)

    def test_prefix_suffix_same_clock_and_exact_target_boundary(self):
        reference = self.reader()
        self.assertFalse(self.check(reference, action(0, 2))["target_reached"])
        self.assertEqual(reference.report()["ticks_integrated"], 2)
        self.assertFalse(self.check(reference, action(1, 2))["target_reached"])
        self.assertEqual(reference.report()["ticks_integrated"], 2)
        self.assertTrue(self.check(reference, action(2, 3))["target_reached"])
        self.assertTrue(reference.report()["all_pass"])
        self.assertEqual(reference.report()["actions_checked"], 3)

    def test_next_tick_counts_and_wrong_accumulated_cost_are_rejected(self):
        for wrong in (dict(waiting_passengers=7, in_vehicle_passengers=8),
                      dict(accrued_wait_seconds=6.)):
            with self.subTest(wrong=wrong):
                reference = self.reader()
                with self.assertRaises(replay.ReplayMismatch):
                    self.check(reference, action(0, 2), **wrong)
                self.assertFalse(reference.report()["all_pass"])

    def test_missing_action_or_tick_cannot_complete(self):
        reference = self.reader(actions=[action(0, 2)])
        self.check(reference, action(0, 2))
        with self.assertRaisesRegex(replay.ReplayMismatch, "actions ended"):
            reference.next_action(action(1, 2)["s"])
        for ticks in ([dict(start=0,end=1,waiting=2,in_vehicle=1)],
                      [dict(start=0,end=1,waiting=2,in_vehicle=1),
                       dict(start=2,end=3,waiting=7,in_vehicle=8)]):
            reference = self.reader(ticks=ticks)
            with self.assertRaises(replay.ReplayMismatch):
                self.check(reference, action(0, 2))

    def test_actual_event_identity_and_raw_state_must_match(self):
        for field, value in (("trip_id", "7X_2"), ("raw_s", [999, 0])):
            reference = self.reader()
            row = action(0, 2)
            reference.next_action(row["s"])
            row[field] = value
            with self.assertRaises(replay.ReplayMismatch):
                reference.check_event(row, {})
            self.assertIn(field, reference.report()["failure"]["fields"])


if __name__ == "__main__":
    unittest.main()
