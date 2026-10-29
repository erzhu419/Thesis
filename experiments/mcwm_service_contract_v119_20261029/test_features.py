"""Pure stdlib contract fixtures, with no native or training dependencies."""
from copy import deepcopy
import unittest

from experiments.mcwm_service_contract_v119_20261029.features import build_context, bind_action


def capture():
    return dict(summary=dict(issue=dict(trip_id="R", station_id="S", sequence=42,
        issue_clock=100., observed_clock=100., event_clock=99.),
        checks=dict(native_event_before_apply=True), receiver=dict(service_ready_time=98.5),
        diagnostic_person_id="p", end_state={"unknown": True}), vehicles=[
            dict(receiver=True, trip_id="R", capacity=50., free_seats=37., counts=dict(onboard=13),
                location=dict(lane_id="lane", lane_position=25.5), people=[]),
            dict(receiver=False, trip_id="F", counts=dict(onboard=2),
                planned_stops=["S", "D"], location=dict(lane_id="lane", lane_position=10.5, speed=.4, at_stop=False),
                people=[dict(person_id="p", current_stage0=dict(known=True, type=3, line="F", destStop="S"),
                             additional_intent=dict(final_stop="D", first_leg_transfer_at_receiver=True)),
                        dict(person_id="q", current_stage0=dict(known=True, type=3, line="F", destStop="D"),
                             additional_intent=dict(final_stop="S"))]),
            dict(receiver=False, trip_id="unrelated", counts=dict(onboard=99), planned_stops=["D"], people=[])])


class FeatureTests(unittest.TestCase):
    def test_intent_future_and_identifiers_do_not_enter_x(self):
        before = capture()
        expected = build_context(before, source_group="root-group")
        changed = deepcopy(before)
        changed["summary"]["diagnostic_person_id"] = "another"
        changed["summary"]["end_state"] = {"completion_time": 1234.}
        changed["future_events"] = [{"boarding_time": 101.}]
        changed["vehicles"][1]["people"][0]["additional_intent"] = {"final_stop": "opposite", "next_leg": "R"}
        changed["vehicles"][1]["people"][1]["additional_intent"] = {}
        self.assertEqual(expected, build_context(changed, source_group="root-group"))
        changed["vehicles"][1]["trip_id"] = "other-feeder-id"
        changed["vehicles"][1]["people"][0]["person_id"] = "other-person-id"
        actual = build_context(changed, source_group="other-source")
        self.assertEqual(expected["feeders"][0]["x"], actual["feeders"][0]["x"])
        self.assertEqual(expected["feeders"][0]["cohort_ids"], ["p"])
        self.assertFalse(expected["feeders"][0]["at_target_stop_at_issue"])
        self.assertEqual(expected["feeders"][0]["x"]["current_leg_dest_count"], 1)
        self.assertEqual(expected["feeders"][0]["x"]["gap_m"], 15.)
        self.assertEqual(expected["receiver"]["service_ready_age_seconds"], 1.5)
        self.assertEqual(len(expected["feeders"]), 1)

    def test_different_lane_and_unknown_measurements_remain_unknown(self):
        value = capture()
        value["vehicles"][1]["location"]["lane_id"] = "different"
        value["vehicles"][1]["location"]["speed"] = None
        value["vehicles"][0]["free_seats"] = None
        value["summary"]["receiver"]["service_ready_time"] = None
        value["vehicles"][1]["people"][1]["current_stage0"] = {"known": False}
        x = build_context(value, source_group="g")["feeders"][0]["x"]
        self.assertFalse(x["same_lane"])
        for key in ("gap_m", "feeder_speed", "receiver_free_seats", "receiver_service_ready_age_seconds", "current_leg_dest_count"):
            self.assertIsNone(x[key])

    def test_only_the_actual_before_action_clock_is_eligible(self):
        for offset in (-1., 1.):
            value = capture()
            value["summary"]["issue"]["observed_clock"] += offset
            with self.assertRaises(ValueError):
                build_context(value, source_group="g")
        value = capture()
        value["summary"]["checks"]["native_event_before_apply"] = False
        with self.assertRaises(ValueError):
            build_context(value, source_group="g")

    def test_valid_zero_hold_and_native_command_evidence(self):
        context = build_context(capture(), source_group="g")
        for name, stop_field in (("hold_seconds", "station_id"), ("requested_additional_hold_seconds", "stop_id")):
            result = bind_action(context, dict(accepted=True, applied_at=100., trip_id="R", **{name: 0., stop_field: "S"}))
            self.assertEqual(result["action"], dict(hold_seconds=0., applied_at=100.))
            self.assertNotIn("action", context)
            result["feeders"][0]["x"]["gap_m"] = -999.
            self.assertEqual(context["feeders"][0]["x"]["gap_m"], 15.)
        for command in (dict(accepted=False, applied_at=100., hold_seconds=0.),
                        dict(accepted=True, applied_at=101., hold_seconds=0.),
                        dict(accepted=True, applied_at=100., hold_seconds=0., trip_id="another-vehicle"),
                        dict(accepted=True, applied_at=100., hold_seconds=0., stop_id="another-stop")):
            command = dict(dict(trip_id="R", stop_id="S"), **command)
            with self.assertRaises(ValueError):
                bind_action(context, command)


if __name__ == "__main__":
    unittest.main()
