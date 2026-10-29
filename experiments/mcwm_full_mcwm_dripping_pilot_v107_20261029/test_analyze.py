"""Small fixtures for missing results, the two cost comparators and prefix gates."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import analyze


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def setup(root):
    p = analyze.read(analyze.HERE / "protocol.json")
    experiment = root / analyze.RELATIVE
    put(experiment / "protocol.json", p)
    return p, experiment


def periods(p, cost):
    result = {}
    for name, bounds in p["periods"].items():
        weight = {"full_day": 1, "early": .1, "post_budget": .2}[name]
        total = cost * weight * p["full_day_departed_passengers"]
        result[name] = dict(status="complete", start=bounds[0], end=bounds[1],
            wait=total / 4, ride=total / 2, cost=total,
            departed=p["full_day_departed_passengers"],
            cost_per_passenger=total / p["full_day_departed_passengers"])
    return result


def run(p, scene, arm, cost):
    counts = analyze.expected_counts(456)
    return dict(status="complete", cell=dict(training_seed=6201, external_seed=106206,
        scenario=scene, arm=arm), integrity=dict(all_pass=True), periods=periods(p, cost),
        counts=counts, action_audit=dict(raw_execution_exact=True, canonical_observation_exact=True),
        execution_wall=dict(native_started=100 if arm == "fresh" else 101, native_finished=200),
        post_budget_counts=dict(frozen_at=18002, last_update_clock=18002,
            post18300_observed_at=18304, before=counts, after=counts, post18300=counts),
        budget=dict(native_seconds=24000, native_seconds_is_lower_bound=False, packets=57,
            feedback_slots=285, policy_updates=456, update_attempts=456,
            partial_update_possible=False, source_collection_events=0, model_refits=0,
            offline_training_updates=0, counterfactual_queries=0,
            target_training_labels=285 if arm == "fresh" else 0))


def frozen(root, p, cost):
    for scene in p["scenarios"]:
        put(root / p["frozen_reference_root"] / scene / "summary.json",
            dict(status="complete", actor_seed=6201, external_seed=106206, scenario=scene,
                 integrity=dict(all_pass=True), periods=periods(p, cost)))


class AnalyzeTests(unittest.TestCase):
    def test_missing_days_remain_unknown_and_closed_preflight_can_hold(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            p, experiment = setup(root)
            frozen(root, p, 100)
            put(experiment / "runs/stable/fresh/summary.json", run(p, "stable", "fresh", 99))
            result = analyze.analyze(root)
            self.assertEqual(result["status"], "pending")
            self.assertFalse((experiment / "summary.json").exists())
            self.assertEqual(analyze.preflight(root, experiment)["status"], "pending")
            self.assertFalse((experiment / "preflight_gate.json").exists())
            put(experiment / "preflight_gate.json", dict(status="failed", all_pass=False))
            result = analyze.analyze(root)
            self.assertEqual(result["decision"]["state"], "HOLD_TECHNICAL_INCOMPLETE")
            self.assertEqual(len(result["missing"]), 3)
            self.assertIsNone(result["cells"][0]["comparisons"]["full_day"]["fresh_minus_matched"])
            self.assertIsNone(result["budget"]["new_native_seconds"])
            self.assertEqual(result["budget"]["known_native_seconds"], 24000)
            self.assertIsNone(result["cost_gates"])

    def test_matched_and_frozen_cost_gates_and_decomposition(self):
        # Better than a degraded no_fresh is insufficient; positive matched
        # difference also fails even when both dynamic policies beat frozen.
        for fresh, nofresh, old, state in ((105, 110, 100, "CLOSE_FIXED_PILOT"),
                                          (101, 100, 110, "CLOSE_FIXED_PILOT"),
                                          (90, 100, 95, "PASS_FIXED_PILOT")):
            with self.subTest(fresh=fresh, nofresh=nofresh, old=old), TemporaryDirectory() as folder:
                root = Path(folder)
                p, experiment = setup(root)
                frozen(root, p, old)
                put(experiment / "preflight_gate.json", dict(status="complete", all_pass=True))
                for scene in p["scenarios"]:
                    for arm, cost in (("fresh", fresh), ("no_fresh", nofresh)):
                        put(experiment / "runs" / scene / arm / "summary.json", run(p, scene, arm, cost))
                result = analyze.analyze(root)
                self.assertTrue(result["integrity"]["all_pass"])
                self.assertEqual(result["decision"]["state"], state)
                means = result["means"]["full_day"]
                self.assertEqual(means["fresh_minus_matched"]["cost_per_passenger"], fresh - nofresh)
                self.assertEqual(means["no_fresh_minus_frozen"]["cost_per_passenger"], nofresh - old)
                self.assertEqual(means["fresh_minus_frozen"]["cost_per_passenger"], fresh - old)
                self.assertEqual(result["budget"]["new_native_seconds"], 96000)
                self.assertLessEqual((experiment / "summary.json").stat().st_size, p["summary_max_bytes"])

    def test_closed_prefix_logs_match_exactly_and_prefix_timing_is_not_full_day_timing(self):
        for change_service in (False, True):
            with self.subTest(change_service=change_service), TemporaryDirectory() as folder:
                root = Path(folder)
                p, experiment = setup(root)
                for index, arm in enumerate(p["arms"]):
                    folder = experiment / "runs/stable" / arm
                    put(folder / "preflight.json", dict(status="complete", integrity=dict(all_pass=True),
                        cell=dict(training_seed=6201, external_seed=106206, scenario="stable", arm=arm),
                        initial_updates=70000, initial_online_critic_updates=20000,
                        counts=analyze.expected_counts(8), mix=dict(real=.75, model=.25), model_batch_sizes=[128],
                        feedback=dict(records_received=5, training_outcomes=5 if arm == "fresh" else 0),
                        prefix=dict(trace_rows=1, service_rows=1, service_events=1, clock=1201),
                        execution_wall=dict(native_started=100 + 20 * index, first_packet_finished=110 + 20 * index)))
                    (folder / "prefix_trace.jsonl").write_text('{"clock":1200,"action":0.2}\n')
                    event = dict(clock=1200, events=[dict(vehicle="a", terminal=False,
                        station=2 if change_service and arm == "no_fresh" else 1)])
                    (folder / "prefix_services.jsonl").write_text(json.dumps(event) + "\n")
                result = analyze.preflight(root, experiment)
                self.assertEqual(result["all_pass"], not change_service)
                self.assertEqual(result["parallel"]["status"], "pending_final_native_intervals")
                self.assertFalse(result["parallel"]["prefix_interval_observation"]["all_pass"])
                self.assertEqual(result["prefixes"]["services"]["unequal_rows"], int(change_service))


if __name__ == "__main__":
    unittest.main()
