"""Final-endpoint readout must not depend on recoverable intermediate weights."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import collect


class ContinuousSummaryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def endpoint(self, endpoint):
        for arm in ("full", "masked"):
            for seed in (42, 123, 456, 789, 1001, 1002, 1003, 1004, 1005, 1006):
                path = self.root / "runs300" / f"{arm}_seed{seed}" / f"results_ep{endpoint}_extended"
                path.mkdir(parents=True)
                done = dict(arm=arm, training_seed=seed, training_episodes=endpoint+1,
                            checkpoint_episode=endpoint, run_training_budget=300,
                            test_plan="extended", evaluation_episodes=50,
                            evaluation_feedback_nonzero=0)
                (path / "complete.json").write_text(json.dumps(done))
                metrics = dict.fromkeys(("composite", "wait_min", "total_min", "onboard_min",
                                         "vehicle_hours", "full_day_holding_vehicle_hours",
                                         "service_km", "peak_fleet", "headway_cv"),
                                        1 if arm == "full" else 2)
                rows = [dict(metrics, episode=i, completed_trips=262,
                             environment_seed=seed*1000+i, fleet_budget=10) for i in range(50)]
                (path / "episodes.jsonl").write_text("\n".join(map(json.dumps, rows)))

    def summarize(self):
        with patch.object(collect, "HERE", self.root), contextlib.redirect_stdout(io.StringIO()):
            collect.compare_continuous_budgets()
        return json.loads((self.root / "continuous300_summary.json").read_text())

    def test_final_without_intermediate_is_reported(self):
        self.endpoint(299)
        result = self.summarize()
        self.assertTrue(result["primary_complete"])
        self.assertEqual(result["full_minus_masked"]["299"]["composite"]["mean"], -1)
        self.assertNotIn("99", result["full_minus_masked"])
        self.assertFalse(result["budget_comparison_complete"])
        self.assertEqual(result["budget300_minus100"], {})

    def test_both_endpoints_enable_budget_comparison(self):
        self.endpoint(99)
        self.endpoint(299)
        result = self.summarize()
        self.assertTrue(result["budget_comparison_complete"])
        self.assertEqual(result["budget300_minus100"]["full"]["composite"]["mean"], 0)

    def test_partial_final_is_not_reported_as_complete(self):
        self.endpoint(299)
        (self.root / "runs300/full_seed42/results_ep299_extended/complete.json").unlink()
        result = self.summarize()
        self.assertFalse(result["primary_complete"])
        self.assertNotIn("299", result["full_minus_masked"])

    def test_schedule_mismatch_still_fails(self):
        self.endpoint(299)
        path = self.root / "runs300/full_seed42/results_ep299_extended/episodes.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["fleet_budget"] = 11
        path.write_text("\n".join(map(json.dumps, rows)))
        with self.assertRaises(AssertionError):
            self.summarize()


if __name__ == "__main__":
    unittest.main()
