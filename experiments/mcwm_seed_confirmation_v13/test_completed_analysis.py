"""Check paired units and missing rule-only measurements in the compact analysis."""

import unittest

from analyze_completed import METRICS, aggregate_pairs, average, paired, stats


class CompletedAnalysisTests(unittest.TestCase):
    def test_seed_level_interval(self):
        result = stats(list(range(20)))
        self.assertEqual(result["n"], 20)
        self.assertEqual(result["mean"], 9.5)
        self.assertLess(result["ci95"][0], 9.5)
        self.assertGreater(result["ci95"][1], 9.5)

    def test_paired_cost_decomposition(self):
        reference = dict(cost_seconds=30, wait_seconds=10, in_vehicle_seconds=10,
            completion_percent=99, inference_ms=1)
        candidate = dict(cost_seconds=25, wait_seconds=8, in_vehicle_seconds=9,
            completion_percent=99.1, inference_ms=1)
        delta = paired(candidate, reference)
        self.assertEqual(delta["cost_seconds"], 2*delta["wait_seconds"]+delta["in_vehicle_seconds"])
        summary = aggregate_pairs([dict(seed=1, **delta), dict(seed=2, **delta)])
        self.assertEqual(summary["better_seeds"], 2)

    def test_missing_rule_inference_is_not_zero(self):
        row = dict.fromkeys(METRICS)
        row["cost_seconds"] = 25
        result = average([row])
        self.assertNotIn("inference_ms", result)
        self.assertEqual(result["cost_seconds"], 25)


if __name__ == "__main__":
    unittest.main()
