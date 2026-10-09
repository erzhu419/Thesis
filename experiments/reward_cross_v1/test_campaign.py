"""Catch omitted objective/seed pairs and an accidental budget increase."""

from collections import Counter
import unittest
from collect_results import placements
from cross_common import P


class CampaignTest(unittest.TestCase):
    def test_frozen_size_and_paired_objectives(self):
        rows = list(placements())
        self.assertEqual(Counter(r["kind"] for r in rows), dict(legacy=20, cs=20, mc=90))
        groups = {}
        for row in rows:
            key = (row["kind"], row["arm"], row["seed"], row["scene"])
            groups.setdefault(key, set()).add(row["objective"])
        self.assertTrue(all(values == {"headway", "passenger"} for values in groups.values()))
        self.assertEqual(P["episodes"], 100)
        self.assertNotIn("jtl311linux", P["training_nodes"])

    def test_smoke_covers_every_method_and_objective(self):
        rows = list(placements(True))
        self.assertEqual(len(rows), 14)
        self.assertEqual({(r["kind"], r["arm"], r["objective"]) for r in rows},
                         {(kind, arm, objective) for kind in ("legacy", "cs", "mc")
                          for arm in P[kind+"_arms"] for objective in P["objectives"]})

    def test_evaluation_does_not_request_a_gpu_or_pin_a_node(self):
        from launch import task
        item = next(i for i in placements(True) if i["kind"] == "legacy" and i["arm"] == "resac")
        row = task(item, case=0)
        self.assertEqual(row["vram"], 0)
        self.assertTrue(row["allow_cpu_training"])
        self.assertNotIn("require_node", row)
        self.assertEqual(len(row["wait_for_files"]), 1)


if __name__ == "__main__":
    unittest.main()
