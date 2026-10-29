"""Use the actual v105 receipt to cover the observed key-type failure."""
import copy
import json
from pathlib import Path
import unittest

from adjudicate import assess


ORIGINAL = Path(__file__).resolve().parent.parent / "mcwm_full_state_recovery_v105_20261029"


class AdjudicateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads((ORIGINAL / "recovery_summary.json").read_text())
        cls.protocol = json.loads((ORIGINAL / "protocol.json").read_text())

    def test_actual_receipt_corrects_only_the_false_episode_gate(self):
        result = assess(self.summary, self.protocol)
        self.assertTrue(result["all_pass"], result)
        self.assertTrue(result["original_gate_error"]["confirmed"])
        self.assertFalse(result["original_gate_error"]["raw_comparison"])
        self.assertTrue(result["original_gate_error"]["normalized_comparison"])
        self.assertFalse(self.summary["identity"]["original_source_episodes"])

    def test_changed_count_and_missing_zero_key_are_rejected(self):
        for change in ("count", "missing_zero_key"):
            with self.subTest(change=change):
                summary = copy.deepcopy(self.summary)
                counts = summary["training"]["demand_scale_episode_counts"]
                if change == "count":
                    counts["1.25"] = 1
                else:
                    del counts["1.25"]
                result = assess(summary, self.protocol)
                self.assertFalse(result["all_pass"])
                self.assertEqual(result["failures"], ["original_source_episodes"])

    def test_float_and_string_keys_are_equivalent(self):
        summary = copy.deepcopy(self.summary)
        summary["training"]["demand_scale_episode_counts"] = {
            float(key): value for key, value in
            summary["training"]["demand_scale_episode_counts"].items()
        }
        self.assertTrue(assess(summary, self.protocol)["all_pass"])

    def test_other_identity_checks_are_retained_and_facts_rechecked(self):
        summary = copy.deepcopy(self.summary)
        summary["identity"]["historical_seed"] = False
        self.assertFalse(assess(summary, self.protocol)["all_pass"])
        summary = copy.deepcopy(self.summary)
        summary["training"]["source_events"] += 1
        self.assertEqual(assess(summary, self.protocol)["failures"], ["original_source_events"])


if __name__ == "__main__":
    unittest.main()
