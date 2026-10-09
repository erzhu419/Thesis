"""Catch budget drift, hardware mixing, and test-driven coefficient selection."""

import json
import unittest
from td_common import HERE, P, FIXED, CONTEXTS, module, smoke_protocol
import collect_results

launch = module("td_upper_test_launcher", HERE / "launch.py")


class ProtocolTests(unittest.TestCase):
    def test_roster_and_hardware_pool(self):
        items = launch.base.placements()
        self.assertEqual(len(items), 36)
        self.assertEqual(len(items)*20, 720)
        self.assertEqual(P["real_td_coefficient"], .75)
        self.assertEqual(set(FIXED.values()), {.125, .25, .375, .5})
        for item in items:
            row = launch.task(item)
            self.assertNotIn("require_node", row)
            self.assertEqual(row["allowed_nodes"], ["jtl110gpu", "jtl110gpu2"])
            self.assertNotIn("wait_for_files", row)
            self.assertIn("wait_for_files", launch.task(item, 10))
        self.assertTrue(all(launch.task(i)["allow_cpu_training"] for i in launch.base.placements(True)))

    def test_new_days_and_unchanged_training_budget(self):
        old = json.loads((HERE.parent / "mcwm_context_v7/protocol.json").read_text())
        old_days = {b["demand_seed"] for k in ["development_blocks", "test_blocks"] for b in old[k]}
        dev = {b["demand_seed"] for b in P["development_blocks"]}
        test = {b["demand_seed"] for b in P["test_blocks"]}
        self.assertTrue(dev.isdisjoint(test) and (dev|test).isdisjoint(old_days))
        self.assertEqual(P["source_event_budget"]-P["warmup_events"], 20000)
        self.assertEqual(P["offline_updates_in_initializer"], 50000)
        self.assertEqual(smoke_protocol(P)["real_td_coefficient"], .75)

    def test_selection_excludes_consumer_and_uses_only_development(self):
        rows = {(scene,arm,seed,c):dict(generalized_seconds_per_departed_passenger=(
            -10 if seed == 6201 and arm == "td0125" else abs(FIXED[arm]-.375)))
            for scene in CONTEXTS for arm in FIXED for seed in P["training_seeds"] for c in range(10)}
        mapping = collect_results.choose(rows)[0]
        self.assertEqual(mapping["consumer_seed"], 6201)
        self.assertEqual(mapping["provider_seeds"], [6202, 6203])
        self.assertEqual(mapping["global_arm"], "td0375")
        self.assertFalse(mapping["test_outcomes_used"])


if __name__ == "__main__":
    unittest.main()
