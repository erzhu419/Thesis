import json
from collections import Counter
from pathlib import Path
import unittest
from regimes import source_schedule, training_regime

P = json.loads((Path(__file__).parent / "protocol.json").read_text())


class SourceCoverageTest(unittest.TestCase):
    def test_budget_and_held_out_intensities(self):
        cycle = P["source_training_cycle"]
        regimes = [training_regime("global_coverage", i, cycle) for i in range(100)]
        self.assertEqual(Counter(regimes), dict(normal=40, static2=20, static3=20, pulse3=10, repeated3=10))
        strengths = {value for regime in regimes for _, value in source_schedule(regime, 18000)}
        self.assertEqual(strengths, {1, 2, 3})
        self.assertTrue(strengths.isdisjoint({5, 10, 50}))
        self.assertEqual({training_regime("global_terminal", i, cycle) for i in range(100)}, {"normal"})

    def test_schedule_and_smoke_switches(self):
        for horizon in [3600, 18000]:
            for regime in set(P["source_training_cycle"]):
                schedule = source_schedule(regime, horizon)
                times = [time for time, _ in schedule]
                self.assertEqual(times[0], 0)
                self.assertEqual(times, sorted(set(times)))
                self.assertLess(times[-1], horizon)
        self.assertEqual(source_schedule("repeated3", 18000),
            [[0, 1], [3600, 3], [5400, 1], [9000, 3], [10800, 1], [14400, 3], [16200, 1]])
        self.assertEqual([training_regime("global_coverage", i, P["source_training_cycle"], True) for i in range(2)],
                         ["pulse3", "repeated3"])


if __name__ == "__main__":
    unittest.main()
