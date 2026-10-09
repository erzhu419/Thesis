import unittest
import numpy as np
from contract import ArrivalFlow, EventCredit, observation


class AlignmentTests(unittest.TestCase):
    def test_same_time_terminal_tail_and_shaping(self):
        credit = EventCredit(100, 10, True)
        state = np.zeros(92, dtype=np.float32)
        credit.begin(state, [0], 10, 20, 3, 2)
        first = credit.finish(state, 10, 20, 3, 2)
        self.assertEqual(first[2], 0)
        credit.begin(state, [1], 10, 20, 3, 2)
        second = credit.finish(state, 40, 80, 4, 2)
        credit.begin(state, [2], 40, 80, 4, 2)
        last = credit.finish(state, 100, 500, 9, 9, terminal=True)
        result = credit.verify(500)
        self.assertEqual(result["raw_reward_sum"], -480)
        self.assertEqual(result["shaped_reward_sum"], 240)
        self.assertEqual(result["zero_time_transitions"], 1)
        self.assertEqual(last[4], 1)
        self.assertAlmostEqual(sum(t[2] for t in [first, second, last]), 24)

    def test_potential_ordering(self):
        returns = []
        for cost in [100, 250]:
            c = EventCredit(100, 1, True)
            c.begin(np.zeros(92), [0], 20, 10, 2, 3)
            c.finish(np.zeros(92), 100, cost, 0, 0, terminal=True)
            c.verify(cost)
            returns.append((c.raw_sum, c.shaped_sum))
        self.assertEqual(returns[0][0] - returns[1][0], returns[0][1] - returns[1][1])

    def test_observation_information_and_committed_commands(self):
        lines = [(100, 30, 3, 20)] * 12
        full = observation(np.ones(15), 30, 60, 50, 100, lines, {2: [30, 60]}, 4)
        local = observation(np.ones(15), 30, 60, 50, 100, lines, {2: [30, 60]}, 4, True)
        self.assertEqual(full.shape, (92,))
        self.assertEqual(full[15], .5)
        self.assertAlmostEqual(float(full[35]), .1)
        self.assertEqual(full[36], .75)
        np.testing.assert_array_equal(full[:15], local[:15])
        self.assertTrue((local[15:] == 0).all())

    def test_baseline_flow_is_shared_and_extra_window_is_local(self):
        normal = ArrivalFlow(78101, 0, [[0, 1]])
        burst = ArrivalFlow(78101, 0, [[0, 1], [3600, 50], [4200, 1]])
        extras = []
        for time in [0, 3600, 4000, 4200]:
            base1, extra1 = normal.draw([1, 2], 20, time)
            base2, extra2 = burst.draw([1, 2], 20, time)
            np.testing.assert_array_equal(base1, base2)
            self.assertEqual(extra1.sum(), 0)
            extras.append(extra2.sum())
        self.assertEqual(extras[0], extras[3])
        self.assertGreater(extras[1], 0)
        self.assertGreater(extras[2], 0)


if __name__ == "__main__":
    unittest.main()
