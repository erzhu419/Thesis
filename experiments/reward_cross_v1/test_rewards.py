"""Reward parity, interval accounting and simultaneous/terminal semantics."""

import unittest
import numpy as np
from rewards import ridge_reward, passenger_reward, canonical_headway_reward, PassengerLedger, resolve_objective


class RewardsTest(unittest.TestCase):
    def test_original_defaults(self):
        self.assertEqual(resolve_objective("mc", "original"), "passenger")
        self.assertEqual(resolve_objective("legacy", "original"), "headway")
        self.assertEqual(resolve_objective("cs", "original"), "headway")

    def test_legacy_ridge_parity(self):
        for f, b in [(360, 360), (400, 320), (800, 360), (10, 700)]:
            w = abs(f-360)/(abs(f-360)+abs(b-360)+1e-6)
            expected = -w*abs(f-360)-(1-w)*abs(b-360)-.5*abs(f-b)
            expected -= 20 if abs(f-360) > 180 or abs(b-360) > 180 else 0
            self.assertAlmostEqual(float(ridge_reward(f,b)), expected)
        self.assertEqual(float(ridge_reward(420, 0, backward_present=False)), -60.)

    def test_ratio_targets(self):
        self.assertEqual(float(ridge_reward(360, 720, 360, 720)), 0.)

    def test_passenger_reward_is_time_integral(self):
        ledger = PassengerLedger()
        self.assertEqual(ledger.settle("a"), 0.)
        ledger.advance(3, 5, 10)
        self.assertEqual(ledger.settle("b"), 0.)
        ledger.advance(1, 4, 7)
        self.assertEqual(ledger.settle("a"), -152.)
        self.assertEqual(ledger.settle("b"), -42.)
        self.assertEqual(ledger.wait, 37.)
        self.assertEqual(ledger.ride, 78.)
        self.assertEqual(float(passenger_reward(ledger.wait, ledger.ride)), -152.)

    def test_canonical_batched_online_parity_and_terminal(self):
        states = np.zeros((3, 229), np.float32)
        slots = states[:, 5:].reshape(3, 16, 14)
        slots[:, 0, 0] = 1.
        slots[:, 0, 3:5] = [1.5, .5]
        slots[:, 0, 5:7] = .5
        slots[:, 0, 11:13] = 1.
        rewards = canonical_headway_reward(states, [0, 0, 1])
        self.assertEqual(float(rewards[0]), canonical_headway_reward(states[0]))
        self.assertEqual(float(rewards[1]), float(rewards[0]))
        self.assertEqual(float(rewards[2]), 0.)
        self.assertLess(float(rewards[0]), 0.)
        # The next event can belong to slot one in a simultaneous batch.
        states[0, 2] = 1/15
        slots[0, 1] = slots[0, 0]
        slots[0, 0, 3] = 4.
        self.assertEqual(canonical_headway_reward(states[0]), float(rewards[1]))


if __name__ == "__main__":
    unittest.main()
