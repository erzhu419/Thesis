"""Check the new intervention and invariants before remote training."""

import unittest
import torch
from mechanisms import TargetSupport, blend_targets, current_features, trust_backup
from usage_common import P


def states(n, line=0):
    x = torch.zeros(n, 229)
    x[:, 1] = 1/16
    x[:, 5] = 1
    x[:, 6] = line
    x[:, 0] = torch.linspace(0, .5, n)
    return x


class MechanismTests(unittest.TestCase):
    def test_confidence_fixed_points(self):
        log_c = torch.tensor(.5).log()
        for normalized, expected in ((False, .5**10), (True, .5)):
            q = torch.zeros(())
            for _ in range(300):
                q = trust_backup(log_c, q, torch.tensor(0.), .9, normalized)
            self.assertAlmostEqual(float(q.exp()), expected, places=6)

    def test_terminal_and_gamma_zero(self):
        torch.testing.assert_close(trust_backup(torch.tensor(-2.), torch.tensor(-50.),
            torch.tensor(1.), .9, True), torch.tensor(-.2))
        self.assertEqual(trust_backup(-2., -50., 0., 0., True), -2.)

    def test_scalar_blend_preserves_endpoints(self):
        s, t = torch.tensor([-2., -3., -4.]), torch.tensor([-8., -9., -10.])
        torch.testing.assert_close(blend_targets(s, t, torch.tensor([0., .5, 1.])),
                                   torch.tensor([-2., -6., -10.]))
        torch.testing.assert_close(s, torch.tensor([-2., -3., -4.]))

    def test_support_route_and_far_state(self):
        s, a = states(20), torch.zeros(20, 1)
        data = dict(s=s, a=a)
        gate = TargetSupport(data, data, 20, .95)
        self.assertTrue(gate.accepted(s, a).all())
        self.assertFalse(gate.accepted(states(20, 1), a).any())
        far = s.clone()
        far[:, 3] = 100
        self.assertFalse(gate.accepted(far, a).any())

    def test_features_use_current_slot_not_first(self):
        s = states(2)
        s[:, 2] = 2/15
        s[:, 5+2*14+1] = 4
        line, features = current_features(s, torch.zeros(2, 1))
        self.assertTrue((line == 4).all())
        self.assertEqual(features.shape, (2, 15))

    def test_budget_and_no_zero_speed(self):
        self.assertEqual(P["source_event_budget"]-P["warmup_events"], 20000)
        self.assertEqual(P["deployment_hold_scale"], 1)
        self.assertEqual(P["model_fraction"], .25)
        self.assertTrue(all(s["speed_factor"] > 0 for s in P["scenarios"]))


if __name__ == "__main__":
    unittest.main()
