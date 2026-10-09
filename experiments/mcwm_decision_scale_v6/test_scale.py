"""Check fixed error units and unpinned campaign coverage."""

import math
import unittest

import torch
from balanced_model import rms_scale
from launch import base


class ScaleTests(unittest.TestCase):
    def test_rms_matches_both_candidate_error_distributions(self):
        errors = [torch.tensor([.5, 1.5]), torch.tensor([1., 2.])]
        self.assertAlmostEqual(float(rms_scale(errors, .001)), math.sqrt(1.875), places=6)

    def test_decision_component_is_in_its_own_units(self):
        errors = [torch.full((4,), .5), torch.full((4,), .5)]
        scale = rms_scale(errors, .001)
        self.assertEqual(float((errors[0]/scale).mean()), 1.)
        self.assertLess(float((errors[0]/60.).mean()), .01)

    def test_zero_errors_use_frozen_floor(self):
        self.assertAlmostEqual(float(rms_scale([torch.zeros(4)], .001)), .001, places=6)

    def test_campaign_coverage_and_no_fixed_nodes(self):
        items = base.placements()
        self.assertEqual(len(items), 18)
        for item in items:
            train = base.task(item)
            self.assertNotIn("require_node", train)
            self.assertNotIn("jtl311linux", train["allowed_nodes"])
            self.assertIn("mcwm_decision_scale_v6/node_python", train["cmd"])
            for case in range(10):
                self.assertNotIn("require_node", base.task(item, case))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
