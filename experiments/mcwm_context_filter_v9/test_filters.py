"""Test bounded evidence, fallback, reset recovery and cross-seed fitting."""

import unittest
from unittest.mock import patch
import numpy as np
import diagnose
from filters import WindowFilter, softmax


MODEL = dict(classes=["calibrated", "joint50", "joint75"], mean=[0.], scale=[1.],
             centers=[[0.], [1.], [2.]], variance=[1.])


class FilterTests(unittest.TestCase):
    def test_repeating_weak_evidence_does_not_inflate_score(self):
        learner = WindowFilter(MODEL, [0., .125, .25], 16, .25)
        learner.observe([.6])
        first = learner.belief.copy()
        for _ in range(100):
            learner.observe([.6])
        np.testing.assert_allclose(learner.belief, first)
        self.assertEqual(len(learner.history), 16)

    def test_reset_discards_previous_episode(self):
        learner = WindowFilter(MODEL, [0., .125, .25], 4, .25)
        for _ in range(4):
            learner.observe([0.])
        learner.reset()
        learner.observe([2.])
        np.testing.assert_allclose(learner.belief, softmax(np.array([-2., -.5, 0.])))
        self.assertEqual(len(learner.history), 1)

    def test_low_margin_falls_back_to_global_coefficient(self):
        learner = WindowFilter(MODEL, [0., .125, .25], 4, .25, .15)
        learner.observe([.5])
        self.assertEqual(learner.decision, (.25, False))
        learner.observe([-10.])
        self.assertTrue(learner.decision[1])
        self.assertLess(learner.coefficient, .01)

    def test_saved_beliefs_recover_actual_reset(self):
        original = diagnose.context.ContextFilter(MODEL, [0., .125, .25], diagnose.P["original_filter"])
        blocks, trace = [], []
        for i, x in enumerate([0., 0., 1., 2., 2.]):
            if i == 3:
                original.reset()
            original.observe([x])
            blocks.append(dict(end_source_row=(i+1)*64, features=[x]))
            trace.append(dict(end_source_row=(i+1)*64, belief=original.belief.tolist()))
        resets, residual = diagnose.recover_resets(blocks, trace, MODEL)
        self.assertEqual(resets, [True, False, False, True, False])
        self.assertEqual(residual, 0.)

    def test_fitting_uses_only_provider_fixed_arms(self):
        calls = []
        def source(seed, scene, arm):
            calls.append((seed, arm))
            i = diagnose.P["contexts"].index(scene)
            return dict(blocks=[dict(features=[float(i)]*8), dict(features=[i+.01]*8)])
        with patch.object(diagnose, "read", side_effect=source):
            model = diagnose.fit(6202, "physical_summary")
        self.assertTrue(all(s == 6202 and a in diagnose.P["fit_arms"] for s, a in calls))
        self.assertEqual(len(model["mean"]), 3)


if __name__ == "__main__":
    unittest.main()
