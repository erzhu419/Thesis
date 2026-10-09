"""Check feature selection and cross-seed isolation without SUMO data."""

import unittest
from unittest.mock import patch
import numpy as np
import diagnose


class DiagnosisTests(unittest.TestCase):
    def test_fit_never_uses_consumer_adaptive_blocks(self):
        calls = []

        def rows(seed, scene, arm):
            calls.append((seed, scene, arm))
            index = diagnose.P["contexts"].index(scene)
            return np.stack([np.full(8,index), np.full(8,index+.01)])

        with patch.object(diagnose,"read_blocks",side_effect=rows):
            r = diagnose.diagnose(6201,"physical_summary")
        self.assertEqual(r["provider_seed"],6202)
        self.assertEqual(r["features"],[0,1,2])
        self.assertFalse(r["sumo_outcomes_used"])
        self.assertTrue(all(a in diagnose.P["fit_arms"] for s,c,a in calls if s==6202))
        self.assertTrue(all(o["balanced_accuracy"]==1. for o in r["outcomes"]))

    def test_single_block_scores_do_not_depend_on_later_blocks(self):
        labeled = {c:np.array([[i]*3,[i+.01]*3]) for i,c in enumerate(diagnose.P["contexts"])}
        model = diagnose.context.fit_recognizer(labeled,diagnose.P["contexts"],.05)
        first = diagnose.score(model,np.zeros((1,3)))
        both = diagnose.score(model,np.array([[0,0,0],[2,2,2]]))
        np.testing.assert_allclose(first[0],both[0])


if __name__ == "__main__":
    unittest.main()
