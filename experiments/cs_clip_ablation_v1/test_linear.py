import importlib.util
from pathlib import Path
import unittest

from linear import WAITING, normalize_linear

spec = importlib.util.spec_from_file_location("frozen_representation", Path(__file__).resolve().parent.parent / "cs_input_training_v1/representation.py")
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)


class LinearTest(unittest.TestCase):
    def state(self, total):
        values = [.25] * 92
        values[17] = total
        for index in WAITING[1:]:
            values[index] = total * 10. / 12.
        return values

    def test_matches_hard_clip_inside_source_range(self):
        for total in [0., .5, 2.]:
            self.assertEqual(normalize_linear(self.state(total), 2.), frozen.transform(self.state(total), "source_clip", 2.))

    def test_removes_only_the_upper_bound(self):
        state = self.state(6.)
        changed, _ = normalize_linear(state, 2.)
        self.assertEqual(changed[17], 3.)
        self.assertEqual(frozen.transform(state, "source_clip", 2.)[0][17], 1.)
        self.assertAlmostEqual(changed[17] * 10., sum(changed[i] for i in WAITING[1:]))

    def test_preserves_input_and_other_features(self):
        state = self.state(6.)
        before = state.copy()
        changed, factor = normalize_linear(state, 2.)
        self.assertEqual(state, before)
        for index in range(92):
            self.assertEqual(changed[index], before[index] * factor if index in WAITING else before[index])


if __name__ == "__main__":
    unittest.main()
