import math
import unittest

from representation import WAITING, transform


class RepresentationTest(unittest.TestCase):
    def state(self, total):
        values = [.25] * 92
        values[17] = total
        for index in WAITING[1:]:
            values[index] = 10. * total / 12.
        return values

    def test_source_scale_and_extrapolation(self):
        for arm in ["source_clip", "source_log"]:
            self.assertEqual(transform(self.state(2.), arm, 2.)[0][17], 1.)
        self.assertEqual(transform(self.state(6.), "source_clip", 2.)[0][17], 1.)
        self.assertAlmostEqual(transform(self.state(6.), "source_log", 2.)[0][17], 2.)
        self.assertLess(transform(self.state(4.), "source_log", 2.)[0][17], 2.)

    def test_coherent_components_and_no_other_changes(self):
        original = self.state(6.)
        before = original.copy()
        for arm in ["identity", "source_clip", "source_log"]:
            changed, factor = transform(original, arm, 2.)
            self.assertEqual(original, before)
            self.assertAlmostEqual(changed[17] * 10., sum(changed[i] for i in WAITING[1:]))
            for i in range(92):
                self.assertEqual(changed[i], original[i] * factor if i in WAITING else original[i])
        self.assertEqual(transform(original, "identity", 2.)[0], original)

    def test_zero_queue_and_finite_small_queue(self):
        for arm in ["source_clip", "source_log"]:
            for total in [0., 1e-8]:
                changed, factor = transform(self.state(total), arm, 2.)
                self.assertTrue(math.isfinite(factor))
                self.assertTrue(all(math.isfinite(v) for v in changed))
                self.assertGreaterEqual(changed[17], 0.)


if __name__ == "__main__":
    unittest.main()
