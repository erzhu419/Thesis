import unittest

from phase_total import WAITING, compress_total


class PhaseTotalTest(unittest.TestCase):
    def setUp(self):
        self.limits = dict(bin_seconds=1800, phase_caps=[[2.] + [.001] * 12, [4.] + [.001] * 12])
        self.state = [0.25] * 92
        self.state[17] = 3.
        for index in WAITING[1:]:
            self.state[index] = 2.5

    def test_time_boundary_and_unchanged_range(self):
        changed, ratio, factor = compress_total(self.state, 1800, self.limits)
        self.assertEqual(changed, self.state)
        self.assertEqual((ratio, factor), (.75, 1.))
        self.assertEqual(compress_total(self.state, 1799, self.limits)[2], 2. / 3.)
        self.assertEqual(compress_total(self.state, 18000, self.limits)[2], 1.)

    def test_coherence_and_no_other_mutation(self):
        before = self.state.copy()
        changed, _, factor = compress_total(self.state, 0, self.limits)
        self.assertEqual(self.state, before)
        self.assertAlmostEqual(changed[17] * 10., sum(changed[i] for i in WAITING[1:]))
        for i in range(92):
            self.assertEqual(changed[i], before[i] * factor if i in WAITING else before[i])

    def test_per_line_caps_cannot_compress_total(self):
        changed, ratio, factor = compress_total(self.state, 0, self.limits)
        self.assertEqual(ratio, 1.5)
        self.assertEqual(factor, 2. / 3.)
        self.assertEqual(changed[17], 2.)


if __name__ == "__main__":
    unittest.main()
