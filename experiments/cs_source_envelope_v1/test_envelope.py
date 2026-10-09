import unittest

from envelope import WAITING, compress


class QueueCompressionTest(unittest.TestCase):
    def setUp(self):
        self.state = [0.25] * 92
        for index in WAITING[1:]:
            self.state[index] = 5.
        self.state[17] = 6.
        self.limits = dict(daily_caps=[3.] + [2.] * 12,
                           phase_caps=[[3.] + [2.] * 12, [6.] + [5.] * 12], bin_seconds=1800)

    def test_identity_and_in_range_are_unchanged(self):
        self.assertEqual(compress(self.state, "identity", 0, self.limits)[0], self.state)
        self.assertEqual(compress(self.state, "phase_envelope", 1800, self.limits)[0], self.state)

    def test_daily_total_preserves_other_features_and_queue_identity(self):
        original = self.state.copy()
        changed, ratio, factor = compress(self.state, "daily_total", 0, self.limits)
        self.assertEqual((ratio, factor), (2., .5))
        self.assertEqual(self.state, original)
        self.assertAlmostEqual(changed[17], sum(changed[i] for i in WAITING[1:]) / 10.)
        for i in range(92):
            self.assertEqual(changed[i], original[i] * .5 if i in WAITING else original[i])

    def test_phase_envelope_respects_all_bounds(self):
        changed, ratio, factor = compress(self.state, "phase_envelope", 1799, self.limits)
        self.assertEqual((ratio, factor), (2.5, .4))
        for index, bound in zip(WAITING, self.limits["phase_caps"][0]):
            self.assertLessEqual(changed[index], bound)
        self.assertAlmostEqual(changed[17], sum(changed[i] for i in WAITING[1:]) / 10.)


if __name__ == "__main__":
    unittest.main()
