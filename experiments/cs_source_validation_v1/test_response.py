import unittest

from response import Response, describe, merge


class ResponseTests(unittest.TestCase):
    def test_same_state_direction_and_buckets(self):
        r = Response()
        r.record(0, .1, 0., 2., 5., 5.)
        r.record(1800, .8, .7, 10., 2., 2.)
        d = describe(r.summary())
        self.assertEqual(d["total"]["new_minus_old_seconds"], -2.5)
        self.assertEqual(d["total"]["new_longer_fraction"], .5)
        self.assertEqual(d["total"]["old_longer_fraction"], .5)
        self.assertEqual(set(d["bins"]["time"]), {"0", "1"})
        self.assertEqual(set(d["bins"]["occupancy"]), {"empty", "above_half"})
        self.assertEqual(set(d["bins"]["queue_ratio"]), {"0.1", "1.0"})

    def test_merge_weights_decisions_not_case_means(self):
        a, b = Response(), Response()
        a.record(0, .2, .4, 1., 3., 3.)
        for _ in range(3):
            b.record(0, .2, .4, 1., 7., 7.)
        d = describe(merge([a.summary(), b.summary()]))
        self.assertEqual(d["total"]["decisions"], 4)
        self.assertEqual(d["total"]["new_holding_seconds"], 6.)
        self.assertEqual(d["bins"]["occupancy"]["up_to_half"]["new_holding_seconds"], 6.)


if __name__ == "__main__":
    unittest.main()
