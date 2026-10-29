"""Fixed-point synthetic checks only: no parameter fitting or real-data reads."""
from copy import deepcopy
import math
import unittest

from experiments.mcwm_cold_start_contract_v128_20261029.model import (
    features, hold_only_features, objective, probability,
)
from experiments.mcwm_cold_start_contract_v128_20261029.timing import causal_events


class ModelContractTests(unittest.TestCase):
    def test_unit_prior_and_sum_likelihood(self):
        samples = [(features(0, 0), 0), (features(2, 30), 1), (features(1, 60), 0)]
        zero = objective([0., 0., 0.], samples)
        self.assertAlmostEqual(zero['loss'], 3 * math.log(2))
        for i in range(3):
            self.assertAlmostEqual(zero['gradient'][i], sum((.5-y)*x[i] for x,y in samples))
            for j in range(3):
                self.assertAlmostEqual(zero['hessian'][i][j],
                    float(i == j) + .25*sum(x[i]*x[j] for x,y in samples))
        beta = [-.7, .2, -.1]
        once, twice = objective(beta, samples), objective(beta, samples * 2)
        self.assertAlmostEqual(twice['loss'], 2*once['loss']-.5*sum(b*b for b in beta))
        for i in range(3):
            self.assertAlmostEqual(twice['gradient'][i], 2*once['gradient'][i]-beta[i])

    def test_single_class_has_finite_root_bracket_without_solving(self):
        samples = [(features(0, 0), 0)] * 3
        left, right = objective([-3., 0., 0.], samples), objective([0., 0., 0.], samples)
        self.assertLess(left['gradient'][0], 0)
        self.assertGreater(right['gradient'][0], 0)
        for beta in ([-3., 0., 0.], [-1., .2, -.4], [0., 0., 0.]):
            value = objective(beta, samples)
            self.assertTrue(math.isfinite(value['loss']))
            self.assertGreaterEqual(value['loss'], .5*sum(b*b for b in beta))
            for vector in ([1., 0., 0.], [1., -2., .5]):
                curvature = sum(vector[i]*value['hessian'][i][j]*vector[j]
                                for i in range(3) for j in range(3))
                self.assertGreaterEqual(curvature, sum(v*v for v in vector))
        self.assertGreater(probability([-3., 0., 0.], features(0, 0)), 0)
        self.assertLess(probability([-3., 0., 0.], features(0, 0)), .5)

    def test_zero_count_is_unidentified_and_matches_hold_only(self):
        samples = [(features(0, h), 0) for h in (0, 15, 60)]
        base = objective([-.8, 0., -.2], samples)
        changed = objective([-.8, 2., -.2], samples)
        self.assertAlmostEqual(changed['loss']-base['loss'], 2.)
        self.assertEqual(changed['gradient'][1], 2.)
        self.assertEqual(changed['hessian'][1], [0., 1., 0.])
        reduced = objective([-.8, -.2], [(hold_only_features(h), 0) for h in (0, 15, 60)])
        self.assertAlmostEqual(base['loss'], reduced['loss'])
        for h in (0, 15, 60):
            self.assertEqual(probability([-.8, 0., -.2], features(0, h)),
                             probability([-.8, -.2], hold_only_features(h)))

    def test_nonzero_feature_derivatives_at_a_fixed_point(self):
        samples = [(features(0, 60), 0), (features(3, 15), 1)]
        beta, step = [-.6, .3, -.2], 1e-5
        value = objective(beta, samples)
        for j in range(3):
            high, low = list(beta), list(beta)
            high[j] += step
            low[j] -= step
            a, b = objective(high, samples), objective(low, samples)
            self.assertAlmostEqual(value['gradient'][j], (a['loss']-b['loss'])/(2*step), places=8)
            for i in range(3):
                self.assertAlmostEqual(value['hessian'][i][j],
                    (a['gradient'][i]-b['gradient'][i])/(2*step), places=8)


def root(sequence, issue, deadline):
    return dict(meta=dict(source_group='synthetic/stable', sequence=sequence, observed_at=issue),
                selection=dict(sequence=sequence, window_start=issue, packet_deadline=deadline),
                target=dict(value=1))


class ChronologyTests(unittest.TestCase):
    def test_strict_release_without_reading_current_or_future_targets(self):
        roots = [root(0, 1000, 1200), root(1, 9100, 9300), root(2, 9300, 9600),
                 root(3, 9301, 9600), root(4, 17910, 18000)]
        before = deepcopy(roots)
        events = causal_events(roots)
        self.assertEqual(roots, before)
        self.assertEqual(events[:3], [dict(kind='initialize', sequences=[0]),
            dict(kind='predict', sequence=1, issue_time=9100),
            dict(kind='predict', sequence=2, issue_time=9300)])
        self.assertEqual(events[3], dict(kind='release', packet_deadline=9300, sequences=[1]))
        self.assertEqual([e['packet_deadline'] for e in events if e['kind']=='release'], [9300, 9600])
        for r in roots:
            r['target'] = object()  # The interface must never inspect targets.
        self.assertEqual(causal_events(roots), events)
        roots[-1]['meta']['source_group'] = 'synthetic/shock'
        with self.assertRaises(ValueError):
            causal_events(roots)

    def test_full_synthetic_schedule_has_only_29_used_suffix_packets(self):
        roots = [root(5*p+i, 1100+300*p+i, 1200+300*p) for p in range(57) for i in range(5)]
        events = causal_events(roots)
        self.assertEqual(len(events[0]['sequences']), 135)
        self.assertEqual(sum(e['kind']=='predict' for e in events), 150)
        releases = [e for e in events if e['kind']=='release']
        self.assertEqual(len(releases), 29)
        self.assertEqual(releases[-1]['packet_deadline'], 17700)
        self.assertEqual(2+2*2*len(releases), 118)  # Future fit cap, no fitting here.


if __name__ == '__main__':
    unittest.main()
