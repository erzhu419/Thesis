"""Small synthetic solver/integration tests, separately counted from the study."""
from copy import deepcopy
import math
import unittest

from experiments.mcwm_cold_start_contract_v128_20261029.model import features
from experiments.mcwm_cold_start_factual_v129_20261029.solver import fit
from experiments.mcwm_cold_start_factual_v129_20261029.forecast import initialize, forecast

FIT_RECEIPTS = []


def checked_fit(samples, dimension, metadata=None):
    value = fit(samples, dimension)
    FIT_RECEIPTS.append(value)
    if not value['converged']:
        raise AssertionError(value)
    return value


class ExecutionTests(unittest.TestCase):
    def test_known_balanced_optimum_and_single_class_scalar_stationarity(self):
        balanced = checked_fit([(features(0,0),0),(features(0,0),1)],3)
        self.assertEqual(balanced['beta'], [0.,0.,0.])
        self.assertEqual(balanced['newton_steps'], 0)
        negative = checked_fit([(features(0,0),0)]*3,3)
        intercept, count_slope, hold_slope = negative['beta']
        self.assertTrue(-3 < intercept < 0)
        self.assertEqual([count_slope,hold_slope],[0.,0.])
        # This scalar derivative is independent of the vector objective code.
        self.assertLessEqual(abs(intercept+3/(1+math.exp(-intercept))), 1e-8)
        for fitted in (balanced,negative):
            self.assertEqual(fitted['objective_evaluations'], 1+fitted['line_search_evaluations'])
            self.assertLessEqual(fitted['newton_steps'], 100)

    def test_future_label_and_same_clock_packet_cannot_change_earlier_predictions(self):
        rows = []
        values = [(1000,1200,0,0,0),(1300,1500,0,30,0),
                  (9100,9300,2,5,1),(9300,9600,0,10,0),
                  (9301,9600,2,10,1),(9700,9900,1,15,0)]
        for sequence,(issue,deadline,n,h,y) in enumerate(values):
            rows.append(dict(meta=dict(source_group='synthetic/stable',sequence=sequence,observed_at=issue),
                selection=dict(sequence=sequence,window_start=issue,packet_deadline=deadline),
                inputs=[n,h],target=dict(value=y)))
        initial = initialize(rows[:2], lambda row:row['target']['value'], checked_fit)
        saved = deepcopy(initial)
        self.assertEqual(initial['full'][1], 0.)
        for a,b in zip((initial['full'][0],initial['full'][2]),initial['hold_only']):
            self.assertAlmostEqual(a,b,places=12)

        def execute(source):
            trace = []
            def truth(row):
                self.assertEqual(trace[-1]['kind'],'prediction')
                self.assertEqual(trace[-1]['sequence'],row['meta']['sequence'])
                return row['target']['value']
            result,stats = forecast(source,initial,truth,checked_fit,trace.append)
            return result,stats,trace

        first,stats,trace = execute(rows)
        changed = deepcopy(rows)
        changed[2]['target']['value'] = 0
        second,_,_ = execute(changed)
        self.assertEqual(initial,saved)
        self.assertEqual([r['probabilities'] for r in first[:2]],
                         [r['probabilities'] for r in second[:2]])
        self.assertNotEqual(first[2]['probabilities']['fresh'], second[2]['probabilities']['fresh'])
        self.assertEqual([r['probabilities']['frozen'] for r in first],
                         [r['probabilities']['frozen'] for r in second])
        self.assertEqual([r['latest_feedback_deadline'] for r in first],[9000,9000,9300,9600])
        self.assertEqual(stats['released_packets'],2)
        self.assertEqual(stats['final_history_rows'],5)


if __name__ == '__main__':
    unittest.main()
