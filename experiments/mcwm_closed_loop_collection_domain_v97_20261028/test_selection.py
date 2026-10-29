from copy import deepcopy
import json
from pathlib import Path
import unittest

from experiments.mcwm_closed_loop_collection_domain_v97_20261028.selection import (
    context_reason, frozen_predictions, lock_first, model_support)

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'mcwm_closed_loop_confidence_collection_v96_20261028'


class SelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = json.loads((HERE / 'source_model.json').read_text())
        cls.coverage = json.loads(
            (PREVIOUS / 'local_inputs/309177151_stable.json').read_text())['coverage']

    def test_unsupported_first_slot_cannot_be_replaced(self):
        slots = {'7X':dict(selected=None, status='missing')}
        self.assertTrue(lock_first(slots, '7X', dict(sequence=1)))
        slots['7X']['status'] = 'unsupported'
        self.assertFalse(lock_first(slots, '7X', dict(sequence=2)))
        self.assertEqual(slots['7X']['selected']['sequence'], 1)

    def test_per_line_empirical_range_only_describes_model_support(self):
        model = dict(common_support=dict(q_training_ranges={'7X':[.01,.46], '705X':[.01,.50]}))
        condition = dict(status='supported', historical_only=True, latest_completed_at=11999.,
            latest_observed_at=11999., latest_age_seconds=1., value=.49)
        service = dict(observed=True)
        self.assertIsNone(context_reason(service, condition, '7X', model, 12000.))
        self.assertIsNone(context_reason(service, condition, '705X', model, 12000.))
        outside = model_support(model, '7X', .49)
        inside = model_support(model, '705X', .49)
        self.assertFalse(outside['in_training_range'])
        self.assertTrue(outside['extrapolation'])
        self.assertEqual(outside['status'], 'outside_training_range')
        self.assertTrue(inside['in_training_range'])
        self.assertFalse(inside['extrapolation'])
        self.assertEqual(inside['status'], 'within_training_range')

    def test_v96_real_coverage_is_collectible_but_outside_old_model_range(self):
        self.assertEqual(set(self.coverage), {'7X', '102X', '705X'})
        for line, row in self.coverage.items():
            with self.subTest(line=line):
                self.assertEqual(row['reason'], 'history_outside_frozen_line_range')
                self.assertTrue(all(row['observation_checks'].values()))
                self.assertIsNone(context_reason(row['service'], row['condition'], line,
                                                 self.model, row['root']['started_at']))
                support = model_support(self.model, line, row['condition']['value'])
                self.assertFalse(support['in_training_range'])
                self.assertTrue(support['extrapolation'])
                self.assertEqual(support['status'], 'outside_training_range')

    def test_real_coverage_still_requires_strict_past_and_fresh_history(self):
        row = self.coverage['102X']
        issue_clock = row['root']['started_at']
        cases = [('latest_completed_at', issue_clock, 'history_availability_or_age'),
                 ('latest_observed_at', issue_clock, 'history_availability_or_age'),
                 ('latest_age_seconds', 901., 'history_availability_or_age'),
                 ('historical_only', False, 'history_not_strictly_past')]
        for key, value, expected in cases:
            with self.subTest(field=key):
                condition = deepcopy(row['condition'])
                condition[key] = value
                self.assertEqual(context_reason(row['service'], condition, '102X',
                                                self.model, issue_clock), expected)

    def test_native_missing_history_and_nonfinite_values_remain_ineligible(self):
        row = self.coverage['102X']
        issue_clock = row['root']['started_at']
        service = dict(row['service'], observed=False, reason='native_read_missing')
        self.assertEqual(context_reason(service, row['condition'], '102X', self.model,
                                        issue_clock), 'native_read_missing')
        condition = dict(row['condition'], status='unsupported',
                         reason='baseline_coverage_missing')
        self.assertEqual(context_reason(row['service'], condition, '102X', self.model,
                                        issue_clock), 'baseline_coverage_missing')
        for value in (None, float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                condition = dict(row['condition'], value=value)
                self.assertEqual(context_reason(row['service'], condition, '102X',
                                                self.model, issue_clock),
                                 'history_value_missing_or_nonfinite')
                support = model_support(self.model, '102X', value)
                self.assertEqual(support['status'], 'unavailable')
                self.assertIsNone(support['q'])
                self.assertIsNone(support['in_training_range'])
                self.assertIsNone(support['extrapolation'])
                json.dumps(support, allow_nan=False)

    def test_v96_real_forward_values_are_unchanged(self):
        expected = {
            '102X':dict(line3=-0.46525131899171607, service5=-0.4683002397426417,
                        service_history6=-1.0120607067782825),
            '7X':dict(line3=-0.22504282001052983, service5=-0.21781596767147057,
                      service_history6=-0.7721361602501323),
            '705X':dict(line3=-0.4640432223301127, service5=-0.5210565187148799,
                        service_history6=-1.1399029711792472)}
        for line, row in self.coverage.items():
            with self.subTest(line=line):
                self.assertEqual(frozen_predictions(self.model, line, row['service'],
                    row['condition']['value'], row['ref_hold'], row['cand_hold']),
                    expected[line])

    def test_forward_only_and_action_scaled_features(self):
        model = dict(hold_scale=7.5, design=dict(line_centers={'7X':[.1,-.5,.2]}),
            models={'line3':dict(coef=[1,2,3]), 'service5':dict(coef=[1,2,3,4,5]),
                    'service_history6':dict(coef=[1,2,3,4,5,6])})
        values = frozen_predictions(model, '7X', dict(features=[.3,-.1]), .4, 3.75, 7.5)
        self.assertAlmostEqual(values['line3'], .5)
        self.assertAlmostEqual(values['service5'], 1.9)
        self.assertAlmostEqual(values['service_history6'], 2.5)


if __name__ == '__main__':
    unittest.main()
