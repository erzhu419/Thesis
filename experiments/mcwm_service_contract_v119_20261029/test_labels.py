"""Detect premature labels, incorrect cohort/visit joins and feedback-time leakage."""
import unittest

from .labels import mature_labels, feedback_deadline


class LabelsTest(unittest.TestCase):
    def setUp(self):
        self.sample = dict(meta=dict(receiver='R', station='S', observed_at=10),
            action=dict(applied_at=10, hold_seconds=7.5),
            feeders=[dict(trip_id='F', cohort_ids=['p', 'q'], current_stage0_complete=True,
                          at_target_stop_at_issue=False)])
        self.evidence = dict(start=10, end=70, continuous=True,
            visit_events=[dict(event='departure', time=21, trip_id='R', stop_id='S')],
            exchanges=[])

    def test_departure_closes_zero_but_not_feeder_arrival(self):
        before = mature_labels(self.sample, self.evidence, available_at=20)
        after = mature_labels(self.sample, self.evidence, available_at=21)
        self.assertIsNone(before['cohort_boarded_count']['value'])
        self.assertEqual(after['cohort_boarded_count']['value'], 0)
        self.assertTrue(after['feeders'][0]['arrival']['right_censored'])
        person = after['feeders'][0]['people'][0]
        self.assertEqual(person['boarded_this_visit']['value'], 0)
        self.assertFalse(person['boarding']['right_censored'])
        self.assertTrue(person['boarding']['closed_without_event'])

    def test_same_tick_membership_and_fixed_cohort_not_future_events(self):
        def event(time, trip, stop, board=(), alight=()):
            return dict(time=time, trip_id=trip, stop_id=stop,
                        boarded_ids=list(board), alighted_ids=list(alight))
        self.evidence['exchanges'] = [event(21,'R','S',['p','outside']),
            event(21,'F','S',alight=['p']), event(22,'R','S',['q']),
            event(19,'other','S',['q']), event(19,'R','other',['q'])]
        labels = mature_labels(self.sample, self.evidence, available_at=70)
        self.assertEqual(labels['cohort_boarded_count']['value'], 1)
        self.assertEqual(labels['receiver_post_issue_boarded_count']['value'], 2)
        self.assertEqual(labels['feeders'][0]['people'][0]['alighting']['value'], 21)
        self.assertEqual(labels['feeders'][0]['people'][1]['boarded_this_visit']['value'], 0)
        self.assertEqual(labels['departure_residual_seconds']['value'], 3.5)

    def test_maturity_feedback_delay_and_closed_budget(self):
        protocol = dict(feedback_first_deadline=1200, feedback_last_deadline=18000,
                        feedback_interval_seconds=300, feedback_delay_seconds=30)
        self.assertEqual(feedback_deadline(14589, protocol), 14700)
        self.assertEqual(feedback_deadline(17700, protocol), 18000)
        self.assertIsNone(feedback_deadline(17971, protocol))
        self.assertIsNone(feedback_deadline(18971, protocol))
        self.assertIsNone(feedback_deadline(None, protocol))
        self.evidence['continuous'] = False
        with self.assertRaises(ValueError):
            mature_labels(self.sample, self.evidence, available_at=21)


if __name__ == '__main__':
    unittest.main()
