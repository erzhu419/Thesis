"""Synthetic ledger/label checks; no simulator or fitting."""
import unittest
from audit import ride_intervals, receiver_label, feeder_label
from core import waiting_intervals, maturity


def selection():
    return dict(sequence=1,window_start=5.,window_end=65.,packet_deadline=1200)


class CostLabels(unittest.TestCase):
    def test_wait_uses_membership_and_ride_ends_at_actual_exit(self):
        events = [dict(time=0,trip_id='a',boarded_ids=['p'],alighted_ids=[],observed_at_stop=True,stop_id='start'),
            dict(time=10,trip_id='a',boarded_ids=[],alighted_ids=['p'],observed_at_stop=True,stop_id='s'),
            dict(time=20,trip_id='b',boarded_ids=['p'],alighted_ids=[],observed_at_stop=True,stop_id='s')]
        rides, boards = ride_intervals(events,horizon=30)
        waiting = waiting_intervals([dict(time=12,entered=['p'],left=[]),dict(time=18,entered=[],left=['p'])],30)
        record = dict(selection=selection(),cohort_definition_complete=True,
            context=dict(meta=dict(station='s'),feeders=[dict(trip_id='a',cohort_ids=['p'])]))
        label = feeder_label(record,{('p','a'):[rides[0]]},boards,waiting,
                             {'p':dict(completion_time=None)},[1300])
        self.assertEqual(label['value'],6)
        self.assertEqual(label['maturity']['closed_at'],20)
        self.assertEqual(label['elapsed_is_not_wait_people'],1)
        ride = receiver_label([rides[0]],{'p'},selection(),[1300])
        self.assertEqual(ride['value'],5)
        rides[0].update(exit_at_stop=False,stop=None)
        self.assertEqual(receiver_label([rides[0]],{'p'},selection(),[1300])['value'],5)
        unknown = feeder_label(record,{('p','a'):[rides[0]]},boards,waiting,
                               {'p':dict(completion_time=None)},[1300])
        self.assertIsNone(unknown['value'])
        self.assertEqual(unknown['unknown_people'],1)

    def test_late_cost_does_not_enter_original_window_or_common_prefix(self):
        fixed = dict(window_start=8810.,window_end=8870.,packet_deadline=9000)
        late = maturity(fixed,8890.,True,[9000,9010,9200])
        self.assertFalse(late['within_original_window'])
        self.assertFalse(late['eligible_original_packet'])
        self.assertFalse(late['common_initialization_available'])
        self.assertEqual(late['earliest_extended_packet'],9000)
        self.assertEqual(late['later_issue_count'],2)
        cross = maturity(fixed,9200.,True,[9300,9301])
        self.assertFalse(cross['common_initialization_available'])
        self.assertEqual(cross['earliest_extended_packet'],9300)
        self.assertEqual(cross['later_issue_count'],1)
        self.assertIsNone(maturity(fixed,None,True,[9301])['earliest_extended_packet'])
        self.assertFalse(maturity(fixed,8810,False,[9301])['evidence'])

    def test_early_closure_cannot_preempt_original_packet(self):
        fixed = dict(window_start=8620.,window_end=8680.,packet_deadline=9000)
        value = maturity(fixed,8630.,True,[8800,9000,9010])
        self.assertTrue(value['eligible_original_packet'])
        self.assertEqual(value['earliest_extended_packet'],9000)
        self.assertEqual(value['later_issue_count'],1)


if __name__=='__main__':
    unittest.main()
