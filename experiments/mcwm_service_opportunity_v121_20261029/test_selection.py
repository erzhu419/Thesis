"""Detect wrong replacement, release-window leakage and outcome-based selection."""
from copy import deepcopy
import unittest

from .select_opportunity import select_packets


def context(sequence, clock, count=0):
    return dict(meta=dict(sequence=sequence, observed_at=clock),
        feeders=[dict(x=dict(feeder_next_stop_matches=True, current_leg_dest_count=count,
                             same_lane=False, gap_m=None))])


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.protocol = dict(feedback_first_deadline=1200, feedback_last_deadline=1200,
            feedback_interval_seconds=300, feedback_records_per_packet=5,
            feedback_delay_seconds=30, target_record_budget=5)
        self.original = [dict(deadline=1200, release_lower_exclusive=None,
            selected_sequences=[2,3,4,5,6])]

    def test_only_oldest_replaced_latest_opportunity_and_boundary(self):
        data = [context(i, 1104+i, 1 if i in [0,1,7] else 0) for i in range(8)]
        plans, chosen = select_packets(data, self.original, self.protocol)
        self.assertEqual(plans[0]['selected_sequences'], [1,3,4,5,6])
        self.assertEqual(plans[0]['substitution'], dict(removed_sequence=2,added_sequence=1))
        self.assertNotIn(7,chosen)  # Its feedback becomes available at1201, not1200.

    def test_existing_opportunity_or_none_preserves_original_five(self):
        data = [context(i, 1104+i) for i in range(7)]
        for count in [0,1]:
            data[-1]['feeders'][0]['x']['current_leg_dest_count'] = count
            plans, _ = select_packets(data,self.original,self.protocol)
            self.assertEqual(plans[0]['selected_sequences'],[2,3,4,5,6])
            self.assertIsNone(plans[0]['substitution'])

    def test_equal_clock_uses_sequence_and_ignores_outcomes(self):
        data = [context(i, 1100, 1 if i<2 else None) for i in range(7)]
        plans, _ = select_packets(data,self.original,self.protocol)
        changed=deepcopy(data)
        for row in changed:
            row['future_labels']={'boarded_count':1000-row['meta']['sequence']}
            row['future_cost']=-1000000
            row['diagnostic_person']='irrelevant'
        other, _ = select_packets(changed,self.original,self.protocol)
        self.assertEqual(plans,other)
        self.assertEqual(plans[0]['selected_sequences'],[1,3,4,5,6])


if __name__=='__main__':
    unittest.main()
