import unittest
import numpy as np
from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
from experiments.mcwm_closed_loop_supervision_plan_v84_20261027.collection import choose_pair


class CollectionTest(unittest.TestCase):
    def setUp(self):
        self.transform = ExecutionAction(dict(max_hold_seconds=60., scheduled_headway_scale_seconds=600.))
        self.state = np.zeros(229, np.float32)
        self.state[0] = .5

    def select(self, proposal):
        return choose_pair(self.transform, self.state, np.asarray([proposal], np.float32),
                           simulation_step=1., action_step=1.)

    def test_observed_interior_reference_is_not_replaced_with_zero(self):
        row = self.select(-.09766614437103271)
        self.assertAlmostEqual(row['reference_hold_seconds'], 3.3837515115737915)
        self.assertEqual(row['candidate_hold_seconds'], 7.5)
        self.assertTrue(row['supported'])

    def test_reference_can_be_larger_than_candidate(self):
        row = self.select(1.)
        self.assertEqual(row['reference_hold_seconds'], 7.5)
        self.assertEqual(row['candidate_hold_seconds'], 0.)
        self.assertTrue(row['supported'])

    def test_completion_floor_single_support_is_not_a_pair(self):
        event = self.state[5:19]
        event[11], event[9] = 1., 1.
        self.assertFalse(self.select(-1.)['supported'])


if __name__ == '__main__':
    unittest.main()
