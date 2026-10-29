"""Check cost subtraction, label timing and the actual native pair contract."""
from copy import deepcopy
import unittest

from experiments.mcwm_closed_loop_supervision_plan_v84_20261027 import contract as C


class ContractTest(unittest.TestCase):
    def pair(self):
        low = dict(root=dict(sequence=120, started_at=12007., line_id="102X",
                             s=[.5, 1.], ledger=dict(wait=1000., ride=2000.)),
                   reference=C.source_snapshot_metadata(seed=6207, scenario="stable",
                       published_clock=12003., max_available_at=11999.),
                   external_seed=106207, executed_hold_seconds=0.,
                   sim_step_seconds=1., action_step_seconds=1.,
                   departure_clock=12008., final_clock=24000,
                   suffix_updates=0, suffix_admitted_labels=0)
        high = deepcopy(low)
        high.update(executed_hold_seconds=15., departure_clock=12023.)
        return low, high

    def test_cost_subtracts_actual_issue_prefix(self):
        start = dict(time_seconds=12007., accrued_wait_seconds=1000.,
                     accrued_in_vehicle_seconds=2000.)
        end = dict(time_seconds=24000., accrued_wait_seconds=1500.,
                   accrued_in_vehicle_seconds=3000.)
        value = C.remaining_cost(start, end)
        self.assertEqual((value["wait"], value["ride"], value["cost"]), (500., 1000., 2000.))
        self.assertEqual(value["dt"], 11993.)
        self.assertEqual(value["available_at"], 24030)
        with self.assertRaises(ValueError):
            C.remaining_cost(start, dict(end, time_seconds=23999.))

    def test_label_cannot_enter_same_day_training(self):
        self.assertFalse(C.label_available(24030, 18000))
        self.assertFalse(C.label_available(24030, 24029))
        self.assertTrue(C.label_available(24030, 24030))
        self.assertFalse(C.label_available(24000, 24030))

    def test_same_root_reference_and_actual_separation_pass(self):
        self.assertTrue(C.pair_gate(*self.pair())["all_pass"])

    def test_different_complete_root_fails(self):
        low, high = self.pair()
        high["root"]["ledger"]["wait"] += 1
        gate = C.pair_gate(low, high)
        self.assertFalse(gate["checks"]["same_complete_root"])
        self.assertFalse(gate["all_pass"])

    def test_different_or_future_reference_fails(self):
        low, high = self.pair()
        high["reference"]["counts"]["actor"] = 304
        self.assertFalse(C.pair_gate(low, high)["all_pass"])
        with self.assertRaises(ValueError):
            C.source_snapshot_metadata(seed=6207, scenario="stable",
                published_clock=12008, max_available_at=12001)

    def test_command_difference_without_physical_difference_fails(self):
        low, high = self.pair()
        high["departure_clock"] = low["departure_clock"]
        gate = C.pair_gate(low, high)
        self.assertTrue(gate["checks"]["command_support"])
        self.assertFalse(gate["checks"]["physical_departure_separation"])
        self.assertFalse(gate["all_pass"])

    def test_missing_departure_or_suffix_update_fails(self):
        low, high = self.pair()
        high["departure_clock"] = None
        self.assertFalse(C.pair_gate(low, high)["all_pass"])
        low, high = self.pair()
        high["suffix_updates"] = 1
        self.assertFalse(C.pair_gate(low, high)["all_pass"])


if __name__ == "__main__":
    unittest.main()
