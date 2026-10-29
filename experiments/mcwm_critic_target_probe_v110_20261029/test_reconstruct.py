"""Detect confusing chronological transitions with same-vehicle next stops."""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'experiments/mcwm_bus/runtime/H2Oplus'))
from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7
from reconstruct import first_packet


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((ROOT/'experiments/mcwm_full_mcwm_dripping_pilot_v107_20261029/protocol.json').read_text())
        self.times = [900, 910, 920, 920, 940, 950, 960, 1180]
        self.trace = [dict(sequence=i, vehicle_id=str(i % 3), s=[i], a=[-.5],
            issue_ledger=dict(time_seconds=t, waiting_passengers=4, in_vehicle_passengers=5,
                accrued_wait_seconds=4*t, accrued_in_vehicle_seconds=5*t))
            for i, t in enumerate(self.times)]
        self.receipt = dict(deadline=1200, processed_at=1212., records=5,
            sequence_ids=[0, 1, 3, 4, 5], max_available_at=990.)

    def test_exact_latest_five_and_raw_reward(self):
        packet = first_packet(self.trace, self.receipt, self.protocol, TransitMDPConfigV7())
        self.assertEqual([r.sequence for r in packet], [0, 1, 3, 4, 5])
        self.assertEqual([r.dt for r in packet], [10, 10, 20, 10, 10])
        self.assertEqual(packet[2].r, -(2*4+5)*20)
        self.assertEqual(packet[2].ns.tolist(), [4])

    def test_missing_successor_or_changed_receipt_stops_probe(self):
        with self.assertRaisesRegex(ValueError, 'Missing next chronological'):
            first_packet(self.trace[:3]+self.trace[4:], self.receipt, self.protocol, TransitMDPConfigV7())
        wrong = dict(self.receipt, sequence_ids=[1, 2, 3, 4, 5])
        with self.assertRaisesRegex(ValueError, 'differs from the actual'):
            first_packet(self.trace, wrong, self.protocol, TransitMDPConfigV7())


if __name__ == '__main__':
    unittest.main()
