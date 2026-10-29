import importlib.util
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

SPEC = importlib.util.spec_from_file_location("v117_support", Path(__file__).with_name("support.py"))
support = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(support)


def ledger(t, w=0., r=0., waiting=2, riding=3):
    return dict(time_seconds=t, waiting_passengers=waiting, in_vehicle_passengers=riding,
                accrued_wait_seconds=w, accrued_in_vehicle_seconds=r)


class SupportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def build(self):
        actions = [dict(sequence=i, started_at=t, trip_id=trip, station_id=stop,
                        proposed_a=[proposal], a=[proposal], s=[t, i], visit_index=i)
                   for i,t,trip,stop,proposal in ((0,1,"a","s",-.8), (1,2,"a","s",.2),
                                                 (2,3,"b","s",.7))]
        paths = []
        for name, values in (("parent",actions[:1]),("suffix",actions[1:])):
            path=self.root/name
            path.write_text("".join(json.dumps(row)+"\n" for row in values))
            paths.append(path)
        tick=self.root/"ticks"
        tick.write_text("".join(json.dumps(dict(start=t,end=t+1,waiting=2,in_vehicle=3))+"\n"
                                for t in range(4)))
        return support.RecordedContinuation(paths,1,4,[tick]), actions

    def test_identity_lookup_survives_order_change_and_tracks_repeat_stop(self):
        continuation, actions = self.build()
        batch=[dict(trip_id="terminal",station_id="last",terminal=True),
               dict(trip_id="b",station_id="s",terminal=False),
               dict(trip_id="a",station_id="s",terminal=False)]
        stream=SimpleNamespace(_batch=lambda:batch)
        self.assertEqual(continuation.proposal([0,0,1/15],stream),[.2])
        self.assertEqual(continuation.note_issue(actions[1]),("a","s",2))
        self.assertEqual(continuation.proposal([0,0,0],stream),[.7])
        continuation.note_issue(actions[2])
        batch[1]["trip_id"]="missing"
        with self.assertRaisesRegex(ValueError,"no recorded continuation"):
            continuation.proposal([0,0,0],stream)

    def test_root_detach_and_real_issue_endpoint_exactness(self):
        continuation, actions = self.build()
        closed=[]
        reference=SimpleNamespace(current=actions[1], ledger_at=lambda clock:ledger(2),
            report=lambda:dict(actions_checked=1),close=lambda:closed.append(True))
        stream=SimpleNamespace(_snapshot=lambda:ledger(2))
        binding=continuation.close_root(reference,stream)
        self.assertEqual(closed,[True])
        self.assertTrue(binding["root_issue_not_yet_observed"])
        self.assertEqual(binding["prefix"]["actions_checked"],1)
        continuation.check_issue(actions[1],ledger(2))
        continuation.check_issue(actions[2],ledger(3,2,3))
        with self.assertRaises(ValueError):
            continuation.check_end(ledger(4,5,6))
        self.assertTrue(continuation.check_end(ledger(4,4,6))["unchanged_branch_exact"])

    def test_missing_issue_and_root_ledger_drift_are_rejected(self):
        continuation, actions = self.build()
        reference=SimpleNamespace(current=actions[1],ledger_at=lambda clock:ledger(2),
                                  report=lambda:{},close=lambda:None)
        with self.assertRaisesRegex(ValueError,"Root ledger differs"):
            continuation.close_root(reference,SimpleNamespace(_snapshot=lambda:ledger(2,1)))
        continuation.close_root(reference,SimpleNamespace(_snapshot=lambda:ledger(2)))
        continuation.check_issue(actions[1],ledger(2))
        with self.assertRaisesRegex(ValueError,"declared issues"):
            continuation.check_end(ledger(4,4,6))

    def test_state_receipt_detects_ledger_counter_and_rng_drift(self):
        snapshot=ledger(2)
        stream=SimpleNamespace(_snapshot=lambda:snapshot,episode_id=0,issued_count=1,
            completed_count=0,released_count=0,thinned_count=0,_pending=None,
            _vehicle_pending={},_queue=[],_interval_queue=[],_batch=lambda:[])
        before=support.state_receipt(stream,dict(applied=1))
        self.assertEqual(before,support.state_receipt(stream,dict(applied=1)))
        self.assertNotEqual(before,support.state_receipt(stream,dict(applied=2)))
        snapshot["accrued_wait_seconds"]=1
        self.assertNotEqual(before,support.state_receipt(stream,dict(applied=1)))
        snapshot["accrued_wait_seconds"]=0
        saved=random.getstate()
        try:
            random.random()
            self.assertNotEqual(before,support.state_receipt(stream,dict(applied=1)))
        finally:
            random.setstate(saved)


if __name__ == "__main__":
    unittest.main()
