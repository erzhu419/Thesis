import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

SPEC = importlib.util.spec_from_file_location("v120_samples", Path(__file__).with_name("samples.py"))
samples = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(samples)


class SamplesTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.contract=dict(feedback_first_deadline=1200,feedback_last_deadline=1500,
            feedback_interval_seconds=300,feedback_records_per_packet=5,
            feedback_delay_seconds=30,target_record_budget=10)

    def write(self,name,values):
        (self.root/name).write_text("".join(json.dumps(v)+"\n" for v in values))

    def fixture(self, starts, *, late=False):
        contexts, commands, visits, exchanges = [], [], [], []
        for sequence,start in enumerate(starts):
            context=dict(meta=dict(sequence=sequence,observed_at=start,receiver=f"R{sequence}",
                station="S",source_group=f"scene/{sequence}"),
                capture=dict(status="complete",missing=[],checks=dict(receiver_native_present=True,
                    receiver_at_declared_stop=True,feeder_selection_complete=True)),
                receiver=dict(capacity=50,onboard=10,free_seats=40,service_ready_age_seconds=None),
                feeders=[dict(trip_id=f"F{sequence}",cohort_ids=[f"p{sequence}"],
                    at_target_stop_at_issue=False,current_stage0_complete=True,
                    x=dict.fromkeys(samples.FEEDER_FIELDS,None))])
            contexts.append(context)
            commands.append(dict(sequence=sequence,trip_id=f"R{sequence}",stop_id="S",
                applied_at=start,accepted=True,requested_additional_hold_seconds=0))
            visits.append(dict(time=start+(61 if late else 20),event="departure",trip_id=f"R{sequence}",stop_id="S"))
            exchanges.append(dict(time=start+(61 if late else 10),trip_id=f"R{sequence}",stop_id="S",
                observed_at_stop=True,boarded_ids=[f"p{sequence}"],alighted_ids=[]))
        for name,rows in (("contexts.jsonl",contexts),("commands.jsonl",commands),
                          ("visit_events.jsonl",visits),("exchanges.jsonl",exchanges)):
            self.write(name,rows)
        (self.root/"observer_summary.json").write_text(json.dumps(dict(status="complete",start=0,end=24000,
            checks=dict(tick_contiguous=True,ledger_exact=True))))

    def read(self,name):
        return [json.loads(line) for line in (self.root/name).read_text().splitlines()]

    def test_latest_five_per_release_window_preserves_unknown_and_zero_hold(self):
        self.fixture(list(range(1000,1006))+list(range(1300,1306)))
        contexts=self.read("contexts.jsonl")
        contexts[4]["capture"]["status"]="missing"
        contexts[4]["capture"]["missing"]=["receiver_service_ready_time"]
        contexts[5]["capture"]=dict(status="missing",missing=["receiver"],checks=dict(
            receiver_native_present=False,receiver_at_declared_stop=False,feeder_selection_complete=True))
        contexts[5]["feeders"]=[]
        self.write("contexts.jsonl",contexts)
        result=samples.select_and_label(self.root,self.contract)
        self.assertTrue(result["all_pass"])
        self.assertEqual([p["selected_sequences"] for p in self.read("packets.jsonl")],
                         [[1,2,3,4,5],[7,8,9,10,11]])
        self.assertEqual(result["statistics"]["selected_with_unknown_inputs"],10)
        self.assertEqual(result["action_seconds"]["zero"],10)
        self.assertEqual(result["budget"]["root_action_records"],10)
        self.assertEqual(result["statistics"]["selected_with_incomplete_capture"],2)
        self.assertEqual(result["statistics"]["valid_mature_records"],9)

    def test_packet_time_cannot_reveal_events_after_sixty_second_window(self):
        self.contract.update(feedback_last_deadline=1200,target_record_budget=5)
        self.fixture(range(1000,1005),late=True)
        result=samples.select_and_label(self.root,self.contract)
        self.assertTrue(result["all_pass"])
        self.assertEqual(result["statistics"]["valid_mature_records"],0)
        for row in self.read("selected_records.jsonl"):
            self.assertIsNone(row["labels"]["receiver_departure"]["value"])
            self.assertEqual(row["labels"]["cohort_boarded_count"]["observed_count_lower_bound"],0)
            self.assertEqual(row["labels"]["available_at"],row["selection"]["window_end"])

    def test_boundary_release_and_failed_selected_action_do_not_backfill(self):
        self.fixture([1106,1107,1108,1109,1110,1111,1300,1301,1302,1303])
        commands=self.read("commands.jsonl")
        commands[4]["accepted"]=False
        self.write("commands.jsonl",commands)
        result=samples.select_and_label(self.root,self.contract)
        self.assertFalse(result["all_pass"])
        self.assertEqual([p["selected_sequences"] for p in self.read("packets.jsonl")],
                         [[0,1,2,3,4],[5,6,7,8,9]])
        self.assertEqual(result["selected_records"],10)
        failed=next(row for row in self.read("selected_records.jsonl") if row["selection"]["sequence"]==4)
        self.assertTrue(failed["errors"])
        self.assertIsNone(failed["action"])


if __name__ == "__main__":
    unittest.main()
