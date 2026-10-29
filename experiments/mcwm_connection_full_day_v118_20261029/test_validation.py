import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

SPEC = importlib.util.spec_from_file_location("v118_validation", Path(__file__).with_name("validation.py"))
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row)+"\n" for row in rows))
        return path

    def fixture(self):
        parent, child, old = self.root/"actual", self.root/"actual/branches/low", self.root/"old"
        original = old/"branches/low"
        rows = [dict(identity=i, value=i+1) for i in range(3)]
        for name in validation.SEGMENTED:
            self.write(parent/(name+".jsonl"), rows[:2])
            self.write(child/(name+".jsonl"), rows[2:])
            self.write(old/(name+".jsonl"), rows[:1])
            self.write(original/(name+".jsonl"), rows[1:])
        for name in validation.CUMULATIVE:
            self.write(child/(name+".jsonl"), rows)
            self.write(original/(name+".jsonl"), rows)
        summary = dict(status="complete", checks=dict(ledger_exact=True), errors=[], ledger_exact=True,
            start=0, end=24000, ticks=3, issues=3, commands=3, passengers=3, visits=3,
            rejected_commands=0, passenger_totals=dict(wait=8, ride=9), periods={},
            ledger=dict(time_seconds=24000, accrued_wait_seconds=8, accrued_in_vehicle_seconds=9),
            maximum_count_gap=0, maximum_ledger_gap=0, traffic_or_other_reasons={})
        for directory in (child, original):
            (directory/"observer_summary.json").write_text(json.dumps(summary))
        return parent, child, old

    def test_different_fork_boundaries_have_equal_full_records(self):
        result = validation.compare_nominal(*self.fixture())
        self.assertTrue(result["all_pass"])
        self.assertEqual(result["files"]["actions.jsonl"]["observed_rows"], [3,3])

    def test_changed_native_tick_and_missing_file_are_rejected(self):
        parent, child, old = self.fixture()
        self.write(child/"ticks.jsonl", [dict(identity=2,value=99)])
        result=validation.compare_nominal(parent,child,old)
        self.assertFalse(result["all_pass"])
        self.assertEqual(result["files"]["ticks.jsonl"]["first_difference"],3)
        (child/"exchanges.jsonl").unlink()
        self.assertFalse(validation.compare_nominal(parent,child,old)["files"]["exchanges.jsonl"]["all_pass"])

    def test_truncated_people_and_changed_ledger_are_rejected(self):
        parent, child, old = self.fixture()
        self.write(child/"passengers.jsonl", [dict(identity=0,value=1)])
        result=validation.compare_nominal(parent,child,old)
        self.assertFalse(result["all_pass"])
        self.assertFalse(result["observer"]["row_counts"]["passengers"])
        path=child/"observer_summary.json"
        row=json.loads(path.read_text())
        row["ledger"]["accrued_wait_seconds"]=10
        path.write_text(json.dumps(row))
        self.assertFalse(validation.compare_nominal(parent,child,old)["observer"]["fields_equal"]["ledger"])


if __name__ == "__main__":
    unittest.main()
