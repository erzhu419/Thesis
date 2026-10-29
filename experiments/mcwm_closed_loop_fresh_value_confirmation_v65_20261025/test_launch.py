"""Offline fixed v64 confirmation dispatch, completion, and receipt reuse."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("v65_launch_tested", HERE / "launch.py")
L = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(L)
OBSERVER = L.module("v65_test_observer", L.V60 / "supervisor.py")
WATCH = L.module("v65_test_existing_watch", L.V62 / "launch.py")
PROTOCOL = dict(name="v65-test", confirmation_seeds=[6203, 6204, 6205],
                scenarios=["stable", "local_speed_drop"])


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.put(self.root / "protocol.json", PROTOCOL)
        self.put(self.root / "execution_ready.json", dict(ready=True))
        self.outputs, self.submissions = {}, []
        self.infra = SimpleNamespace(remote=str, task=self.task,
            presence=mock.Mock(side_effect=lambda paths: {p: p in self.outputs for p in paths}),
            collect=mock.Mock(side_effect=lambda p: self.outputs[p]),
            status=mock.Mock(side_effect=lambda ids: [dict(id=key, status="running") for key in ids]),
            submit=mock.Mock(side_effect=self.submit))
        patcher = mock.patch.object(L, "HERE", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(L, "module", side_effect=lambda name, path:
            OBSERVER if path.name == "supervisor.py" else WATCH if path == L.V62 / "launch.py" else self.infra)
        patcher.start()
        self.addCleanup(patcher.stop)

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def task(self, label, runner, argv):
        return dict(signature=PROTOCOL["name"] + "/" + label, runner=str(runner), argv=argv, priority="high")

    def submit(self, jobs, batch):
        path = self.root / (batch + "_submission.json")
        if path.exists():
            return json.loads(path.read_text())
        value = dict(submitted=[dict(id="t_" + str(len(self.submissions)) + "_" + str(index),
                                    signature=row["signature"]) for index, row in enumerate(jobs)])
        self.submissions.append(dict(batch=batch, jobs=jobs))
        self.put(path, value)
        return value

    def native_complete(self):
        for cell in L.cells(PROTOCOL):
            self.outputs[cell["relative"] + "/summary.json"] = dict(status="complete")

    def finish(self, *, passed=True, integrity=True):
        self.native_complete()
        self.outputs["results/confirmation.json"] = dict(status="complete" if integrity else "integrity_failed",
            integrity=dict(all_pass=integrity), decision=dict(confirmed_control_improvement=passed,
                ordinary_dripping_goal_achieved=False))

    def test_ready_gate_prevents_submission(self):
        self.put(self.root / "execution_ready.json", dict(ready=False))
        current = L.advance()
        self.assertEqual(current["stage"], "preparing_fresh_value_confirmation")
        self.assertFalse(current["done"])
        self.infra.submit.assert_not_called()

    def test_twelve_days_only_use_frozen_v64_runner_without_candidate_argument(self):
        current = L.advance()
        jobs = self.submissions[0]["jobs"]
        self.assertEqual(self.submissions[0]["batch"], "fresh_value_confirmation")
        self.assertEqual(len(jobs), 12)
        self.assertEqual({Path(job["runner"]) for job in jobs}, {L.V64 / "control.py"})
        self.assertEqual({job["argv"][1] for job in jobs}, {6203, 6204, 6205})
        self.assertEqual({job["argv"][5] for job in jobs}, {"fresh", "no_fresh"})
        self.assertTrue(all("--candidate" not in job["argv"] for job in jobs))
        self.assertTrue(all("/runs/fresh_value/confirmation/" in job["signature"] for job in jobs))
        self.assertTrue(all(job["priority"] == "high" for job in jobs))
        self.assertEqual(len(current["pending"]), 12)

    def test_existing_receipt_reused_without_duplicate_native_dispatch(self):
        L.advance()
        L.advance()
        self.infra.submit.assert_called_once()

    def test_analysis_waits_for_all_twelve(self):
        cell = L.cells(PROTOCOL)[0]
        self.outputs[cell["relative"] + "/summary.json"] = dict(status="complete")
        current = L.advance()
        self.assertEqual(current["completed"], 1)
        self.assertEqual(len(current["pending"]), 11)
        self.assertEqual(len(self.submissions), 1)

    def test_summary_wait_returns_current_twelve_completed_and_empty_pending(self):
        self.native_complete()
        current = L.advance()
        self.assertEqual(current["stage"], "fresh_value_confirmation_summary")
        self.assertEqual(current["completed"], 12)
        self.assertEqual(current["pending"], [])
        self.assertTrue(current["all_terminal"])
        self.assertFalse(current["done"])
        self.assertEqual(self.submissions[1]["jobs"][0]["argv"],
                         ["--out", str(self.root / "results/confirmation.json")])

    def test_confirmed_result_completes_batch_but_never_global_goal(self):
        self.finish()
        current = L.advance()
        self.assertEqual(current["stage"], "confirmation_gain_requires_root_review")
        self.assertTrue(current["done"])
        self.assertTrue(current["confirmed_candidate"])
        self.assertTrue(current["requires_agent_review"])
        self.assertFalse(current["goal_achieved"])
        self.assertEqual(current["completed"], 12)
        self.assertEqual(current["pending"], [])
        self.assertEqual(current["failures"], [])

    def test_failed_result_also_completes_batch_without_more_seeds(self):
        self.finish(passed=False)
        first = L.advance()
        count = self.infra.submit.call_count
        second = L.advance()
        self.assertEqual(first["stage"], "confirmation_candidate_closed")
        self.assertTrue(first["done"])
        self.assertFalse(first["confirmed_candidate"])
        self.assertFalse(first["goal_achieved"])
        self.assertEqual(second["completed"], 12)
        self.assertEqual(self.infra.submit.call_count, count)

    def test_completed_technical_failure_is_distinct_and_stops_batch(self):
        self.finish(integrity=False)
        current = L.advance()
        self.assertEqual(current["stage"], "confirmation_technical_attention")
        self.assertEqual(current["performance_verdict"], "not_evaluated_incomplete")
        self.assertTrue(current["done"])
        self.assertFalse(current["goal_achieved"])

    def test_missing_boolean_confirmation_gate_is_error(self):
        self.finish()
        self.outputs["results/confirmation.json"]["decision"].pop("confirmed_control_improvement")
        with self.assertRaisesRegex(ValueError, "boolean confirmed_control_improvement"):
            L.advance()

    def test_read_lock_and_ssh_recover_without_duplicate_submission(self):
        for error in (RuntimeError("waiting for scheduler state lock"),
                      RuntimeError("ssh: connect to host node001: Connection timed out")):
            self.infra.status.side_effect = error
            self.assertEqual(L.advance()["stage"], "waiting_transient_confirmation_read")
        self.infra.status.side_effect = lambda ids: [dict(id=key, status="running") for key in ids]
        self.assertEqual(L.advance()["stage"], "fresh_value_confirmation_native")
        self.infra.submit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
