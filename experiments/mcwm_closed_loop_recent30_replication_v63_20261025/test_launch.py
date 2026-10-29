"""Offline independent replication dispatch and review gates."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("v63_launch_tested", HERE / "launch.py")
L = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(L)
OBSERVER = L.module("v63_test_observer", L.V60 / "supervisor.py")
WATCH = L.module("v63_test_existing_watch", L.V62 / "launch.py")
PROTOCOL = dict(name="v63-test", confirmation_seeds=[6203, 6204, 6205],
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

    def finish(self, *, passed=True, integrity=True):
        for cell in L.cells(PROTOCOL):
            self.outputs[cell["relative"] + "/summary.json"] = dict(status="complete")
        self.outputs["results/replication.json"] = dict(status="complete" if integrity else "integrity_failed",
            integrity=dict(all_pass=integrity), decision=dict(confirmed_control_improvement=passed,
                ordinary_dripping_goal_achieved=False))

    def test_execution_ready_required(self):
        self.put(self.root / "execution_ready.json", dict(ready=False))
        self.assertEqual(L.advance()["stage"], "preparing_recent30_replication")
        self.infra.submit.assert_not_called()

    def test_exact_twelve_new_days_with_frozen_recent30_runner_without_old_promotion_gate(self):
        current = L.advance()
        jobs = self.submissions[0]["jobs"]
        self.assertEqual(len(jobs), 12)
        self.assertEqual({Path(job["runner"]) for job in jobs}, {L.V60 / "control.py"})
        self.assertEqual({job["argv"][3] for job in jobs}, {6203, 6204, 6205})
        self.assertEqual({job["argv"][7] for job in jobs}, {"fresh", "no_fresh"})
        self.assertTrue(all(job["argv"][:2] == ["--candidate", "recent30"] for job in jobs))
        self.assertTrue(all(job["priority"] == "high" for job in jobs))
        self.assertEqual(current["stage"], "recent30_replication_native")
        self.assertFalse(current["goal_achieved"])

    def test_native_receipt_reused_after_repeated_advance(self):
        L.advance()
        L.advance()
        self.infra.submit.assert_called_once()

    def test_partial_completion_waits_for_all_twelve_no_early_summary(self):
        cell = L.cells(PROTOCOL)[0]
        self.outputs[cell["relative"] + "/summary.json"] = dict(status="complete")
        current = L.advance()
        self.assertEqual(current["completed"], 1)
        self.assertEqual(len(current["pending"]), 11)
        self.assertEqual(len(self.submissions), 1)

    def test_all_completed_submits_one_summary_with_out_only(self):
        self.finish()
        current = L.advance()
        summary = self.submissions[1]
        self.assertEqual(summary["batch"], "recent30_confirmation_analysis")
        self.assertEqual(len(summary["jobs"]), 1)
        self.assertEqual(summary["jobs"][0]["argv"], ["--out", str(self.root / "results/replication.json")])
        self.assertEqual(current["stage"], "replication_gain_requires_root_review")
        self.assertTrue(current["confirmed_candidate"])
        self.assertFalse(current["goal_achieved"])
        self.assertFalse(current["done"])
        self.assertTrue(current["requires_agent_review"])

    def test_failed_replication_closes_without_adding_seeds_or_submissions(self):
        self.finish(passed=False)
        first = L.advance()
        count = self.infra.submit.call_count
        second = L.advance()
        self.assertEqual(first["stage"], "replication_candidate_closed")
        self.assertFalse(first["confirmed_candidate"])
        self.assertFalse(first["goal_achieved"])
        self.assertEqual(second["stage"], first["stage"])
        self.assertEqual(self.infra.submit.call_count, count)

    def test_technical_failure_never_becomes_scientific_failure(self):
        self.finish(integrity=False)
        current = L.advance()
        self.assertEqual(current["stage"], "replication_technical_attention")
        self.assertEqual(current["performance_verdict"], "not_evaluated_incomplete")
        self.assertFalse(current["goal_achieved"])

    def test_missing_boolean_confirmation_gate_is_an_error(self):
        self.finish()
        self.outputs["results/replication.json"]["decision"].pop("confirmed_control_improvement")
        with self.assertRaisesRegex(ValueError, "boolean confirmed_control_improvement"):
            L.advance()

    def test_read_lock_and_ssh_failures_recover_without_duplicate_native(self):
        for error in (RuntimeError("waiting for scheduler state lock"),
                      RuntimeError("ssh: connect to host node001: Connection timed out")):
            self.infra.status.side_effect = error
            self.assertEqual(L.advance()["stage"], "waiting_transient_replication_read")
        self.infra.status.side_effect = lambda ids: [dict(id=key, status="running") for key in ids]
        self.assertEqual(L.advance()["stage"], "recent30_replication_native")
        self.infra.submit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
