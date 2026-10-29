"""Receipt reuse, actual first-pair admission, and fixed eight-arm development."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import subprocess

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("v70_launch_tested", HERE / "launch.py")
L = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(L)
OBSERVER = L.module("v70_test_observer", L.V60 / "supervisor.py")
WATCH = L.module("v70_test_existing_watch", L.V62 / "launch.py")
PROTOCOL = dict(name="v70-test", development_seeds=[6201, 6203],
    scenarios=["stable", "local_speed_drop"], native_preflight=dict(seed=6201, scenario="stable",
        reuse_full_day=True, feedback_slots=5, policy_updates=8, critic_updates=8,
        value_updates=8, actor_updates=8, alpha_updates=8))


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
        patcher.start(); self.addCleanup(patcher.stop)
        patcher = mock.patch.object(L, "module", side_effect=lambda name, path:
            OBSERVER if path.name == "supervisor.py" else WATCH if path == L.V62 / "launch.py" else self.infra)
        patcher.start(); self.addCleanup(patcher.stop)

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def task(self, label, runner, argv):
        return dict(signature=PROTOCOL["name"] + "/" + label, runner=str(runner), argv=argv, priority="high")

    def submit(self, jobs, batch):
        path = self.root / (batch + "_submission.json")
        if path.exists(): return json.loads(path.read_text())
        value = dict(submitted=[dict(id="t_" + str(len(self.submissions)) + "_" + str(index),
                                    signature=row["signature"]) for index, row in enumerate(jobs)])
        self.submissions.append(dict(batch=batch, jobs=jobs))
        self.put(path, value)
        return value

    def preflight_ready(self):
        for cell in L.cells(PROTOCOL)[:2]:
            self.outputs[cell["relative"] + "/preflight.json"] = dict(technical_gate_pass=True,
                seed=cell["seed"], scenario=cell["scenario"], arm=cell["arm"], feedback_slots=5,
                policy_updates=8, critic_updates=8, value_updates=8, actor_updates=8, alpha_updates=8,
                packet_schedule=[8, 8, 8, 8], initializer=dict(updates=50000),
                first_feedback=dict(time=9000), execution_contract=dict(F="unchanged"))

    def native_complete(self):
        for cell in L.cells(PROTOCOL):
            self.outputs[cell["relative"] + "/summary.json"] = dict(status="complete")

    def finish(self, *, passed=True, integrity=True):
        self.preflight_ready(); self.native_complete()
        self.outputs["results/development.json"] = dict(status="complete" if integrity else "integrity_failed",
            integrity=dict(all_pass=integrity), decision=dict(advance_to_confirmation=passed,
                ordinary_dripping_goal_achieved=False))
        self.outputs["results/policy_drift.json"] = dict(native_evidence=[])
        self.outputs["results/recency.json"] = dict(native_evidence=[])

    def test_ready_gate_prevents_submission(self):
        self.put(self.root / "execution_ready.json", dict(ready=False))
        self.assertEqual(L.advance()["stage"], "preparing_executed_cql_development")
        self.infra.submit.assert_not_called()

    def test_first_pair_runs_full_days_and_receipt_is_reused(self):
        first = L.advance(); L.advance()
        self.assertEqual(first["stage"], "executed_cql_native_preflight")
        self.infra.submit.assert_called_once()
        jobs = self.submissions[0]["jobs"]
        self.assertEqual(len(jobs), 2)
        self.assertTrue(all(job["argv"][1] == 6201 and job["argv"][3] == "stable" for job in jobs))
        self.assertTrue(all("--stop" not in job["argv"] for job in jobs))
        self.assertEqual({Path(job["runner"]) for job in jobs}, {self.root / "control.py"})

    def test_preflight_allows_exactly_six_other_days_without_repeating_first_pair(self):
        self.preflight_ready()
        current = L.advance(); L.advance()
        jobs = [job for batch in self.submissions for job in batch["jobs"]]
        self.assertEqual([len(batch["jobs"]) for batch in self.submissions], [2, 6])
        self.assertEqual(len({job["signature"] for job in jobs}), 8)
        self.assertEqual({job["argv"][1] for job in jobs}, {6201, 6203})
        self.assertEqual({job["argv"][5] for job in jobs}, {"fresh", "no_fresh"})
        self.assertTrue(all(job["priority"] == "high" for job in jobs))
        self.assertTrue(current["preflight"]["all_pass"])
        self.assertEqual(len(current["pending"]), 8)

    def test_one_preflight_arm_never_opens_remaining_batch(self):
        self.preflight_ready()
        self.outputs.pop(L.cells(PROTOCOL)[1]["relative"] + "/preflight.json")
        self.assertFalse(L.advance()["done"])
        self.assertEqual(len(self.submissions), 1)

    def test_failed_or_mismatched_preflight_stops_expansion_without_cancelling_running_days(self):
        for field, value in (("technical_gate_pass", False), ("initializer", dict(updates=1)),
                             ("first_feedback", dict(time=9001)), ("execution_contract", {}),
                             ("feedback_slots", 6), ("seed", 6203), ("policy_updates", 1),
                             ("critic_updates", 7), ("value_updates", 7), ("actor_updates", 1),
                             ("alpha_updates", 1), ("packet_schedule", [1, 1, 1, 1])):
            self.preflight_ready()
            self.outputs[L.cells(PROTOCOL)[1]["relative"] + "/preflight.json"][field] = value
            result = L.advance()
            self.assertTrue(result["done"])
            self.assertTrue(result["native_tasks_may_still_run"])
            self.assertFalse(result["preflight"]["all_pass"])
            self.assertEqual(len(self.submissions), 1)

    def test_terminal_first_pair_without_preflight_is_attention_without_rerun(self):
        for cell in L.cells(PROTOCOL)[:2]:
            self.outputs[cell["relative"] + "/summary.json"] = dict(status="complete")
        current = L.advance()
        self.assertTrue(current["done"])
        self.assertEqual(current["stage"], "executed_cql_preflight_attention")
        self.assertEqual(len(self.submissions), 1)

    def test_summary_waits_for_all_eight_and_has_one_receipt(self):
        self.preflight_ready()
        for cell in L.cells(PROTOCOL)[:-1]:
            self.outputs[cell["relative"] + "/summary.json"] = dict(status="complete")
        self.assertEqual(L.advance()["completed"], 7)
        self.assertEqual(len(self.submissions), 2)
        self.native_complete()
        current = L.advance(); L.advance()
        self.assertEqual(current["stage"], "executed_cql_development_summary")
        self.assertEqual(current["completed"], 8)
        self.assertEqual(current["pending"], [])
        self.assertEqual(len(self.submissions), 3)
        self.assertEqual(self.submissions[-1]["jobs"][0]["argv"],
                         ["--out", str(self.root / "results/development.json")])

    def test_both_scientific_outcomes_close_batch_without_automatic_confirmation(self):
        for passed in (True, False):
            self.finish(passed=passed)
            result = L.advance()
            self.assertTrue(result["done"])
            self.assertEqual(result["candidate_passed"], passed)
            self.assertFalse(result["goal_achieved"])
            self.assertEqual(result["completed"], 8)
        self.assertEqual(len(self.submissions), 3)

    def test_integrity_failure_is_not_a_cost_verdict(self):
        self.finish(integrity=False)
        result = L.advance()
        self.assertTrue(result["done"])
        self.assertEqual(result["performance_verdict"], "not_evaluated_incomplete")

    def test_split_summary_requires_all_three_small_files(self):
        self.preflight_ready(); self.native_complete()
        self.outputs["results/development.json"] = dict(status="complete", integrity=dict(all_pass=True),
            decision=dict(advance_to_confirmation=False))
        first = L.advance()
        self.assertFalse(first["done"])
        self.assertEqual(first["missing_summary"], ["results/policy_drift.json", "results/recency.json"])
        self.assertNotIn(mock.call("results/development.json"), self.infra.collect.call_args_list)
        self.outputs["results/policy_drift.json"] = dict(native_evidence=[])
        self.outputs["results/recency.json"] = dict(native_evidence=[])
        self.assertTrue(L.advance()["done"])
        self.infra.collect.assert_any_call("results/development.json")
        self.infra.collect.assert_any_call("results/policy_drift.json")
        self.infra.collect.assert_any_call("results/recency.json")
        self.assertEqual(len(self.submissions), 3)

    def test_recency_alone_missing_waits_then_terminal_stops_without_collecting_partial_files(self):
        self.finish()
        self.outputs.pop("results/recency.json")
        pending = L.advance()
        self.assertFalse(pending["done"])
        self.assertEqual(pending["missing_summary"], ["results/recency.json"])
        for path in ("results/development.json", "results/policy_drift.json"):
            self.assertNotIn(mock.call(path), self.infra.collect.call_args_list)
        self.infra.status.side_effect = lambda ids: [dict(id=key, status="done") for key in ids]
        stopped = L.advance()
        self.assertTrue(stopped["done"])
        self.assertEqual(stopped["stage"], "executed_cql_summary_attention")
        self.assertEqual(stopped["missing_summary"], ["results/recency.json"])
        self.assertEqual(len(self.submissions), 3)

    def test_terminal_summary_failure_or_missing_output_stops_without_replacement(self):
        self.preflight_ready(); self.native_complete()
        for status in ("failed", "cancelled", "done"):
            self.infra.status.side_effect = lambda ids, status=status: [dict(id=key, status=status) for key in ids]
            result = L.advance()
            self.assertTrue(result["done"])
            self.assertTrue(result["requires_agent_review"])
            self.assertEqual(result["stage"], "executed_cql_summary_attention")
            self.assertEqual(result["performance_verdict"], "not_evaluated_incomplete")
        self.assertEqual(len(self.submissions), 3)

    def test_failed_summary_with_existing_files_is_still_attention(self):
        self.finish()
        self.infra.status.side_effect = lambda ids: [dict(id=key, status="failed") for key in ids]
        result = L.advance()
        self.assertTrue(result["done"])
        self.assertEqual(result["stage"], "executed_cql_summary_attention")
        self.assertEqual(result["missing_summary"], [])

    def test_read_lock_and_connection_errors_recover_without_duplicate_submission(self):
        for error in (RuntimeError("waiting for scheduler state lock"),
                      RuntimeError("ssh: connect to host node001: Connection timed out")):
            self.infra.status.side_effect = error
            self.assertEqual(L.advance()["stage"], "waiting_transient_development_read")
        self.infra.status.side_effect = lambda ids: [dict(id=key, status="running") for key in ids]
        self.assertEqual(L.advance()["stage"], "executed_cql_native_preflight")
        self.infra.submit.assert_called_once()


class InfrastructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.infra = L.module("v70_receipted_infra_test", HERE / "infra.py")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        patcher = mock.patch.object(self.infra, "HERE", self.root)
        patcher.start(); self.addCleanup(patcher.stop)

    def test_known_submission_receipt_is_reused_without_another_scheduler_call(self):
        receipt = dict(submitted=[dict(id="t_existing", signature="v70/first")])
        (self.root / "first_submission.json").write_text(json.dumps(receipt))
        with mock.patch.object(self.infra.subprocess, "run") as run:
            self.assertEqual(self.infra.submit([dict(signature="v70/first")], "first"), receipt)
        run.assert_not_called()

    def test_unresolved_attempt_prevents_blind_resubmission(self):
        (self.root / "first_submission_attempt.json").write_text("{}")
        with mock.patch.object(self.infra.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "Unresolved submission"):
                self.infra.submit([dict(signature="v70/first")], "first")
        run.assert_not_called()

    def test_submit_waits_for_real_queue_lock_and_dispatches_only_receipted_id(self):
        receipt = dict(submitted=[dict(id="t_v70", signature="v70/first")])
        outputs = [subprocess.CompletedProcess([], 0, json.dumps(receipt), ""),
                   subprocess.CompletedProcess([], 0, "dispatched", "")]
        with mock.patch.object(self.infra.subprocess, "run", side_effect=outputs) as run:
            self.assertEqual(self.infra.submit([dict(signature="v70/first")], "first"), receipt)
        submit = run.call_args_list[0]
        self.assertEqual(submit.args[0][-2:], ["--lock-timeout", "300"])
        self.assertEqual(submit.kwargs["timeout"], 360)
        dispatch = run.call_args_list[1]
        self.assertEqual(dispatch.args[0][-2:], ["--task-id", "t_v70"])
        self.assertIn("--no-bulk-window", dispatch.args[0])
        self.assertEqual(dispatch.args[0][dispatch.args[0].index("--lock-timeout")+1], "300")
        self.assertEqual(dispatch.kwargs["timeout"], 360)
        self.assertTrue((self.root / "first_submission.json").exists())


    def test_dispatch_failure_retains_receipt_and_next_submit_does_not_resubmit(self):
        receipt = dict(submitted=[dict(id="t_v70", signature="v70/first")])
        outputs = [subprocess.CompletedProcess([], 0, json.dumps(receipt), ""),
                   subprocess.CompletedProcess([], 2, "", "lock timeout")]
        with mock.patch.object(self.infra.subprocess, "run", side_effect=outputs) as run:
            with self.assertRaisesRegex(RuntimeError, "Known submitted tasks need dispatch review"):
                self.infra.submit([dict(signature="v70/first")], "first")
            self.assertEqual(self.infra.submit([dict(signature="v70/first")], "first"), receipt)
        self.assertEqual(run.call_count, 2)
        self.assertTrue((self.root / "first_submission.json").exists())
        self.assertTrue((self.root / "first_dispatch.json").exists())


if __name__ == "__main__": unittest.main()
