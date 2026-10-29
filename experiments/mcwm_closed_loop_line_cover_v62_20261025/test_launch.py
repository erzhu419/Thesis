"""Offline full-day dispatch gates, receipt reuse, and read recovery."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("v62_launch_tested", HERE / "launch.py")
L = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(L)
OBSERVER = L.module("v62_launch_test_observer", L.V60 / "supervisor.py")
PROTOCOL = dict(name="v62-test", development_seeds=[6201, 6202],
                confirmation_seeds=[6203, 6204, 6205], scenarios=["stable", "local_speed_drop"])


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.put(self.root / "protocol.json", PROTOCOL)
        self.put(self.root / "execution_ready.json", dict(ready=True))
        self.outputs, self.submissions = {}, []
        self.infra = SimpleNamespace(remote=lambda path: str(path), task=self.task,
            presence=mock.Mock(side_effect=lambda paths: {path: path in self.outputs for path in paths}),
            collect=mock.Mock(side_effect=lambda path: self.outputs[path]),
            status=mock.Mock(side_effect=lambda ids: [dict(id=key, status="running") for key in ids]),
            submit=mock.Mock(side_effect=self.submit))
        patcher = mock.patch.object(L, "HERE", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(L, "module", side_effect=lambda name, path:
            OBSERVER if path.name == "supervisor.py" else self.infra)
        patcher.start()
        self.addCleanup(patcher.stop)

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def task(self, label, runner, argv):
        return dict(signature=PROTOCOL["name"] + "/" + label, runner=str(runner), argv=argv,
                    priority="high")

    def submit(self, jobs, batch):
        path = self.root / (batch + "_submission.json")
        if path.exists():
            return json.loads(path.read_text())
        value = dict(submitted=[dict(id="t_" + str(len(self.submissions)) + "_" + str(index),
                                    signature=row["signature"]) for index, row in enumerate(jobs)])
        self.submissions.append(dict(batch=batch, jobs=jobs))
        self.put(path, value)
        return value

    def finish(self, stage, *, passed=True, integrity=True):
        for cell in L.cells(PROTOCOL, stage):
            self.outputs[cell["relative"] + "/summary.json"] = dict(status="complete")
        self.outputs["results/line_cover/" + stage + ".json"] = dict(
            status="complete" if integrity else "integrity_failed", integrity=dict(all_pass=integrity),
            decision=dict(advance_to_confirmation=passed if stage == "development" else False,
                          confirmed_control_improvement=passed if stage == "confirmation" else False,
                          ordinary_dripping_goal_ready_for_root_review=passed if stage == "confirmation" else False,
                          ordinary_dripping_goal_achieved=False))

    def test_not_ready_never_submits(self):
        self.put(self.root / "execution_ready.json", dict(ready=False))
        current = L.advance()
        self.assertEqual(current["stage"], "preparing_line_cover_control")
        self.infra.submit.assert_not_called()

    def test_development_eight_native_jobs_without_reused_third_baseline_rerun(self):
        current = L.advance()
        jobs = self.submissions[0]["jobs"]
        self.assertEqual(len(jobs), 8)
        self.assertEqual({Path(job["runner"]) for job in jobs}, {self.root / "control.py"})
        self.assertEqual({job["argv"][5] for job in jobs}, {"fresh", "no_fresh"})
        self.assertTrue(all(job["priority"] == "high" for job in jobs))
        self.assertEqual(current["stage"], "line_cover_control_development")

    def test_restart_reuses_native_receipt_without_duplicate_dispatch(self):
        first, second = L.advance(), L.advance()
        self.assertEqual(first["stage"], second["stage"])
        self.infra.submit.assert_called_once()

    def test_partial_native_completion_keeps_other_real_days_running(self):
        cell = L.cells(PROTOCOL, "development")[0]
        self.outputs[cell["relative"] + "/summary.json"] = dict(status="complete")
        current = L.advance()
        self.assertEqual(current["completed"], 1)
        self.assertEqual(len(current["pending"]), 7)
        self.assertEqual([row["batch"] for row in self.submissions], ["line_cover_development"])

    def test_scientific_gate_failure_closes_without_confirmation(self):
        self.finish("development", passed=False)
        current = L.advance()
        self.assertEqual(current["stage"], "line_cover_candidate_closed_requires_next_control")
        self.assertTrue(current["requires_agent_review"])
        self.assertFalse(current["done"])
        self.assertEqual([row["batch"] for row in self.submissions],
                         ["line_cover_development", "line_cover_development_analysis"])

    def test_technical_failure_is_not_scientific_failure(self):
        self.finish("development", integrity=False)
        current = L.advance()
        self.assertEqual(current["stage"], "line_cover_technical_attention")
        self.assertEqual(current["performance_verdict"], "not_evaluated_incomplete")
        self.assertFalse(current["goal_achieved"])
        self.assertFalse(any(row["batch"] == "line_cover_confirmation" for row in self.submissions))

    def test_development_pass_dispatches_all_eighteen_fixed_confirmation_days(self):
        self.finish("development")
        current = L.advance()
        self.assertEqual(current["stage"], "line_cover_control_confirmation")
        jobs = next(row["jobs"] for row in self.submissions if row["batch"] == "line_cover_confirmation")
        self.assertEqual(len(jobs), 18)
        original = [job for job in jobs if job["signature"].endswith("/original_no_fresh")]
        self.assertEqual(len(original), 6)
        self.assertTrue(all(Path(job["runner"]) == L.V60 / "control.py" for job in original))
        self.assertTrue(all(job["argv"][5] == "no_fresh" and job["argv"][-2:] == ["--candidate", "recent30"]
                            for job in original))
        self.assertTrue(all(job["priority"] == "high" for job in jobs))
        self.assertEqual({job["argv"][1] for job in jobs}, {6203, 6204, 6205})

    def test_confirmation_gain_requires_root_review_and_never_auto_completes(self):
        self.finish("development")
        self.finish("confirmation")
        current = L.advance()
        self.assertEqual(current["stage"], "line_cover_gain_requires_root_review")
        self.assertFalse(current["goal_achieved"])
        self.assertTrue(current["confirmed_candidate"])
        self.assertTrue(current["requires_agent_review"])
        self.assertFalse(current["done"])

    def test_confirmation_failure_stays_closed_without_resampling(self):
        self.finish("development")
        self.finish("confirmation", passed=False)
        first = L.advance()
        count = self.infra.submit.call_count
        second = L.advance()
        self.assertEqual(first["stage"], "line_cover_candidate_closed_after_confirmation")
        self.assertFalse(first["confirmed_candidate"])
        self.assertFalse(first["goal_achieved"])
        self.assertEqual(second["stage"], first["stage"])
        self.assertEqual(self.infra.submit.call_count, count)

    def test_confirmation_requires_explicit_boolean_control_improvement(self):
        self.finish("development")
        self.finish("confirmation")
        decision = self.outputs["results/line_cover/confirmation.json"]["decision"]
        decision.pop("confirmed_control_improvement")
        with self.assertRaisesRegex(ValueError, "confirmed_control_improvement"):
            L.advance()

    def test_source_module_read_lock_recovers_without_another_submission(self):
        self.infra.status.side_effect = RuntimeError("waiting for scheduler state lock")
        current = L.advance()
        self.assertEqual(current["stage"], "waiting_transient_line_cover_read")
        self.infra.status.side_effect = lambda ids: [dict(id=key, status="running") for key in ids]
        self.assertEqual(L.advance()["stage"], "line_cover_control_development")
        self.infra.submit.assert_called_once()

    def test_terminal_status_rechecks_resident_output_before_declaring_failure(self):
        cell = L.cells(PROTOCOL, "development")[0]
        path = cell["relative"] + "/summary.json"
        self.infra.presence.side_effect = [{}, {path: True}]
        self.outputs[path] = dict(status="complete")
        self.infra.status.side_effect = lambda ids: [dict(id=key, status="done") for key in ids]
        receipt = dict(submitted=[dict(id="t1", signature=PROTOCOL["name"] + "/" + cell["relative"])])
        current = L.watch(receipt, [cell], self.infra, OBSERVER.read_only)
        self.assertEqual(current["completed"], 1)
        self.assertFalse(current["failures"])

    def test_missing_existing_task_never_creates_replacement(self):
        self.infra.status.return_value = []
        self.infra.status.side_effect = None
        with self.assertRaisesRegex(ValueError, "omitted an existing task"):
            L.advance()
        self.infra.submit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
