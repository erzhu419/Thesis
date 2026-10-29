"""Offline fresh-value actual-control dispatch and cost gates."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("v64_launch_tested", HERE / "launch.py")
L = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(L)
OBSERVER = L.module("v64_test_observer", L.V60 / "supervisor.py")
WATCH = L.module("v64_test_existing_watch", L.V62 / "launch.py")
PROTOCOL = dict(name="v64-test", development_seeds=[6201, 6202], confirmation_seeds=[6203, 6204, 6205],
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

    def finish(self, stage, *, passed=True, integrity=True):
        for cell in L.cells(PROTOCOL, stage):
            self.outputs[cell["relative"] + "/summary.json"] = dict(status="complete")
        self.outputs["results/fresh_value/" + stage + ".json"] = dict(
            status="complete" if integrity else "integrity_failed", integrity=dict(all_pass=integrity),
            decision=dict(advance_to_confirmation=passed if stage == "development" else False,
                          confirmed_control_improvement=passed if stage == "confirmation" else False,
                          ordinary_dripping_goal_achieved=False))

    def test_ready_gate_prevents_submission(self):
        self.put(self.root / "execution_ready.json", dict(ready=False))
        self.assertEqual(L.advance()["stage"], "preparing_fresh_value_control")
        self.infra.submit.assert_not_called()

    def test_development_dispatches_only_eight_local_two_arm_days_without_candidate_cli(self):
        current = L.advance()
        jobs = self.submissions[0]["jobs"]
        self.assertEqual(len(jobs), 8)
        self.assertEqual({Path(job["runner"]) for job in jobs}, {self.root / "control.py"})
        self.assertTrue(all("--candidate" not in job["argv"] for job in jobs))
        self.assertEqual({job["argv"][5] for job in jobs}, {"fresh", "no_fresh"})
        self.assertTrue(all(job["priority"] == "high" for job in jobs))
        self.assertEqual(current["stage"], "fresh_value_control_development")

    def test_existing_receipt_prevents_duplicate_native_submission(self):
        L.advance()
        L.advance()
        self.infra.submit.assert_called_once()

    def test_failed_development_never_promotes(self):
        self.finish("development", passed=False)
        current = L.advance()
        self.assertEqual(current["stage"], "fresh_value_candidate_closed_requires_next_control")
        self.assertFalse(current["goal_achieved"])
        self.assertFalse(any(row["batch"] == "fresh_value_confirmation" for row in self.submissions))

    def test_technical_failure_does_not_become_performance_failure(self):
        self.finish("development", integrity=False)
        current = L.advance()
        self.assertEqual(current["stage"], "fresh_value_technical_attention")
        self.assertEqual(current["performance_verdict"], "not_evaluated_incomplete")

    def test_passed_development_dispatches_twelve_confirmation_days_without_third_arm(self):
        self.finish("development")
        current = L.advance()
        jobs = next(row["jobs"] for row in self.submissions if row["batch"] == "fresh_value_confirmation")
        self.assertEqual(len(jobs), 12)
        self.assertEqual({job["argv"][1] for job in jobs}, {6203, 6204, 6205})
        self.assertEqual({job["argv"][5] for job in jobs}, {"fresh", "no_fresh"})
        self.assertEqual({Path(job["runner"]) for job in jobs}, {self.root / "control.py"})
        self.assertEqual(current["stage"], "fresh_value_control_confirmation")

    def test_confirmation_success_is_root_review_not_goal_completion(self):
        self.finish("development")
        self.finish("confirmation")
        current = L.advance()
        self.assertEqual(current["stage"], "fresh_value_gain_requires_root_review")
        self.assertTrue(current["confirmed_candidate"])
        self.assertTrue(current["requires_agent_review"])
        self.assertFalse(current["goal_achieved"])
        self.assertFalse(current["done"])

    def test_confirmation_failure_is_closed_and_does_not_resubmit(self):
        self.finish("development")
        self.finish("confirmation", passed=False)
        first = L.advance()
        count = self.infra.submit.call_count
        second = L.advance()
        self.assertEqual(first["stage"], "fresh_value_candidate_closed_after_confirmation")
        self.assertEqual(first["stage"], second["stage"])
        self.assertFalse(first["confirmed_candidate"])
        self.assertFalse(first["goal_achieved"])
        self.assertEqual(count, self.infra.submit.call_count)

    def test_missing_boolean_confirmation_gate_is_error(self):
        self.finish("development")
        self.finish("confirmation")
        self.outputs["results/fresh_value/confirmation.json"]["decision"].pop("confirmed_control_improvement")
        with self.assertRaisesRegex(ValueError, "confirmed_control_improvement"):
            L.advance()

    def test_read_lock_recovers_using_existing_native_receipt(self):
        self.infra.status.side_effect = RuntimeError("waiting for scheduler state lock")
        self.assertEqual(L.advance()["stage"], "waiting_transient_fresh_value_read")
        self.infra.status.side_effect = lambda ids: [dict(id=key, status="running") for key in ids]
        self.assertEqual(L.advance()["stage"], "fresh_value_control_development")
        self.infra.submit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
