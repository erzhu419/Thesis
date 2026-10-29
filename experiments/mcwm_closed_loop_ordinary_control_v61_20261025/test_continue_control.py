"""Offline continuation gates; no scheduler writes or server access."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("v61_continue_tested", HERE / "continue_control.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)
OBSERVER = C.load("v61_source_observer_tested", C.V60 / "supervisor.py")


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root, self.source = Path(self.tmp.name) / "v61", Path(self.tmp.name) / "v60"
        self.root.mkdir()
        self.source.mkdir()
        self.infra = SimpleNamespace(status=mock.Mock(side_effect=lambda ids:
            [dict(id=identifier, status="done") for identifier in ids]),
            presence=mock.Mock(return_value={}), collect=mock.Mock())
        self.advance = mock.Mock(return_value=dict(stage="ordinary_development", done=False,
                                                  goal_achieved=False))
        self.pid_alive = mock.Mock(return_value=True)
        self.put(self.source / "supervisor_process.json", dict(pid=12345))
        self.put(self.source / "supervisor_state.json", dict(stage="actual_control_development"))
        self.receipt("development")
        self.put(self.root / "execution_ready.json", dict(ready=True))
        self.continuation = self.make()

    def make(self):
        return C.Continuation(self.root, self.source, infra=self.infra, advance=self.advance,
                              observer=OBSERVER, pid_alive=self.pid_alive)

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def receipt(self, stage, group="native"):
        suffix = "_analysis" if group == "analysis" else ""
        self.put(self.source / ("paired_readout_" + stage + suffix + "_submission.json"),
                 dict(submitted=[dict(id="t_" + stage + suffix)]))

    def result(self, stage, passed):
        value = dict(status="complete", candidate="paired_readout", stage=stage,
                     integrity=dict(all_pass=True, failed_checks=[]),
                     decision={"advance_to_confirmation" if stage == "development"
                               else "confirmed_control_improvement": passed})
        self.put(self.source / "results/paired_readout" / (stage + ".json"), value)
        return value

    def test_paired_development_pass_waits_confirmation_not_goal_completion(self):
        self.result("development", True)
        current = self.continuation.step()
        self.assertEqual(current["stage"], "waiting_paired_confirmation_submission")
        self.assertFalse(current["done"])
        self.assertFalse(current["goal_achieved"])
        self.advance.assert_not_called()

    def test_existing_ordinary_receipt_advances_without_waiting_for_paired_cost(self):
        self.put(self.root / "recent30_development_submission.json",
                 dict(submitted=[dict(id="t_ordinary")]))
        self.infra.status.side_effect = AssertionError("Paired status must not gate an already running ordinary trial")
        current = self.continuation.step()
        self.advance.assert_called_once_with()
        self.assertEqual(current["stage"], "ordinary_development")
        self.assertEqual(current["ordinary_admission_reason"], "predeclared_parallel_actual_control")
        self.assertEqual(current["paired_performance_verdict"], "not_used_for_ordinary_admission")
        self.assertFalse(current["done"])
        self.assertFalse(current["goal_achieved"])
        self.infra.presence.assert_not_called()

    def test_existing_ordinary_receipt_survives_restart_without_source_receipt(self):
        self.put(self.root / "recent30_development_submission.json",
                 dict(submitted=[dict(id="t_ordinary")]))
        (self.source / "paired_readout_development_submission.json").unlink()
        self.continuation.step()
        current = self.make().step()
        self.assertEqual(self.advance.call_count, 2)
        self.assertEqual(current["ordinary_admission_reason"], "predeclared_parallel_actual_control")
        self.infra.status.assert_not_called()

    def test_empty_ordinary_receipt_is_not_a_submission_handle(self):
        self.put(self.root / "recent30_development_submission.json", dict(submitted=[]))
        with self.assertRaisesRegex(C.Attention, "no task handles"):
            self.continuation.step()
        self.advance.assert_not_called()

    def test_paired_confirmation_win_still_advances_ordinary_goal(self):
        self.result("development", True)
        self.receipt("confirmation")
        self.result("confirmation", True)
        current = self.continuation.step()
        self.advance.assert_called_once_with()
        self.assertEqual(current["paired_disposition"], "confirmed_counterfactual_assisted_only")
        self.assertFalse(current["done"])
        self.assertFalse(current["goal_achieved"])

    def test_paired_scientific_fail_immediately_advances_ordinary(self):
        self.result("development", False)
        current = self.continuation.step()
        self.assertEqual(current["paired_disposition"], "closed_failed_development_gate")
        self.advance.assert_called_once_with()
        self.assertFalse(current["done"])

    def test_terminal_missing_summary_is_incomplete_not_performance_failure(self):
        self.receipt("development", "analysis")
        self.put(self.source / "results/paired_readout/development.json",
            dict(status="pending", integrity=dict(all_pass=False, failed_checks=[]),
                 missing=["external106360/stable/readout"], failures=[]))
        current = self.continuation.step()
        self.assertEqual(current["paired_disposition"], "not_admitted_incomplete")
        self.assertEqual(current["paired_performance_verdict"], "not_evaluated_incomplete")
        self.advance.assert_called_once_with()

    def test_missing_result_while_native_running_never_triggers_ordinary(self):
        self.infra.status.side_effect = lambda ids: [dict(id=key, status="running") for key in ids]
        self.put(self.source / "results/paired_readout/development.json",
            dict(status="pending", integrity=dict(all_pass=False), missing=["one"], failures=[]))
        current = self.continuation.step()
        self.assertEqual(current["stage"], "waiting_paired_development")
        self.advance.assert_not_called()

    def test_actual_integrity_failure_holds_for_root_repair(self):
        self.put(self.source / "results/paired_readout/development.json",
            dict(status="integrity_failed", integrity=dict(all_pass=False,
                 failed_checks=["prefix_exact"]), failures=[]))
        current = self.continuation.run()
        self.assertEqual(current["stage"], "attention")
        self.assertIn("No performance failure inferred", current["message"])
        self.advance.assert_not_called()

    def test_missing_execution_ready_is_preparing_not_success_or_dispatch(self):
        (self.root / "execution_ready.json").unlink()
        self.result("development", False)
        current = self.continuation.step()
        self.assertEqual(current["stage"], "preparing_ordinary_control_code")
        self.assertFalse(current["goal_achieved"])
        self.advance.assert_not_called()

    def test_ordinary_receipt_is_reused_on_watcher_restart(self):
        self.result("development", False)
        dispatches = []

        def advance():
            receipt = self.root / "development_submission.json"
            if not receipt.exists():
                dispatches.append("ordinary")
                self.put(receipt, dict(submitted=[dict(id="t_ordinary")]))
            return dict(stage="ordinary_development", done=False, goal_achieved=False)

        self.advance.side_effect = advance
        self.continuation.step()
        self.make().step()
        self.assertEqual(dispatches, ["ordinary"])
        self.assertEqual(self.advance.call_count, 2)

    def test_source_unique_read_exception_is_translated_and_recovers(self):
        self.infra.status.side_effect = RuntimeError("waiting for scheduler state lock")
        first = self.continuation.run()
        self.assertEqual(first["stage"], "waiting_transient_read")
        self.advance.assert_not_called()
        self.infra.status.side_effect = lambda ids: [dict(id=key, status="running") for key in ids]
        second = self.continuation.run()
        self.assertEqual(second["stage"], "waiting_paired_development")
        self.assertFalse(second["done"])
        self.assertFalse((self.root / "continue_control_failure.json").exists())

    def test_read_timeout_keeps_watch_alive(self):
        self.continuation.step = mock.Mock(side_effect=[C.ReadUnavailable("connection reset"),
            dict(stage="waiting_paired_development", done=False, goal_achieved=False)])
        with mock.patch.object(C.time, "sleep", side_effect=[None, StopIteration]) as sleep:
            with self.assertRaises(StopIteration):
                self.continuation.run(watch=True)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [60, 30])
        self.assertEqual(self.continuation.state["stage"], "waiting_paired_development")

    def test_dead_supervisor_does_not_restart_receipted_native_tasks(self):
        self.pid_alive.return_value = False
        self.infra.status.side_effect = lambda ids: [dict(id=key, status="running") for key in ids]
        current = self.continuation.step()
        self.pid_alive.assert_called_once_with(12345)
        self.assertEqual(current["paired_observation"]["task_counts"], dict(native=dict(running=1)))
        self.assertIn("without rerun", current["attention"])
        self.advance.assert_not_called()

    def test_unknown_submission_from_launcher_is_never_retried(self):
        self.result("development", False)
        self.advance.side_effect = subprocess.TimeoutExpired(["submit-jsonl"], 180)
        first = self.continuation.run()
        second = self.make().run()
        self.assertEqual(first["stage"], "attention")
        self.assertEqual(second["stage"], "attention")
        self.assertTrue(self.continuation.state["advance_uncertain"])
        self.advance.assert_called_once_with()

    def test_ordinary_gain_is_left_for_root_review(self):
        self.result("development", False)
        self.advance.return_value = dict(stage="confirmed_ordinary_dripping_gain", done=True,
                                        goal_achieved=True)
        current = self.continuation.step()
        self.assertEqual(current["stage"], "ordinary_gain_requires_root_review")
        self.assertTrue(current["requires_agent_review"])
        self.assertTrue(current["ordinary_campaign_done"])
        self.assertFalse(current["done"])

    def test_no_paired_receipt_never_creates_a_replacement(self):
        (self.source / "paired_readout_development_submission.json").unlink()
        with self.assertRaisesRegex(C.Attention, "never replace"):
            self.continuation.step()
        self.infra.status.assert_not_called()
        self.advance.assert_not_called()

    def test_small_summary_limit(self):
        path = self.source / "oversize.json"
        path.write_text(" " * 48001)
        with self.assertRaisesRegex(ValueError, "48000"):
            C.read_json(path)


if __name__ == "__main__":
    unittest.main()
