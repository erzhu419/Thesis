"""Offline watcher recovery and uncertainty handling; no server calls."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("v62_continue_tested", HERE / "continue_control.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "execution_ready.json").write_text(json.dumps(dict(ready=True)))
        self.advance = mock.Mock(return_value=dict(stage="line_cover_control_development",
                                                  done=False, goal_achieved=False))
        self.continuation = C.Continuation(self.root, advance=self.advance)

    def test_execution_gate_blocks_dispatch_when_code_not_ready(self):
        (self.root / "execution_ready.json").unlink()
        self.assertEqual(self.continuation.step()["stage"], "preparing_line_cover_control")
        self.advance.assert_not_called()

    def test_existing_receipts_are_advanced_on_restart(self):
        first = self.continuation.run()
        second = C.Continuation(self.root, advance=self.advance).run()
        self.assertEqual(first["stage"], second["stage"])
        self.assertEqual(self.advance.call_count, 2)

    def test_read_only_transient_recovery_keeps_watcher_running(self):
        self.advance.side_effect = [dict(stage="waiting_transient_line_cover_read", done=False,
            goal_achieved=False, message="scheduler lock"), dict(stage="line_cover_control_development",
            done=False, goal_achieved=False)]
        with mock.patch.object(C.time, "sleep", side_effect=[None, StopIteration]) as sleep:
            with self.assertRaises(StopIteration):
                self.continuation.run(watch=True)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [60, 30])
        self.assertNotIn("advance_uncertain", self.continuation.state)

    def test_unknown_submission_is_not_retried_after_restart(self):
        self.advance.side_effect = subprocess.TimeoutExpired(["submit-jsonl"], 180)
        self.assertEqual(self.continuation.run()["stage"], "attention")
        current = C.Continuation(self.root, advance=self.advance).run()
        self.assertEqual(current["stage"], "attention")
        self.advance.assert_called_once_with()
        self.assertTrue(self.continuation.state["advance_uncertain"])

    def test_gain_is_a_root_review_request_not_automatic_goal_completion(self):
        self.advance.return_value = dict(stage="confirmed", done=True, goal_achieved=True)
        current = self.continuation.step()
        self.assertTrue(current["requires_agent_review"])
        self.assertFalse(current["done"])
        self.assertTrue(current["campaign_done"])

    def test_scientific_closure_persists_without_claiming_success(self):
        self.advance.return_value = dict(stage="line_cover_candidate_closed_requires_next_control",
                                        done=False, goal_achieved=False, requires_agent_review=True)
        current = self.continuation.run()
        self.assertFalse(current["goal_achieved"])
        self.assertEqual(json.loads((self.root / "continue_control_state.json").read_text())["stage"], current["stage"])

    def test_oversized_state_is_rejected(self):
        path = self.root / "too_large.json"
        path.write_text(" " * 48001)
        with self.assertRaisesRegex(ValueError, "48000"):
            C.read_json(path)


if __name__ == "__main__":
    unittest.main()
