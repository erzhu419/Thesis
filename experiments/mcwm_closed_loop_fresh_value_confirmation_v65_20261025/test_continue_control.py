"""Offline finite-batch watcher completion and transient-read recovery."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("v65_continue_tested", HERE / "continue_control.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "execution_ready.json").write_text(json.dumps(dict(ready=True)))
        self.advance = mock.Mock(return_value=dict(stage="fresh_value_confirmation_native", done=False,
            goal_achieved=False, completed=0, pending=[dict(id="t1")], failures=[], all_terminal=False))
        self.continuation = C.Continuation(self.root, advance=self.advance)

    def test_not_ready_never_dispatches(self):
        (self.root / "execution_ready.json").unlink()
        self.assertEqual(self.continuation.step()["stage"], "preparing_fresh_value_confirmation")
        self.advance.assert_not_called()

    def test_unknown_submission_is_not_retried_after_restart(self):
        self.advance.side_effect = subprocess.TimeoutExpired(["submit-jsonl"], 180)
        self.assertEqual(self.continuation.run()["stage"], "attention")
        self.assertEqual(C.Continuation(self.root, advance=self.advance).run()["stage"], "attention")
        self.advance.assert_called_once_with()

    def test_transient_read_wait_recovers_and_exits_on_completed_batch(self):
        final = dict(stage="confirmation_candidate_closed", done=True, goal_achieved=False,
                     completed=12, pending=[], failures=[], all_terminal=True, confirmed_candidate=False)
        self.advance.side_effect = [dict(stage="waiting_transient_confirmation_read", done=False,
                                        goal_achieved=False), final]
        with mock.patch.object(C.time, "sleep") as sleep:
            current = self.continuation.run(watch=True)
        self.assertTrue(current["done"])
        sleep.assert_called_once_with(60)
        self.assertNotIn("advance_uncertain", self.continuation.state)

    def test_final_state_replaces_actual_stale_nine_and_pending_three(self):
        self.continuation.save(dict(completed=9, pending=[dict(id=key) for key in ("t10", "t11", "t12")]))
        self.advance.return_value = dict(stage="confirmation_gain_requires_root_review", done=True,
            goal_achieved=False, confirmed_candidate=True, requires_agent_review=True,
            completed=12, pending=[], failures=[], all_terminal=True)
        with mock.patch.object(C.time, "sleep") as sleep:
            current = self.continuation.run(watch=True)
        self.assertTrue(current["campaign_done"])
        self.assertFalse(current["goal_achieved"])
        self.assertEqual(self.continuation.state["completed"], 12)
        self.assertEqual(self.continuation.state["pending"], [])
        self.assertEqual(self.continuation.state["next_poll_seconds"], 0)
        sleep.assert_not_called()
        self.advance.assert_called_once_with()

    def test_confirmation_failure_stops_watch_without_claiming_goal(self):
        self.advance.return_value = dict(stage="confirmation_candidate_closed", done=True,
            goal_achieved=False, confirmed_candidate=False, completed=12, pending=[], failures=[], all_terminal=True)
        with mock.patch.object(C.time, "sleep") as sleep:
            current = self.continuation.run(watch=True)
        self.assertTrue(current["done"])
        self.assertFalse(current["goal_achieved"])
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
