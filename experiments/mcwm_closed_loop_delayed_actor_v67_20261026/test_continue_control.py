"""Offline finite-batch watcher completion and transient-read recovery."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("v67_continue_tested", HERE / "continue_control.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "execution_ready.json").write_text(json.dumps(dict(ready=True)))
        self.advance = mock.Mock(return_value=dict(stage="delayed_actor_development_native", done=False,
            goal_achieved=False, completed=0, pending=[dict(id="t1")], failures=[], all_terminal=False))
        self.continuation = C.Continuation(self.root, advance=self.advance)

    def test_not_ready_never_dispatches(self):
        (self.root / "execution_ready.json").unlink()
        self.assertEqual(self.continuation.step()["stage"], "preparing_delayed_actor_development")
        self.advance.assert_not_called()

    def test_unknown_submission_is_not_retried_after_restart(self):
        self.advance.side_effect = subprocess.TimeoutExpired(["submit-jsonl"], 180)
        with mock.patch.object(C.time, "sleep") as sleep:
            current = self.continuation.run(watch=True)
        self.assertEqual(current["stage"], "attention")
        self.assertTrue(current["done"])
        self.assertTrue(current["requires_agent_review"])
        sleep.assert_not_called()
        self.assertEqual(C.Continuation(self.root, advance=self.advance).run()["stage"], "attention")
        self.advance.assert_called_once_with()

    def test_transient_read_wait_recovers_and_exits_on_completed_batch(self):
        final = dict(stage="development_candidate_closed", done=True, goal_achieved=False,
                     completed=8, pending=[], failures=[], all_terminal=True, candidate_passed=False)
        self.advance.side_effect = [dict(stage="waiting_transient_development_read", done=False,
                                        goal_achieved=False), final]
        with mock.patch.object(C.time, "sleep") as sleep:
            current = self.continuation.run(watch=True)
        self.assertTrue(current["done"])
        sleep.assert_called_once_with(60)
        self.assertNotIn("advance_uncertain", self.continuation.state)

    def test_final_state_replaces_stale_pending_count(self):
        self.continuation.save(dict(completed=5, pending=[dict(id=key) for key in ("t10", "t11", "t12")]))
        self.advance.return_value = dict(stage="development_gain_requires_root_review", done=True,
            goal_achieved=False, candidate_passed=True, requires_agent_review=True,
            completed=8, pending=[], failures=[], all_terminal=True)
        with mock.patch.object(C.time, "sleep") as sleep:
            current = self.continuation.run(watch=True)
        self.assertTrue(current["campaign_done"])
        self.assertFalse(current["goal_achieved"])
        self.assertEqual(self.continuation.state["completed"], 8)
        self.assertEqual(self.continuation.state["pending"], [])
        self.assertEqual(self.continuation.state["next_poll_seconds"], 0)
        sleep.assert_not_called()
        self.advance.assert_called_once_with()

    def test_failed_development_stops_watch_without_claiming_goal(self):
        self.advance.return_value = dict(stage="development_candidate_closed", done=True,
            goal_achieved=False, candidate_passed=False, completed=8, pending=[], failures=[], all_terminal=True)
        with mock.patch.object(C.time, "sleep") as sleep:
            current = self.continuation.run(watch=True)
        self.assertTrue(current["done"])
        self.assertFalse(current["goal_achieved"])
        sleep.assert_not_called()

    def test_poll_timing_is_saved_even_if_submission_outcome_is_unknown(self):
        self.advance.side_effect = subprocess.TimeoutExpired(["submit-jsonl"], 180)
        self.continuation.run()
        state = self.continuation.state
        self.assertLessEqual(state["last_poll_started_unix"], state["last_poll_finished_unix"])
        self.assertGreaterEqual(state["last_poll_elapsed_seconds"], 0)
        self.assertTrue(state["advance_uncertain"])


if __name__ == "__main__":
    unittest.main()
