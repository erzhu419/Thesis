"""Offline fresh_value watcher recovery and submission uncertainty."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("v64_continue_tested", HERE / "continue_control.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "execution_ready.json").write_text(json.dumps(dict(ready=True)))
        self.advance = mock.Mock(return_value=dict(stage="fresh_value_control_development", done=False, goal_achieved=False))
        self.continuation = C.Continuation(self.root, advance=self.advance)

    def test_not_ready_never_dispatches(self):
        (self.root / "execution_ready.json").unlink()
        self.assertEqual(self.continuation.step()["stage"], "preparing_fresh_value_control")
        self.advance.assert_not_called()

    def test_unknown_submission_is_not_retried_after_restart(self):
        self.advance.side_effect = subprocess.TimeoutExpired(["submit-jsonl"], 180)
        self.assertEqual(self.continuation.run()["stage"], "attention")
        self.assertEqual(C.Continuation(self.root, advance=self.advance).run()["stage"], "attention")
        self.advance.assert_called_once_with()
        self.assertTrue(self.continuation.state["advance_uncertain"])

    def test_transient_read_keeps_watch_running_then_recovers(self):
        self.advance.side_effect = [dict(stage="waiting_transient_fresh_value_read", done=False, goal_achieved=False),
            dict(stage="fresh_value_control_development", done=False, goal_achieved=False)]
        with mock.patch.object(C.time, "sleep", side_effect=[None, StopIteration]) as sleep:
            with self.assertRaises(StopIteration):
                self.continuation.run(watch=True)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [60, 30])
        self.assertNotIn("advance_uncertain", self.continuation.state)

    def test_confirmed_candidate_stays_root_review_without_goal_completion(self):
        self.advance.return_value = dict(stage="fresh_value_gain_requires_root_review", done=False,
            goal_achieved=False, confirmed_candidate=True, requires_agent_review=True)
        current = self.continuation.run()
        self.assertTrue(current["confirmed_candidate"])
        self.assertTrue(current["requires_agent_review"])
        self.assertFalse(current["goal_achieved"])
        self.assertFalse(current["done"])

    def test_closed_candidate_does_not_stop_watch_or_claim_gain(self):
        self.advance.return_value = dict(stage="fresh_value_candidate_closed", done=False,
            goal_achieved=False, confirmed_candidate=False, requires_agent_review=True)
        first = self.continuation.run()
        second = C.Continuation(self.root, advance=self.advance).run()
        self.assertEqual(first["stage"], second["stage"])
        self.assertFalse(second["goal_achieved"])
        self.assertEqual(self.advance.call_count, 2)


if __name__ == "__main__":
    unittest.main()

