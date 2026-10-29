"""Offline continuation tests: no scheduler submissions, SSH, or native branches."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("v60_supervisor_tested", HERE / "supervisor.py")
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)
P = dict(name="v59-test", scenarios=["stable", "local_speed_drop"])
ARMS = ("fresh", "no_fresh")


def receipt(labels):
    return dict(submitted=[dict(id="t_" + label.replace("/", "_"),
                               signature=P["name"] + "/" + label) for label in labels])


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "v60"
        self.source = Path(self.tmp.name) / "v59"
        self.source.mkdir()
        self.infra = SimpleNamespace(presence=mock.Mock(return_value={}),
            collect=mock.Mock(), status=mock.Mock(return_value=[]), publish_state=mock.Mock())
        self.launch = SimpleNamespace(ARMS=ARMS, protocol=mock.Mock(return_value=P),
            learning=mock.Mock(), finalize=mock.Mock(), native=mock.Mock())
        self.advance = mock.Mock(return_value=dict(stage="waiting_control", done=False))
        self.supervisor = S.Supervisor(self.root, self.source, infra=self.infra,
                                      launch=self.launch, advance=self.advance)
        self.native = self.put_receipt("native", ["collection/" + case for case in P["scenarios"]])

    def put_receipt(self, batch, labels):
        value = receipt(labels)
        (self.source / (batch + "_submission.json")).write_text(json.dumps(value))
        return value

    def pending_status(self, ids):
        return [dict(id=key, status="running", node="node001") for key in ids]

    def all_resident(self, *, gate=True):
        for case in P["scenarios"]:
            self.put_receipt("learning_" + case, ["learning/" + case + "/" + arm for arm in ARMS])
        self.put_receipt("finalize", ["small-summary"])
        self.infra.presence.side_effect = lambda paths: {path: path.endswith("/summary.json") for path in paths}
        final = dict(status="complete", integrity=dict(all_pass=True),
                     next_control_gate=dict(passed=gate, eligible_models=["readout_line"] if gate else []))
        self.infra.collect.side_effect = lambda path, **kwargs: final if path == "analysis/summary.json" else dict(status="complete")
        return final

    def test_scheduler_lock_is_read_wait_and_does_not_fail_or_resubmit_native(self):
        self.infra.status.side_effect = RuntimeError(
            "timed out after30s waiting for scheduler state lock (status:readonly)")
        result = self.supervisor.run()
        self.assertEqual(result["stage"], "waiting_transient_read")
        self.assertFalse(result["done"])
        self.launch.learning.assert_not_called()
        self.launch.native.assert_not_called()
        self.assertFalse((self.root / "supervisor_failure.json").exists())

    def test_read_backoff_recovers_without_restarting_a_worker(self):
        self.supervisor.step = mock.Mock(side_effect=[S.ReadUnavailable("connection reset"),
            S.ReadUnavailable("connection reset"), dict(stage="complete", done=True)])
        with mock.patch.object(S.time, "sleep") as sleep:
            result = self.supervisor.run(watch=True)
        self.assertTrue(result["done"])
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [60, 120])
        self.launch.native.assert_not_called()

    def test_one_complete_case_starts_learning_while_other_native_continues(self):
        self.infra.presence.return_value = {"runs/collection/stable/summary.json": True}
        self.infra.collect.return_value = dict(status="complete")
        self.infra.status.side_effect = self.pending_status
        self.launch.learning.side_effect = lambda case: self.put_receipt("learning_" + case,
            ["learning/" + case + "/" + arm for arm in ARMS])
        result = self.supervisor.step()
        self.launch.learning.assert_called_once_with("stable")
        self.assertEqual(set(result["cases"]["stable"]["learning"]), set(ARMS))
        self.assertEqual(result["cases"]["local_speed_drop"]["collection"]["state"], "pending")
        self.launch.native.assert_not_called()

    def test_resident_outputs_override_status_reads_and_reuse_all_receipts(self):
        final = self.all_resident()
        self.infra.status.side_effect = RuntimeError("waiting for scheduler state lock")
        result = self.supervisor.step()
        self.infra.status.assert_not_called()
        self.launch.learning.assert_not_called()
        self.launch.finalize.assert_not_called()
        self.advance.assert_called_once_with(final)
        self.assertEqual(result["v59_disposition"], "candidate_admitted")
        self.assertTrue(all(call.kwargs["max_bytes"] == 48000 for call in self.infra.collect.call_args_list))

    def test_failed_gate_closes_readout_and_hands_original_summary_to_control(self):
        final = self.all_resident(gate=False)
        result = self.supervisor.step()
        self.assertEqual(result["v59_disposition"], "candidate_closed")
        self.advance.assert_called_once_with(final)
        self.assertFalse(final["next_control_gate"]["passed"])
        restarted = S.Supervisor(self.root, self.source, infra=self.infra,
                                 launch=self.launch, advance=self.advance)
        self.assertEqual(restarted.state["v59_disposition"], "candidate_closed")

    def test_partial_evidence_closes_admission_and_advances_without_performance_verdict(self):
        final = self.all_resident(gate=False)
        final.update(status="partial_complete", integrity=dict(all_pass=False))
        final.pop("next_control_gate")
        result = self.supervisor.step()
        self.assertEqual(result["v59_disposition"], "candidate_not_admitted_incomplete")
        self.advance.assert_called_once_with(final)
        events = [json.loads(line) for line in (self.root / "supervisor_events.jsonl").read_text().splitlines()]
        self.assertEqual(events[-1]["performance_verdict"], "not_evaluated_incomplete")
        self.assertNotIn("next_control_gate", final)

    def test_integrity_failure_also_advances_recent_candidate_without_failed_gate_claim(self):
        final = self.all_resident(gate=True)
        final["integrity"]["all_pass"] = False
        result = self.supervisor.step()
        self.assertEqual(result["v59_disposition"], "candidate_not_admitted_incomplete")
        self.advance.assert_called_once_with(final)

    def test_unknown_submission_attempt_is_never_retried(self):
        (self.source / "learning_stable_submission_attempt.json").write_text("{}")
        with self.assertRaisesRegex(S.Attention, "Unresolved submission"):
            self.supervisor.submit_once("learning_stable", self.launch.learning)
        self.launch.learning.assert_not_called()

    def test_submit_timeout_survives_restart_and_blocks_duplicate_attempt(self):
        self.launch.learning.side_effect = subprocess.TimeoutExpired(["submit-jsonl"], 180)
        with self.assertRaisesRegex(S.Attention, "outcome unknown"):
            self.supervisor.submit_once("learning_stable", self.launch.learning)
        restarted = S.Supervisor(self.root, self.source, infra=self.infra,
                                 launch=self.launch, advance=self.advance)
        with self.assertRaisesRegex(S.Attention, "Unresolved submission"):
            restarted.submit_once("learning_stable", self.launch.learning)
        self.assertEqual(self.launch.learning.call_count, 1)

    def test_finalizer_submits_once_and_receipt_survives_restart(self):
        for case in P["scenarios"]:
            self.put_receipt("learning_" + case, ["learning/" + case + "/" + arm for arm in ARMS])
        self.infra.presence.side_effect = lambda paths: {
            path: path.endswith("/summary.json") and not path.startswith("analysis/") for path in paths}
        self.infra.collect.return_value = dict(status="complete")
        self.infra.status.side_effect = self.pending_status
        self.launch.finalize.side_effect = lambda: self.put_receipt("finalize", ["small-summary"])
        first = self.supervisor.step()
        restarted = S.Supervisor(self.root, self.source, infra=self.infra,
                                 launch=self.launch, advance=self.advance)
        second = restarted.step()
        self.assertEqual(first["stage"], "waiting_v59_summary")
        self.assertEqual(second["stage"], "waiting_v59_summary")
        self.launch.finalize.assert_called_once_with()
        self.assertTrue(self.infra.publish_state.call_args.args[0]["all_terminal"])

    def test_terminal_status_gets_a_second_resident_read_before_failure(self):
        self.infra.presence.side_effect = [{}, {"runs/test/summary.json": True}]
        self.infra.status.return_value = [dict(id="t1", status="done")]
        self.infra.collect.return_value = dict(status="complete")
        result = self.supervisor.outcomes(dict(t1="runs/test"))
        self.assertEqual(result["t1"]["state"], "complete")
        self.assertEqual(self.infra.presence.call_count, 2)

    def test_unknown_task_cannot_trigger_replacement(self):
        with self.assertRaisesRegex(S.Attention, "omitted a receipted task"):
            self.supervisor.step()
        self.launch.learning.assert_not_called()
        self.launch.native.assert_not_called()

    def test_coding_error_is_recorded_and_raised_not_treated_as_retry(self):
        self.infra.presence.side_effect = ValueError("bad JSON schema")
        with self.assertRaisesRegex(ValueError, "bad JSON schema"):
            self.supervisor.run(watch=True)
        state = json.loads((self.root / "supervisor_state.json").read_text())
        self.assertTrue(state["unexpected_error"])
        self.assertEqual(state["stage"], "attention")

    def test_only_explicit_read_failures_are_transient(self):
        self.assertTrue(S.transient_read(RuntimeError("ssh: connect to host node001: Connection timed out")))
        self.assertFalse(S.transient_read(RuntimeError("Permission denied (publickey)")))
        self.assertFalse(S.transient_read(ValueError("unexpected source signature")))

    def test_v59_launch_import_uses_own_infra_and_restores_existing_module(self):
        unrelated, native_infra = object(), object()
        launch = SimpleNamespace()

        def loader(name, path):
            if path.name == "infra.py":
                return native_infra
            self.assertIs(sys.modules["infra"], native_infra)
            return launch

        with mock.patch.dict(sys.modules, {"infra": unrelated}), mock.patch.object(S, "load", side_effect=loader):
            self.assertEqual(S.v59_modules(self.source), (native_infra, launch))
            self.assertIs(sys.modules["infra"], unrelated)


if __name__ == "__main__":
    unittest.main()
