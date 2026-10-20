"""A failed native gate cannot launch; existing receipts cannot duplicate work."""

import importlib.util
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("v4_finite_continuation_test", HERE / "continue_campaign.py")
continuation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuation)


def test_native_gate_mismatch_stops_before_any_submission(tmp_path, monkeypatch):
    monkeypatch.setattr(continuation, "HERE", tmp_path)
    (tmp_path / "preflight_submission.json").write_text("{}")
    monkeypatch.setattr(continuation, "resident_metadata", lambda **kwargs: dict(preflights=[
        dict(summary=True, failure=False, all_gates_pass=True),
        dict(summary=True, failure=False, all_gates_pass=False)]))
    submitted = []
    monkeypatch.setattr(continuation.launch, "submit_campaign", lambda: submitted.append("campaign"))
    monkeypatch.setattr(continuation.launch, "submit_finalizer", lambda: submitted.append("finalizer"))
    with pytest.raises(RuntimeError, match="gate mismatch"):
        continuation.main()
    assert submitted == []
    assert (tmp_path / "native_preflight_gate.json").is_file()


def test_existing_receipts_and_result_resume_without_resubmit_or_repull(tmp_path, monkeypatch):
    monkeypatch.setattr(continuation, "HERE", tmp_path)
    for name in ("campaign_submission.json", "finalize_submission.json", continuation.RESULT_NAME):
        (tmp_path / name).write_text("{}")
    observed = []
    def metadata(**kwargs):
        observed.append(kwargs)
        return dict(result_exists=True, result_metadata=dict(status="complete", errors=0,
            complete_new_runs=20, bytes=100))
    monkeypatch.setattr(continuation, "resident_metadata", metadata)
    def unexpected():
        raise AssertionError("Existing receipts/result must not repeat actions")
    monkeypatch.setattr(continuation.launch, "submit_campaign", unexpected)
    monkeypatch.setattr(continuation.launch, "submit_finalizer", unexpected)
    monkeypatch.setattr(continuation.infra, "collect_json", unexpected)
    continuation.main()
    assert observed == [dict(preflights=False)]


def test_incomplete_or_oversized_remote_result_stays_resident(tmp_path, monkeypatch):
    monkeypatch.setattr(continuation, "HERE", tmp_path)
    monkeypatch.setattr(continuation, "resident_metadata", lambda **kwargs: dict(result_exists=True,
        result_metadata=dict(status="complete", errors=0, complete_new_runs=20, bytes=1_000_001)))
    with pytest.raises(RuntimeError, match="exceeds1MB"):
        continuation.wait_result()
    assert not (tmp_path / continuation.RESULT_NAME).exists()
