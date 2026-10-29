"""The first arm is reusable only through its separately recorded clock audit."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v19_review_finalize_test', HERE/'finalize.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def test_audit_overlay_preserves_raw_failure_and_uses_only_the_audited_path(tmp_path, monkeypatch):
    root, original = tmp_path/'review', tmp_path/'v19'
    root.mkdir()
    (root/'first_arm_clock_audit.json').write_text(json.dumps(dict(reusable=True)))
    raw = dict(status='integrity_failed', integrity=dict(all_pass=False,
        checks=dict(full_native_prefix_exact=True, actor_real_clock=False)),
        phase_intervention=dict(actor_real_clock=False))
    before = deepcopy(raw)
    monkeypatch.setattr(review.analysis, 'load_branch', lambda path: raw)

    def analyze(source, target):
        first = review.analysis.load_branch(original/'branches_retest/7X/low')
        assert first['status'] == 'complete' and first['integrity']['all_pass'] is True
        assert review.analysis.load_branch(original/'branches/7X/high') is raw
        assert raw == before
        return dict(status='complete')

    monkeypatch.setattr(review.analysis, 'analyze', analyze)
    value = review.compare(tmp_path/'source', root, original)
    assert value['first_arm_clock_audit']['original_status'] == 'integrity_failed'
    assert value['first_arm_clock_audit']['suffix_per_issue_clock_records'] is False
    assert raw == before


def test_failed_audit_and_additional_native_failure_stop_reuse(tmp_path, monkeypatch):
    audit = tmp_path/'first_arm_clock_audit.json'
    audit.write_text('{"reusable": false}')
    with pytest.raises(RuntimeError, match='has not passed'):
        review.compare(tmp_path, tmp_path, tmp_path)
    audit.write_text('{"reusable": true}')
    monkeypatch.setattr(review.analysis, 'load_branch', lambda path: dict(status='integrity_failed',
        integrity=dict(checks=dict(actor_real_clock=False, actual_action_exact=False))))
    monkeypatch.setattr(review.analysis, 'analyze', lambda source, root:
        review.analysis.load_branch(tmp_path/'branches_retest/7X/low'))
    with pytest.raises(RuntimeError, match='differs from its audited failure'):
        review.compare(tmp_path, tmp_path, tmp_path)


def test_last_terminal_failure_does_not_publish_a_complete_summary(tmp_path, monkeypatch):
    out = tmp_path/'summary.json'
    monkeypatch.setattr(sys, 'argv', ['finalize.py', '--root', str(tmp_path), '--source', str(tmp_path),
        '--original', str(tmp_path), '--out', str(out), '--wait'])
    monkeypatch.setattr(review, 'compare', lambda *args: dict(status='complete',
        branch_statuses=dict(complete=11, integrity_failed=1), integrity=dict(all_pass=False)))
    with pytest.raises(RuntimeError, match='remaining branch failed'):
        review.main()
    assert not out.exists()
