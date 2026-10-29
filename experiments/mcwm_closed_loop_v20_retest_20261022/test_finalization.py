"""The complete native suffix must reach evidence writing without a clock-key error."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('v20_native_retest_runner', HERE/'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def original_phase():
    # The actual runner has not recorded the post-suffix batch counter yet.
    return dict(mapping='native_all_suffix', fork_batch_native=True,
        tail_policy_issues=0, first_tail_sequence=None, actor_real_clock=True)


def test_completed_native_suffix_without_saved_batch_counter_can_be_finalized():
    phase = original_phase()
    assert 'tail_batches' not in phase
    stream = SimpleNamespace(tail_batches=0, native_suffix_steps=300)
    assert runner.finalize_native_phase(phase, stream) is True
    assert phase['tail_batches'] == 0
    # These are the phase fields subsequently written in the real summary.
    assert phase['mapping'] == 'native_all_suffix' and phase['actor_real_clock'] is True
    assert stream.native_suffix_steps == 300


def test_actual_tail_batches_override_stale_native_evidence_and_fail_admission():
    phase = original_phase()
    phase['tail_batches'] = 0
    assert runner.finalize_native_phase(phase, SimpleNamespace(tail_batches=1)) is False
    assert phase['tail_batches'] == 1


@pytest.mark.parametrize('change', [dict(fork_batch_native=False),
    dict(tail_policy_issues=1), dict(first_tail_sequence=11756)])
def test_suffix_execution_changes_remain_rejected(change):
    phase = original_phase()
    phase.update(change)
    original = deepcopy(phase)
    assert runner.finalize_native_phase(phase, SimpleNamespace(tail_batches=0)) is False
    assert {k:phase[k] for k in original} == original
