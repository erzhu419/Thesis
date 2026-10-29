"""Native float32 parity, matched-subset Q, and direct admitted-row checks."""
from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

os.environ.setdefault('THESIS_H2O_ROOT', str(Path(__file__).resolve().parents[1]/'sumo_eval/runtime/H2Oplus'))

from experiments.mcwm_closed_loop_trace_target_v44_20261024 import probe
from experiments.mcwm_closed_loop_trace_target_v44_20261024.trace import decompose_trace
from experiments.mcwm_closed_loop_multievent_credit_v38_20261024.test_frozen import learner
from experiments.mcwm_closed_loop_trace_support_v39_20261024.test_probe import chain
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng, _state


def test_native_rows_feed_lambda_zero_exact_and_keep_model_adam_rng():
    model = learner()
    first, second = chain()
    saved, rng = deepcopy(_state(model)), _rng()
    rows = [probe._trace_row(model, first, second, 336202),
            probe._trace_row(model, second, None, 336202)]
    result = decompose_trace(rows, [.7], 0.)
    assert result['lambda_zero_native_exact']
    assert np.array_equal(np.asarray(result['targets'], np.float32).view(np.uint32),
                          np.asarray([row['native_target'] for row in rows], np.float32).view(np.uint32))
    assert rows[0]['subset_statistics']['baseline_slot'] == 0
    assert rows[0]['subset_statistics']['baseline_uses_native_subset_path']
    assert rows[0]['recorded_next_target_q'] == min(rows[0]['subset_statistics']['selected_heads'])
    assert rows[1]['recorded_next_target_q'] is None
    assert rows[1]['baseline_missing_reason'] == 'no_next_real_recorded_action'
    assert _equal(saved, _state(model)) and _equal(rng, _rng())


def test_noncausal_internal_link_is_not_silently_treated_as_a_cutoff():
    model = learner()
    first, second = chain()
    second['memory_s']['vector'][0] += .01
    with pytest.raises(RuntimeError, match='no legal matching-subset baseline'):
        probe._trace_row(model, first, second, 336202)


def test_admitted_replay_compares_actual_outcomes_and_causal_vectors():
    first, _ = chain()
    record = {field: np.asarray(first[field]).copy() for field in probe.REPLAY_FIELDS}
    record.update({field: np.asarray(first[field]['vector'], np.float32)
                   for field in ('memory_s', 'memory_ns')})
    model = SimpleNamespace(replay_buffer=SimpleNamespace(recent=[record], target_received=1))
    result = probe.replay_parity(model, [first])
    assert result['all_pass'] and result['replay_modified'] is False
    expected = deepcopy(first)
    expected['r'] += 1.
    with pytest.raises(RuntimeError, match='row0/r'):
        probe.replay_parity(model, [expected])
    expected = deepcopy(first)
    expected['memory_ns']['vector'][0] += .1
    with pytest.raises(RuntimeError, match='row0/memory_ns'):
        probe.replay_parity(model, [expected])


def test_loading_freezes_only_new_copy_and_preserves_rng(tmp_path):
    model = learner()
    model.actor.requires_grad_(True)
    model.actor.train()
    path = tmp_path/'resident_test.pt'
    torch.save({'learner': model}, path)
    rng = _rng()
    loaded, _ = probe._loaded(path)
    assert _equal(rng, _rng())
    assert loaded is not model and not loaded.actor.training
    assert all(not value.requires_grad and value.grad is None for value in loaded.actor.parameters())
    assert model.actor.training and all(value.requires_grad for value in model.actor.parameters())
