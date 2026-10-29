"""Check native cached parity, subset baseline selection, and causal endpoints."""
from copy import deepcopy
import os
from pathlib import Path

import pytest
import torch

os.environ.setdefault('THESIS_H2O_ROOT', str(Path(__file__).resolve().parents[1]/'sumo_eval/runtime/H2Oplus'))

from experiments.mcwm_closed_loop_trace_recovery_v39r1_20261024.probe import audit_probe
from experiments.mcwm_closed_loop_multievent_credit_v38_20261024 import frozen
from experiments.mcwm_closed_loop_multievent_credit_v38_20261024.test_frozen import learner
from experiments.mcwm_closed_loop_paired_recovery_v37r1_20261024.test_run import transition


def chain():
    first = transition(1, 18400., arm='low')
    second = transition(2, 18400.)
    second['s'] = deepcopy(first['ns'])
    second['memory_s'] = deepcopy(first['memory_ns'])
    return first, second


def test_native_target_and_cached_parts_exact_without_model_or_rng_mutation():
    model = learner()
    first, second = chain()
    saved = frozen.score_chain(model, [first])['rows'][0]
    result = audit_probe(model, first, second, saved)
    assert result['parts_exact'] and result['native_target'] == saved['target']
    assert result['target_subset'] == saved['target_subset']
    assert result['integrity']['all_pass'] and result['integrity']['new_updates'] == 0
    assert result['subset_statistics']['duplicate_actions_exact']
    assert result['subset_statistics']['target_native_two_action_batch']
    assert result['subset_statistics']['baseline_uses_native_subset_path']
    assert result['subset_statistics']['duplicate_inputs_exact']
    assert result['subset_statistics']['repeated_forward_exact']
    assert result['subset_statistics']['baseline_slot'] == 0
    assert result['recorded_next_target_q'] == min(result['subset_statistics']['selected_heads'])


def test_budget_row_without_saved_probe_builds_baseline_without_claiming_cache_parity():
    model = learner()
    first, second = chain()
    result = audit_probe(model, first, second, None)
    assert result['cached_reference_available'] is False
    assert result['parts_exact'] is None and result['changed_parts'] == []
    assert result['native_target_equals_actual_learner_target'] is True
    assert result['recorded_next_target_q'] == min(result['subset_statistics']['selected_heads'])
    assert result['subset_statistics']['baseline_uses_native_subset_path']
    assert result['integrity']['all_pass']


def test_recorded_q_uses_previous_targets_subset_rather_than_full_head_min(monkeypatch):
    model = learner()
    # Force a strict gap between the selected head and the unused lower head.
    with torch.no_grad():
        model.target_critic.action_weight[0].fill_(-10.)
        model.target_critic.action_weight[1].fill_(10.)
    monkeypatch.setattr(model, '_sample_target_subset', lambda: torch.tensor([1]))
    first, second = chain()
    saved = frozen.score_chain(model, [first])['rows'][0]
    result = audit_probe(model, first, second, saved)
    stats = result['subset_statistics']
    assert result['target_subset'] == [1] and stats['head_indices'] == [1]
    assert result['recorded_next_target_q'] == stats['selected_heads'][0]
    assert stats['full_head_min_differs']
    assert stats['full_head_min'] < result['recorded_next_target_q']


def test_repeatable_position_ulp_difference_is_recorded_without_changing_slot0(monkeypatch):
    model = learner()
    original = model.target_critic.forward_features_subset

    def position_rounding(features, actions, chosen):
        result = original(features, actions, chosen).clone()
        result[:, 1] = torch.nextafter(result[:, 1], torch.full_like(result[:, 1], torch.inf))
        return result

    monkeypatch.setattr(model.target_critic, 'forward_features_subset', position_rounding)
    first, second = chain()
    saved = frozen.score_chain(model, [first])['rows'][0]
    result = audit_probe(model, first, second, saved)
    stats = result['subset_statistics']
    assert result['parts_exact'] and result['integrity']['all_pass']
    assert stats['duplicate_inputs_exact'] and stats['repeated_forward_exact']
    assert stats['duplicate_actions_exact'] is False
    assert stats['duplicate_outputs_max_abs'] > 0.
    assert stats['duplicate_outputs_max_ulp'] == 1
    assert stats['duplicate_output_first_difference']['ulp_difference'] == 1
    assert stats['baseline_slot'] == 0
    assert result['recorded_next_target_q'] == min(row[0] for row in stats['selected_heads_both_slots'])
    assert stats['same_encoded_full_subset_path_exact'] is False


def test_nonrepeatable_forward_keeps_error_evidence_and_rejects_baseline(monkeypatch):
    model = learner()
    first, second = chain()
    saved = frozen.score_chain(model, [first])['rows'][0]
    original, calls = model.target_critic.forward_features_subset, 0

    def transient_rounding(features, actions, chosen):
        nonlocal calls
        calls += 1
        result = original(features, actions, chosen).clone()
        # Calls 1/2 are the unchanged decomposition/native target parity check;
        # calls 3/4 compare the actual recorded-action baseline's repeatability.
        if calls == 4:
            result[0, 0] = torch.nextafter(result[0, 0], torch.full_like(result[0, 0], torch.inf))
        return result

    monkeypatch.setattr(model.target_critic, 'forward_features_subset', transient_rounding)
    result = audit_probe(model, first, second, saved)
    stats = result['subset_statistics']
    assert result['parts_exact'] and stats['duplicate_inputs_exact']
    assert stats['repeated_forward_exact'] is False
    assert stats['repeated_forward_max_ulp'] == 1
    assert stats['repeated_forward_first_difference']['ulp_difference'] == 1
    assert result['recorded_next_target_q'] is None
    assert result['baseline_missing_reason'] == 'same_native_subset_forward_not_repeatable'


@pytest.mark.parametrize('damage', ['missing', 'terminal', 'memory', 'cache'])
def test_missing_or_noncausal_baseline_never_gets_a_substitute(damage):
    model = learner()
    first, second = chain()
    if damage == 'terminal':
        first['terminal'] = True
    saved = frozen.score_chain(model, [first])['rows'][0]
    if damage == 'missing':
        second = None
    elif damage == 'memory':
        second['memory_s']['vector'][0] += .01
    elif damage == 'cache':
        saved['target'] += .01
    result = audit_probe(model, first, second, saved)
    assert result['recorded_next_target_q'] is None
    assert result['subset_statistics'] is None
    assert result['baseline_missing_reason'] is not None
    assert result['parts_exact'] is (damage != 'cache')
