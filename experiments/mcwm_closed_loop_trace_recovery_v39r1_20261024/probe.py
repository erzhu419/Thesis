"""Preserve native target parity; audit repeatable, slot-0 subset Qbar."""
from __future__ import annotations

import numpy as np
import torch

from experiments.mcwm_closed_loop_multievent_credit_v38_20261024.frozen import (
    BELL_FIELDS, frozen_state, row_from_record, unchanged,
)
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.learning import critic_states
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.probes import probe
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _rng


def _row(value):
    return row_from_record(value.get('transition', value)) if isinstance(value, dict) else value


def _ulp_gap(left, right):
    """Signed float32 values in this model may cross zero; order before differencing."""
    def ordered(value):
        bits = value.detach().cpu().numpy().astype(np.float32).view(np.int32).astype(np.int64)
        return np.where(bits < 0, np.iinfo(np.int32).min-bits, bits)
    return int(np.abs(ordered(left)-ordered(right)).max(initial=0))


def _first_difference(left, right, chosen):
    """Keep actual head/slot evidence resident for a numerical-path difference."""
    different = torch.nonzero(left != right, as_tuple=False)
    if not len(different):
        return None
    coordinate = different[0].tolist()
    head = int(coordinate[0])
    first, second = left[tuple(coordinate)], right[tuple(coordinate)]
    return dict(target_head=int(chosen[head]), coordinate=coordinate,
                first=float(first), second=float(second),
                absolute_difference=abs(float(first)-float(second)),
                ulp_difference=_ulp_gap(first.reshape(1), second.reshape(1)))


@torch.no_grad()
def audit_probe(learner, row, next_row_or_none, cached_v38_score, seed=336202):
    """Audit one native transition without training or changing model/RNG state.

    Native one-step targets are recomputed through v33's equality-checked probe.
    Qbar uses exactly that target's sampled head subset at the next recorded
    *executed* action, not the next row's independently sampled subset or online
    full-head min.  Fixed300 costs are neither accepted nor read by this API.
    """
    current = _row(row)
    following = None if next_row_or_none is None else _row(next_row_or_none)
    before, rng = frozen_state(learner), _rng()
    try:
        parts = probe(learner, [current], seed=seed)[0]
        cached = cached_v38_score is not None
        changed = ([field for field in BELL_FIELDS if parts[field] != cached_v38_score.get(field)]
                   if cached else [])
        result = dict(native_target=parts['target'], target_subset=parts['target_subset'],
            parts={field: parts[field] for field in BELL_FIELDS},
            cached_reference_available=cached, parts_exact=(not changed) if cached else None,
            changed_parts=changed, native_target_equals_actual_learner_target=True,
            recorded_next_target_q=None, subset_statistics=None,
            baseline_missing_reason=None, synthetic300_state=False, new_training_updates=0)
        if current.terminal:
            result['baseline_missing_reason'] = 'terminal_transition'
        elif following is None:
            result['baseline_missing_reason'] = 'no_next_real_recorded_action'
        elif not (current.completed_at == following.started_at
                  and np.array_equal(current.ns, following.s)
                  and current.memory_ns['observed_at'] == following.memory_s['observed_at']
                  and current.memory_ns['vector'] == following.memory_s['vector']):
            result['baseline_missing_reason'] = 'next_real_state_or_causal_memory_not_contiguous'
        elif changed:
            result['baseline_missing_reason'] = 'cached_native_parts_changed'
        else:
            states = torch.as_tensor(current.ns, dtype=torch.float32, device=learner.device).reshape(1, 229)
            memory = torch.as_tensor(current.memory_ns['vector'], dtype=states.dtype,
                                     device=learner.device).reshape(1, 84)
            features = learner.target_critic.encode_states(critic_states(learner, states, memory))
            # Native SAC evaluates its zero/positive alternatives in a two-action
            # batch.  Preserve that shape for the recorded-action replacement.
            repeated = torch.cat((features, features))
            action = torch.as_tensor(following.a, dtype=states.dtype, device=learner.device).reshape(1, 1)
            actions = torch.cat((action, action))
            chosen = torch.as_tensor(parts['target_subset'], dtype=torch.long, device=learner.device)
            duplicate_inputs_exact = (torch.equal(repeated[0], repeated[1])
                                      and torch.equal(actions[0], actions[1]))
            selected = learner.target_critic.forward_features_subset(repeated, actions, chosen)
            repeated_selected = learner.target_critic.forward_features_subset(repeated, actions, chosen)
            repeat_exact = torch.equal(selected, repeated_selected)
            duplicate_exact = torch.equal(selected[:, 0], selected[:, 1])
            full = learner.target_critic.forward_features(repeated, actions)
            indexed_full = full.index_select(0, chosen)
            baseline = selected[:, 0].min()
            selected_full_min = indexed_full[:, 0].min()
            full_min = full[:, 0].min()
            # Native float32 kernels can round two batch positions differently
            # despite identical inputs.  Keep the position explicit; require the
            # same native computation to reproduce it, never average the slots.
            if duplicate_inputs_exact and repeat_exact:
                result['recorded_next_target_q'] = float(baseline)
            else:
                result['baseline_missing_reason'] = ('duplicate_native_inputs_differ'
                    if not duplicate_inputs_exact else 'same_native_subset_forward_not_repeatable')
            result['subset_statistics'] = dict(
                target_network=True, input='current.ns_and_causal_memory_ns',
                action='next_row_saved_executed_action_without_reapplying_F',
                head_indices=chosen.tolist(), reduction='min',
                target_native_two_action_batch=True, duplicate_actions_exact=duplicate_exact,
                duplicate_inputs_exact=duplicate_inputs_exact,
                repeated_forward_exact=repeat_exact, baseline_slot=0,
                duplicate_outputs_max_abs=float((selected[:, 0]-selected[:, 1]).abs().max()),
                duplicate_outputs_max_ulp=_ulp_gap(selected[:, 0], selected[:, 1]),
                duplicate_output_first_difference=_first_difference(selected[:, 0], selected[:, 1], chosen),
                repeated_forward_max_abs=float((selected-repeated_selected).abs().max()),
                repeated_forward_max_ulp=_ulp_gap(selected, repeated_selected),
                repeated_forward_first_difference=_first_difference(selected, repeated_selected, chosen),
                selected_heads_both_slots=selected.tolist(),
                repeated_selected_heads_both_slots=repeated_selected.tolist(),
                selected_heads=selected[:, 0].tolist(),
                full_heads=full[:, 0].tolist(), full_head_min=float(full_min),
                same_encoded_full_selected_min=float(selected_full_min),
                full_head_min_differs=float(full_min) != float(baseline),
                full_head_min_minus_matched_subset=float(full_min)-float(baseline),
                same_encoded_full_subset_path_exact=torch.equal(selected, indexed_full),
                same_encoded_full_subset_max_abs=float((selected-indexed_full).abs().max()),
                same_encoded_full_subset_max_ulp=_ulp_gap(selected, indexed_full),
                baseline_uses_native_subset_path=True)
    finally:
        proof = unchanged(learner, before, rng)
    result['integrity'] = proof
    return result
