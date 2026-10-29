"""Recover the actual nominal-prior input used by the frozen suffix actor."""
from __future__ import annotations

import numpy as np


def behavior_context(row, selected, execution, contract, stations):
    result = reconstruct_behavior_state(row, selected, execution.get('nominal_action'),
        contract, stations, forced_action=execution.get('executed_action'))
    result['reconstructed'] = result.pop('repaired')
    return result


def reconstruct_behavior_state(row, selected, nominal_action, contract, stations,
                               *, forced_action=None):
    """Return ``state, known, reason`` without changing the recorded target state.

    ForkEventStream keeps the selected action nominal only in the actor input
    for the remaining controls in that same atomic batch.  Its recorded state
    and executed-action memory correctly retain the intervention.  Rebuild
    just that actor input; target pi and target F must still use ``row.s``.
    """
    from bus_h2o.mdp_contract_v7 import (
        RAW_PREFIX_NAMES, RAW_SLOT_WIDTH, RAW_STATE_INDEX, raw_state_to_canonical_v7,
    )

    row = row.get('transition', row)
    actual = np.asarray(row['s'], dtype=np.float32).copy()

    def result(known, reason, state=None, repaired=False):
        return dict(state=actual if state is None else state, known=known,
                    reason=reason, repaired=repaired)

    sequence, first_sequence = row['sequence'], selected['sequence']
    if sequence == first_sequence:
        return result(False, 'intervened_root_excluded_from_behavior_ratio')
    if sequence < first_sequence or row['started_at'] != selected['started_at']:
        return result(True, 'original_policy_input_outside_intervention_batch')
    raw_value, selected_value = row.get('raw_s'), selected.get('raw_s')
    if raw_value is None or selected_value is None:
        return result(False, 'same_clock_batch_raw_state_missing')
    raw = np.asarray(raw_value, dtype=np.float32)
    first = np.asarray(selected_value, dtype=np.float32)
    count_index = RAW_STATE_INDEX['simultaneous_decision_count']
    current_index = RAW_STATE_INDEX['current_decision_index']
    first_count, first_slot = int(first[count_index]), int(first[current_index])
    offset = sequence-first_sequence
    if offset > first_count-1-first_slot:
        # Every issue in the first atomic batch has one consecutive sequence.
        # A later sequence cannot inherit its temporary nominal_prior buffer.
        return result(True, 'later_atomic_batch_at_same_clock')
    count, current = int(raw[count_index]), int(raw[current_index])
    if count != first_count or current != first_slot+offset:
        return result(False, 'same_clock_batch_identity_not_proven')
    start = len(RAW_PREFIX_NAMES)
    prior_columns = np.arange(start+RAW_SLOT_WIDTH-1, raw.size, RAW_SLOT_WIDTH)
    physical = np.ones(raw.size, dtype=bool)
    physical[prior_columns], physical[current_index] = False, False
    if (raw.shape != first.shape or not np.array_equal(raw[physical], first[physical])
            or raw[start+first_slot*RAW_SLOT_WIDTH] != 1.):
        return result(False, 'same_clock_physical_batch_mismatch')
    if not np.array_equal(raw_state_to_canonical_v7(raw, stations, contract), actual):
        return result(False, 'recorded_state_does_not_match_native_raw')
    forced = forced_action if forced_action is not None else selected.get('a')
    if forced is None or nominal_action is None:
        return result(False, 'selected_executed_or_nominal_action_missing')
    forced = np.asarray(forced, dtype=np.float32).reshape(-1)[0]
    nominal = np.asarray(nominal_action, dtype=np.float32).reshape(-1)[0]
    prior_index = start+first_slot*RAW_SLOT_WIDTH+RAW_SLOT_WIDTH-1
    if raw[prior_index] != forced:
        return result(False, 'selected_prior_does_not_match_actual_intervention')
    corrected = raw.copy()
    corrected[prior_index] = nominal
    state = raw_state_to_canonical_v7(corrected, stations, contract)
    return result(True, 'recovered_original_nominal_prior_actor_input',
                  state=state, repaired=not np.array_equal(state, actual))
