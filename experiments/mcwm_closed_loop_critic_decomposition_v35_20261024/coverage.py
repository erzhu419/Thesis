"""Descriptive support of the admitted factual executed actions and contexts.

No proposal is transformed or counted here. Source/history rows with unavailable
memory do not enter this target-285 support analysis. Selected-state truth and
model outputs are neither read nor used to choose rows or thresholds.
"""
from __future__ import annotations

import numpy as np

from experiments.mcwm_closed_loop_v1.stream import policy_action_to_hold_seconds_v7


NEIGHBORHOOD_SECONDS = .5
LOW_HOLD_SECONDS = 1.875
FOCUS_LINES = ('7X', '102X', '705X')
V33_WINDOWS = {'before9000': (0., 9000.), '9000-12000': (9000., 12000.),
    '12000-18000': (12000., 18000.), 'tail': (18000., float('inf'))}
V34_WINDOWS = {'early': (9300., 9900.), 'tail': (18300., 18900.)}
ACTION_COLUMNS = ['count', 'min', 'q10', 'q25', 'median', 'q75', 'q90', 'max',
    'mean', 'std', 'exact_zero', 'near_zero', 'near_1.875', 'floor60', 'low_le_1.875']
SUPPORT_COLUMNS = ['count', 'min', 'max', 'near_low', 'near_high', 'nearest_low_seconds',
    'nearest_high_seconds', 'low_in_range', 'high_in_range', 'memory_outside_count', 'memory_outside_indices']


def _action_stats(actions):
    if not len(actions):
        return [0]+[None]*9+[0]*5
    q = np.quantile(actions, [.1, .25, .5, .75, .9]).tolist()
    return [len(actions), float(actions.min()), *q, float(actions.max()),
        float(actions.mean()), float(actions.std()), int(np.count_nonzero(actions == 0.)),
        int(np.count_nonzero(abs(actions) <= NEIGHBORHOOD_SECONDS)),
        int(np.count_nonzero(abs(actions-LOW_HOLD_SECONDS) <= NEIGHBORHOOD_SECONDS)),
        int(np.count_nonzero(actions == 60.)), int(np.count_nonzero(actions <= LOW_HOLD_SECONDS))]


def _design(actions, memory, max_hold):
    """SVD of the actual residual inputs, retaining constant nonzero context.

    A constant presence feature still creates the action column z*presence.
    Centering memory would remove that global slope, so it is not used here.
    """
    nonzero_features = np.flatnonzero(np.any(memory != 0., axis=0))
    varying = np.flatnonzero(np.ptp(memory, axis=0) != 0.)
    base = memory[:, nonzero_features]
    design = np.concatenate((base, actions[:, None]/max_hold*base), axis=1)
    feature_counts = dict(nonzero_features=len(nonzero_features), varying_features=len(varying),
        constant_nonzero_features=len(nonzero_features)-len(varying),
        all_zero_features=memory.shape[1]-len(nonzero_features))
    if not design.shape[1]:
        return dict(rows=len(actions), **feature_counts, columns=0, rank=0,
            full_rank_condition=None, nonzero_subspace_condition=None)
    singular = np.linalg.svd(design, compute_uv=False)
    tolerance = np.finfo(np.float64).eps*max(design.shape)*singular[0]
    nonzero = singular[singular > tolerance]
    rank = len(nonzero)
    condition = float(nonzero[0]/nonzero[-1]) if rank else None
    return dict(rows=len(actions), **feature_counts, columns=design.shape[1], rank=rank,
        full_rank_condition=condition if rank == design.shape[1] else None,
        nonzero_subspace_condition=condition)


def _support(indices, actions, memory, selected_memory, holds):
    if not len(indices):
        return [0, None, None, 0, 0, None, None, None, None, None, None]
    observed, context = actions[indices], memory[indices]
    low, high = map(float, holds)
    distances = [abs(observed-action) for action in (low, high)]
    lower, upper = context.min(0), context.max(0)
    outside = np.flatnonzero((selected_memory < lower) | (selected_memory > upper)).tolist()
    return [len(indices), float(observed.min()), float(observed.max()),
        *[int(np.count_nonzero(distance <= NEIGHBORHOOD_SECONDS)) for distance in distances],
        *[float(distance.min()) for distance in distances],
        bool(observed.min() <= low <= observed.max()), bool(observed.min() <= high <= observed.max()),
        len(outside), outside]


def coverage(facts, selected_records, contract):
    """Summarize one scenario's original facts and raw v34 selected records.

    facts is the dictionary returned by v33.facts.load_facts. selected_records
    contain slot, selected (including memory_s), and holds[low/high]. All action
    neighborhoods are descriptive with the fixed 0.5-second radius, not gates.
    """
    rows = [row for packet in facts['packets'] for row in packet['rows']]
    if not rows:
        raise ValueError('Factual action coverage requires admitted target rows')
    actions = np.asarray(policy_action_to_hold_seconds_v7(
        np.asarray([row.a for row in rows], np.float32).reshape(-1), contract), np.float64)
    memory = np.asarray([row.memory_s['vector'] for row in rows], np.float64)
    feature_names = rows[0].memory_s['feature_names']
    lines = np.asarray([str(row.line_id) for row in rows])
    clocks = np.asarray([float(row.started_at) for row in rows])
    line_ids = sorted(set(lines.tolist()) | set(FOCUS_LINES))
    groups, line_groups, v34_groups = {}, {}, {}
    indices_by_line = {line: np.flatnonzero(lines == line) for line in line_ids}
    for line, indices in indices_by_line.items():
        line_groups[line] = _action_stats(actions[indices])
        groups[line+'/full'] = indices
        for window, (lower, upper) in V33_WINDOWS.items():
            selected = indices[(clocks[indices] >= lower) & (clocks[indices] < upper)]
            groups[line+'/'+window] = selected
        for window, (lower, upper) in V34_WINDOWS.items():
            selected = indices[(clocks[indices] >= lower) & (clocks[indices] < upper)]
            v34_groups[line+'/'+window] = _action_stats(actions[selected])
    selected_support = []
    for record in selected_records:
        state = record['selected']
        if state is None:
            selected_support.append(dict(slot=record['slot'], status='unselected'))
            continue
        line = str(state['line_id'])
        holds = [float(record['holds'][arm]) for arm in ('low', 'high')]
        context = np.asarray(state['memory_s']['vector'], np.float64)
        support = {window: _support(groups.get(line+'/'+window, np.asarray([], int)),
            actions, memory, context, holds) for window in ('full', '9000-12000', '12000-18000')}
        selected_support.append(dict(slot=record['slot'], sequence=state['sequence'],
            issue=state['started_at'], line_id=line, holds=holds, supports=support))
    design = {'all_target285': _design(actions, memory, float(contract.max_hold_seconds))}
    for line in FOCUS_LINES:
        indices = indices_by_line[line]
        if len(indices):
            design[line] = _design(actions[indices], memory[indices], float(contract.max_hold_seconds))
    return dict(metadata=dict(admitted_target_rows=len(rows), nonzero_memory_rows=int(np.count_nonzero(
            np.any(memory != 0., axis=1))), source_history_rows_included=0,
            action_semantics='archived executed a converted directly to seconds; no proposal and no repeated F',
            neighborhood_seconds=NEIGHBORHOOD_SECONDS, neighborhood_is_gate=False,
            low_action_upper_seconds=LOW_HOLD_SECONDS,
            v34_training_windows={key:list(value) for key,value in V34_WINDOWS.items()},
            design='Actual residual inputs [m,z*m], z=executed_hold/max_hold; all-zero columns omitted, constant nonzero features retained'),
        columns=dict(action=ACTION_COLUMNS, selected_support=SUPPORT_COLUMNS),
        all_actions=_action_stats(actions), by_line=line_groups,
        by_line_window={key:_action_stats(actions[indices]) for key,indices in groups.items() if not key.endswith('/full')},
        v34_window_training=v34_groups,
        memory=dict(feature_names=feature_names, min=memory.min(0).tolist(), max=memory.max(0).tolist()),
        selected=selected_support, residual_design=design)
