"""Read-only credit scoring with an actual frozen post24 behavior actor.

New physical event chains are sampled by the frozen actual post24 stochastic
actor.  Their root intervention is excluded from the behavior ratio.  Each
continuation records its actual actor and F inputs; the current Bellman actor
and F use the recorded transition state.  Admission of a factual transition
to training is a separate collector/release decision.
"""
from __future__ import annotations

import math

import numpy as np

from experiments.mcwm_closed_loop_trace_support_v39_20261024.support import atom_geometry
from experiments.mcwm_closed_loop_trace_design_v40_20261024.behavior import (
    actor_parameters, execution_measure, ratio_from_measures,
)
from experiments.mcwm_closed_loop_overlap_v41_20261024.overlap import (
    conditional_overlap, geometry_at,
)


def prepare_chain(reference, rows, selected):
    """Cache frozen mu24 parameters and F geometry once, before target updates.

    ``rows`` contains the real root and its next44 recorded event transitions.
    A prepared chain is an in-memory object; save only score output.  No weights
    or actor observations are changed by this helper.
    """
    records = [row.get('transition', row) for row in rows]
    if len(records) != 45 or records[0]['sequence'] != selected['sequence']:
        raise ValueError('Expected the selected root and44 real continuation events')
    prepared = []
    for event, row in enumerate(records[1:], 1):
        context = row['behavior_s']
        if context['actor_snapshot'] != 'actual_post24' or context.get('intervention', False):
            raise ValueError('Continuation was not sampled by the frozen actual post24 actor')
        state = np.asarray(row['s'], dtype=np.float32)
        actual = np.asarray(context['policy_input'], dtype=np.float32).reshape(229)
        execution_state = np.asarray(context['execution_input'], dtype=np.float32).reshape(229)
        behavior_geometry = geometry_at(reference.action_transform, execution_state)
        target_geometry = geometry_at(reference.action_transform, state)
        if behavior_geometry['map_key'] != target_geometry['map_key']:
            raise ValueError('Atomic actor input also changed F; overlap is unsupported')
        parameters = actor_parameters(reference, actual)
        if parameters != context['parameters']:
            raise ValueError('Captured behavior distribution does not match the actual post24 snapshot')
        nominal = reference.action_transform.numpy(execution_state, context['proposal'])
        if not (np.array_equal(nominal, np.asarray(context['nominal_executed_action'], np.float32))
                and np.array_equal(nominal, np.asarray(row['a'], np.float32))):
            raise ValueError('Captured continuation action does not match its actual F input')
        self_score = conditional_overlap(reference.action_transform, execution_state,
            parameters, parameters, geometry=behavior_geometry)
        if (abs(self_score['total_overlap']-1.) > 5e-12 or
                abs(self_score['expected_log_c_under_behavior']) > 5e-12):
            raise ValueError('Frozen behavior does not normalize on the actual F')
        prepared.append(dict(event=event, sequence=row['sequence'], dt=row['dt'],
            state=state, action=row['a'], behavior_state=actual.copy(),
            behavior_execution_state=execution_state.copy(),
            behavior_input_differs=not np.array_equal(actual, state),
            behavior_F_input_differs=not np.array_equal(execution_state, state),
            execution_constants=(reference.action_transform.max_hold,
                                 reference.action_transform.headway_scale),
            behavior_parameters=parameters, geometry=behavior_geometry,
            point_geometry=atom_geometry(reference.action_transform, execution_state, row['a'])))
    return prepared


def score_parameters(transform, prepared, target_parameters):
    """Score one real continuation action and its whole conditional F measure."""
    denominator = execution_measure(transform, prepared['behavior_execution_state'], prepared['action'],
        prepared['behavior_parameters'], geometry=prepared['point_geometry'])
    numerator = execution_measure(transform, prepared['state'], prepared['action'],
        target_parameters, geometry=prepared['point_geometry'])
    ratio = ratio_from_measures(denominator, numerator)
    overlap = conditional_overlap(transform, prepared['behavior_execution_state'],
        prepared['behavior_parameters'], target_parameters, geometry=prepared['geometry'])
    if (ratio['status'] != 'supported' or overlap['status'] != 'supported' or
            ratio['log_c'] is None or overlap['expected_log_c_under_behavior'] is None):
        raise ValueError('Unresolved or infinite credit cannot be represented as zero loss')
    return dict(event=prepared['event'], sequence=prepared['sequence'], dt=prepared['dt'],
        behavior_input_differs=prepared['behavior_input_differs'],
        behavior_F_input_differs=prepared['behavior_F_input_differs'],
        behavior_parameters=dict(prepared['behavior_parameters']),
        target_parameters=dict(target_parameters),
        log_c=ratio['log_c'], log_ratio=ratio['log_ratio'], c=ratio['c'],
        overlap=overlap['total_overlap'], atom_overlap=overlap['atom_overlap'],
        continuous_overlap=overlap['continuous_overlap'],
        expected_log_c=overlap['expected_log_c_under_behavior'],
        log_overlap=overlap['log_overlap'], jensen_gap=overlap['jensen_gap'],
        normalization_error=overlap['normalization_error'],
        expectation_identity_error=overlap['expectation_identity_error'],
        source_new_stochastic_mu24=True, state_conditioned_probe=True,
        trajectory_product_expectation=False, scorer_authorizes_training=False)


def prefix_metrics(scores, count, *, mixed_path_weight=.9):
    """Three fixed-state log summaries, separate from the mixed-backup factor.

    L_geom sums E_mu[log c] after re-integrating under frozen actual mu24;
    L_arith sums log E_mu[c].  Neither is a trajectory-level expectation.
    ``mixed_path_weight`` is the original mixed-backup path weight, not gamma.
    """
    if not 0 <= count <= len(scores):
        raise ValueError('The prefix must contain real continuation events')
    chosen = scores[:count]
    old = math.fsum(row['log_c'] for row in chosen)
    geom = math.fsum(row['expected_log_c'] for row in chosen)
    arith = math.fsum(row['log_overlap'] for row in chosen)
    path = count*math.log(mixed_path_weight)
    return dict(links=count, L_old=old, L_geom=geom, L_arith=arith,
        jensen_gap=arith-geom,
        overlap_mean=math.fsum(row['overlap'] for row in chosen)/count if count else None,
        overlap_min=min(row['overlap'] for row in chosen) if count else None,
        overlap_max=max(row['overlap'] for row in chosen) if count else None,
        atom_overlap_mean=math.fsum(row['atom_overlap'] for row in chosen)/count if count else None,
        continuous_overlap_mean=math.fsum(row['continuous_overlap'] for row in chosen)/count if count else None,
        mixed_path_log_weight=path, full_old_log_weight=old+path,
        full_fixed_state_geom_log_weight=geom+path,
        full_fixed_state_arith_log_weight=arith+path,
        trajectory_product_expectation=False)


def score_chain(learner, prepared, *, prefixes=None):
    """Read current actor at one packet boundary; caller retains its optimizer.

    ``prefixes`` maps names to event counts, e.g. ``{'first_reward': 23}``.
    The full44 continuation prefix is always returned.  The actual F geometry
    is cached at issue inputs, while current actor parameters are recomputed.
    """
    # F is untrained and its two physical constants are fixed.  The actual and
    # target maps were compared at every captured state when preparing this
    # chain, so repeated atom enumeration at each of54 packet boundaries adds
    # no information.
    if (learner.action_transform.max_hold, learner.action_transform.headway_scale) != prepared[0]['execution_constants']:
        raise ValueError('Current target execution F changed')
    scores = []
    for row in prepared:
        scores.append(score_parameters(learner.action_transform, row,
            actor_parameters(learner, row['state'])))
    counts = dict(prefixes or {})
    counts['full44'] = len(prepared)
    return dict(rows=scores, prefixes={name: prefix_metrics(scores, count)
        for name, count in counts.items()},
        maximum_normalization_error=max(row['normalization_error'] for row in scores),
        maximum_expectation_identity_error=max(row['expectation_identity_error'] for row in scores),
        reconstructed_inputs=sum(row['behavior_input_differs'] for row in scores),
        captured_contexts=len(prepared),
        new_updates=0, new_feedback=0, native_seconds=0)
