"""Detect action-history leakage, changing issue snapshots and clock regressions."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.mcwm_closed_loop_assignment_innovation_v14_20261021.innovation import AssignmentInnovationReadout
from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.test_endpoints import issue
from experiments.mcwm_closed_loop_response_residual_v11_20261021.test_residual import baseline_record, observable_context
from experiments.mcwm_closed_loop_v1.stream import EI


def records(model, *, offset=0, started_at=0., blocks=4):
    rows = []
    for line in ('7X', '102X', '705X'):
        for block in range(blocks):
            for position in (0, 1):
                hold = 7.5 if (block+position)%2 else 0.
                sequence = offset+len(rows)
                row = baseline_record(sequence, line, hold)
                row.started_at = started_at
                row.available_at = started_at+330.
                row.context.update(assignment_position=position, assignment_success_index=2*block+position)
                row.event[EI['forward_headway_seconds']] = 200.+50.*block
                query = issue(sequence, line, now=started_at, hold=hold)
                query['event'] = row.event.copy()
                model.capture_context(query, row.context)
                u = 2.*hold/7.5-1.
                for endpoint in row.endpoints.values():
                    endpoint['wait'] += u*(1000.+100.*block)
                    endpoint['ride'] += u*500.
                    endpoint['cost'] = 2.*endpoint['wait']+endpoint['ride']
                rows.append(row)
    return rows


def fit(*, line_only=False):
    model = AssignmentInnovationReadout(line_only=line_only)
    rows = records(model)
    model.accept(rows, now=330.)
    model.refit()
    return model, rows


def test_only_first_random_bits_fit_actions_second_labels_cannot_change_beta():
    model, rows = fit()
    changed = deepcopy(rows)
    for row in changed:
        if row.context['assignment_position'] == 1:
            for endpoint in row.endpoints.values():
                endpoint['wait'] += 1e8
                endpoint['ride'] -= 1e7
                endpoint['cost'] = 2.*endpoint['wait']+endpoint['ride']
    other = AssignmentInnovationReadout()
    other.accept(changed, now=330.)
    other.refit()
    for endpoint in model.readouts:
        assert np.array_equal(model.readouts[endpoint][12:], other.readouts[endpoint][12:])
        assert not np.array_equal(model.background_readouts[endpoint], other.background_readouts[endpoint])
        diag = model.innovation_diagnostics[endpoint]
        assert diag['first_rows'] == diag['second_rows'] == 12
        assert diag['cold_start_first_rows'] == 12
        assert diag['cold_start_action_coefficient_physical'] == diag['action_coefficient_physical']
        assert diag['cold_warm_beta_reconstruction_error'] < 1e-6
        assert diag['snapshot_scope_valid']
        assert set(diag['first_sequence_ids']).isdisjoint(diag['second_sequence_ids'])


def test_issue_background_is_frozen_across_later_outcomes_and_same_for_both_arms():
    full, _ = fit()
    line, _ = fit(line_only=True)
    query = issue(1000, now=1000.)
    full_context, line_context = observable_context(), observable_context()
    captured = full.capture_context(query, full_context)
    assert captured == line.capture_context(query, line_context)
    saved = deepcopy(captured)
    rows = records(full, offset=1001, started_at=1500., blocks=2)
    for row in rows:
        for endpoint in row.endpoints.values():
            endpoint['wait'] += 100000.
    full.accept(rows, now=1830.)
    full.refit()
    assert captured == saved and full_context['innovation_background'] == saved
    assert all(diag['cold_warm_beta_reconstruction_error'] < 1e-6
               for diag in full.innovation_diagnostics.values())
    assert full.capture_context(issue(2000, now=2000.), observable_context()) != saved


def test_line_only_keeps_six_interactions_exactly_zero_and_actual_first_support():
    model, _ = fit(line_only=True)
    for endpoint in model.readouts:
        assert np.count_nonzero(model.readouts[endpoint][15:]) == 0
        assert model.innovation_diagnostics[endpoint]['action_design']['columns'] == 3
    for line in ('7X', '102X', '705X'):
        support = model.endpoint_coverage()['fixed300'][line]
        assert support == dict(n=4, zero_holding=2, high_holding=2)
    only_low = AssignmentInnovationReadout()
    rows = [row for row in records(only_low) if row.hold_seconds == 0. or row.context['assignment_position'] == 1]
    only_low.accept(rows, now=330.)
    only_low.refit()
    assert only_low.predict_issue(issue(), observable_context())['endpoint_status']['fixed300'] == 'unsupported'


def test_background_levels_cannot_change_gain_and_fixed300_clock_stays_exact():
    model, _ = fit()
    context = observable_context()
    before = model.predict_issue(issue(), context)
    for weight in model.readouts.values():
        weight[:12] += 1e14
    after = model.predict_issue(issue(), context)
    assert before['gains'] == after['gains']
    for branch in ('low', 'high'):
        assert after[branch]['endpoints']['fixed300']['dt'] == 300.
    assert after['high_minus_low']['fixed300']['dt'] == 0.


def test_heldout_is_rejected_and_future_snapshot_scope_is_exposed():
    model = AssignmentInnovationReadout()
    heldout = baseline_record(99, '7X', 0., role='heldout')
    with pytest.raises(ValueError, match='Held-out'):
        model.accept([heldout], now=330.)
    rows = records(model)
    rows[0].context['innovation_background']['endpoints']['fixed300']['max_available_at'] = 1.
    model.accept(rows, now=330.)
    model.refit()
    assert model.innovation_diagnostics['fixed300']['snapshot_scope_valid'] is False
