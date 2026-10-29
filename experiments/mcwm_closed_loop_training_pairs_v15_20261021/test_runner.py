"""Exercise source action isolation and the training-period intervention boundary."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def helpers():
    names = {'find_selection', 'nominal_issue', 'choose_prefix_calibration',
             'background_scope_valid', 'expected_prefix_counts'}
    tree = ast.parse((HERE/'run.py').read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {'np': np}
    exec(compile(tree, str(HERE/'run.py'), 'exec'), scope)
    return SimpleNamespace(**{name: scope[name] for name in names})


def test_nominal_prediction_cannot_overwrite_actual_intervention():
    h = helpers()
    actual = dict(sequence=17, a=np.asarray([-1.], np.float32), s=np.asarray([3., 5.]))
    before = actual['a'].copy()
    predicted = h.nominal_issue(actual, [-.75])
    assert predicted['a'][0] == -.75
    assert np.array_equal(actual['a'], before)
    predicted['a'][0] = .5
    assert np.array_equal(actual['a'], before)
    assert predicted['s'] is actual['s']


def test_fork_consumes_old_assignment_but_reserves_no_diagnostic_label():
    h = helpers()
    calls, reserved, decisions = [], [], {}
    calibration = SimpleNamespace(choose=lambda event, role: calls.append((event['sequence'], role))
        or dict(action=np.asarray([-.75]), assignment='high'))
    budget = SimpleNamespace(reserve=reserved.append)
    for sequence in (4, 5):
        action = h.choose_prefix_calibration(dict(sequence=sequence, started_at=1200.),
            calibration, budget, decisions, selected_sequence=5, already_forked=False, last_deadline=18000.)
        assert action[0] == -.75
    assert calls == [(4, 'train'), (5, 'train')]
    assert reserved == [4]
    assert set(decisions) == {4, 5}
    assert h.choose_prefix_calibration(dict(sequence=6, started_at=1200.), calibration, budget,
        decisions, selected_sequence=5, already_forked=True, last_deadline=18000.) is None
    assert len(calls) == 2 and reserved == [4]  # Suffix never asks for another calibration assignment.


def test_prefix_counts_include_only_processed_slots_and_actual_global_support():
    h = helpers()
    p = dict(policy_updates_per_packet=8, model_refit_updates=10)
    receipts = [dict(calibration_records=1, learned_global_records=4),
                dict(calibration_records=0, learned_global_records=5)]
    counts = h.expected_prefix_counts(p, receipts, 50000)
    assert counts['policy'] == 50016 and counts['trust'] == 16
    assert counts['world'] == 40 and counts['global58_labels'] == 9
    assert counts['innovation_labels'] == counts['legacy_labels'] == 1
    assert counts['response_refits'] == 2 and counts['physical'] == 0
    p['model_refit_updates'] = 1
    assert h.expected_prefix_counts(p, receipts, 50000)['world'] == 4  # Old native preflight budget.


def test_background_snapshot_allows_same_clock_prior_release_but_never_future():
    h = helpers()
    event = dict(sequence=7, started_at=1200.)
    snapshot = dict(sequence=7, issued_at=1200., endpoints={
        'fixed300': dict(max_available_at=1200., max_sequence=6),
        'global_next': dict(max_available_at=None, max_sequence=None)})
    context = dict(innovation_background=snapshot)
    assert h.background_scope_valid(context, event)
    snapshot['endpoints']['fixed300']['max_available_at'] = 1201.
    assert not h.background_scope_valid(context, event)
    snapshot['endpoints']['fixed300']['max_available_at'] = 1200.
    snapshot['endpoints']['fixed300']['max_sequence'] = 7
    assert not h.background_scope_valid(context, event)


def test_manifest_missing_cell_is_preserved_and_window_cannot_alias_another_state():
    h = helpers()
    rows = [dict(cell=dict(seed=6201, scenario='stable', target_line='7X', window_name=window),
                 status='selection_missing' if window == 'post_shock' else 'selected')
            for window in ('pre_shock', 'post_shock')]
    args = SimpleNamespace(seed=6201, scenario='stable', target_line='7X', window='post_shock')
    assert h.find_selection(dict(states=rows), args) is rows[1]
    args.window = 'absent'
    with pytest.raises(ValueError, match='exactly one'):
        h.find_selection(dict(states=rows), args)
