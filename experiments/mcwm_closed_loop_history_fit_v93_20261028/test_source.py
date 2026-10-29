"""Source rejection and no-solve checks; no real coefficient fits."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.mcwm_closed_loop_history_fit_v93_20261028.source import build_training, load_training

SOURCE = Path(__file__).resolve().parent.parent/'mcwm_closed_loop_completion_v92_20261028'


def documents():
    return (json.loads((SOURCE/'summary.json').read_text()),
            json.loads((SOURCE/'source_protocol.json').read_text()))


def test_frozen_source_rebuilds_without_any_coefficient_solve(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('The source loader must not fit coefficients')
    monkeypatch.setattr(np.linalg, 'solve', forbidden)
    monkeypatch.setattr(np.linalg, 'lstsq', forbidden)
    value = load_training(SOURCE/'summary.json', SOURCE/'source_protocol.json')
    assert value['all_pass'] and all(value['checks'].values())
    assert list(value['matrices']) == ['line3', 'service5', 'service_history6']
    assert [matrix.shape for matrix in value['matrices'].values()] == [(12, 3), (12, 5), (12, 6)]
    summary, _ = documents()
    assert [row['root']['sequence'] for row in value['rows']] == [row['root']['sequence'] for row in summary['pairs']]


def missing_pair(summary):
    summary['pairs'].pop()


def duplicate_root(summary):
    summary['pairs'][-1] = deepcopy(summary['pairs'][0])


def invalid_pair(summary):
    summary['pairs'][0]['checks']['real_terminal_maturity'] = False


def inconsistent_label(summary):
    summary['pairs'][0]['label_delta_per_passenger']['cost'] += .01


def unsupported_action(summary):
    summary['pairs'][0]['reference_hold_seconds'] = 3.75


def future_condition(summary):
    summary['pairs'][0]['condition']['latest_sources'][0]['completed_at'] = summary['pairs'][0]['root']['started_at']


def future_baseline(summary):
    summary['pairs'][0]['condition']['baseline_provenance']['baseline_last_observed_at'] = 9000


def relabeled_new_source(summary):
    summary['pairs'][-1]['origin'] = 'old_v90_fixed'


def changed_budget(summary):
    summary['budget']['cumulative_native_seconds'] = 314090


def changed_row_order(summary):
    summary['pairs'].reverse()


@pytest.mark.parametrize('change', [missing_pair, duplicate_root, invalid_pair, inconsistent_label,
    unsupported_action, future_condition, future_baseline, relabeled_new_source, changed_budget, changed_row_order])
def test_bad_source_refuses_fitting_inputs(change):
    summary, protocol = documents()
    change(summary)
    with pytest.raises(ValueError):
        build_training(summary, protocol)


def test_source_initializer_is_frozen():
    summary, protocol = documents()
    protocol['reference']['initializer_seed'] = 6208
    with pytest.raises(ValueError):
        build_training(summary, protocol)
