"""Reference anchoring and execution support under the actual frozen F."""
import numpy as np

from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
from experiments.mcwm_closed_loop_reference_collection_v84_20261027.action_pair import choose_pair


def choose(proposal, *, completion=False):
    transform = ExecutionAction(dict(max_hold_seconds=60., scheduled_headway_scale_seconds=600.))
    state = np.zeros(229, np.float32)
    state[0] = .5
    state[5] = 1.
    if completion:
        event = state[5:19]
        event[11], event[9] = 1., 1.
    return choose_pair(transform, state, np.asarray([proposal], np.float32),
                       simulation_step=1., action_step=1.)


def test_observed_interior_reference_remains_its_actual_executed_action():
    result = choose(-.09766614437103271)
    assert np.isclose(result['reference_hold_seconds'], 3.3837515115737915)
    assert result['reference_action'] != [-1.]
    assert result['candidate_hold_seconds'] == 7.5
    assert result['supported']


def test_candidate_can_be_shorter_than_the_reference():
    result = choose(1.)
    assert result['reference_hold_seconds'] == 7.5
    assert result['candidate_hold_seconds'] == 0.
    assert result['gap_seconds'] == 7.5
    assert result['supported']


def test_exact_distance_tie_chooses_the_smaller_holding():
    result = choose(0.)
    assert result['reference_hold_seconds'] == 3.75
    assert result['candidate_hold_seconds'] == 0.
    assert result['gap_seconds'] == 3.75


def test_completion_floor_singleton_is_not_supported_as_a_pair():
    result = choose(-1., completion=True)
    assert result['reference_hold_seconds'] == result['candidate_hold_seconds'] == 60.
    assert result['gap_seconds'] == 0.
    assert not result['supported']
