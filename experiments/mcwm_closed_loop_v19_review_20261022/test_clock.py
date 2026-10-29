"""Clock diagnostics must accept native delivery lag and detect rewritten input."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from clock_policy import NativeClockPolicy, tail_action
from run_remaining import run


class Actor:
    def __init__(self):
        self.inputs = []

    def act(self, state, deterministic):
        assert deterministic is True
        self.inputs.append(state.copy())
        return np.asarray([[.5]], np.float32)


class Transform:
    max_hold = 60.

    @staticmethod
    def hold_seconds(proposal):
        return 30. * (proposal + 1.)

    @staticmethod
    def numpy(state, proposal):
        return np.asarray(proposal, np.float32).copy()


def fixture(*, tail=False, state_time=16864.):
    stream = SimpleNamespace(now=16865., tail_policy_active=tail,
        env=SimpleNamespace(_current_events=[SimpleNamespace(sim_time=16864.)]))
    evidence = dict(actor_real_clock=True, actor_clock_max_abs_error=0.,
        native_policy_issues=0, tail_policy_issues=0)
    actor = Actor()
    policy = NativeClockPolicy(actor, Transform(), stream, 24000., evidence)
    state = np.zeros(229, np.float32)
    state[0] = np.float32(state_time / 24000.)
    return policy, actor, state, evidence


def test_native_one_second_delivery_lag_passes_without_changing_actor_input():
    policy, actor, state, evidence = fixture()
    original = state.copy()
    result = policy(state)
    assert evidence['actor_real_clock'] is True
    assert evidence['actor_clock_max_abs_error'] == 0.
    assert evidence['ledger_vs_event_lag'] == dict(count=1, min_seconds=1., max_seconds=1.)
    assert np.array_equal(state, original)
    assert np.array_equal(actor.inputs[0][0], original)
    assert np.array_equal(result, np.asarray([.5], np.float32))


def test_replacing_event_time_with_ledger_or_tail_time_is_reported():
    for altered_time in (16865., 19000.):
        policy, actor, state, evidence = fixture(state_time=altered_time)
        original = state.copy()
        policy(state)
        assert evidence['actor_real_clock'] is False
        assert evidence['actor_clock_max_abs_error'] > 0.
        assert np.array_equal(actor.inputs[0][0], original)


@pytest.mark.parametrize('tail', [False, True])
def test_clock_fix_keeps_native_and_tail_actions(tail):
    policy, actor, state, evidence = fixture(tail=tail)
    action = policy(state)
    proposal = np.asarray([.5], np.float32)
    expected = tail_action(Transform(), proposal) if tail else proposal
    assert np.array_equal(action, expected)
    assert evidence['tail_policy_issues'] == int(tail)
    assert evidence['native_policy_issues'] == int(not tail)


def test_completed_first_arm_cannot_be_requeried():
    with pytest.raises(ValueError, match='must be reused'):
        run(SimpleNamespace(target_id='train/7X/late', branch='low'))
