"""A middle-of-batch intervention changes stored state, not continuation actor input."""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault('THESIS_H2O_ROOT', str(Path(__file__).resolve().parents[1]/'sumo_eval/runtime/H2Oplus'))

from experiments.mcwm_closed_loop_action_fork_v7_20261020.test_fork_stream import (
    FakeNativeEnv, stream,
)
from experiments.mcwm_closed_loop_trace_support_v39_20261024.context import (
    behavior_context, reconstruct_behavior_state,
)


def collected():
    env = FakeNativeEnv([(10, [('A',0),('B',1),('C',2),('D',0)]), (20,[('E',0)])])
    issued, inputs = [], []

    def actor(state):
        inputs.append(state.copy())
        # This observes the middle selected slot once the focal issue passes it.
        return np.asarray([state[32]*.5], np.float32)

    branch = stream(env, selector=lambda event:[1.] if event['sequence'] == 1 else None,
                    duration=15, issued=issued)
    branch.step(actor)
    return branch, issued, inputs


def recover(branch, issued, index, **changes):
    arguments = dict(row=issued[index], selected=issued[1],
        nominal_action=branch.fork['nominal_a'], contract=branch.contract,
        stations=[3], forced_action=branch.fork['a'])
    arguments.update(changes)
    return reconstruct_behavior_state(**arguments)


def test_actual_native_middle_selection_recovers_each_original_actor_input():
    branch, issued, inputs = collected()
    assert branch.fork['sequence'] == 1
    before = [row['s'].copy() for row in issued]
    for index in (2, 3):
        result = recover(branch, issued, index)
        assert result['known'] and result['repaired']
        np.testing.assert_array_equal(result['state'], inputs[index])
        assert not np.array_equal(result['state'], issued[index]['s'])
        # Other prior actions retain the genuine native continuation outputs.
        selected_prior_column = 5+1*14+13
        different = np.flatnonzero(result['state'] != issued[index]['s']).tolist()
        assert different == [selected_prior_column]
    for old, row in zip(before, issued):
        np.testing.assert_array_equal(old, row['s'])


def test_root_has_no_behavior_ratio_and_later_batch_does_not_inherit_repair():
    branch, issued, inputs = collected()
    root = recover(branch, issued, 1)
    assert root['known'] is False
    assert root['reason'] == 'intervened_root_excluded_from_behavior_ratio'
    later = dict(issued[2], sequence=issued[1]['sequence']+4,
                 started_at=issued[1]['started_at'])
    result = recover(branch, issued, 2, row=later)
    assert result['known'] and not result['repaired']
    assert result['reason'] == 'later_atomic_batch_at_same_clock'
    result = recover(branch, issued, 2, row=dict(issued[2], started_at=20.))
    assert result['known'] and not result['repaired']


def test_unproven_same_batch_or_wrong_forced_prior_is_unknown_not_guessed():
    branch, issued, _ = collected()
    result = recover(branch, issued, 2, forced_action=[-.9375])
    assert not result['known']
    assert result['reason'] == 'selected_prior_does_not_match_actual_intervention'
    raw = issued[2]['raw_s'].copy()
    raw[2] = 1.
    result = recover(branch, issued, 2, row=dict(issued[2], raw_s=raw))
    assert not result['known']
    assert result['reason'] == 'same_clock_batch_identity_not_proven'


def test_support_uses_reconstructed_mu_input_but_actual_state_for_target_pi_and_F():
    from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
    from experiments.mcwm_closed_loop_trace_support_v39_20261024.support import support_at
    from experiments.mcwm_closed_loop_trace_support_v39_20261024.test_support import Actor, Behavior

    branch, issued, inputs = collected()

    class ObservedF(ExecutionAction):
        def __init__(self, contract):
            super().__init__(contract)
            self.inputs = []

        def __call__(self, states, proposals):
            self.inputs.append(states.numpy().copy())
            return super().__call__(states, proposals)

    class ObservedActor(Actor):
        def distribution_parameters(self, states):
            self.input = states.numpy().copy()
            return super().distribution_parameters(states)

    actual = issued[2]['s'].copy()
    reconstructed = behavior_context(issued[2], issued[1],
        dict(nominal_action=branch.fork['nominal_a'], executed_action=branch.fork['a']),
        branch.contract, [3])
    behavior = Behavior(ObservedF(branch.contract), proposal=lambda states:states[0,32]*.5)
    learner = SimpleNamespace(actor=ObservedActor(), action_transform=ObservedF(branch.contract), device='cpu')
    proof = support_at(behavior, learner, issued[2],
        behavior_state=reconstructed['state'], behavior_state_known=reconstructed['known'])
    assert proof['behavior_action_exact'] and reconstructed['reconstructed']
    np.testing.assert_array_equal(behavior.action_transform.inputs[0][0], inputs[2])
    np.testing.assert_array_equal(learner.actor.input[0], actual)
    assert learner.action_transform.inputs
    for observed in learner.action_transform.inputs:
        np.testing.assert_array_equal(observed[0], actual)
    np.testing.assert_array_equal(issued[2]['s'], actual)
