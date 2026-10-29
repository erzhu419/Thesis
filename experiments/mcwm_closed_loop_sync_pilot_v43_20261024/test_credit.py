"""Check root exclusion, frozen denominator and honest fixed-state credit."""
import copy
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'sumo_eval/runtime/H2Oplus'))

from experiments.mcwm_closed_loop_trace_support_v39_20261024.test_support import case
from experiments.mcwm_closed_loop_trace_design_v40_20261024.behavior import actor_parameters
from experiments.mcwm_closed_loop_sync_pilot_v43_20261024.credit import (
    prepare_chain, score_chain,
)


def setup_chain(**config):
    _, learner, row = case(**config)
    rows = []
    for sequence in range(45):
        value = copy.deepcopy(row)
        value.update(sequence=sequence, started_at=100.+sequence, dt=1.)
        value['behavior_s'] = dict(policy_input=copy.deepcopy(row['s']),
            execution_input=copy.deepcopy(row['s']), parameters=actor_parameters(learner, row['s']),
            proposal=[config.get('proposal', 0.)], nominal_executed_action=copy.deepcopy(row['a']),
            actor_snapshot='actual_post24', intervention=sequence == 0,
            action_rng=dict(stream_seed=230, draw_ordinal=sequence))
        rows.append(value)
    selected = dict(sequence=0, started_at=100.)
    return learner, rows, selected


def test_identical_actual_actor_has_unit_credit_without_rng_or_actor_mutation():
    learner, rows, selected = setup_chain(gate=-.4, mean=.2, log_std=-.3)
    rng = torch.random.get_rng_state().clone()
    before = copy.deepcopy(vars(learner.actor))
    chain = prepare_chain(learner, rows, selected)
    result = score_chain(learner, chain, prefixes={'first_reward': 17})
    assert [row['sequence'] for row in result['rows']] == list(range(1, 45))
    assert result['prefixes']['full44']['links'] == 44
    assert result['prefixes']['first_reward']['L_geom'] == 0.
    assert result['prefixes']['full44']['L_old'] == 0.
    assert result['prefixes']['full44']['overlap_mean'] == pytest.approx(1.)
    assert not result['prefixes']['full44']['trajectory_product_expectation']
    assert result['prefixes']['full44']['mixed_path_log_weight'] == 44*math.log(.9)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert vars(learner.actor) == before


def test_cached_behavior_stays_frozen_when_current_gate_drifts_between_packets():
    learner, rows, selected = setup_chain(gate=math.log(.2/.8), proposal=-1.)
    chain = prepare_chain(learner, rows, selected)
    before = [copy.deepcopy(row['behavior_parameters']) for row in chain]
    result0 = score_chain(learner, chain)
    learner.actor.gate = math.log(.8/.2)
    result1 = score_chain(learner, chain)
    mu = before[0]['positive_probability']
    pi = result1['rows'][0]['target_parameters']['positive_probability']
    log_ratio = math.log((1.-pi)/(1.-mu))
    expected = (1.-mu)*log_ratio
    assert result1['prefixes']['full44']['L_geom'] == pytest.approx(44*expected, abs=2e-12)
    assert result1['prefixes']['full44']['L_old'] == pytest.approx(44*log_ratio, abs=2e-12)
    assert result1['prefixes']['full44']['overlap_mean'] == pytest.approx(1.-pi+mu)
    assert result0['prefixes']['full44']['L_geom'] == 0.
    assert [row['behavior_parameters'] for row in chain] == before


def test_actual_atomic_actor_input_is_kept_separate_from_recorded_target():
    learner, rows, selected = setup_chain()
    original = np.asarray(rows[1]['s'], dtype=np.float32)
    actual = original.copy()
    actual[1] = .7  # A policy-visible column without changing F.

    class StateActor:
        def distribution_parameters(self, states):
            return states[:, 1], states[:, 1]*0., states[:, 1]*0.

    learner.actor = StateActor()
    for row in rows:
        context = row['behavior_s']
        context['policy_input'] = actual.copy() if row['sequence'] == 1 else row['s']
        context['parameters'] = actor_parameters(learner, context['policy_input'])
    chain = prepare_chain(learner, rows, selected)
    result = score_chain(learner, chain)
    assert result['reconstructed_inputs'] == 1
    assert chain[0]['behavior_state'][1] == pytest.approx(.7)
    assert chain[0]['state'][1] == 0.
    assert result['rows'][0]['behavior_parameters']['positive_probability'] > .5
    assert result['rows'][0]['target_parameters']['positive_probability'] == .5
    assert result['rows'][0]['expected_log_c'] < 0.


def test_incomplete_chain_or_changed_target_F_is_not_scored_as_neutral_credit():
    learner, rows, selected = setup_chain(time=.3, reserve=5.)
    with pytest.raises(ValueError, match='44 real continuation'):
        prepare_chain(learner, rows[:-1], selected)
    chain = prepare_chain(learner, rows, selected)
    learner.action_transform.max_hold = 120.
    with pytest.raises(ValueError, match='execution F changed'):
        score_chain(learner, chain)


def test_root_immediate_reward_difference_has_zero_continuation_factors():
    learner, rows, selected = setup_chain()
    chain = prepare_chain(learner, rows, selected)
    result = score_chain(learner, chain, prefixes={'root_reward': 0})
    root = result['prefixes']['root_reward']
    assert root['links'] == 0
    assert root['L_old'] == root['L_geom'] == root['L_arith'] == 0.
    assert root['mixed_path_log_weight'] == root['full_old_log_weight'] == 0.
    assert root['overlap_mean'] is root['overlap_min'] is root['overlap_max'] is None
    assert result['prefixes']['full44']['links'] == 44


def test_captured_parameter_or_action_mismatch_stops_credit_precheck():
    learner, rows, selected = setup_chain()
    rows[1]['behavior_s']['parameters']['mean'] += .1
    with pytest.raises(ValueError, match='Captured behavior distribution'):
        prepare_chain(learner, rows, selected)
    learner, rows, selected = setup_chain()
    rows[1]['behavior_s']['nominal_executed_action'] = [-1.]
    with pytest.raises(ValueError, match='Captured continuation action'):
        prepare_chain(learner, rows, selected)
