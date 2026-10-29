"""Exact old scores, direct shared-level cancellation and min-head attribution."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.test_learning import initial, rows
from experiments.mcwm_closed_loop_shared_level_v36_20261024.learning import attach_shared_level
from experiments.mcwm_closed_loop_physical_truth_v55_20261025.frozen import score_frozen
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng, _state
from experiments.mcwm_closed_loop_action_attribution_v56_20261025.decomposition import (
    decompose_frozen, compare_frozen)


def inputs():
    learner = attach_shared_level(initial())
    with torch.no_grad():
        learner.critic.level_weight.fill_(.02)
        learner.critic.action_weight[0].fill_(.1)
        learner.critic.action_weight[1].fill_(-.2)
    return learner, rows(.85)[0].s, np.full(84, .4, np.float32), [-1., -.75]


def test_complete_score_exact_native_addition_and_no_training_or_rng_changes():
    learner, state, memory, actions = inputs()
    before, rng = deepcopy(_state(learner)), _rng()
    reference = score_frozen(dict(one_step=learner, trace=learner), state, memory, actions)["one_step"]
    result = decompose_frozen(learner, state, memory, actions, reference=reference)
    assert result["variants"]["full"]["q_heads"] == reference["q_heads"]
    assert result["executed_actions"] == [-1., -.75]  # No second F application.
    assert result["numeric"]["native_float32_forward_max_abs_error"] == 0.
    assert result["integrity"]["all_pass"]
    assert _equal(before, _state(learner)) and _equal(rng, _rng())
    assert result["component_delta_heads"]["shared_level"] == [0., 0.]


def test_large_shared_level_cannot_directly_reorder_heads_or_give_action_slope():
    learner, state, memory, actions = inputs()
    with torch.no_grad():
        learner.critic.level_weight.fill_(15.)
    result = decompose_frozen(learner, state, memory, actions)
    full, deleted = result["variants"]["full"], result["variants"]["no_level"]
    assert full["min_head"] == deleted["min_head"]
    bound = result["numeric"]["native_float32_roundoff_bound"]
    assert abs(full["delta_min"]-deleted["delta_min"]) <= 2.*bound
    assert result["component_delta_heads"]["shared_level"] == [0., 0.]
    assert result["component_delta_heads"]["action_interaction"][0] > 0.
    assert result["component_delta_heads"]["action_interaction"][1] < 0.


def test_between_arm_level_drift_is_not_mislabeled_as_action_contrast():
    one, state, memory, actions = inputs()
    trace = deepcopy(one)
    with torch.no_grad():
        trace.critic.level_weight.add_(.2)
        trace.critic.action_weight[0].add_(.3)
        trace.critic.base.network.layers[-1].bias.add_(.25)
    left, right = (decompose_frozen(item, state, memory, actions) for item in (one, trace))
    result = compare_frozen(left, right)
    bound = sum(item["numeric"]["native_float32_roundoff_bound"] for item in (left, right))
    np.testing.assert_allclose(result["parts"]["shared_level"]["head_action_contrast_increment"], 0., atol=bound)
    np.testing.assert_allclose(result["parts"]["base"]["head_common_level_shift"], .25, atol=1e-6)
    np.testing.assert_allclose(result["parts"]["base"]["head_action_contrast_increment"], 0., atol=1e-6)
    assert result["parts"]["action_interaction"]["head_action_contrast_increment"][0] > 0.
    assert result["min_advantage"]["decomposition_error"] == 0.
    assert abs(result["reference_head_component_sum_error"]) < 1e-5


def test_duplicate_executed_actions_keep_zero_contrasts_and_saved_mismatch_fails():
    learner, state, memory, actions = inputs()
    result = decompose_frozen(learner, state, memory, [actions[1], actions[1]])
    assert result["alias_groups"] == [0, 0]
    assert result["variants"]["full"]["delta_heads"] == [0., 0.]
    reference = score_frozen(dict(one_step=learner, trace=learner), state, memory, actions)["one_step"]
    reference["q_heads"][0][0] += 1.
    with pytest.raises(RuntimeError, match="complete_saved_prediction_exact"):
        decompose_frozen(learner, state, memory, actions, reference=reference)


def test_min_head_switch_increment_is_separate_from_fixed_head_slope():
    def value(matrix):
        q = torch.tensor(matrix, dtype=torch.float32)
        from experiments.mcwm_closed_loop_action_attribution_v56_20261025.decomposition import summarize
        return dict(executed_actions=[-1., -.75], variants=dict(full=summarize(q)),
            components=dict(base=q.tolist(), shared_level=torch.zeros_like(q).tolist(),
                action_interaction=torch.zeros_like(q).tolist()))
    result = compare_frozen(value([[0., 2.], [1., 1.]]), value([[0., 2.], [-2., 1.]]))
    assert result["min_advantage"]["increment"] == 2.
    assert result["min_advantage"]["fixed_reference_head_contrast_increment"] == 0.
    assert result["min_advantage"]["min_head_selection_gap_increment"] == 2.
    assert result["min_advantage"]["decomposition_error"] == 0.
