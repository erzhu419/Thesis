"""Native replay, level-induced min-head changes and read-only decomposition."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_memory_truth_v34_20261024.test_frozen import learners, packets
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.frozen import score_frozen
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _state, _rng
from experiments.mcwm_closed_loop_critic_decomposition_v35_20261024.decomposition import decompose


def inputs():
    pair = learners()
    row = packets()[-1]["rows"][0]
    selected = dict(s=row.s.tolist(), memory_s=row.memory_s)
    actions = dict(low=[-1.], high=[-.9375])
    return pair, selected, actions


def reference(pair, selected, actions):
    return score_frozen(pair, selected["s"], selected["memory_s"]["vector"],
                        [actions["low"], actions["high"]])


def test_complete_native_prediction_replay_and_per_head_closure_preserve_state_and_rng():
    pair, selected, actions = inputs()
    expected = reference(pair, selected, actions)
    before = {name: deepcopy(_state(learner)) for name, learner in pair.items()}
    private = {name: deepcopy(learner.diagnostic_rng.bit_generator.state) for name, learner in pair.items()}
    rng = _rng()
    result = decompose(pair, selected, actions, reference=expected)
    assert result["integrity"]["all_pass"]
    for name, arm in (("full", "memory"), ("zero_memory", "zero_memory")):
        assert result["variants"][name]["q_min"] == expected[arm]["q_min"]
        assert result["variants"][name]["q_mean"] == expected[arm]["q_mean"]
        assert result["variants"][name]["choice"] == expected[arm]["predicted_choice"]
        assert len(result["variants"][name]["q_heads"]) == 2
    components = result["components"]
    assert components["level_algebraic_delta_heads"] == [0., 0.]
    assert components["level_delta_max_roundoff"] <= components["native_order_tolerance"]
    assert components["native_order_max_error"] <= components["native_order_tolerance"]
    base = torch.tensor(result["variants"]["memory_base"]["q_heads"])
    level = torch.tensor(components["level_heads"])
    action = torch.tensor(components["action_heads"])
    full = torch.tensor(result["variants"]["full"]["q_heads"])
    torch.testing.assert_close(base+(level+action), full,
        atol=components["native_order_tolerance"], rtol=0)
    assert _equal(rng, _rng())
    for name, learner in pair.items():
        assert _equal(before[name], _state(learner))
        assert _equal(private[name], learner.diagnostic_rng.bit_generator.state)


def test_level_can_change_ensemble_min_head_and_advantage_without_same_head_action_effect():
    pair, selected, actions = inputs()
    selected["memory_s"]["vector"] = [1.]+[0.]*83
    critic = pair["memory"].critic
    states = torch.tensor(selected["s"])[None].expand(2, -1)
    with torch.no_grad():
        critic.level_weight.zero_()
        critic.action_weight.zero_()
        base = critic.base(states, torch.tensor([[-1.], [-.9375]]))
        # Place head1 between its two crossings with head0. The native network
        # has different per-head action slopes; a constant level can select
        # different heads at low/high without contributing any own slope.
        gaps = base[0]-base[1]
        assert gaps[0] != gaps[1]
        critic.level_weight[1, 0] = gaps.mean()
    result = decompose(pair, selected, actions, reference=reference(pair, selected, actions))
    assert result["components"]["level_delta_heads"] == [0., 0.]
    level = result["variants"]["level_only"]
    assert level["min_head_changes"]
    assert level["min_heads"][0] != level["min_heads"][1]
    assert level["selection_gap"] < 0
    assert level["delta_min"] != result["variants"]["memory_base"]["delta_min"]
    assert result["variants"]["full"] == level


def test_action_interaction_changes_head_slopes_and_base_drift_is_separate():
    pair, selected, actions = inputs()
    with torch.no_grad():
        pair["memory"].critic.level_weight.zero_()
        pair["memory"].critic.action_weight.fill_(.4)
        # This is a different learned base, not an input-mask intervention.
        pair["memory"].critic.base.network.layers[-1].bias.add_(.25)
    result = decompose(pair, selected, actions, reference=reference(pair, selected, actions))
    components = result["components"]
    assert all(value > 0 for value in components["action_delta_heads"])
    full = torch.tensor(result["variants"]["full"]["delta_heads"])
    base = torch.tensor(result["variants"]["memory_base"]["delta_heads"])
    torch.testing.assert_close(full-base, torch.tensor(components["action_delta_heads"]), atol=1e-6, rtol=0)
    assert result["variants"]["full"] == result["variants"]["action_only"]
    drift = np.asarray(result["between_learner_base_drift"]["head_q_difference"])
    np.testing.assert_allclose(drift, .25, atol=1e-6, rtol=0)
    assert result["direct_memory_same_weights"]["full_minus_base_min"] > 0


def test_saved_prediction_mismatch_rejected_without_side_effects():
    pair, selected, actions = inputs()
    expected = reference(pair, selected, actions)
    expected["memory"]["q_min"][0] += 1.
    before = _rng()
    with pytest.raises(RuntimeError, match="memory_full_saved_prediction_exact"):
        decompose(pair, selected, actions, reference=expected)
    assert _equal(before, _rng())
