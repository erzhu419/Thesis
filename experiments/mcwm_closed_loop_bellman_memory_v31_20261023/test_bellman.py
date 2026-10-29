"""Check real trainer parity, causal target blindness and zero probe side effects."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_bellman_memory_v31_20261023.bellman import FrozenBellman
from experiments.mcwm_closed_loop_context_shadow_v5_20261020.shadow import (
    restore_rng, rng_state, _same)
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.test_learning import (
    FIELDS, make_trainer, table)
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import (
    _batch, _state)


def row():
    data = table(.72, count=2)
    return SimpleNamespace(**{field: data[field][0].copy() for field in FIELDS.values()},
        started_at=17000., completed_at=17005., available_at=17005., sequence=12775)


def pair():
    learner = make_trainer("fresh")
    learner.continue_step()  # Exercise nonempty Adam/replay state, rather than an unused model.
    return {"fresh": learner, "no_fresh": deepcopy(learner)}


def private_rngs(learners):
    return {arm: {name: deepcopy(getattr(learner, name).bit_generator.state)
                  for name in ("rng", "diagnostic_rng")}
            for arm, learner in learners.items()}


def replay_state(learner):
    state = deepcopy(learner.replay_buffer.__dict__)
    state["recent"] = list(state["recent"])
    return state


def test_repeated_common_draw_probes_preserve_parameters_adam_replay_and_rng():
    learners = pair()
    evaluator = FrozenBellman(learners)
    state_before = {arm: deepcopy(_state(learner)) for arm, learner in learners.items()}
    replay_before = {arm: replay_state(learner) for arm, learner in learners.items()}
    globals_before, private_before = rng_state(), private_rngs(learners)
    first = evaluator.evaluate(row())
    assert all(first["integrity"].values())
    assert _same(globals_before, rng_state())
    assert _same(private_before, private_rngs(learners))
    # Randomness consumed elsewhere must not alter the paired common-random probe.
    torch.rand(7)
    np.random.random(3)
    second_before = rng_state()
    second = evaluator.evaluate(row())
    assert first == second
    assert _same(second_before, rng_state())
    for arm, learner in learners.items():
        assert _same(state_before[arm], _state(learner))
        assert _same(replay_before[arm], replay_state(learner))
    assert evaluator.calls == 2
    assert first["diagnostic_budget"] == dict(optimizer_updates=0, training_labels=0)


@pytest.mark.parametrize("entropy", [True, False])
def test_target_components_match_actual_trainer_for_ordinary_zero_and_terminal_rows(entropy):
    learners = pair()
    for learner in learners.values():
        learner.config = replace(learner.config, backup_entropy=entropy)
    evaluator = FrozenBellman(learners)
    for dt, terminal in ((5., False), (0., False), (5., True)):
        transition = row()
        transition.dt, transition.terminal = dt, terminal
        result = evaluator.evaluate(transition)
        for arm, learner in learners.items():
            global_before, private_before = rng_state(), private_rngs(learners)
            try:
                evaluator._common_draw(learner)
                actual = learner._target(_batch([transition], learner.device), backup="h2oplus")
            finally:
                restore_rng(global_before)
                for name, state in private_before[arm].items():
                    getattr(learner, name).bit_generator.state = state
            point = result["models"][arm]
            assert point["target"] == float(actual[0])
            assert point["target_path_exact"]
            assert point["effective_discount"] == float(not terminal)
            assert point["reward_scaled"] + point["bootstrap_contribution"] == pytest.approx(point["target"])
            assert point["value_contribution"] + point["sac_contribution"] == pytest.approx(
                point["bootstrap_contribution"])
            assert point["sac_q_without_entropy"] + point["sac_entropy_bonus"] == pytest.approx(
                point["sac_bootstrap"])
            if not entropy:
                assert point["sac_entropy_bonus"] == 0.
            if terminal:
                assert point["bootstrap_contribution"] == 0.


def test_same_next_state_reward_and_terminal_make_target_blind_to_current_action_and_dt():
    evaluator = FrozenBellman(pair())
    low, high = row(), row()
    low.a, high.a = np.array([-1.], np.float32), np.array([-.9375], np.float32)
    low.dt, high.dt = 0., 2.
    a, b = evaluator(low), evaluator(high)
    assert a["zero_duration"] and not b["zero_duration"]
    for arm in ("fresh", "no_fresh"):
        left, right = a["models"][arm], b["models"][arm]
        for key in ("target", "next_value", "sac_bootstrap", "next_actions", "target_subset"):
            assert left[key] == right[key]
        assert left["q_heads"] != right["q_heads"]


def test_probe_exception_still_restores_global_and_private_rng(monkeypatch):
    learners = pair()
    evaluator = FrozenBellman(learners)
    global_before, private_before = rng_state(), private_rngs(learners)

    def broken(*args):
        raise RuntimeError("failed diagnostic value forward")

    monkeypatch.setattr(learners["fresh"].value, "forward", broken)
    with pytest.raises(RuntimeError, match="failed diagnostic"):
        evaluator.evaluate(row())
    assert _same(global_before, rng_state())
    assert _same(private_before, private_rngs(learners))
