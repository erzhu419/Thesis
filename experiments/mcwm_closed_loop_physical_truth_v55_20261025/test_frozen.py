"""Catch double-F scoring, lost memory, changed final models and RNG drift."""
from collections import deque
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.learning import (
    attach_memory, critic_states)
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.test_learning import initial, rows
from experiments.mcwm_closed_loop_physical_shadow_v54_20261025.score import actor_candidates, score_bank
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _state, _rng
from experiments.mcwm_closed_loop_physical_truth_v55_20261025 import frozen


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def bank():
    result = []
    for line, row in zip(('7X', '102X', '705X'), rows(.4)):
        row.line_id, row.sequence, row.started_at = line, len(result), 9010.+len(result)
        row.memory_s['observed_at'] = row.started_at
        result.append(dict(bank_id=f'{line}/root', line_id=line, row=row))
    return result


def fixture(tmp_path, monkeypatch):
    protocol = dict(scenarios=['stable'], training_seed=6202, source_version='v53')
    write(tmp_path/'protocol.json', protocol)
    source = attach_memory(initial())
    source.total_updates = 50456
    source.replay_buffer.target_received = 285
    source.replay_buffer.recent = deque([{} for _ in range(285)])
    learners = {arm: deepcopy(source) for arm in frozen.ARMS}
    with torch.no_grad():
        learners['trace'].critic.level_weight.fill_(.03)
        learners['trace'].critic.action_weight.fill_(.2)
    items = bank()
    monkeypatch.setattr(frozen, 'load_bank', lambda *args: items)
    shared = {arm: actor_candidates(learner, items) for arm, learner in learners.items()}
    for arm, learner in learners.items():
        root = tmp_path/'runs'/'stable'/arm
        write(root/'summary.json', dict(status='complete', integrity=dict(all_pass=True),
            scenario='stable', arm=arm, budget=dict(initializer_updates=50000,
                feedback_slots=285, optimizer_updates=456, source_horizon_seconds=24000)))
        write(root/'candidates_54.json', dict(packet=54, candidates=shared[arm]))
        write(root/'scores.jsonl', dict(packet=54, cumulative_updates=456,
            rows=score_bank(learner, items, packet=54, cumulative_updates=456,
                shared_actor_proposals=shared, own_candidates=shared[arm], arm=arm)))
        torch.save(dict(learner=learner, updates=456, feedback_slots=285,
            protocol=protocol, source_ready='v53/runs/stable/feedback_ready_model.pt',
            contract={'max_hold_seconds': 60}, stations=np.array([1, 2])), root/'post456_model.pt')
    return learners


def test_final_models_and_complete_old_prediction_gate_are_exact(tmp_path, monkeypatch):
    expected = fixture(tmp_path, monkeypatch)
    before = _rng()
    learners, contract, stations, proof = frozen.build_frozen('stable', source_root=tmp_path)
    assert _equal(before, _rng()) and proof['all_pass']
    assert proof['new_optimizer_updates'] == proof['new_feedback_slots'] == 0
    assert contract['max_hold_seconds'] == 60 and list(stations) == [1, 2]
    for arm, learner in learners.items():
        assert _equal(_state(expected[arm]), _state(learner))
        assert proof['old_final_predictions'][arm]['final_three_line_prediction_rows'] == 3
        assert not learner.log_alpha.requires_grad
        assert all(not parameter.requires_grad for parameter in learner.critic.parameters())


def test_changed_final_prediction_or_budget_blocks_new_truth(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    path = tmp_path/'runs'/'stable'/'trace'/'scores.jsonl'
    changed = json.loads(path.read_text())
    changed['rows'][0]['q_min'][0] += 1
    write(path, changed)
    with pytest.raises(RuntimeError, match='full prediction differs'):
        frozen.build_frozen('stable', source_root=tmp_path)
    path = tmp_path/'runs'/'stable'/'one_step'/'summary.json'
    changed = json.loads(path.read_text())
    changed['budget']['optimizer_updates'] = 455
    write(path, changed)
    with pytest.raises(RuntimeError, match='budget/scenario gate differs'):
        frozen.build_frozen('stable', source_root=tmp_path)


def test_inverse_F_retains_actual_support_aliases_and_floor():
    learner = initial()
    state = rows(.85)[0].s
    candidates = frozen.executable_candidates(learner.action_transform, state)
    assert candidates['holding_seconds'] == [0., 7.5]
    assert candidates['proposal_actions'] == [-1., 1.]
    assert candidates['supported'] == [True, True] and candidates['physically_distinct']
    # The project's real completion floor aliases both requested actions to60s.
    state[0] = .4
    slot = state[5:].reshape(16, 14)[0]
    slot[11], slot[12], slot[9], slot[8] = 1., 0., .1, .2
    blocked = frozen.executable_candidates(learner.action_transform, state)
    assert blocked['holding_seconds'] == [60., 60.]
    assert blocked['supported'] == [False, False]
    assert blocked['alias_groups'] == [0, 0] and not blocked['physically_distinct']


def test_issue_scoring_keeps_memory_heads_Adam_RNG_and_avoids_double_F():
    learner = attach_memory(initial())
    with torch.no_grad():
        learner.critic.level_weight.fill_(.03)
        learner.critic.action_weight.fill_(.3)
    learner = frozen.freeze(learner)
    pair = {arm: deepcopy(learner) for arm in frozen.ARMS}
    state, context = rows(.85)[0].s, np.full(84, .4, np.float32)
    actions = [-1., -.75]  # Already executed0/7.5 seconds; F would shrink again.
    states_before = {arm: deepcopy(_state(item)) for arm, item in pair.items()}
    rng_before = _rng()
    scored = frozen.score_frozen(pair, state, context, actions)
    explicit = critic_states(learner, torch.tensor(state)[None].repeat(2, 1),
        torch.tensor(context)[None].repeat(2, 1))
    with torch.no_grad():
        q = learner.critic(explicit, torch.tensor(actions)[:, None])
    assert scored['trace']['q_heads'] == q.tolist()
    assert scored['trace']['q_min'] == q.min(0).values.tolist()
    assert scored['trace']['holding_seconds'] == [0., 7.5]
    assert not scored['trace']['scoring_uses_target_estimator']
    for arm, item in pair.items():
        assert _equal(states_before[arm], _state(item))
    assert _equal(rng_before, _rng())
    masked = frozen.score_frozen(pair, state, np.zeros(84, np.float32), actions)
    assert masked['trace']['q_heads'] != scored['trace']['q_heads']
    duplicate = frozen.score_frozen(pair, state, context, [actions[1], actions[1]])
    assert duplicate['trace']['advantage_min'] == [0., 0.]
    assert duplicate['trace']['unique_executed_candidates'] == 1
