"""Verify physical feedback admission and the exact24+432 update order."""
from collections import deque
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_sync_pilot_v43_20261024 import learning


class Replay:
    def __init__(self):
        self.recent, self.target_received, self.last_sampling = deque(maxlen=285), 0, {}

    def add_target(self, rows):
        self.recent.extend(rows)
        self.target_received += len(rows)


class Learner:
    def __init__(self, replay):
        self.total_updates, self.replay, self.calls = 50000, replay, []
        self.rng, self.diagnostic_rng = np.random.default_rng(1), np.random.default_rng(2)
        for name in ('actor', 'critic', 'target_critic', 'value', 'ratio'):
            setattr(self, name, torch.nn.Linear(1, 1))
        for name in ('actor_optimizer', 'critic_optimizer', 'value_optimizer', 'alpha_optimizer'):
            parameter = torch.nn.Parameter(torch.ones(1))
            setattr(self, name, SimpleNamespace(param_groups=[{'params': [parameter]}],
                state={parameter: {'step': torch.tensor(50000.)}}))

    def continue_step(self, collect_diagnostics=False):
        self.calls.append(self.replay.target_received)
        self.total_updates += 1
        for name in ('actor_optimizer', 'critic_optimizer', 'value_optimizer', 'alpha_optimizer'):
            for value in getattr(self, name).state.values():
                value['step'] += 1
        self.replay.last_sampling = {'recent': [int(self.rng.integers(self.replay.target_received))]}
        return {'train/update': self.total_updates}


def warmed(tmp_path):
    replay = Replay()
    learner = Learner(replay)
    engine = learning.PilotUpdates(tmp_path/'learning', {'credit_score_after_updates': [24, 456]})
    for packet in range(3):
        rows = [SimpleNamespace(available_at=1000.+300*packet, feedback_id=f'O{5*packet+i}') for i in range(5)]
        engine.warmup(learner, replay, rows, {'processed_at': 1200.+300*packet})
    return engine, learner, replay


def test_all270_response_slots_precede432_updates_and_adam_is_retained(tmp_path, monkeypatch):
    engine, learner, replay = warmed(tmp_path)
    pairs = []
    for line in ('7X', '102X', '705X'):
        branches = {}
        for arm in ('low', 'high'):
            path = tmp_path/f'{line}_{arm}.jsonl'
            path.write_text(json.dumps({'transition': {'r': int(arm == 'high')}})+'\n')
            branches[arm] = {'events_path': str(path)}
        pairs.append({'line_id': line, 'selected': {}, 'branches': branches})
    scored = []
    engine._score = lambda value: scored.append(value.total_updates)
    monkeypatch.setattr(learning, 'prepare_chain', lambda *args: [])
    saved = []
    monkeypatch.setattr(torch, 'save', lambda value, target: saved.append(value['updates']))
    rows = [SimpleNamespace(feedback_id=f'R{i}', available_at=9500., dt=0.) for i in range(270)]
    engine.response(learner, replay, rows, {'processed_at': 9630., 'pairs': pairs}, object())
    assert learner.calls == [5]*8+[10]*8+[15]*8+[285]*432
    assert learner.total_updates == 50456 and len(engine.packets) == 57
    assert len(replay.recent) == replay.target_received == 285
    assert [row.feedback_id for row in list(replay.recent)[:15]] == [f'O{i}' for i in range(15)]
    assert scored == list(range(50024, 50457, 8))
    assert saved == [456]
    assert engine.summary(learner, replay)['integrity']['all_pass']
    assert all(not parameter.requires_grad for parameter in learner.actor.parameters())


@pytest.mark.parametrize('problem', ['late', 'duplicate'])
def test_unreleased_or_duplicate_response_does_not_spend_updates(tmp_path, problem):
    engine, learner, replay = warmed(tmp_path)
    rows = [SimpleNamespace(feedback_id=f'R{i}', available_at=9500., dt=0.) for i in range(270)]
    if problem == 'late':
        rows[-1].available_at = 9631.
    else:
        rows[-1].feedback_id = rows[0].feedback_id
    with pytest.raises(RuntimeError, match='duplicated or physically unavailable'):
        engine.response(learner, replay, rows, {'processed_at': 9630.}, object())
    assert learner.total_updates == 50024 and replay.target_received == 15
