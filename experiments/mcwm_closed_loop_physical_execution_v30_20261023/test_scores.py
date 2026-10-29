"""The added physical candidate keeps the saved grid and training RNG intact."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location('v30_candidate_runner_test', HERE/'run.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class Critic:
    def __init__(self, offset, *, batch_dependent=False):
        self.offset, self.batch_dependent = offset, batch_dependent
        self.shapes = []

    def __call__(self, states, actions):
        self.shapes.append((tuple(states.shape), tuple(actions.shape)))
        score = states[:, 0]+states[:, 1]/16.+2.*actions[:, 0]+self.offset
        if self.batch_dependent:
            score = score+actions.mean()
        return torch.stack((score, score-1.), dim=0)


def bank(*, batch_dependent=False):
    states = torch.zeros((12, 229), dtype=torch.float32)
    states[:, 0] = .8  # Both selected states are in the unchanged tail F regime.
    states[:, 1] = torch.arange(12, dtype=torch.float32)/16.
    transform = ExecutionAction(dict(max_hold_seconds=60., scheduled_headway_scale_seconds=600.))
    learners = {arm: SimpleNamespace(action_transform=transform,
        critic=Critic(offset, batch_dependent=batch_dependent))
        for arm, offset in (('fresh', 0.), ('no_fresh', .25))}
    pair = SimpleNamespace(states=states, learners=learners)
    proposals = torch.tensor([-.5, 0., .25, -1., -.75]).repeat(12, 1)
    grid = states[:, None, :].expand(-1, 5, -1).reshape(-1, 229)
    executed = transform(grid, proposals.reshape(-1, 1))
    scores = {arm: learner.critic(grid, executed).reshape(-1, 12, 5).min(dim=0).values
        for arm, learner in learners.items()}
    rows = [dict(bank_id=f'fixed/{index}', proposal_action=proposals[index].tolist(),
        q_min={arm: values[index].tolist() for arm, values in scores.items()})
        for index in range(12)]
    return pair, rows


def test_added_15_second_proposal_executes_1_875_without_changing_old_scores_or_rng():
    pair, rows = bank()
    saved_rng = _rng()
    result = runner.cross_tick_candidate(pair, rows, 'fixed/9')
    assert result['action'] == [-.9375]
    assert result['holding_seconds'] == 1.875
    assert _equal(saved_rng, _rng())
    expected_states = pair.states[9:10]
    for arm, learner in pair.learners.items():
        expected = learner.critic(expected_states, torch.tensor([[-.9375]])).min().item()
        assert result['min_q'][arm] == expected
        # The scoring invocation retains the original 12 states x 5 candidates.
        assert learner.critic.shapes[:2] == [((60, 229), (60, 1))]*2


def test_batch_dependent_change_to_an_unchanged_candidate_is_rejected():
    pair, rows = bank(batch_dependent=True)
    with pytest.raises(RuntimeError, match='changed unchanged candidate predictions'):
        runner.cross_tick_candidate(pair, rows, 'fixed/9')
