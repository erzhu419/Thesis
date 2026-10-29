"""Native closure stop, private action RNG, real-clock admission and label rights."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('THESIS_H2O_ROOT', str(Path(__file__).resolve().parents[1]/'sumo_eval/runtime/H2Oplus'))
import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_sync_pilot_v43_20261024.collect import (
    StochasticController, ResponseStream, branch_rows, transition_dict,
    install_release_clock, observed_branch_budget,
)
from experiments.mcwm_closed_loop_critic_memory_v32_20261023.test_memory import MemoryNativeEnv
from experiments.mcwm_closed_loop_critic_memory_v32_20261023.memory import ExecutionMemory
from experiments.mcwm_closed_loop_paired_coverage_v37_20261024.test_collect import native_row
from experiments.mcwm_closed_loop_v1.stream import TransitMDPConfigV7


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.2))

    def distribution_parameters(self, states):
        batch = len(states)
        return (self.weight.expand(batch), self.weight.expand(batch), torch.zeros(batch))

    def forward(self, states, deterministic=False):
        assert deterministic is False
        return torch.tanh(self.weight+torch.randn((len(states), 1))), None


class Transform:
    def numpy(self, state, proposal):
        return proposal.copy()


def test_private_native_sampling_does_not_change_sgd_rng_and_frozen_snapshot_is_immutable():
    learner = SimpleNamespace(actor=Actor(), action_transform=Transform(), device='cpu', total_updates=50000)
    controller = StochasticController(learner, seed=42)
    another = StochasticController(learner, seed=42)
    state = np.zeros(229, np.float32)
    saved = torch.get_rng_state().clone()
    first = controller(state)
    assert torch.equal(torch.get_rng_state(), saved)
    np.testing.assert_array_equal(first, another(state))
    with torch.no_grad():
        learner.actor.weight.fill_(7.)
    assert controller.reference.actor.weight.item() == pytest.approx(.2)
    assert controller.last['actor_snapshot'] == 'initial50000'
    learner.total_updates = 50024
    controller.switch_post24(learner)
    assert controller.reference.actor.weight.item() == 7.
    assert controller.draws == 1
    controller(state)
    assert controller.last['actor_snapshot'] == 'actual_post24'
    assert controller.last['action_rng'] == {'stream_seed': 42, 'draw_ordinal': 1}
    assert torch.equal(torch.get_rng_state(), saved)
    learner.total_updates = 50032
    with pytest.raises(RuntimeError, match='actual24th'):
        controller.switch_post24(learner)


def test_first45_real_closures_include_zero_dt_and_do_not_issue46th_action(tmp_path):
    # Two simultaneous first controls guarantee the legitimate zero-dt root.
    batches = [(10, [('A', 0), ('B', 0)])]
    batches += [(clock, [(f'B{clock}', 0)]) for clock in range(11, 70)]
    env = MemoryNativeEnv(batches)
    issues, state_inputs = [], []

    def policy(state):
        state_inputs.append(state.copy())
        return np.asarray([-1.], np.float32)

    def issue(event):
        issues.append(event)
        event['behavior_s'] = dict(policy_input=state_inputs[-1].tolist(),
            execution_input=state_inputs[-1].tolist(), actor_snapshot='actual_post24')

    stream = ResponseStream(env, [3], TransitMDPConfigV7(horizon_seconds=24000),
        selector=lambda event: [-1.], on_issue=issue, execute_transform=False,
        fork_duration_seconds=300., delay_seconds=30.)
    stream.reset()
    stream.memory = ExecutionMemory(env._bridge)
    stream.response_rows = []
    stream.response_line, stream.response_arm = '7X', 'low'
    with (tmp_path/'events.jsonl').open('x') as handle:
        stream.response_file = handle
        while not stream.cutoff_reached:
            stream.step(policy)
    assert stream.response_finished
    assert len(stream.response_rows) == 45
    assert stream.response_rows[0].dt == 0.
    assert stream.issued_count == len(issues) == len(state_inputs) == 45
    assert stream.now == stream.response_rows[-1].completed_at < 310.
    assert stream._pending is None
    assert not stream.done and stream.final_snapshot is None
    records = [json.loads(line)['transition'] for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert [row['sequence'] for row in records] == list(range(45))
    assert len({row['feedback_id'] for row in records}) == 45
    assert all(row['available_at'] == row['completed_at']+30. for row in records)
    assert all(row['ns'] == nxt['s'] for row, nxt in zip(records, records[1:]))
    stream.memory.close()


def test_real9630_tick_admission_does_not_require_a_control_event_and_skips_suffix():
    bridge = SimpleNamespace(clock=9628.)

    def native_tick():
        bridge.clock += 1.
        return 'native'

    bridge._advance_one_step = native_tick

    class Stream:
        @property
        def now(self):
            return bridge.clock

    child, calls = [False], []
    install_release_clock(bridge, Stream(), lambda: child[0], lambda: calls.append(bridge.clock))
    assert bridge._advance_one_step() == 'native'
    assert calls == []
    bridge._advance_one_step()
    assert calls == [9630.]
    bridge.clock = 9629.
    child[0] = True
    bridge._advance_one_step()
    assert calls == [9630.]


def test_response_admission_uses_all270_in_declared_order_without_short_or_unreleased_fill(tmp_path):
    branches = []
    for line in ('7X', '102X', '705X'):
        for arm in ('low', 'high'):
            path = tmp_path/f'{line}_{arm}.jsonl'
            records = []
            for event in range(45):
                row = native_row(100+event, 9100.+event, 9100.+event, -1.)
                row.line_id = line
                row.behavior_s = dict(actor_snapshot='actual_post24')
                row.feedback_id = f'response/{line}/{arm}/{event}'
                records.append(dict(transition=transition_dict(row)))
            path.write_text(''.join(json.dumps(record)+'\n' for record in records))
            branches.append(dict(line_id=line, arm=arm, events_path=str(path)))
    rows = branch_rows(branches, release_clock=9630.)
    assert len(rows) == len({row.feedback_id for row in rows}) == 270
    assert [row.feedback_id for row in rows[:6]] == [f'response/{line}/{arm}/0'
        for line in ('7X', '102X', '705X') for arm in ('low', 'high')]
    assert all(row.dt == 0. for row in rows)
    with pytest.raises(RuntimeError, match='short or unreleased'):
        branch_rows(branches, release_clock=9100.)
    path = Path(branches[0]['events_path'])
    path.write_text('\n'.join(path.read_text().splitlines()[:-1])+'\n')
    with pytest.raises(RuntimeError, match='short or unreleased'):
        branch_rows(branches, release_clock=9630.)


def test_failed_pair_keeps_completed_and_partial_suffix_costs_separate(tmp_path):
    low = tmp_path/'branches/7X/low'
    high = tmp_path/'branches/7X/high'
    low.mkdir(parents=True)
    high.mkdir(parents=True)
    (low/'summary.json').write_text(json.dumps(dict(status='complete', rows=45,
        budget=dict(native_seconds=82))))
    (high/'events.jsonl').write_text(json.dumps(dict(transition=dict(started_at=9100., completed_at=9117.)))+'\n')
    result = observed_branch_budget(tmp_path)
    assert result['observed_suffixes'] == 2
    assert result['observed_suffix_native_seconds'] == 99.
    assert result['real_closed_rows'] == 46
    assert result['unknown_or_lower_bound_suffixes'] == 1
    assert result['arms'][1]['status'] == 'interrupted'
