"""Native suffix observation without changing old truth or causal state."""
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from experiments.mcwm_closed_loop_action_fork_v7_20261020.test_fork_stream import policy
from experiments.mcwm_closed_loop_v1.stream import TransitMDPConfigV7
from experiments.mcwm_closed_loop_physical_execution_v30_20261023.test_suffix import (
    AppliedNativeEnv, run_suffix as old_run_suffix,
)
from experiments.mcwm_closed_loop_bellman_memory_v31_20261023.suffix import NativeTruthTriple, HOLD_SECONDS
from experiments.mcwm_closed_loop_bellman_memory_v31_20261023.visibility import VisibilityEventStream, VisibilityRecorder


class FrozenProbe:
    def __init__(self):
        self.calls = []

    def evaluate(self, row):
        self.calls.append(row)
        return dict(raw_reward=row.r, models=dict(fresh=dict(target=row.r)),
            integrity=dict(rng_restored=True, learner_rng_restored=True, target_path_exact=True))


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def run_suffix(tmp_path, arm, batches, *, physical_cost=False):
    contract = TransitMDPConfigV7(horizon_seconds=24000)
    actions = {name: [2*seconds/contract.max_hold_seconds-1] for name, seconds in HOLD_SECONDS.items()}
    counts = dict(policy=456, feedback=285, fresh_shadow=248, no_fresh_shadow=248, world=0, trust=0)
    (tmp_path/arm).mkdir(parents=True)
    probe = FrozenProbe()
    pair = NativeTruthTriple(tmp_path, dict(selected=dict(sequence=0), actions=actions),
        lambda: counts.copy(), 106202, evaluator=probe)
    pair.in_child, pair.arm, pair.before, pair.started_wall = True, arm, counts.copy(), 1.
    env = AppliedNativeEnv(batches)
    stream = VisibilityEventStream(env, [3], contract, execute_transform=False,
        selector=lambda event: actions[arm], fork_duration_seconds=300.,
        on_issue=lambda event: pair.observe_issue(event, stream), delay_seconds=30.)
    stream.reset()
    if physical_cost:
        original_snapshot = env._bridge.passenger_time_ledger.snapshot

        def snapshot():
            row = original_snapshot()
            extra = int(arm == 'high' and row.time_seconds >= 12)
            return replace(row, accrued_wait_seconds=row.accrued_wait_seconds+extra,
                accrued_in_vehicle_seconds=row.accrued_in_vehicle_seconds+3*extra)

        env._bridge.passenger_time_ledger.snapshot = snapshot
    pair._install_native_observer(stream, traci=env.traci)
    while not stream.cutoff_reached:
        stream.step(policy)
        pair.observe_progress(stream)
    summary = pair.build_summary(stream)
    return pair, stream, summary, read_rows(tmp_path/arm/'visibility.jsonl'), read_rows(tmp_path/arm/'ledger_ticks.jsonl'), probe


def test_global_zero_dt_and_other_bus_memory_are_observed_after_real_apply(tmp_path):
    batches = [(10, [('A', 0), ('B', 0)]), (20, [('C', 0)]), (30, [('D', 0)]), (1000, [('E', 0)])]
    branches = [run_suffix(tmp_path/arm, arm, batches) for arm in ('low', 'high')]
    for _, _, summary, rows, ticks, probe in branches:
        assert summary['integrity']['all_pass']
        assert rows[0]['transition']['dt'] == 0
        assert rows[0]['memory_s']['execution_applied'] is False
        assert rows[0]['memory_ns']['execution_applied'] is False
        assert rows[1]['memory_s']['execution_applied'] is False
        assert rows[1]['memory_ns']['execution_applied'] is True
        assert rows[2]['memory_s']['actor_is_selected_vehicle'] is False
        assert len(ticks) == 301
        assert ticks[0]['relative_seconds'] == 0 and ticks[-1]['relative_seconds'] == 300
        assert len(probe.calls) == 3
        assert rows[-1]['kind'] == 'cutoff_partial'
    left, right = branches[0][3][2], branches[1][3][2]
    np.testing.assert_array_equal(left['transition']['s'], right['transition']['s'])
    assert left['memory_s']['executed_hold_seconds'] == 0.
    assert right['memory_s']['executed_hold_seconds'] == 1.875
    assert left['memory_s'] != right['memory_s']


def test_cutoff_partial_keeps_state_and_bootstrap_missing_and_pending_unchanged(tmp_path):
    _, stream, summary, rows, _, probe = run_suffix(tmp_path, 'mid',
        [(10, [('A', 0)]), (20, [('B', 0)]), (1000, [('C', 0)])])
    partial = rows[-1]
    assert partial['kind'] == 'cutoff_partial'
    assert partial['transition']['ns'] is partial['transition']['r'] is partial['bellman'] is None
    assert partial['partial_cost'] == dict(wait=580., ride=870., cost=2030.)
    assert stream._pending['vehicle_id'] == 'B'
    assert stream.issued_count == 2 and len(probe.calls) == 1
    assert summary['truth']['fixed300'] == dict(status='complete', dt=300., wait=600., ride=900., cost=2100.)
    assert not stream.done and stream.final_snapshot is None


def test_real_available_cutoff_state_is_evaluated_without_issuing_or_queue_mutation(tmp_path):
    _, stream, summary, rows, _, probe = run_suffix(tmp_path, 'high',
        [(10, [('A', 0)]), (20, [('B', 0)]), (310, [('C', 0)]), (1000, [('D', 0)])])
    assert rows[-1]['kind'] == 'cutoff_actual_without_issue'
    assert rows[-1]['transition']['completed_at'] == 310.
    assert rows[-1]['transition']['next_vehicle_id'] == 'C'
    assert rows[-1]['transition']['ns'] is not None and rows[-1]['bellman'] is not None
    assert stream.issued_count == 2 and stream._pending['vehicle_id'] == 'B'
    assert stream.completed_count == 1 and len(probe.calls) == 2
    assert summary['visibility']['cutoff_partial_count'] == 0
    assert summary['integrity']['all_pass']


def test_history_observation_never_backfills_an_earlier_memory(tmp_path):
    physical = SimpleNamespace(previous=None, departure=None, next_arrival=None,
        history_departure=dict(observed_at=30., clock_seconds=15.), history_arrival=None)
    selected = dict(trip_id='A', vehicle_id='L::A', station_id='S0', started_at=10.)
    recorder = VisibilityRecorder(tmp_path, selected, physical, FrozenProbe())
    earlier = recorder.memory(20., 'L::B')
    assert earlier['departed_observed'] is False
    assert recorder.memory(30., 'L::B')['departed_observed'] is True
    assert earlier['departed_observed'] is False
    assert earlier['executed_hold_seconds'] is None
    recorder.observe_apply(dict(applied_hold_seconds=.9375, real_clock_after=30., time=29., issue_ledger_time=30.))
    assert recorder.memory(31.)['executed_hold_seconds'] == .9375
    assert earlier['execution_applied'] is False


def test_common_clock_cost_onset_is_independent_of_control_event_alignment(tmp_path):
    batches = [(10, [('A', 0)]), (20, [('B', 0)]), (1000, [('C', 0)])]
    low = run_suffix(tmp_path/'low', 'low', batches, physical_cost=True)
    high = run_suffix(tmp_path/'high', 'high', batches, physical_cost=True)
    paired = list(zip(low[4], high[4]))
    changed = [(left, right) for left, right in paired if left['cost'] != right['cost']]
    assert changed[0][0]['relative_seconds'] == 2.
    assert changed[0][1]['cost']-changed[0][0]['cost'] == 5.
    assert high[3][0]['transition']['completed_at'] == 20.
    assert high[2]['truth']['fixed300']['cost']-low[2]['truth']['fixed300']['cost'] == 5.


def test_old_truth_physical_and_first_transition_remain_exact(tmp_path):
    batches = [(10, [('A', 0), ('B', 0)]), (20, [('C', 0)]), (1000, [('D', 0)])]
    for arm in ('low', 'mid', 'high'):
        _, _, old = old_run_suffix(tmp_path/'old'/arm, arm, batches)
        _, _, new, _, _, _ = run_suffix(tmp_path/'new'/arm, arm, batches)
        for key in ('selected', 'execution', 'physical', 'truth', 'global_transition', 'native', 'counters', 'budget'):
            assert new[key] == old[key], key
