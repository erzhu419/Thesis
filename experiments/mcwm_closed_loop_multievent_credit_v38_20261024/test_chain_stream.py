"""Real endpoints preserve native chains and never manufacture a300s state."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault('THESIS_H2O_ROOT', str(Path(__file__).resolve().parents[1]/'sumo_eval/runtime/H2Oplus'))

from experiments.mcwm_closed_loop_action_fork_v7_20261020.test_fork_stream import policy
from experiments.mcwm_closed_loop_critic_memory_v32_20261023.test_memory import MemoryNativeEnv
from experiments.mcwm_closed_loop_critic_memory_v32_20261023.memory import ExecutionMemory
from experiments.mcwm_closed_loop_v1.stream import TransitMDPConfigV7
from experiments.mcwm_closed_loop_multievent_credit_v38_20261024.chain_stream import (
    ChainRecorder, CreditEventStream, ledger,
)
from experiments.mcwm_closed_loop_multievent_credit_v38_20261024.collect import CreditPair, failure_budget
from experiments.mcwm_closed_loop_physical_execution_v30_20261023.test_suffix import AppliedNativeEnv


def run_chain(tmp_path, endpoint):
    env = MemoryNativeEnv([(10, [('A', 0), ('B', 0)]), (20, [('C', 0)]),
        (30, [('D', 0)]), (endpoint, [('E', 0)]), (2000, [('F', 0)])])
    issues = []
    stream = None

    def issue(event):
        issues.append(event)
        if stream.recorder is None:
            selected = dict(sequence=event['sequence'], started_at=event['started_at'],
                issue_ledger=ledger(stream._snapshot()))
            stream.recorder = ChainRecorder(tmp_path, selected)
        stream.recorder.observe_tick(stream)

    stream = CreditEventStream(env, [3], TransitMDPConfigV7(horizon_seconds=24000),
        selector=lambda event: [-1.], on_issue=issue, execute_transform=False,
        fork_duration_seconds=300., delay_seconds=30., max_overrun_seconds=120.)
    stream.reset()
    stream.memory = ExecutionMemory(env._bridge)
    advance = env._bridge._advance_one_step

    def tick():
        value = advance()
        stream.recorder.observe_tick(stream)
        return value

    env._bridge._advance_one_step = tick
    while not stream.cutoff_reached:
        stream.step(policy)
    stream.recorder.events.flush(); stream.recorder.ticks.flush()
    rows = [json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    return stream, issues, rows


@pytest.mark.parametrize('endpoint', [310., 327.])
def test_first_real_endpoint_closes_pending_without_issuing_or_executing(tmp_path, endpoint):
    stream, issues, rows = run_chain(tmp_path, endpoint)
    assert stream.now == endpoint and not stream.done
    assert stream.boundary300['clock'] == stream.cutoff_snapshot.time_seconds == 310.
    assert stream.real_endpoint['status'] == 'observed'
    assert stream.real_endpoint['clock'] == endpoint
    assert stream.real_endpoint['issued'] is stream.real_endpoint['executed'] is False
    assert stream.issued_count == len(issues) == 4
    assert [row['transition']['sequence'] for row in rows] == [0, 1, 2, 3]
    assert rows[0]['transition']['dt'] == 0.
    assert rows[-1]['kind'] == 'endpoint_actual_without_issue'
    assert rows[-1]['transition']['completed_at'] == endpoint
    assert rows[-1]['transition']['next_vehicle_id'] == 'E'
    assert rows[-1]['transition']['ns'] == stream.real_endpoint['s']
    assert rows[-1]['transition']['memory_ns'] == stream.real_endpoint['memory']
    assert all(row['transition']['r'] == -(2.*(row['ledger_ns']['accrued_wait_seconds']-
        row['ledger_s']['accrued_wait_seconds'])+row['ledger_ns']['accrued_in_vehicle_seconds']-
        row['ledger_s']['accrued_in_vehicle_seconds']) for row in rows)
    assert stream.recorder.summary()['integrity']['all_pass']
    assert stream.recorder.tick_count == int(endpoint-10.)+1
    if endpoint > 310:
        assert stream.boundary300['state'] is None
        assert stream.boundary300['actual_control_at_clock'] is False
    else:
        assert stream.boundary300['state'] == stream.real_endpoint['s']
    stream.memory.close()


def test_no_real_endpoint_by420_keeps_partial_unlabelled(tmp_path):
    stream, issues, rows = run_chain(tmp_path, 1000.)
    assert stream.now == 430. and stream.cutoff_snapshot.time_seconds == 310.
    assert stream.real_endpoint['status'] == 'missing'
    assert stream.real_endpoint['s'] is stream.real_endpoint['memory'] is None
    assert stream.issued_count == len(issues) == 4
    assert len(rows) == 3
    assert stream.endpoint_pending_partial['ns'] is None
    assert stream.endpoint_pending_partial['r'] is None
    assert stream.endpoint_pending_partial['bootstrap'] is None
    assert stream.endpoint_pending_partial['completed_at'] == 430.
    assert stream.recorder.summary()['integrity']['all_pass']
    stream.memory.close()


@pytest.mark.parametrize('endpoint,overrun', [(327., 17.), (1000., 120.)])
def test_pair_summary_separates_exact300_truth_from_real_endpoint_and_extra_budget(tmp_path, endpoint, overrun):
    (tmp_path/'high').mkdir()
    counts = dict(total_updates=50000, new_training_labels=0, new_optimizer_updates=0)
    holds = dict(low=0., high=1.875)
    actions = {arm: [2.*value/60.-1.] for arm, value in holds.items()}
    pair = CreditPair(tmp_path, dict(selected=dict(sequence=0), actions=actions, holds=holds),
        lambda: counts.copy(), 106237)
    pair.in_child, pair.arm, pair.before, pair.started_wall = True, 'high', counts.copy(), 1.
    env = AppliedNativeEnv([(10, [('A', 0), ('B', 0)]), (20, [('C', 0)]),
        (30, [('D', 0)]), (endpoint, [('E', 0)]), (2000, [('F', 0)])])
    stream = CreditEventStream(env, [3], TransitMDPConfigV7(horizon_seconds=24000),
        selector=lambda event: actions['high'], execute_transform=False,
        on_issue=lambda event: pair.observe_issue(event, stream), fork_duration_seconds=300.,
        delay_seconds=30., max_overrun_seconds=120.)
    stream.reset()
    stream.memory = SimpleNamespace(snapshot=lambda clock: dict(observed_at=clock, vector=[0.]*84))
    pair._install_native_observer(stream, traci=env.traci)
    while not stream.cutoff_reached:
        stream.step(policy)
        pair.observe_progress(stream)
    summary = pair.build_summary(stream)
    assert summary['integrity']['all_pass']
    assert summary['truth']['fixed300'] == dict(status='complete', dt=300., wait=600., ride=900., cost=2100.)
    assert summary['truth']['next_event']['dt'] == 0.
    assert summary['execution']['native_applied_hold_seconds'] == 1.875
    assert summary['budget']['actual_suffix_native_seconds'] == 300.+overrun
    assert summary['budget']['overrun_native_seconds'] == overrun
    assert summary['budget']['new_optimizer_updates'] == summary['budget']['new_training_labels'] == 0
    if endpoint < 430.:
        assert summary['cost300_to_endpoint'] == dict(dt=17., wait=34., ride=51., cost=119.)
    else:
        assert summary['cost300_to_endpoint'] is None
        assert summary['pending_partial']['ns'] is None


def test_failure_budget_counts_real_work_without_promoting_interrupted_arm_to_full_suffix(tmp_path):
    branch = tmp_path/'states/7X_early'
    (branch/'low').mkdir(parents=True)
    (branch/'high').mkdir()
    (branch/'low/summary.json').write_text(json.dumps(dict(status='complete',
        budget=dict(actual_suffix_native_seconds=317.))))
    (branch/'high/ticks.jsonl').write_text(json.dumps(dict(relative_seconds=47.))+'\n')
    stream = SimpleNamespace(now=18302., initial_snapshot=object())
    value = failure_budget(tmp_path, stream, None)
    assert value['prefix_native_seconds'] == 18302.
    assert value['observed_suffixes'] == 2 and value['completed_suffixes'] == 1
    assert value['observed_suffix_native_seconds'] == 364.
    assert value['observed_overrun_native_seconds'] == 17.
    assert value['unknown_or_lower_bound_suffixes'] == 1
    assert value['diagnostic_truth_outcomes'] == 1
    assert value['new_training_labels'] == value['new_optimizer_updates'] == 0
    child = SimpleNamespace(in_child=True, expected=dict(selected=dict(started_at=18302.)))
    actual = failure_budget(tmp_path, SimpleNamespace(now=18349., initial_snapshot=object()), child)
    assert actual['actual_suffix_native_seconds'] == 47.
    assert actual['overrun_native_seconds'] == 0.

