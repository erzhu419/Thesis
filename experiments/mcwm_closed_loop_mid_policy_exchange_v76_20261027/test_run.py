"""Exercise the actual native loop's packet51 cutoff without running SUMO."""
import ast
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from experiments.mcwm_closed_loop_mid_policy_exchange_v76_20261027 import run as R


def node_named(name):
    module = ast.parse(Path(R.__file__).read_text())
    return next(node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name == name)


def test_actual_loop_stops_admission_and_all_updates_after_packet51(tmp_path):
    native_loop = next(node for node in ast.walk(node_named('run'))
        if isinstance(node, ast.While) and ast.unparse(node.test) == 'not stream.done')
    ns = dict(PACKETS=51, p=dict(critic_updates_per_packet=8), args=SimpleNamespace(seed=6208, arm='fresh'),
        branch=None, frozen_counts=None, freeze_clock=None, own_snapshot=None,
        clock_audit=dict(first_feedback_applied_at=1200), update_rows_matched=100,
        suffix_feedback_discarded=0, root_out=tmp_path, time=time, json=json, last_progress=time.monotonic(),
        updates=io.StringIO(), policy_records=io.StringIO(), policy=lambda state:None,
        progress=lambda *a,**k:None, strip_profiler=R.strip_profiler)
    class Budget:
        index = 50
        ingested = taken = 0
        receipts = [dict(max_available_at=15900)]*50
        def ingest(self, rows):
            assert ns['branch'] is None
            self.ingested += 1
        def due(self, now):
            assert ns['branch'] is None, 'The frozen suffix must never inspect a due packet'
            return True
        def take(self, now):
            self.taken += 1; self.index += 1
            self.receipts.append(dict(max_available_at=16200))
            return [1]*5, [2]*5
    class Replay:
        target_received = 250
        def add_target(self, covariates, outcomes):
            assert ns['branch'] is None
            self.target_received += len(covariates)
    metrics = dict(td_diagnostic=dict(real_scale=1, model_scale=0), loss=.5)
    class Learner:
        updates = 400
        def begin_actor_packet(self):pass
        def continue_step(self, **kwargs):
            assert ns['branch'] is None
            self.updates += 1
            return metrics
    class Stream:
        done = False
        now = 16199
        calls = 0
        def step(self, policy):
            self.calls += 1
            self.now = 16203 if self.calls == 1 else 24000
            if self.calls == 2:
                ns['branch']='low';self.done=True
            return [object()]*9
        def release_intervals(self):pass
    learner, replay, budget, stream = Learner(), Replay(), Budget(), Stream()
    expected = [dict(packet=51, step_in_packet=step, processed_at=16203, **metrics) for step in (1,8)]
    ns.update(learner=learner, replay=replay, budget=budget, stream=stream,
        reference_updates=io.StringIO(''.join(json.dumps(row)+'\n' for row in expected)),
        counts=lambda:dict(R.COUNTS, **{k:learner.updates for k in ('critic','value','actor','alpha','policy')}),
        policy_probe=SimpleNamespace(record=lambda *a,**k:dict(packet=51)),
        publish_actor=mock.Mock(return_value=dict(packets=51)))
    ast.fix_missing_locations(native_loop)
    exec(compile(ast.Module(body=[native_loop],type_ignores=[]), R.__file__, 'exec'),ns)
    assert budget.index == 51 and budget.ingested == budget.taken == 1
    assert learner.updates == 408 and replay.target_received == 255
    assert ns['update_rows_matched'] == 102
    assert ns['suffix_feedback_discarded'] == 9
    assert ns['frozen_counts'] == R.COUNTS
    ns['publish_actor'].assert_called_once()
    assert ns['publish_actor'].call_args.kwargs['max_available_at'] == 16200


def policy_namespace(tmp_path):
    policy = node_named('policy')
    for node in ast.walk(policy):
        if isinstance(node, ast.Nonlocal):
            node.__class__ = ast.Global
    module = ast.fix_missing_locations(ast.Module(body=[policy], type_ignores=[]))
    clock = 16210.
    state = np.asarray([clock/24000, .2],dtype=np.float32)
    source = dict(started_at=clock,s=state.tolist())
    ledger = SimpleNamespace(time_seconds=clock,accrued_wait_seconds=10.,
        accrued_in_vehicle_seconds=20.,waiting_passengers=1,in_vehicle_passengers=2)
    lock = (tmp_path/'entry.lock').open('w')
    ns = dict(np=np,json=json,time=time,fcntl=R.fcntl,
        last_proposal=None,branch=None,out=tmp_path,trace=SimpleNamespace(reference=io.StringIO(json.dumps(source)+'\n'),
            observed_issues=10,matched_issues=10,comparison_exact=True),
        native_started_wall=None,started=0,switch_time=None,cut_snapshot=None,
        original_next_issue=None,cut_input_exact=False,prefix_trace_rows=None,prefix_matched_rows=None,
        selected_snapshot=None,env=SimpleNamespace(_current_events=[SimpleNamespace(sim_time=clock)]),
        p=dict(horizon_seconds=24000),clock_audit=dict(actor_real_clock=True,max_actor_clock_error_seconds=0.,native_policy_issues=0),
        budget=SimpleNamespace(index=51,receipts=[{}]*51),counts=lambda:dict(R.COUNTS),COUNTS=R.COUNTS,PACKETS=51,RECORDS=255,
        replay=SimpleNamespace(target_received=255),own_snapshot=dict(packets=51),update_rows_matched=102,
        reference=dict(receipts=[{}]*51),stream=SimpleNamespace(now=clock,_snapshot=lambda:ledger),
        snapshot=R.snapshot,args=SimpleNamespace(seed=6208,arm='fresh'),root_out=tmp_path/'fresh',
        opposite_actor=mock.Mock(return_value=({},dict(packets=51))),
        acquire_tail_slot=mock.Mock(return_value=lock),updates=io.StringIO(),policy_records=io.StringIO(),
        service=SimpleNamespace(flush=lambda:None),learner=SimpleNamespace(act=mock.Mock()),
        transform=SimpleNamespace(numpy=mock.Mock()),ARMS=R.ARMS,PrefixComplete=R.PrefixComplete)
    class PausedAtFork(Exception):pass
    def fork(out, trace, native_clock, counters, prepare_child, **kwargs):
        assert trace.reference.tell() == 0
        assert native_clock() == clock and counters() == R.COUNTS
        raise PausedAtFork
    ns['fork_children']=fork
    exec(compile(module,R.__file__,'exec'),ns)
    return ns,state,lock,PausedAtFork


def test_first_post_packet51_state_is_forked_before_issuing_a_new_action(tmp_path):
    ns,state,lock,paused = policy_namespace(tmp_path)
    with pytest.raises(paused):
        ns['policy'](state)
    ns['learner'].act.assert_not_called()
    assert ns['cut_input_exact'] and ns['cut_snapshot']['time_seconds'] == 16210
    assert ns['prefix_trace_rows'] == ns['prefix_matched_rows'] == 10
    assert lock.closed
    assert ns['opposite_actor'].call_args.args[0].name == 'actor_at_packet51.pt'


def test_first_cut_rejects_a_future456_actor_budget(tmp_path):
    ns,state,lock,_ = policy_namespace(tmp_path)
    ns['counts']=lambda:dict(R.COUNTS,actor=456)
    with pytest.raises(RuntimeError,match='exact visible actor budget'):
        ns['policy'](state)
    ns['opposite_actor'].assert_not_called()
    ns['learner'].act.assert_not_called()
    lock.close()
