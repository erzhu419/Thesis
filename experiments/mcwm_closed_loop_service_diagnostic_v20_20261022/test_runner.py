"""The service diagnostic must reproduce the original native action response."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('v20_service_runner_test', HERE/'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

from experiments.mcwm_closed_loop_action_fork_v7_20261020.test_fork_stream import FakeNativeEnv
from experiments.mcwm_closed_loop_action_response_v8_20261021.action_response import ActionResponseMemory
from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.endpoints import ResponseEndpointCollector
from experiments.mcwm_closed_loop_response_residual_v11_20261021.paired_stream import PairedControlledStream


def test_native_suffix_keeps_original_actions_and_exact_300_clock():
    observed = []
    for cls in (PairedControlledStream, runner.NativeControlledStream):
        env = FakeNativeEnv([(10, [('A', 0), ('B', 0)]),
                             (20, [('A', 1), ('B', 1)]), (40, [('C', 0)]),
                             (500, [('A', 2)])])
        endpoint = ResponseEndpointCollector(duration_seconds=300.)
        stream = None

        def register(event):
            if event['sequence'] == 0:
                endpoint.register(event, stream._snapshot(), role='heldout', hold_seconds=7.5)

        stream = cls(env, [3], runner.formal.TransitMDPConfigV7(horizon_seconds=24000),
            selector=lambda event: None,
            fork_selector=lambda event: [-.75] if event['sequence'] == 0 else None,
            endpoints=endpoint, response_memory=ActionResponseMemory(),
            execute_transform=False, on_issue=register)
        stream.reset()
        stream.attach_native_clock()
        policy_calls = []

        def native_policy(state):
            policy_calls.append(state.copy())
            return np.asarray([.5], np.float32)

        while not stream.cutoff_reached:
            stream.step(native_policy)
        endpoint.finish(stream.now, reason='exact_native300_cutoff')
        assert stream.now == 310. and stream.native_suffix_steps == 300
        if cls is runner.NativeControlledStream:
            assert stream.tail_policy_active is False and stream.tail_batches == 0
        observed.append((env._bridge.actions, stream.fork_batch_other_controls,
                         endpoint.records[0].endpoints, policy_calls))
    assert observed[0][:3] == observed[1][:3]
    assert len(observed[0][3]) == len(observed[1][3])
    for old, current in zip(observed[0][3], observed[1][3]):
        assert np.array_equal(old, current)


def test_reference_check_detects_service_observer_changes_without_replacing_truth():
    original = dict(truth=dict(fixed300=dict(dt=300., wait=100., ride=200.)),
        global_transition=dict(s=[.1], a=[-.75], ns=[.2]),
        execution=dict(requested_hold_seconds=7.5, executed_hold_seconds=7.5),
        native=dict(other_batch_actions=[dict(sequence=3, a=[.5])]))
    unchanged = deepcopy(original)
    assert runner.exact_native_reference(unchanged, original)
    for field in ('truth', 'global_transition', 'execution', 'native'):
        changed = deepcopy(original)
        changed[field] = {}
        assert not runner.exact_native_reference(changed, original)
    assert unchanged == original


def test_fixed_clock_observation_must_reconstruct_the_original_cost_endpoint():
    endpoint = dict(dt=300., wait=100., ride=200., cost=400., status='complete')
    observation = dict(fixed300={key: endpoint[key] for key in ('dt', 'wait', 'ride', 'cost')})
    assert runner.exact_service_ledger(observation, endpoint)
    observation['fixed300']['dt'] = 299.
    assert not runner.exact_service_ledger(observation, endpoint)
    assert not runner.exact_service_ledger(observation, None)


def test_only_the_three_predeclared_native_states_are_admitted(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    (source/'protocol.json').write_text('{"protocol":{}}')
    for seed, scene, target in ((6202, 'stable', 'train/7X/late'),
            (6201, 'local_speed_drop', 'train/7X/late'),
            (6201, 'stable', 'train/7X/early')):
        args = SimpleNamespace(source=source, seed=seed, scenario=scene,
            target_id=target, out=tmp_path/'out', branch='low', mode='branch')
        with pytest.raises(ValueError, match='predeclared'):
            runner.run(args)
        assert not args.out.exists()
