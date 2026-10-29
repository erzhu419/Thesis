"""Check the intervention boundary, unchanged actor clock and prefix contracts."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
spec = importlib.util.spec_from_file_location('v19_native_runner_test',HERE/'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
from phase_stream import PhaseControlledStream, PhasePolicy, tail_action
from experiments.mcwm_closed_loop_response_residual_v11_20261021.paired_stream import PairedControlledStream


def state(at=6000.):
    value = np.zeros(229,np.float32)
    value[0] = np.float32(at/24000.)
    slot = value[5:19]
    slot[3:7] = [1.,1.2,1.,1.]
    slot[8:10] = [.1,.1]
    slot[11:13] = [1.,0.]
    return value


def evidence():
    return dict(actor_real_clock=True,actor_clock_max_abs_error=0.,
        tail_policy_issues=0,native_policy_issues=0)


class Actor:
    def __init__(self):
        self.inputs = []

    def act(self,values,deterministic):
        assert deterministic is True
        self.inputs.append(values.copy())
        return np.ones((1,1),np.float32)


def transform():
    return runner.ExecutionAction(dict(max_hold_seconds=60.,scheduled_headway_scale_seconds=360.))


def test_tail_formula_matches_original_native_tail_and_keeps_proposal():
    rule = transform()
    current = np.stack([state(19000.) for _ in range(3)])
    proposals = np.asarray([[-1.],[-.5],[1.]],np.float32)
    original = proposals.copy()
    assert np.array_equal(tail_action(rule,proposals),rule.numpy(current,proposals).reshape(-1))
    assert np.array_equal(proposals,original)


def test_fork_and_siblings_stay_native_until_next_batch_even_at_same_clock(monkeypatch):
    stream = PhaseControlledStream.__new__(PhaseControlledStream)
    stream.fork,stream.tail_policy_active,stream.tail_batches = None,False,0
    # A branch can have consecutive event batches at the same native second.
    clock_stream = SimpleNamespace(now=6000.,tail_policy_active=False)
    actor,stats = Actor(),evidence()
    policy = PhasePolicy(actor,transform(),clock_stream,24000.,stats)
    observed = []
    current = state()

    def native_step(self,callback):
        clock_stream.tail_policy_active = self.tail_policy_active
        observed.append(float(transform().hold_seconds(torch.as_tensor(callback(current)))[0]))
        if self.fork is None:
            # These calls represent native controls before/at/after the fork
            # within one batch; creating the fork must not switch the last one.
            self.fork = dict(started_at=6000.)
            observed.append(float(transform().hold_seconds(torch.as_tensor(callback(current)))[0]))
            observed.append(float(transform().hold_seconds(torch.as_tensor(callback(current)))[0]))
        return []

    monkeypatch.setattr(PairedControlledStream,'step',native_step)
    before = current.copy()
    stream.step(policy)
    stream.step(policy)
    assert observed == [60.,60.,60.,7.5]
    assert stats['native_policy_issues'] == 3 and stats['tail_policy_issues'] == 1
    assert stats['actor_real_clock'] is True
    assert all(np.array_equal(values[0],before) for values in actor.inputs)
    assert np.array_equal(current,before)
    assert stream.tail_batches == 1


def test_actor_clock_mismatch_is_reported_without_faking_its_input():
    actor,stats = Actor(),evidence()
    current = state(19000.)
    original = current.copy()
    policy = PhasePolicy(actor,transform(),SimpleNamespace(now=6000.,tail_policy_active=True),24000.,stats)
    policy(current)
    assert stats['actor_real_clock'] is False and stats['actor_clock_max_abs_error'] > 12000.
    assert np.array_equal(actor.inputs[0][0],original)
    assert np.array_equal(current,original)


def test_first_zero_and_seven_point_five_commands_are_original_supported_actions():
    current = state()
    current[5+12] = 1.  # Avoid the original completion floor at the chosen anchor.
    current[5+8] = .4
    rule = transform()
    choices = runner.action_choices(rule,current)
    assert 30.*(1.+float(choices['low'][0])) == 0.
    assert 30.*(1.+float(choices['high'][0])) == 7.5


@pytest.mark.parametrize('role,nominal',(('train',-.75),('heldout',-.875)))
def test_source_acquisition_replays_saved_command_without_unsaved_plan_fields(role,nominal):
    # Acquisitions save nominal_a and both allowed interventions, but not the
    # plan's nominal_arm. Replay must follow the saved factual command exactly.
    row = dict(id='train/7X/early',role=role,index=0,window_name='early',
        nominal_a=[nominal],actions=dict(low=[-1.],high=[-.75]),
        selected=dict(sequence=352,started_at=2703.))
    assert 'nominal_arm' not in row
    assert np.array_equal(runner.saved_command(row),np.asarray([nominal],np.float32))


def test_runner_rejects_nonfixed_cohorts_before_creating_output(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    (source/'protocol.json').write_text('{"protocol": {}}')
    for seed,scene,target in ((6202,'stable','train/7X/late'),
            (6201,'local_speed_drop','train/7X/late'),(6201,'stable','train/7X/early')):
        args = SimpleNamespace(source=source,seed=seed,scenario=scene,target_id=target,
            out=tmp_path/'out',branch='low',mode='branch')
        with pytest.raises(ValueError,match='predeclared'):
            runner.run(args)
        assert not args.out.exists()
