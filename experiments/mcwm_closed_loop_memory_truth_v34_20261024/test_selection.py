"""Selection uses issue-visible native clocks/actions; causal snapshot precedes apply."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.mcwm_closed_loop_memory_truth_v34_20261024.selection import FirstEligible, MemoryForkStream
from experiments.mcwm_closed_loop_action_fork_v7_20261020.test_fork_stream import FakeNativeEnv, policy
from experiments.mcwm_closed_loop_v1.stream import TransitMDPConfigV7


def declared():
    return json.loads((Path(__file__).parent/'protocol.json').read_text())


def event(line='7X',time=9300.,sequence=0):
    return dict(line_id=line,started_at=time,sequence=sequence)


def test_first_eligible_per_line_window_and_no_late_replacement():
    selection=FirstEligible(declared())
    assert selection.active(event(time=9299)) is None
    e=event();key=selection.active(e)
    assert key=='7X/early'
    assert not selection.consider(key,e,[0.,.9375],at_stop=True,simulation_step=1.,action_step=1.)
    assert selection.consider(key,event(sequence=1),[0.,1.875],at_stop=True,simulation_step=1.,action_step=1.)
    assert selection.active(event(time=9400)) is None
    assert selection.active(event('102X',9400))=='102X/early'
    assert selection.active(event('705X',9900)) is None
    assert selection.active(event(time=18300))=='7X/tail'
    report=selection.summary()
    assert len(report)==6 and report[key]['selected']['sequence']==1
    assert report[key]['reasons']=={'F_alias_or_insufficient_hold_gap':1}
    assert report['705X/early']['status']=='missing'
    assert report['705X/early']['selected'] is None


@pytest.mark.parametrize('stop,step,action_step',[(False,1.,1.),(None,None,None),(True,1.,2.)])
def test_native_stop_and_both_physical_step_lengths_bound_eligibility(stop,step,action_step):
    selection=FirstEligible(declared());e=event();key=selection.active(e)
    assert not selection.consider(key,e,[0.,1.875],at_stop=stop,simulation_step=step,action_step=action_step)
    assert selection.summary()[key]['selected'] is None


def test_issue_context_is_captured_before_current_batch_actions_and_not_resampled_in_child():
    env=FakeNativeEnv([(10,[('A',0),('B',0)]),(20,[('C',0)]),(1000,[('D',0)])])
    observed=[]
    stream=MemoryForkStream(env,[3],TransitMDPConfigV7(horizon_seconds=24000),
        execute_transform=False,selector=lambda e: [-1.] if e['sequence']==0 else None,
        on_issue=lambda e: observed.append(e),fork_duration_seconds=300.)
    stream.reset()
    snapshots=[]
    def snapshot(clock):
        snapshots.append((clock,len(env._bridge.actions)))
        return dict(observed_at=clock,vector=[len(env._bridge.actions)])
    stream.memory=SimpleNamespace(snapshot=snapshot)
    stream.step(policy)
    assert snapshots==[(10.,0)]
    assert observed[0]['memory_s']==dict(observed_at=10.,vector=[0])
    assert stream.fork['memory_s']==observed[0]['memory_s']
    assert len(env._bridge.actions)==2
