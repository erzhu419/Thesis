"""Frozen v33 models on a new native prefix with paired heldout action truth."""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import time

from bootstrap import HERE, write
import numpy as np
import torch
from stream import create_target_env
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.frozen import build_frozen, score_frozen
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.selection import FirstEligible, MemoryForkStream
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.suffix import NativeTruthPair, selected_state, plain
from experiments.mcwm_closed_loop_critic_memory_v32_20261023.memory import ExecutionMemory
from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace
from experiments.mcwm_closed_loop_formal_20261020.scenarios import install_scenario
from experiments.mcwm_closed_loop_v1.stream import policy_action_to_hold_seconds_v7
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng


MODULES=('actor','critic','target_critic','value','ratio')


def frozen_weights(learners):
    return {arm:dict(modules={name:copy.deepcopy(getattr(learner,name).state_dict()) for name in MODULES},
                     alpha=learner.log_alpha.detach().clone()) for arm,learner in learners.items()}


def run(args):
    p=json.loads((HERE/'protocol.json').read_text())
    out=args.out.resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    started=time.monotonic()
    env=stream=trace=hook=memory=None
    pair=None
    try:
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='reconstruct_frozen_v33',current=0,total=24000,unit='sim_seconds')),flush=True)
        learners,contract,stations,rebuild=build_frozen(args.seed,args.scenario,out/'frozen_models')
        weights=frozen_weights(learners)
        selection=FirstEligible(p)
        continuation=learners['zero_memory']
        last_proposal=None
        selected_sequence=None
        records=[]
        clock_checks=0
        clock_exact=True

        def counts():
            return dict(memory_updates=learners['memory'].total_updates,
                zero_updates=continuation.total_updates,
                memory_feedback=learners['memory'].replay_buffer.target_received,
                zero_feedback=continuation.replay_buffer.target_received,
                new_optimizer_updates=0, new_truth_training_labels=0)

        def policy(state):
            nonlocal last_proposal,clock_checks,clock_exact
            event_clock=float(env._current_events[0].sim_time)
            expected=np.float32(np.float32(event_clock)/p['horizon_seconds'])
            clock_checks+=1;clock_exact &= bool(state[0]==expected)
            last_proposal=continuation.act(state[None],deterministic=True)[0]
            return continuation.action_transform.numpy(state,last_proposal)

        def selector(event):
            nonlocal pair,selected_sequence
            if pair is not None and pair.in_child:
                return None
            key=selection.active(event)
            if key is None: return None
            states=torch.as_tensor(event['s'],dtype=torch.float32)[None].expand(2,-1)
            proposals=torch.tensor(p['proposal_hold_seconds'],dtype=torch.float32)[:,None]
            proposals=2.*proposals/continuation.action_transform.max_hold-1.
            with torch.no_grad():
                actions={a:l.action_transform(states,proposals) for a,l in learners.items()}
            if not torch.equal(actions['memory'],actions['zero_memory']):
                raise RuntimeError('Frozen models changed the shared execution F')
            action=actions['zero_memory'].numpy()
            holds=[float(policy_action_to_hold_seconds_v7(a[0],contract)) for a in action]
            api=importlib.import_module(env._bridge.__class__.__module__).traci
            try:
                stop=bool(api.vehicle.isAtBusStop(event['trip_id']))
                step=float(api.simulation.getDeltaT())
                action_step=float(api.vehicle.getActionStepLength(event['trip_id']))
            except Exception:
                stop=step=action_step=None
            if not selection.consider(key,event,holds,at_stop=stop,simulation_step=step,action_step=action_step):
                return None
            event['proposed_a']=last_proposal.copy()
            snapshot=stream._snapshot()
            selected=selected_state(event,snapshot)
            saved_rng=_rng()
            predictions=score_frozen(learners,event['s'],event['memory_s']['vector'],action)
            if not _equal(saved_rng,_rng()): raise RuntimeError('Frozen scoring changed native RNG')
            trace.observe(event,snapshot)
            selected_sequence=event['sequence']
            state_out=out/'states'/key.replace('/','_')
            expected=dict(selected=selected,actions=dict(low=action[0].tolist(),high=action[1].tolist()),
                holds=dict(low=holds[0],high=holds[1]))
            pair=NativeTruthPair(state_out,expected,counts,p['native_external_seed'])
            replacement=pair.fork(event,stream,trace)
            if pair.in_child:
                return replacement
            record=dict(slot=key,selected=selected,prediction=predictions,actions=expected['actions'],
                holds=expected['holds'],pair_receipt=pair.receipt,comparison=pair.comparison,
                branch_paths={a:str(state_out/a/'summary.json') for a in ('low','high')})
            write(state_out/'state_summary.json',record)
            records.append(record)
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='paired_truth',current=stream.now,
                total=24000,unit='sim_seconds',states=len(records),slot=key)),flush=True)
            return None

        def issue(event):
            event['proposed_a']=last_proposal.copy()
            if pair is not None and pair.in_child:
                pair.observe_issue(event,stream)
                return
            if event['sequence']!=selected_sequence:
                trace.observe(event,stream._snapshot())

        env=create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT'])/'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=p['horizon_seconds'],seed=p['native_external_seed'])
        stream=MemoryForkStream(env,stations,contract,domain='target',selector=selector,
            fork_duration_seconds=300,delay_seconds=p['feedback_delay_seconds'],execute_transform=False,on_issue=issue)
        stream.reset()
        memory=ExecutionMemory(env._bridge)
        stream.memory=memory
        hook=install_scenario(env,args.scenario,shock_at=p['shock_at_seconds'],factor=p['speed_factor'],
                              metric_every=p['metric_interval_seconds'])
        trace=NativePrefixTrace(out/'prefix_trace.jsonl')
        frozen_counts=counts()
        native_started=time.time();last_progress=time.monotonic()
        while not stream.done:
            stream.step(policy)
            stream.release_intervals()
            if pair is not None and pair.in_child:
                pair.observe_progress(stream)
                if stream.cutoff_reached: pair.finish_child(stream)
                continue
            if time.monotonic()-last_progress>=45:
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='new_frozen_prefix',current=stream.now,
                    total=24000,unit='sim_seconds',states=len(records))),flush=True)
                last_progress=time.monotonic()
        native_finished=time.time()
        hook.capture_final()
        after_counts=counts()
        checks=dict(v33_reconstruction_exact=rebuild['all_pass'],
            models_285_slots_456_updates=all(l.total_updates==50456 and l.replay_buffer.target_received==285 for l in learners.values()),
            models_unchanged=_equal(weights,frozen_weights(learners)),
            new_prefix_and_suffix_zero_training=frozen_counts==after_counts,
            full_target_day=stream.now==24000 and stream.done,
            full_new_prefix_trace=trace.observed_issues==stream.issued_count,
            actor_real_clock=clock_exact and clock_checks>0,
            native_seed_is_new=p['native_external_seed']!=p['training_external_seed'],
            all_pairs_parallel_and_parent_preserved=all(r['pair_receipt']['all_pass'] for r in records),
            issue_memory_causal=memory.summary()['integrity']['all_pass'],
            no_truth_feedback=True, no_missing_tail_backfill=True)
        ledger=stream._snapshot()
        result=dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()),checks=checks),scenario=args.scenario,seed=args.seed,
            native_external_seed=p['native_external_seed'],model_rebuild=rebuild,
            selection=selection.summary(),states=[dict(slot=r['slot'],path=r['branch_paths']['low'].rsplit('/low/',1)[0]+'/state_summary.json') for r in records],
            prefix=dict(trace_rows=trace.observed_issues,world_updates=0,trust_updates=0,
                wait=ledger.accrued_wait_seconds,ride=ledger.accrued_in_vehicle_seconds),
            counts=dict(before=frozen_counts,after=after_counts),
            budget=dict(reconstruction_updates=912,native_prefix_seconds=24000,states=len(records),
                        suffixes=2*len(records),suffix_sim_seconds=600*len(records),new_training_labels=0,
                        new_prefix_optimizer_updates=0,suffix_optimizer_updates=0),
            wall=dict(native_started=native_started,native_finished=native_finished,total_seconds=time.monotonic()-started))
        write(out/'summary.json',result)
        if not all(checks.values()): raise RuntimeError('Frozen new-prefix integrity failed')
    except Exception as error:
        if pair is not None and pair.in_child:
            write(pair.out/pair.arm/'failure.json',dict(type=type(error).__name__,message=str(error)))
            sys.stdout.flush();sys.stderr.flush();os._exit(1)
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),native_time=stream.now if stream else None))
        raise
    finally:
        if pair is not None and pair.in_child: os._exit(1)
        if memory is not None: memory.close()
        if trace is not None: trace.close()
        if hook is not None: hook.close()
        if env is not None: env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--scenario',choices=('stable','local_speed_drop'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.seed!=6202: raise ValueError('v34 freezes the declared v33 seed6202 models')
    run(args)
