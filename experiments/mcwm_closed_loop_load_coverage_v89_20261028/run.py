"""Replay the frozen parent and observe7X native load, with no label creation."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from experiments.mcwm_closed_loop_supervision_plan_v84_20261027.run import (
    snapshot,NativePrefixTrace,install_scenario,_equal,create_target_env)
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _rng
from experiments.mcwm_closed_loop_reference_collection_v84_20261027.action_pair import choose_pair
from experiments.mcwm_closed_loop_reference_collection_v84_20261027.stream import CollectionStream
from experiments.mcwm_closed_loop_cost_advantage_v85_20261027.native import load_reference
from experiments.mcwm_closed_loop_service_response_v88_20261028.service import action_support
from experiments.mcwm_closed_loop_v1.stream import EI
import numpy as np
import torch


def write(path,value):
    with Path(path).open('x') as handle:
        json.dump(value,handle,ensure_ascii=False,separators=(',',':'),allow_nan=False)
        handle.write('\n')


class ObservationStream(CollectionStream):
    """Keep native event/action semantics and suppress unused transition labels."""
    def _close_physical(self,event):
        return None

    def _close(self,raw_ns,point,next_vehicle_id='',terminal=False):
        return None


def run(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((HERE/'protocol.json').read_text())
    audit=json.loads((HERE/'freeze_audit.json').read_text())
    source_trace=HERE.parent/'mcwm_closed_loop_service_response_v88_20261028/runs/preflight/prefix_trace.jsonl'
    if not (audit['all_pass'] and all(audit['checks'].values())
            and audit['source_initializer']==protocol['source_initializer']==6207
            and audit['external_seed']==protocol['external_seed']==716207
            and audit['scenario']==protocol['scenario']=='stable'
            and audit['reference_trace_path']==str(source_trace)
            and audit['new_fits']==audit['new_neural_updates']==audit['new_suffixes']==audit['new_labels']==audit['v87_truth_used']==0):
        raise RuntimeError('The frozen observation audit does not match this declaration')
    env=stream=trace=hook=controller=None
    rows,matched_sequences=[],[]
    observation_checks=dict(clock_unchanged=True,ledger_unchanged=True,rng_unchanged=True,
        issue_clock_exact=True,event_clock_exact=True,before_action=True)
    actor_clock_exact=True
    started=time.monotonic()

    def progress(phase):
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase=phase,
            current=stream.now if stream else 0.,total=12600,unit='sim_seconds',
            observed_7x=len(rows),elapsed_seconds=time.monotonic()-started)),flush=True)

    def policy(state):
        nonlocal actor_clock_exact
        clock=float(env._current_events[0].sim_time)
        actor_clock_exact &= bool(state[0]==np.float32(np.float32(clock)/24000))
        return controller(state)

    def selector(event):
        if event['line_id']!='7X' or not 12000<=event['started_at']<12600:
            return None
        api=importlib.import_module(env._bridge.__class__.__module__).traci
        before=dict(clock=float(api.simulation.getTime()),ledger=snapshot(stream._snapshot()),rng=_rng())
        onboard=capacity=load=None
        reason=None
        try:
            onboard=len(api.vehicle.getPersonIDList(event['trip_id']))
            capacity=float(api.vehicle.getPersonCapacity(event['trip_id']))
            if np.isfinite(capacity) and capacity>0:
                load=float(onboard/capacity)
            else:
                reason='native_capacity_invalid'
        except Exception:
            reason='native_load_unavailable'
        step=float(api.simulation.getDeltaT())
        action_step=float(api.vehicle.getActionStepLength(event['trip_id']))
        at_stop=bool(api.vehicle.isAtBusStop(event['trip_id']))
        choice=choose_pair(controller.action_transform,event['s'],controller.last_proposal,
            simulation_step=step,action_step=action_step)
        support=action_support(choice['reference_hold_seconds'],choice['candidate_hold_seconds'],step,action_step)
        domain_reason=support['reason'] if at_stop else 'native_not_at_stop'
        after=dict(clock=float(api.simulation.getTime()),ledger=snapshot(stream._snapshot()),rng=_rng())
        event_clock=float(event['event'][EI['sim_time_seconds']])
        native=[row for row in env._current_events if (str(row.bus_id),str(row.line_id),str(row.stop_id))
                ==(event['trip_id'],event['line_id'],event['station_id'])]
        guards=dict(clock_unchanged=before['clock']==after['clock'],
            ledger_unchanged=before['ledger']==after['ledger'],rng_unchanged=_equal(before['rng'],after['rng']),
            issue_clock_exact=before['clock']==after['clock']==event['started_at'],
            event_clock_exact=len(native)==1 and event_clock==float(native[0].sim_time),before_action=True)
        for key,value in guards.items():observation_checks[key]&=value
        if not np.array_equal(event['a'],np.asarray(choice['reference_action'],np.float32)):
            raise RuntimeError('Observation changed the original actor/F execution')
        rows.append(dict(sequence=int(event['sequence']),trip_id=event['trip_id'],station_id=event['station_id'],
            issue_clock=float(event['started_at']),event_clock=event_clock,
            obs_start=before['clock'],obs_end=after['clock'],
            in_domain=bool(at_stop and support['supported']),domain_reason=domain_reason,at_stop=at_stop,
            sim_step_seconds=step,action_step_seconds=action_step,onboard=onboard,capacity=capacity,load=load,
            observed=reason is None,reason=reason,guard_checks=guards,
            ref_hold=choice['reference_hold_seconds'],cand_hold=choice['candidate_hold_seconds']))
        if not all(guards.values()):
            raise RuntimeError('Native load sampling changed clock, ledger or RNG')
        return None

    def issue(event):
        event['proposed_a']=controller.last_proposal.copy()
        trace.observe(event,stream._snapshot())
        if event['line_id']=='7X' and 12000<=event['started_at']<12600:
            matched_sequences.append(int(event['sequence']))

    try:
        torch.set_num_threads(1)
        controller,mdp,stations,reference=load_reference('stable')
        reference['heldout_external_seed']=716207
        actor_before=controller.snapshot_state()
        reproduced={}
        for version in ('mcwm_closed_loop_reference_collection_v84_20261027',
                        'mcwm_closed_loop_cost_advantage_v85_20261027'):
            for line in ('7X','102X','705X'):
                d=json.loads((HERE.parent/version/'runs/stable/pairs'/line/'root.json').read_text())
                actual=controller(np.asarray(d['root']['s'],np.float32))
                reproduced[version+'/'+line]=bool(
                    np.array_equal(actual,np.asarray(d['choice']['reference_action'],np.float32))
                    and np.array_equal(controller.last_proposal,np.asarray(d['root']['reference_proposal'],np.float32)))
        if not all(reproduced.values()):raise RuntimeError('Original frozen actor/F inputs do not reproduce')
        controller.last_proposal=None
        trace=NativePrefixTrace(out/'prefix_trace.jsonl',reference=source_trace)
        env=create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT'])/'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=24000,seed=716207)
        stream=ObservationStream(env,stations,mdp,domain='target',selector=selector,
            delay_seconds=30,execute_transform=False,on_issue=issue)
        stream.reset();stream.enable_prefix_limit()
        hook=install_scenario(env,'stable',shock_at=9000,factor=.65,metric_every=300)
        last_progress=time.monotonic()
        while not stream.done and not stream.prefix_limit_reached:
            stream.step(policy)
            if time.monotonic()-last_progress>=45:
                progress('frozen_prefix_observation');last_progress=time.monotonic()
        exhausted=trace.reference.readline()==''
        replay_checks=dict(comparison_exact=trace.comparison_exact,reference_exhausted=exhausted,
            all_issues_match=trace.observed_issues==trace.matched_issues==stream.issued_count>0)
        replay=dict(all_pass=all(replay_checks.values()),**replay_checks,
            observed_issues=trace.observed_issues,matched_issues=trace.matched_issues,
            reference_trace_path=str(source_trace),window_7x_sequences=matched_sequences,
            window_7x_issues=len(matched_sequences))
        labels_created=stream.completed_count+len(stream.physical_intervals)+len(stream._interval_queue)
        checks=dict(exact_v88_prefix=all(replay_checks.values()),real_cutoff=stream.now==12600 and stream.prefix_limit_reached,
            original_horizon=mdp.horizon_seconds==24000,nonterminal=not stream.done and stream.final_snapshot is None,
            actor_frozen=_equal(actor_before,controller.snapshot_state()),actor_real_clock=actor_clock_exact,
            original_actions_reproduced=all(reproduced.values()),source_actor_exact=reference['source_actor_path']==audit['source_actor_path'],
            observation_unchanged=all(observation_checks.values()),native_load_complete=all(row['observed'] for row in rows),
            complete_7x_window=[row['sequence'] for row in rows]==matched_sequences,
            zero_created_labels=labels_created==stream.released_count==0,
            no_training_or_branching=True)
        value=dict(status='complete',integrity=dict(all_pass=all(checks.values()),checks=checks),
            seed=6207,external_seed=716207,scenario='stable',prefix_id='v89_load_observation/stable/external716207',
            reference_snapshot=reference,reference_reproduction=reproduced,trace_replay=replay,
            observation_checks=observation_checks,rows=rows,training_started=False,labels_used_for_selection=False,
            budget=dict(actual_simulated_seconds=stream.now,prefixes=1,suffixes=0,labels=labels_created,
                new_initialization_updates=0,new_fits=0,new_neural_updates=0,feedback_admission=0,v87_truth_used=0))
        write(out/'collector_summary.json',value)
        progress('native_observation_complete_no_labels')
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),
            actual_simulated_seconds=stream.now if stream else 0.,suffixes=0,labels=0,new_fits=0,rows=len(rows)))
        raise
    finally:
        if hook is not None:hook.close()
        if trace is not None:trace.close()
        if env is not None:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.out)
