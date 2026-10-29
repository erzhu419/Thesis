"""Two frozen action prefixes with twelve passive current-leg observations."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


class ObservedAllTargets(Exception):
    pass


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        handle.write('\n')


def run(args):
    p=json.loads((HERE/'protocol.json').read_text())
    case=next(c for c in p['cases'] if c['scenario']==args.scenario)
    targets={r['sequence']:r for r in case['targets']}
    last=targets[case['stop_before_apply_sequence']]
    root,out=args.thesis_root.resolve(),args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    captures=out/'captures'
    captures.mkdir()
    env=stream=hook=reference=reference_commands=command_log=None
    native_started=False
    started_wall=finished_wall=None
    started=last_progress=time.monotonic()
    applied_calls=commands_checked=discarded=discarded_intervals=0
    observed_sequences=[]
    result=dict(status='failed',scenario=args.scenario,integrity=dict(all_pass=False))

    def progress(phase):
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase=phase,
            current=stream.now if native_started and stream is not None else 0,
            total=case['stop_before_apply_clock'],unit='sim_seconds',
            observed_roots=len(observed_sequences),applied_commands=applied_calls,
            elapsed_seconds=time.monotonic()-started)),flush=True)

    try:
        import numpy as np
        import torch
        torch.set_num_threads(1)
        from experiments.mcwm_logged_action_observation_v116_20261029.replay import ReplayReference, records
        from experiments.mcwm_logged_action_observation_v116_20261029.observe import _guard, _checks
        from experiments.mcwm_receiver_service_v124_20261029.observe import capture_receiver_service
        from experiments.mcwm_closed_loop_v1.stream import EventStream, create_target_env
        from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7
        from analyze import describe_captures

        old=root/p['reference_root']/args.scenario
        qualified=json.loads((root/p['qualified_root']/args.scenario/'summary.json').read_text())
        if qualified['status']!='complete' or not qualified['integrity']['all_pass']:
            raise ValueError('Original replay is not qualified')
        selected=root/p['selected_root']/args.scenario/'selected_records.jsonl'
        contexts={r['selection']['sequence']:r['context'] for r in records([selected])
                  if r['selection']['sequence'] in targets}
        if set(contexts)!=set(targets):
            raise ValueError('Fixed observation roots are absent from the original selection')
        metadata=json.loads((root/p['metadata']).read_text())
        contract=TransitMDPConfigV7(**metadata['contract'])
        stations=np.asarray(metadata['station_max_by_line'],np.float32)
        reference=ReplayReference([old/'actions.jsonl'],[old/'ticks.jsonl'],
            case['stop_before_apply_sequence'],case['stop_before_apply_clock'])
        reference_commands=records([old/'commands.jsonl'])
        command_log=(out/'applied_commands.jsonl').open('x')

        def policy(state):
            return np.asarray(reference.next_action(state),np.float32)

        def issue(event):
            old_visit=reference.current['visit_index']
            checked=reference.check_event(event,asdict(stream._snapshot()))
            seq=event['sequence']
            if seq not in targets:
                return
            target,context=targets[seq],contexts[seq]
            if (event['trip_id']!=target['receiver'] or event['station_id']!=target['station']
                    or event['started_at']!=target['issue_clock'] or old_visit!=target['visit_index']):
                raise RuntimeError('Fixed target identity differs')
            api=sys.modules[type(env._bridge).__module__].traci
            before_calls=applied_calls
            before=_guard(env._bridge,api)
            observation=capture_receiver_service(env._bridge,event,
                source_group=context['meta']['source_group'])
            after=_guard(env._bridge,api)
            checks=_checks(before,after,event['started_at'])
            observed_count=observation['receiver']['onboard_count']
            observed_age=observation['receiver']['estimated_service_ready_age_seconds']
            field_checks=dict(original_onboard_count=None if observed_count is None else
                observed_count==context['receiver']['onboard'],
                original_ready_estimate=None if observed_age is None else
                observed_age==context['receiver']['service_ready_age_seconds'])
            checks.update(apply_count_unchanged=applied_calls==before_calls,
                original_root_identity=observation['meta']==context['meta'],
                readable_reference_fields_match=all(v is not False for v in field_checks.values()))
            # Persist current-only observations before opening any diagnostic future events.
            write(captures/(str(seq)+'.json'),dict(observation=observation,passivity=checks,
                reference_field_checks=field_checks,
                expected_root=target,reference_ledger=checked['reference_ledger']))
            if not all(checks.values()):
                raise RuntimeError('Observation changed the state or mismatched the frozen root')
            observed_sequences.append(seq)
            progress('receiver_observed')
            if checked['target_reached']:
                result.update(target_ledger=asdict(stream._snapshot()),
                    target_reference_ledger=checked['reference_ledger'],
                    target_command_applied=applied_calls!=before_calls)
                raise ObservedAllTargets()

        env=create_target_env(os.environ['THESIS_SUMO_ROOT'],root/p['edge_xml'],
            episode_seconds=p['horizon_seconds'],seed=p['external_seed'])
        env._ensure_bridge()
        original_apply=env._bridge.apply_action

        def apply(event,action):
            nonlocal applied_calls,commands_checked
            api=original_apply.__func__.__globals__['traci']
            now=float(api.simulation.getTime())
            if (str(event.bus_id),str(event.stop_id),now)==(last['receiver'],last['station'],last['issue_clock']):
                raise RuntimeError('Final target must remain unapplied')
            setter=api.vehicle.setStopParameter
            calls=successes=0

            def record_setter(*argv,**kwargs):
                nonlocal calls,successes
                calls+=1
                value=setter(*argv,**kwargs)
                successes+=1
                return value

            api.vehicle.setStopParameter=record_setter
            try:
                value=original_apply(event,action)
            finally:
                api.vehicle.setStopParameter=setter
            applied_calls+=1
            actual=dict(trip_id=str(event.bus_id),stop_id=str(event.stop_id),applied_at=now,
                requested_additional_hold_seconds=float(event.assigned_action),accepted=calls==successes==1)
            expected=next(reference_commands,None)
            exact=expected is not None and all(expected.get(k)==v for k,v in actual.items())
            command_log.write(json.dumps(dict(actual=actual,reference_sequence=None if expected is None else expected['sequence'],
                exact=exact),separators=(',',':'))+'\n')
            if not exact:
                command_log.flush()
                raise RuntimeError('Applied native command differs from recorded prefix')
            commands_checked+=1
            return value

        env._bridge.apply_action=apply
        stream=EventStream(env,stations,contract,domain='target',delay_seconds=30,
            execute_transform=True,on_issue=issue)
        native_started,started_wall=True,time.time()
        stream.reset()
        from bus_h2o.sumo_env import rl_bridge
        from experiments.mcwm_closed_loop_formal_20261020.scenarios import install_scenario
        version=rl_bridge.traci.getVersion()
        if '1.25.0' not in str(version[1]):
            raise RuntimeError('Expected original SUMO1.25.0 runtime')
        hook=install_scenario(env,args.scenario,**p['scenario_settings'])
        result['runtime']=dict(sumo=version,h2o_root=os.environ['THESIS_H2O_ROOT'],
            sumo_root=os.environ['THESIS_SUMO_ROOT'])
        progress('recorded_prefix')
        try:
            while not stream.done:
                discarded+=len(stream.step(policy))
                discarded_intervals+=len(stream.release_intervals())
                if stream.now>case['stop_before_apply_clock']:
                    raise RuntimeError('Passed the final observation target')
                if time.monotonic()-last_progress>=30:
                    command_log.flush()
                    progress('recorded_prefix')
                    last_progress=time.monotonic()
            raise RuntimeError('Episode ended before the observation target')
        except ObservedAllTargets:
            finished_wall=time.time()
            checks=dict(replay_exact=reference.report()['all_pass'],
                actual_commands_exact=commands_checked==applied_calls,
                fixed_roots_observed=observed_sequences==list(targets),
                exact_stop_clock=stream.now==case['stop_before_apply_clock'],
                final_target_unapplied=result['target_command_applied'] is False,
                nonterminal_stop=stream.done is False,
                scenario_applied=(hook.applied_at==9000. if args.scenario=='local_speed_drop' else hook.applied_at is None))
            result.update(status='complete' if all(checks.values()) else 'integrity_failed',
                integrity=dict(all_pass=all(checks.values()),checks=checks),replay=reference.report(),
                commands_checked=commands_checked,observed_sequences=observed_sequences,
                scenario_metadata=hook.metadata())
            if all(checks.values()):
                result['observation']=describe_captures(captures,
                    root/p['evidence_root']/args.scenario/'full_evidence.json')
    except Exception as error:
        result['status']='failed'
        result['integrity']['all_pass']=False
        result['error']=dict(type=type(error).__name__,message=str(error))
        if reference is not None:
            result['replay']=reference.report()
        raise
    finally:
        from experiments.mcwm_closed_loop_historical_comparison_v104_20261028.run import consumed
        seconds,lower=consumed(env,native_started)
        exact_stop=(finished_wall is not None and result.get('replay',{}).get('all_pass') is True
                    and result.get('target_command_applied') is False
                    and seconds==case['stop_before_apply_clock'])
        result['budget']=dict(native_seconds=seconds,native_seconds_is_lower_bound=lower and not exact_stop,
            native_seconds_exact_stop=exact_stop,
            learner_created=False,optimizer_updates=0,feedback_admitted=0,new_predictor_fits=0,
            checkpoint_loaded=False,world_refits=0,new_policy_actions=0,
            applied_commands=applied_calls,observed_roots=len(observed_sequences),
            released_transitions_discarded=discarded,physical_intervals_discarded=discarded_intervals,
            full_day_completed=False)
        result['execution_wall']=dict(native_started_wall=started_wall,native_finished_wall=finished_wall,
            elapsed_seconds=time.monotonic()-started)
        write(out/'summary.json',result)
        progress(result['status'])
        if command_log is not None: command_log.close()
        if reference_commands is not None: reference_commands.close()
        if reference is not None: reference.close()
        if hook is not None: hook.close()
        if env is not None: env.close()
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--thesis-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--scenario',choices=('stable','local_speed_drop'),required=True)
    result=run(parser.parse_args())
    raise SystemExit(0 if result['status']=='complete' else 1)
