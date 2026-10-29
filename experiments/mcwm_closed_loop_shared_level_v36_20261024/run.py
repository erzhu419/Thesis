"""Three persistent learners; old controls reproduce before frozen issue scoring."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__:
    from .bootstrap import HERE, write
else:
    from bootstrap import HERE, write

import numpy as np
import torch

from experiments.mcwm_closed_loop_shared_level_v36_20261024.learning import (
    ARMS, critic_states, initial_preflight, make_learner,
)
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.facts import load_facts
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.probes import probe, window
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.run import (
    jsonline, optimizer_steps, persistent_objects, plain,
)
from experiments.mcwm_closed_loop_critic_decomposition_v35_20261024.decomposition import summarize
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.frozen import freeze
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng, _restore


OLD_ARMS = {'zero_memory': 'zero_memory', 'per_head_level': 'memory'}
MODULES = ('actor', 'critic', 'target_critic', 'value', 'ratio')


def _read_lines(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source]


def _objects(learner):
    return dict(optimizers=persistent_objects(learner),
        modules={name:(id(getattr(learner, name)), tuple(id(value) for value in
            getattr(learner, name).parameters())) for name in MODULES},
        alpha=id(learner.log_alpha))


def _training_rng(learner):
    return dict(global_state=plain(_rng()), sampler=plain(learner.rng.bit_generator.state),
        diagnostic=plain(learner.diagnostic_rng.bit_generator.state))


def _check_packet(record, schedule_reference, *, exact_reference=None):
    """All arms retain the original eight draws; old arms retain exact updates."""
    checks = dict(packet_and_receipt=record['packet'] == schedule_reference['packet']
        and record['receipt'] == schedule_reference['receipt'],
        all_eight_sampling_draws=record['sampling'] == schedule_reference['sampling'],
        global_and_private_rng=record['rng'] == schedule_reference['rng'],
        update_count=record['updates'] == schedule_reference['updates'])
    step_reference = exact_reference if exact_reference is not None else schedule_reference
    checks['optimizer_steps'] = record['optimizer_steps'] == step_reference['optimizer_steps']
    if exact_reference is not None:
        checks['old_metrics_exact'] = record['metrics'] == exact_reference['metrics']
    if not all(checks.values()):
        raise RuntimeError(f'Packet {record["packet"]} replay differs: '
            +str([key for key,value in checks.items() if not value]))
    return checks


def optimizer_overlap(records_by_arm):
    """Measure actual three-way update intersection, excluding probe gaps."""
    intervals={arm:[(float(row['update_started_wall']),float(row['update_finished_wall']))
        for row in records] for arm,records in records_by_arm.items()}
    if len(intervals)!=3:
        raise ValueError('Optimizer overlap requires three worker interval streams')
    positions=dict.fromkeys(intervals,0)
    total=0.
    while all(positions[arm]<len(rows) for arm,rows in intervals.items()):
        current={arm:rows[positions[arm]] for arm,rows in intervals.items()}
        left=max(value[0] for value in current.values())
        right=min(value[1] for value in current.values())
        total+=max(0.,right-left)
        # At least one current interval ends here. Advancing every tied end
        # preserves union intersections without counting a boundary twice.
        for arm,value in current.items():
            if value[1]==right:positions[arm]+=1
    return total


@torch.no_grad()
def heldout_prediction(learner, selected, actions):
    """Read one frozen issue and its saved executed actions, without another F."""
    saved = _rng()
    private = {name:deepcopy(getattr(learner, name).bit_generator.state)
        for name in ('rng', 'diagnostic_rng')}
    try:
        state = torch.as_tensor(selected['s'], dtype=torch.float32, device=learner.device).reshape(1,229)
        memory = torch.as_tensor(selected['memory_s']['vector'], dtype=state.dtype,
            device=learner.device).reshape(1,84)
        executed = torch.as_tensor([actions[arm] for arm in ('low','high')],
            dtype=state.dtype, device=learner.device).reshape(2,1)
        explicit = critic_states(learner, state.expand(2,-1), memory.expand(2,-1))
        full = learner.critic(explicit, executed)
        variants = {'full':summarize(full)}
        components = {}
        if hasattr(learner.critic, 'base'):
            critic = learner.critic
            # Retain native views and base+(level+action) addition order.
            canonical, context = explicit[:,:229], explicit[:,229:]
            base = critic.base(canonical, executed)
            level = (context@critic.level_weight.T).T
            interaction = ((executed[:,0]+1.)*.5)[None,:]*(context@critic.action_weight.T).T
            variants.update(base=summarize(base), level_only=summarize(base+level),
                action_only=summarize(base+interaction))
            components = dict(level_heads=level.tolist(), action_heads=interaction.tolist())
        prediction = dict(variants['full'])
        proposal = learner.actor(state, deterministic=True)[0]
        actor_executed = learner.action_transform(state, proposal)
        prediction.update(predicted_choice=prediction['choice'],
            executed_actions=executed[:,0].tolist(),
            holding_seconds=learner.action_transform.hold_seconds(executed[:,0]).tolist(),
            actor_proposal_action=float(proposal[0,0]),
            actor_proposal_holding_seconds=float(learner.action_transform.hold_seconds(proposal[0,0])),
            actor_executed_action=float(actor_executed[0,0]),
            actor_holding_seconds=float(learner.action_transform.hold_seconds(actor_executed[0,0])))
        return dict(prediction=prediction, variants=variants, components=components)
    finally:
        for name,value in private.items():
            getattr(learner,name).bit_generator.state=value
        _restore(saved)


def _export_heldout(learner, scenario, p, out):
    """Called only after all training and freeze; no outcome fields are accessed."""
    if any(value.requires_grad for name in MODULES for value in getattr(learner,name).parameters()):
        raise RuntimeError('Heldout issues cannot be read before the learner is frozen')
    before_updates, before_steps = learner.total_updates, optimizer_steps(learner)
    saved_rng, private_rng = _rng(), _training_rng(learner)
    root = HERE.parent/p['v34_directory']/'runs'/f'seed{p["training_seeds"][0]}'/scenario
    parent = json.loads((root/'summary.json').read_text())
    if not parent['integrity']['all_pass'] or len(parent['states']) != 6:
        raise RuntimeError('Frozen scoring requires the six existing v34 state records')
    count = 0
    with (out/'heldout.jsonl').open('x') as handle:
        for reference in parent['states']:
            source = Path(reference['path'])
            record = json.loads(source.read_text())
            # These four fields are the complete scoring interface. Prediction,
            # comparison, branch paths and realized outcomes are not accessed.
            view = {key:record[key] for key in ('slot','selected','actions','holds')}
            values = heldout_prediction(learner, view['selected'], view['actions'])
            prediction = values['prediction']
            if prediction['holding_seconds'] != [view['holds'][arm] for arm in ('low','high')]:
                raise RuntimeError('Saved executed actions changed physical interpretation')
            jsonline(handle, dict(slot=view['slot'], sequence=view['selected']['sequence'],
                issue=view['selected']['started_at'], line_id=view['selected']['line_id'],
                source=str(source), **values))
            count += 1
    checks = dict(six_existing_issues=count==6,
        post_freeze_zero_updates=learner.total_updates==before_updates and optimizer_steps(learner)==before_steps,
        global_rng_unchanged=_equal(saved_rng,_rng()), private_rng_unchanged=private_rng==_training_rng(learner))
    if not all(checks.values()):
        raise RuntimeError('Frozen issue export changed training state')
    return dict(all_pass=True, checks=checks, rows=count, selected_fields_only=True, truth_used=False)


def worker(args, p):
    out=args.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    started=time.time()
    try:
        learner,replay,contract,stations,initializer=make_learner(args.seed,arm=args.child_arm)
        initial_updates=learner.total_updates
        preflight=initial_preflight(learner)
        saved_rng=_rng()
        facts=load_facts(HERE.parents[1]/p['original_runs'][args.scenario],contract,p)
        if not facts['proof']['all_pass'] or not _equal(saved_rng,_rng()):
            raise RuntimeError('Factual reconstruction failed or changed RNG')
        if initial_updates!=p['initializer_updates']:
            raise RuntimeError('Wrong shared initialization')
        reference_root=HERE.parent/p['v33_directory']/'runs'/f'seed{args.seed}'/args.scenario
        schedule=_read_lines(reference_root/'memory'/'packets.jsonl')
        old_arm=OLD_ARMS.get(args.child_arm)
        exact_packets=_read_lines(reference_root/old_arm/'packets.jsonl') if old_arm else None
        exact_scores={}
        old_summary=None
        if old_arm:
            old_summary=json.loads((reference_root/old_arm/'summary.json').read_text())
            if not old_summary['integrity']['all_pass']:
                raise RuntimeError('Old reference arm did not pass its existing gate')
            exact_scores={(value['packet'],value['phase']):value['rows']
                for value in _read_lines(reference_root/old_arm/'scores.jsonl')}
        if len(schedule)!=len(facts['packets']) or (exact_packets is not None and len(exact_packets)!=len(schedule)):
            raise RuntimeError('Old reference packet count differs')
        identities=_objects(learner)
        seen,bank=set(),[]
        records=[]; score_comparisons=0; final_prediction_rows=0
        training_started=None
        last_steps=optimizer_steps(learner)
        with (out/'packets.jsonl').open('x') as packets,(out/'scores.jsonl').open('x') as scores:
            for packet_index,packet in enumerate(facts['packets']):
                receipt,rows=packet['receipt'],packet['rows']
                if _objects(learner)!=identities or optimizer_steps(learner)!=last_steps:
                    raise RuntimeError('Parameters or retained Adam state changed between packets')

                def score(phase, inputs):
                    nonlocal score_comparisons,final_prediction_rows
                    values=probe(learner,inputs,seed=p['diagnostic_seed'])
                    if old_arm:
                        expected=exact_scores.get((packet['index'],phase))
                        if plain(values)!=expected:
                            raise RuntimeError(f'Old {old_arm} {phase} scores differ at packet {packet["index"]}')
                        score_comparisons+=len(values)
                        if packet['index']==len(facts['packets']) and phase in ('packet_after','bank_after'):
                            final_prediction_rows+=len(values)
                    jsonline(scores,dict(packet=packet['index'],phase=phase,
                        processed_at=receipt['processed_at'],rows=values))

                score('packet_before',rows)
                replay.add_target(rows)
                if training_started is None: training_started=time.time()
                update_started=time.time(); sampling=[]; metrics=None
                for step in range(p['policy_updates_per_packet']):
                    values=learner.continue_step(collect_diagnostics=step==0)
                    sampling.append(deepcopy(replay.last_sampling))
                    if step==0:
                        metrics=dict(packet=packet['index'],processed_at=receipt['processed_at'],**values)
                update_finished=time.time()
                expected_updates=initial_updates+packet['index']*p['policy_updates_per_packet']
                if learner.total_updates!=expected_updates or replay.target_received!=packet['index']*p['feedback_records_per_packet']:
                    raise RuntimeError('Feedback/update budget differs')
                for row in rows:
                    key=row.line_id,window(row.started_at)
                    if row.line_id in p['focus_lines'] and key not in seen:
                        seen.add(key);bank.append(row)
                score('packet_after',rows); score('bank_after',bank)
                last_steps=optimizer_steps(learner)
                record=dict(packet=packet['index'],receipt=receipt,updates=learner.total_updates-initial_updates,
                    optimizer_steps=last_steps,optimizer_objects_retained=_objects(learner)==identities,
                    update_started_wall=update_started,update_finished_wall=update_finished,
                    metrics=metrics,sampling=sampling,rng=_training_rng(learner))
                _check_packet(record,schedule[packet_index],
                    exact_reference=exact_packets[packet_index] if exact_packets is not None else None)
                jsonline(packets,record);records.append(record)
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='shared_level_packet',arm=args.child_arm,
                    current=packet['index'],total=len(facts['packets']),unit='packets',
                    updates=learner.total_updates-initial_updates)),flush=True)
        training_finished=time.time()
        frozen_updates,frozen_steps=learner.total_updates,optimizer_steps(learner)
        if old_arm and final_prediction_rows!=14:
            raise RuntimeError('The old final fourteen prediction rows were not exactly reproduced')
        freeze(learner)
        checkpoint=out/'final_model.pt'
        with checkpoint.open('xb') as handle:
            torch.save(dict(learner=learner,contract=contract,stations=stations,seed=args.seed,
                scenario=args.scenario,arm=args.child_arm,initial_preflight=preflight,
                old_reference_arm=old_arm,factual_proof=facts['proof']),handle)
        # No selected v34 issue is loaded until the learner has completed its
        # whole factual training budget and the frozen checkpoint is written.
        heldout=_export_heldout(learner,args.scenario,p,out)
        checks=dict(initial_native_parity=preflight['all_pass'], factual_reconstruction_exact=facts['proof']['all_pass'],
            factual_slots_285=replay.target_received==285,
            updates_456=learner.total_updates-initial_updates==456, original57_packets=len(records)==57,
            original_sampling_and_global_private_rng_exact=True, old_control_metrics_exact=True,
            old_final14_predictions_exact=not old_arm or final_prediction_rows==14,
            all_old_probe_chunks_exact=not old_arm or score_comparisons>0,
            reused_zero_actor_trace_gate=not old_arm or
                old_summary['integrity']['checks']['zero_full_actor_trace_exact'],
            optimizer_objects_retained=_objects(learner)==identities,
            post_budget_zero_training=learner.total_updates==frozen_updates and optimizer_steps(learner)==frozen_steps,
            frozen_heldout_export=heldout['all_pass'], no_world=learner.world is None,
            no_new_native_interaction=True,no_heldout_read_during_training=True,no_truth_training=True)
        summary=dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()),checks=checks),arm=args.child_arm,scenario=args.scenario,
            seed=args.seed,initializer=str(initializer),initial_preflight=preflight,
            facts=facts['metadata'], factual_proof=facts['proof'],
            old_reference_arm=old_arm,old_prediction_rows_compared=score_comparisons,
            old_final_prediction_rows=final_prediction_rows,heldout=heldout,
            resident_checkpoint=str(checkpoint),final_optimizer_steps=optimizer_steps(learner),
            budget=dict(feedback_slots=replay.target_received,shadow_updates=learner.total_updates-initial_updates,
                new_labels=0,native_seconds=0,native_suffixes=0,heldout_predictions=heldout['rows'],
                recorded_target_horizon_seconds=p['horizon_seconds']),
            bank_rows=[dict(sequence=row.sequence,line_id=row.line_id,issue=row.started_at,
                window=window(row.started_at)) for row in bank],
            wall=dict(pid=os.getpid(),started=started,training_started=training_started,
                training_finished=training_finished,finished=time.time()))
        write(out/'summary.json',summary)
        if not all(checks.values()): raise RuntimeError('Persistent shared-level worker failed integrity')
    except Exception as error:
        write(out/'failure.json',dict(status='failed',type=type(error).__name__,message=str(error)))
        raise


def triplet(args,p):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    processes,handles={},[]
    try:
        for arm in p['arms']:
            handle=(out/(arm+'.log')).open('x');handles.append(handle)
            command=[sys.executable,'-u',str(HERE/'run.py'),'--seed',str(args.seed),
                '--scenario',args.scenario,'--out',str(out/arm),'--child-arm',arm]
            processes[arm]=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT)
        last_progress={};last_poll=0.
        while any(process.poll() is None for process in processes.values()):
            failed={arm:process.returncode for arm,process in processes.items() if process.poll() not in (None,0)}
            if failed: raise RuntimeError(f'A parallel worker failed: {failed}')
            if time.monotonic()-last_poll>=5.:
                for arm in processes:
                    lines=[line for line in (out/(arm+'.log')).read_text().splitlines()
                        if line.startswith('SCHEDULEURM_PROGRESS')]
                    if lines and lines[-1]!=last_progress.get(arm):
                        print(lines[-1],flush=True);last_progress[arm]=lines[-1]
                last_poll=time.monotonic()
            time.sleep(.5)
        if any(process.returncode for process in processes.values()):
            raise RuntimeError('A parallel worker did not complete')
        values={arm:json.loads((out/arm/'summary.json').read_text()) for arm in p['arms']}
        overlap=min(value['wall']['training_finished'] for value in values.values())-max(
            value['wall']['training_started'] for value in values.values())
        update_overlap=optimizer_overlap({arm:_read_lines(out/arm/'packets.jsonl') for arm in p['arms']})
        checks=dict(workers_complete=all(value['integrity']['all_pass'] for value in values.values()),
            exactly_three_arms=set(values)==set(ARMS),
            distinct_workers=len({value['wall']['pid'] for value in values.values()})==3,
            actual_three_way_training_overlap=update_overlap>0,
            matched_original_budgets=all(value['budget']['feedback_slots']==285 and
                value['budget']['shadow_updates']==456 for value in values.values()))
        write(out/'summary.json',dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()),checks=checks),scenario=args.scenario,
            training_overlap_wall_seconds=overlap,optimizer_overlap_wall_seconds=update_overlap,
            workers={arm:value['wall'] for arm,value in values.items()},
            budget=dict(unique_original_labels=285,admissions_per_arm=285,shadow_updates=1368,
                new_labels=0,native_seconds=0,native_suffixes=0)))
        if not all(checks.values()): raise RuntimeError('Three-way parallel proof failed')
    except Exception as error:
        for process in processes.values():
            if process.poll() is None: process.terminate()
        for process in processes.values(): process.wait()
        write(out/'failure.json',dict(status='failed',type=type(error).__name__,message=str(error)))
        raise
    finally:
        for handle in handles: handle.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--scenario',choices=('stable','local_speed_drop'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--child-arm',choices=ARMS)
    args=parser.parse_args()
    protocol=json.loads((HERE/'protocol.json').read_text())
    if args.seed not in protocol['training_seeds']:raise ValueError('Seed outside the fixed v36 pilot')
    worker(args,protocol) if args.child_arm else triplet(args,protocol)
