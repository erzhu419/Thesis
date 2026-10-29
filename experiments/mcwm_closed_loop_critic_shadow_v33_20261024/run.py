"""Parallel, persistent shadow learners driven only by existing factual logs."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from bootstrap import HERE, write
import numpy as np
import torch

from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.learning import make_learner
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.facts import load_facts, iter_issues
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.probes import probe, window
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _rng, _equal


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def jsonline(handle, value):
    handle.write(json.dumps(plain(value), separators=(',', ':'), allow_nan=False)+'\n')
    handle.flush()


def optimizer_steps(learner):
    result = {}
    for name in ('actor_optimizer', 'critic_optimizer', 'value_optimizer', 'alpha_optimizer'):
        opt = getattr(learner, name)
        steps = [float(state['step']) for state in opt.state.values() if 'step' in state]
        result[name] = dict(states=len(steps), min=min(steps, default=None), max=max(steps, default=None))
    return result


def persistent_objects(learner):
    return {name: (id(getattr(learner, name)), tuple(id(p) for p in
        getattr(learner, name).param_groups[0]['params']))
        for name in ('actor_optimizer', 'critic_optimizer', 'value_optimizer', 'alpha_optimizer')}


def worker(args, p):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        learner, replay, contract, stations, initializer = make_learner(args.seed, arm=args.child_arm)
        initial_updates = learner.total_updates
        saved_rng = _rng()
        original = HERE.parents[1]/p['original_runs'][args.scenario]
        facts = load_facts(original, contract, p)
        if not facts['proof']['all_pass'] or not _equal(saved_rng, _rng()):
            raise RuntimeError('Factual reconstruction failed or changed training RNG')
        if initial_updates != p['initializer_updates']:
            raise RuntimeError('Wrong shared initializer')
        identities = persistent_objects(learner)
        initial_critic_base = learner.critic.base if args.child_arm=='memory' else learner.critic
        base_parameter_ids = tuple(id(v) for v in initial_critic_base.parameters())
        issues = iter_issues(original) if args.child_arm=='zero_memory' else iter(())
        current_issue = next(issues, None)
        checked_issues = 0

        def check_issue(issue):
            nonlocal checked_issues
            actual = learner.act(np.asarray(issue['s'], np.float32)[None], deterministic=True)[0]
            if actual.tolist() != issue['proposed_a']:
                raise RuntimeError(f'Zero-memory factual actor differs at sequence {issue["sequence"]}')
            executed = learner.action_transform.numpy(np.asarray(issue['s'], np.float32), actual)
            if executed.tolist() != issue['a']:
                raise RuntimeError('Zero-memory executed action differs from its factual trace')
            checked_issues += 1

        seen, bank, records = set(), [], []
        update_exact = True
        training_started = None
        last_steps = optimizer_steps(learner)
        with (out/'packets.jsonl').open('x') as packets, (out/'scores.jsonl').open('x') as scores:
            for packet in facts['packets']:
                receipt, rows = packet['receipt'], packet['rows']
                while current_issue is not None and current_issue['sequence'] < packet['processed_before_sequence']:
                    check_issue(current_issue)
                    current_issue = next(issues, None)
                if optimizer_steps(learner) != last_steps or persistent_objects(learner) != identities:
                    raise RuntimeError('Optimizer or its accumulated state was reset between packets')
                before_rows = probe(learner, rows, seed=p['diagnostic_seed'])
                jsonline(scores, dict(packet=packet['index'], phase='packet_before',
                    processed_at=receipt['processed_at'], rows=before_rows))
                replay.add_target(rows)
                if training_started is None:
                    training_started = time.time()
                update_started = time.time()
                sampling = []
                metrics = None
                for step in range(p['policy_updates_per_packet']):
                    result = learner.continue_step(collect_diagnostics=step==0)
                    sampling.append(deepcopy(replay.last_sampling))
                    if step==0:
                        metrics = dict(packet=packet['index'], processed_at=receipt['processed_at'], **result)
                        if args.child_arm=='zero_memory':
                            update_exact &= metrics == packet['reference_update']
                            if not update_exact:
                                differing = [key for key in metrics if metrics[key] != packet['reference_update'].get(key)]
                                raise RuntimeError(f'Zero-memory packet {packet["index"]} differs: {differing}')
                update_finished = time.time()
                expected = initial_updates+packet['index']*8
                if learner.total_updates != expected or replay.target_received != packet['index']*5:
                    raise RuntimeError('The unchanged feedback/update budget diverged')
                if tuple(id(v) for v in initial_critic_base.parameters()) != base_parameter_ids:
                    raise RuntimeError('Base critic parameters were replaced between packets')
                for row in rows:
                    key = (row.line_id, window(row.started_at))
                    if row.line_id in p['focus_lines'] and key not in seen:
                        seen.add(key); bank.append(row)
                jsonline(scores, dict(packet=packet['index'], phase='packet_after',
                    processed_at=receipt['processed_at'], rows=probe(learner, rows, seed=p['diagnostic_seed'])))
                jsonline(scores, dict(packet=packet['index'], phase='bank_after',
                    processed_at=receipt['processed_at'], rows=probe(learner, bank, seed=p['diagnostic_seed'])))
                last_steps = optimizer_steps(learner)
                record = dict(packet=packet['index'], receipt=receipt,
                    updates=learner.total_updates-initial_updates, optimizer_steps=last_steps,
                    optimizer_objects_retained=persistent_objects(learner)==identities,
                    update_started_wall=update_started, update_finished_wall=update_finished,
                    metrics=metrics, sampling=sampling,
                    rng=dict(global_state=plain(_rng()), sampler=plain(learner.rng.bit_generator.state),
                             diagnostic=plain(learner.diagnostic_rng.bit_generator.state)))
                jsonline(packets, record)
                records.append(record)
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='shadow_packet',
                    arm=args.child_arm,current=packet['index'],total=57,unit='packets',
                    updates=learner.total_updates-initial_updates)), flush=True)
        training_finished = time.time()
        frozen_updates, frozen_steps = learner.total_updates, optimizer_steps(learner)
        # The zero comparator checks the rest of the actual day with its final
        # frozen weights; neither arm invents missing tail execution contexts.
        while current_issue is not None:
            check_issue(current_issue)
            current_issue = next(issues, None)
        checks = dict(factual_reconstruction_exact=facts['proof']['all_pass'],
            factual_slots_285=replay.target_received==285, updates_456=learner.total_updates-initial_updates==456,
            original_57_packets=len(records)==57,
            zero_reference_updates_exact=update_exact,
            zero_full_actor_trace_exact=args.child_arm!='zero_memory' or checked_issues==facts['metadata']['factual_trace_rows'],
            optimizer_objects_retained=persistent_objects(learner)==identities,
            post_budget_zero_training=learner.total_updates==frozen_updates and optimizer_steps(learner)==frozen_steps,
            no_world=learner.world is None, no_new_native_interaction=True,
            no_diagnostic_truth=True, no_tail_context_backfill=True)
        summary = dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()),checks=checks), arm=args.child_arm,
            scenario=args.scenario, seed=args.seed, initializer=str(initializer),
            facts=facts['metadata'], factual_proof=facts['proof'],
            budget=dict(feedback_slots=replay.target_received,shadow_updates=456,new_labels=0,native_seconds=0,
                        native_suffixes=0,recorded_target_horizon_seconds=24000),
            zero_actor_issues_checked=checked_issues, final_optimizer_steps=optimizer_steps(learner),
            bank_rows=[dict(sequence=r.sequence,line_id=r.line_id,issue=r.started_at,window=window(r.started_at)) for r in bank],
            wall=dict(pid=os.getpid(), started=started, training_started=training_started,
                      training_finished=training_finished, finished=time.time()))
        write(out/'summary.json', summary)
        if not all(checks.values()):
            raise RuntimeError('Cumulative shadow integrity failed')
    except Exception as error:
        write(out/'failure.json', dict(status='failed', type=type(error).__name__, message=str(error)))
        raise


def pair(args, p):
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    processes, handles = {}, []
    try:
        for arm in p['arms']:
            handle = (out/(arm+'.log')).open('x'); handles.append(handle)
            command = [sys.executable, '-u', str(HERE/'run.py'), '--seed', str(args.seed),
                '--scenario', args.scenario, '--out', str(out/arm), '--child-arm', arm]
            processes[arm] = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        last_progress={}; last_poll=0.
        while any(proc.poll() is None for proc in processes.values()):
            failed = {arm:proc.returncode for arm,proc in processes.items() if proc.poll() not in (None,0)}
            if failed:
                raise RuntimeError(f'A parallel worker failed: {failed}')
            if time.monotonic()-last_poll>=5.:
                for arm in processes:
                    lines=[line for line in (out/(arm+'.log')).read_text().splitlines()
                           if line.startswith('SCHEDULEURM_PROGRESS')]
                    if lines and lines[-1]!=last_progress.get(arm):
                        print(lines[-1],flush=True);last_progress[arm]=lines[-1]
                last_poll=time.monotonic()
            time.sleep(.5)
        if any(proc.returncode for proc in processes.values()):
            raise RuntimeError('A parallel worker did not complete')
        values = {arm:json.loads((out/arm/'summary.json').read_text()) for arm in p['arms']}
        overlap = min(v['wall']['training_finished'] for v in values.values())-max(
            v['wall']['training_started'] for v in values.values())
        checks = dict(workers_complete=all(v['integrity']['all_pass'] for v in values.values()),
                      distinct_workers=len({v['wall']['pid'] for v in values.values()})==2,
                      actual_training_overlap=overlap>0)
        write(out/'summary.json', dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()),checks=checks),scenario=args.scenario,
            training_overlap_wall_seconds=overlap, workers={a:v['wall'] for a,v in values.items()},
            budget=dict(unique_original_labels=285,admissions_per_arm=285,shadow_updates=912,
                        new_labels=0,native_seconds=0,native_suffixes=0)))
        if not all(checks.values()):
            raise RuntimeError('Parallel-worker proof failed')
    except Exception as error:
        for proc in processes.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in processes.values():
            proc.wait()
        write(out/'failure.json',dict(status='failed',type=type(error).__name__,message=str(error)))
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--scenario',choices=('stable','local_speed_drop'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--child-arm',choices=('zero_memory','memory'))
    args=parser.parse_args()
    protocol=json.loads((HERE/'protocol.json').read_text())
    if args.seed not in protocol['training_seeds']:
        raise ValueError('Seed outside fixed v33 pilot')
    worker(args,protocol) if args.child_arm else pair(args,protocol)
