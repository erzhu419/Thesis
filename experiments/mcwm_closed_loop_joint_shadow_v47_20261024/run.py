"""Replay existing facts for a single M1/M8 shadow arm; never start SUMO."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, separators=(',', ':'), allow_nan=False)
        handle.write('\n')


def jsonline(handle, value):
    handle.write(json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n')
    handle.flush()


def run(scenario, arm, phase):
    protocol = json.loads((HERE/'protocol.json').read_text())
    if scenario not in protocol['scenarios'] or arm not in protocol['arms']:
        raise ValueError('Undeclared scenario/arm')
    if phase == 'preflight' and (scenario != protocol['preflight_scenario']
                                or arm != protocol['preflight_arm']):
        raise ValueError('The preflight is the single declared stableM1 replay')
    out = HERE/('preflight' if phase == 'preflight' else 'runs')/scenario/arm
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        import torch
        from experiments.mcwm_closed_loop_joint_shadow_v47_20261024.replay import (
            load_source, initial, certify_checkpoint, ReplayEngine,
        )
        from experiments.mcwm_closed_loop_joint_shadow_v47_20261024.mean_target import (
            install_mean8, mean8_stats, mean8_state,
        )
        from experiments.mcwm_closed_loop_joint_shadow_v47_20261024.score import load_bank, score_bank
        from experiments.mcwm_closed_loop_joint_shadow_v47_20261024.summary import compact_worker
        from experiments.mcwm_closed_loop_sync_pilot_v43_20261024.learning import steps, objects
        from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _rng, _equal
        torch.set_num_threads(1)
        source = load_source(HERE.parent/protocol['source_version']/'runs'/scenario,
                             scenario=scenario, seed=protocol['training_seed'])
        learner, replay, contract, stations, initializer, initial_proof = initial(protocol['training_seed'])
        initializer_rng = _rng()
        bank = load_bank(source['root']/'response_receipt.json')
        if len(bank) != 6 or not _equal(initializer_rng, _rng()):
            raise RuntimeError('Fixed existing issue-bank loading changed training RNG or scope')
        if arm == 'M8':
            install_mean8(learner, seed=protocol['extra_seed'])
        if not _equal(initializer_rng, _rng()):
            raise RuntimeError('The extra target stream changed the original training RNG at initialization')
        engine = ReplayEngine(learner, replay, arm=arm)
        identities = objects(learner)
        certifications, packet_audits = [], []
        training_started = None

        def save_checkpoint(update):
            path = out/f'post{update}_model.pt'
            with path.open('xb') as handle:
                torch.save(dict(learner=learner, contract=contract, stations=stations,
                    protocol=protocol, seed=protocol['training_seed'], scenario=scenario,
                    arm=arm, phase=phase, updates=update, rng=_rng(),
                    mean8_state=mean8_state(learner) if arm == 'M8' else None,
                    source='v47_same_factual_stream_shadow'), handle)

        with (out/'packets.jsonl').open('x') as packets, (out/'scores.jsonl').open('x') as scores:
            def score(packet, clock):
                rng_before, steps_before = _rng(), steps(learner)
                target_before = mean8_state(learner) if arm == 'M8' else None
                values = score_bank(learner, bank, packet=packet,
                    cumulative_updates=learner.total_updates-50000, processed_at=clock)
                if (not _equal(rng_before, _rng()) or steps_before != steps(learner)
                        or not _equal(target_before, mean8_state(learner) if arm == 'M8' else None)):
                    raise RuntimeError('Read-only bank scoring advanced training or the extra target stream')
                jsonline(scores, dict(packet=packet, cumulative_updates=learner.total_updates-50000,
                                     processed_at=clock, rows=values))

            score(0, 0.)
            if phase == 'formal':
                write(out/'ready.json', dict(arm=arm, updates=learner.total_updates,
                    feedback_slots=replay.target_received, ready_wall=time.time()))
                peer = out.parent/('M8' if arm == 'M1' else 'M1')
                while not (peer/'ready.json').exists():
                    if (peer/'failure.json').exists():
                        raise RuntimeError('Paired arm failed before training became ready')
                    time.sleep(.5)
            for packet in source['schedule']:
                if training_started is None:
                    training_started = time.time()
                begin = time.time()
                record = engine.advance(packet)
                record['update_started_wall'], record['update_finished_wall'] = begin, time.time()
                jsonline(packets, record)
                update = learner.total_updates-50000
                if update in (24, 456):
                    if arm == 'M1':
                        certifications.append(certify_checkpoint(learner,
                            source['checkpoint_paths'][update], engine.admitted))
                    save_checkpoint(update)
                score(packet['index'], packet['processed_at'])
                packet_audits.append(dict(packet=packet['index'], updates=update,
                    feedback_slots=replay.target_received, source_checks=record['source_checks']))
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='v47_'+phase,
                    scenario=scenario, arm=arm, current=packet['index'], total=57,
                    updates=update, feedback_slots=replay.target_received)), flush=True)
        training_finished = record['update_finished_wall']
        final_steps, final_updates = record['optimizer_steps'], 50000+record['updates']
        checks = dict(initial_native_parity=initial_proof['all_pass'], slots285=replay.target_received == 285,
            updates456=learner.total_updates == 50456, packets57=engine.packet_count == 57,
            original_sampling456_exact=engine.audit['sampling_steps'] == 456,
            original_training_rng57_exact=engine.audit['rng_packets'] == 57,
            original_adam_steps57_exact=engine.audit['optimizer_step_packets'] == 57,
            optimizer_objects_retained=objects(learner) == identities,
            all270_admitted_before_response_updates=True, no_world_or_trust=learner.world is None,
            unchanged_single_step_lambda0_structure=protocol['lambda_trace'] == 0,
            F_unchanged=True, issue_bank_fixed_six_records=True,
            no_new_native_or_feedback=True,
            post_budget_zero_training=final_updates == learner.total_updates and final_steps == steps(learner))
        if arm == 'M1':
            checks['first_step_metrics57_exact'] = engine.audit['first_step_metric_packets'] == 57
            checks['saved_post24_post456_modules_adam_exact'] = len(certifications) == 2
        else:
            checks['only_positive_target_estimator_attached'] = learner.__class__.__name__ == 'Mean8OutcomeTrainer'
        proof = dict(all_pass=all(checks.values()), checks=checks, audit_counts=engine.audit,
            source_certifications=certifications,
            boundaries=dict(full_module_Adam_source_snapshots=[24, 456] if arm == 'M1' else [],
                per_packet_full_Adam_snapshots_available=False,
                source_rng_scope='torch_and_private_sampler_and_diagnostic_only',
                historic_global_python_numpy_rng_available=False,
                per_packet_rng_forced=False, M1_metrics_reference=True if arm == 'M1' else None,
                M8_parameter_divergence_is_treatment=arm == 'M8',
                independent_physical_root_states=3, issue_records=6,
                new_300_second_truth=False, policy_deployed=False))
        if not proof['all_pass']:
            raise RuntimeError('Shadow replay final implementation gate failed')
        finished = time.time()
        value = dict(status='complete', scenario=scenario, arm=arm, phase=phase, priority='high',
            proof=proof, budget=dict(initializer_updates=50000, feedback_slots=285,
                optimizer_updates=456, horizon_seconds=24000, source_labels_reused=285,
                new_feedback_slots=0, new_native_branches=0, new_native_seconds=0,
                preflight_extra_optimizer_updates=456 if phase == 'preflight' else 0,
                formal_optimizer_updates=456 if phase == 'formal' else 0),
            extra_compute=mean8_stats(learner) if arm == 'M8' else dict(positive_samples=1,
                extra_positive_samples=0, extra_q_forwards=0, extra_seconds=0.),
            actual_target_path=dict(offline_objective=learner.config.offline_objective,
                online_backup=learner.config.online_backup,
                observed_calls=len(engine.target_calls),
                first_call=engine.target_calls[0],
                last_call=engine.target_calls[-1],
                target_dispatch='one combined history64/fresh64/source128 call per online update',
                applies_to='all three partitions in the actual combined target',
                original_source_batch_tensors_logged=False),
            initializer=str(initializer),
            wall=dict(started=started, training_started=training_started,
                training_finished=training_finished, finished=finished, seconds=finished-started),
            paths=dict(scores=str(out/'scores.jsonl'), packets=str(out/'packets.jsonl'),
                post24_model=str(out/'post24_model.pt'), final_model=str(out/'post456_model.pt')),
            decision=dict(policy_deployed=False, native_branch_started=False,
                next_gate='Compare same-candidate action advantages and F holding; noise reduction is not action correctness.'))
        write(out/'replay_result.json', dict(**value, packet_audits=packet_audits))
        write(out/'summary.json', compact_worker(value))
        print(json.dumps(dict(status=value['status'], scenario=scenario, arm=arm,
                              phase=phase, proof=proof)), flush=True)
        return value
    except Exception as error:
        write(out/'failure.json', dict(status='failed', type=type(error).__name__, message=str(error),
            scenario=scenario, arm=arm, phase=phase, elapsed_seconds=time.time()-started,
            new_feedback_slots=0, new_native_branches=0))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=('stable', 'local_speed_drop'), required=True)
    parser.add_argument('--arm', choices=('M1', 'M8'), required=True)
    parser.add_argument('--phase', choices=('preflight', 'formal'), required=True)
    args = parser.parse_args()
    run(args.scenario, args.arm, args.phase)
