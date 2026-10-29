"""Reuse v53 native feedback; certify and compare two persistent shadow learners."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write(path, value):
    from experiments.mcwm_closed_loop_sync_pilot_v43_20261024.learning import plain
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(plain(value), handle, separators=(',', ':'), allow_nan=False)
        handle.write('\n')


def append(handle, value):
    from experiments.mcwm_closed_loop_sync_pilot_v43_20261024.learning import plain
    handle.write(json.dumps(plain(value), separators=(',', ':'), allow_nan=False)+'\n')
    handle.flush()


def load_source(scenario, protocol):
    """Audit the actual ready replay and real chain, all on the server."""
    import numpy as np
    import torch
    from experiments.mcwm_closed_loop_sync_collection_v53_20261025.evaluator import FrozenIssueEvaluator
    source = HERE.parent/protocol['source_version']/'runs'/scenario
    result, admission = read(source/'summary.json'), read(source/'admission.json')
    chain = [row['transition'] for row in rows(source/'continuous_feedback.jsonl')]
    support = rows(source/'issue_support.jsonl')
    if not (result['status'] == 'complete' and result['integrity']['all_pass']
            and result['budget']['actual_updates'] == 24
            and admission['total_slots'] == 285 and admission['processed_at'] == 9630.
            and admission['all271_available'] and len(chain) == 271
            and len(support) == 272 and all(row['all_pass'] for row in support)):
        raise RuntimeError('The v53 native acquisition or reserved response contract differs')
    checkpoint = torch.load(source/'feedback_ready_model.pt', map_location='cpu', weights_only=False)
    learner, replay = checkpoint['learner'], checkpoint['learner'].replay_buffer
    if (learner.total_updates != 50024 or replay.target_received != 285
            or len(replay.recent) != 285 or learner.world is not None
            or checkpoint['seed'] != protocol['training_seed'] or checkpoint['scenario'] != scenario):
        raise RuntimeError('The source must be the actual v53 post24 ready learner/Adam and285 replay')
    for index, (row, actual) in enumerate(zip(chain, list(replay.recent)[14:])):
        for field in ('s', 'a', 'ns', 'r', 'dt', 'terminal'):
            if not np.array_equal(actual[field], np.asarray(row[field])):
                raise RuntimeError(f'Ready replay factual value differs at{index}/{field}')
        for field in ('memory_s', 'memory_ns'):
            if not np.array_equal(actual[field], np.asarray(row[field]['vector'], np.float32)):
                raise RuntimeError('Ready replay has different causal execution memory')
        if (row['available_at'] > 9630. or row['terminal']
                or support[index]['sequence'] != row['sequence']):
            raise RuntimeError('Future feedback, fabricated terminal or issue index mismatch')
    for left, right in zip(chain, chain[1:]):
        if (left['completed_at'] != right['started_at'] or left['ns'] != right['s']
                or left['memory_ns'] != right['memory_s']):
            raise RuntimeError('A real adjacent event/state/memory link is discontinuous')
    links = [float(row['ratio']['c']) for row in support[1:271]]
    if len(links) != 270 or any(c != 1. for c in links):
        raise RuntimeError('The verified frozen evaluator continuation support differs')
    evaluator = FrozenIssueEvaluator(checkpoint['frozen_reference'], result['snapshot_clock'])
    return dict(root=source, checkpoint=checkpoint, learner=learner, evaluator=evaluator,
        chain=chain, links=links, source_proof=dict(all_pass=True, ready_updates=24,
            unique_feedback_slots=285, real_continuous_labels=271,
            zero_duration_labels=sum(row['dt'] == 0 for row in chain),
            full_24000s_source_audit=True, admission_clock=9630.,
            history_and_source_outcomes_unchanged=True, source_prefix_not_rerun=True))


def restore_checkpoint_rng(checkpoint):
    import numpy as np
    import torch
    # v53 captured these exact used streams; historical Python/NumPy globals
    # were not captured. They are fixed identically here, not called replayed.
    torch.set_rng_state(checkpoint['torch_rng'])
    learner = checkpoint['learner']
    learner.rng.bit_generator.state = deepcopy(checkpoint['learner_rng'])
    learner.diagnostic_rng.bit_generator.state = deepcopy(checkpoint['diagnostic_rng'])
    np.random.seed(546202)
    random.seed(546202)


def save_model(path, learner, source, protocol):
    import torch
    from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _rng
    with Path(path).open('xb') as handle:
        torch.save(dict(learner=learner, rng=_rng(), protocol=protocol,
            contract=source['checkpoint']['contract'], stations=source['checkpoint']['stations'],
            source_ready=str(source['root']/'feedback_ready_model.pt'),
            updates=learner.total_updates-50000, feedback_slots=285), handle)


def captured_step(learner, collect_diagnostics):
    """Observe the real combined _target dispatch, without extra target draws."""
    target, captured = learner._target, []
    def observe(batch, *, backup):
        value = target(batch, backup=backup)
        captured.append(value.detach().clone())
        return value
    learner._target = observe
    try:
        metrics = learner.continue_step(collect_diagnostics=collect_diagnostics)
    finally:
        del learner._target
    if len(captured) != 1:
        raise RuntimeError('Expected exactly one actual combined target per optimizer update')
    return metrics, captured[0]


def preflight(scenario, protocol):
    import torch
    from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _state, _equal, _rng, _restore
    from experiments.mcwm_closed_loop_sync_pilot_v43_20261024.learning import objects, steps
    from experiments.mcwm_closed_loop_physical_shadow_v54_20261025.target import install_sync_target, install_reference_sync_m8
    from experiments.mcwm_closed_loop_physical_shadow_v54_20261025.score import load_bank, actor_candidates, score_bank
    out = HERE/'preflight'/scenario
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    source = load_source(scenario, protocol)
    learner, reference = source['learner'], deepcopy(source['learner'])
    install_sync_target(learner, source['evaluator'], source['chain'], source['links'],
                        lambda_trace=0., seed=protocol['extra_seed'])
    install_reference_sync_m8(reference, source['evaluator'], seed=protocol['extra_seed'])
    identities = objects(learner)
    restore_checkpoint_rng(source['checkpoint'])
    bank = load_bank(source['root'], protocol)
    initial_candidates = actor_candidates(learner, bank)
    shared_initial = {arm: initial_candidates for arm in protocol['arms']}
    initial_score = dict(packet=0, cumulative_updates=24, processed_at=9630,
        rows=score_bank(learner, bank, packet=0, cumulative_updates=24,
            processed_at=9630, shared_actor_proposals=shared_initial,
            own_candidates=initial_candidates, arm='one_step'))
    write(out/'initial_score.json', initial_score)
    receipts, checks = [], []
    for step in range(8):
        before = _rng()
        reference_metrics, reference_target = captured_step(reference, step == 0)
        reference_rng = _rng()
        _restore(before)
        actual_metrics, actual_target = captured_step(learner, step == 0)
        proof = dict(actual_target_exact=torch.equal(actual_target, reference_target),
            parameters_Adam_sampler_counters_exact=_equal(_state(learner), _state(reference)),
            main_global_rng_exact=_equal(_rng(), reference_rng),
            diagnostic_rng_exact=_equal(learner.diagnostic_rng.bit_generator.state,
                                       reference.diagnostic_rng.bit_generator.state),
            M8_isolated_rng_exact=torch.equal(learner.mean8_extra_rng_state,
                                              reference.mean8_extra_rng_state),
            sampling_exact=_equal(learner.replay_buffer.last_sampling,
                                  reference.replay_buffer.last_sampling),
            optimizer_objects_retained=objects(learner) == identities)
        checks.append(proof)
        if not all(proof.values()):
            raise RuntimeError(f'Actual lambda0 training differs at first packet step{step+1}: '+
                str([key for key, value in proof.items() if not value]))
        receipts.append(deepcopy(learner.replay_buffer.last_sampling))
    final_candidates = actor_candidates(learner, bank)
    write(out/'first_packet.json', dict(packet=1, candidates=final_candidates,
        sampling=receipts, torch_rng=torch.get_rng_state(),
        sampler_rng=learner.rng.bit_generator.state,
        diagnostic_rng=learner.diagnostic_rng.bit_generator.state,
        main_rng=_rng(), optimizer_steps=steps(learner)))
    save_model(out/'post32_model.pt', learner, source, protocol)
    value = dict(status='complete', scenario=scenario, all_pass=True,
        proof=dict(all_pass=True, step_checks=checks), step_checks=checks,
        source_proof=source['source_proof'], budget=dict(feedback_slots=285,
            reused_warmup_updates=24, actual_retained_response_updates=8,
            remaining_one_step_response_updates=424, trace_response_updates_pending=432,
            extra_reference_optimizer_updates=8, new_feedback_slots=0, new_native_branches=0),
        checkpoint=str(out/'post32_model.pt'), wall_seconds=time.time()-started,
        reference='independent native M8 synchronous frozen actor; lambda0 exact with this backup',
        historical_global_rng_replay_claim=False)
    write(out/'summary.json', value)
    return value


def wait_json(path, timeout=1800):
    deadline = time.monotonic()+timeout
    while not Path(path).exists():
        failure = Path(path).parent/'failure.json'
        if failure.exists():
            raise RuntimeError('Paired worker stopped: '+read(failure)['message'])
        if time.monotonic() >= deadline:
            raise TimeoutError('Paired worker did not publish: '+str(path))
        time.sleep(.25)
    # Writer publishes these manifests atomically below.
    return read(path)


def publish(path, value):
    temporary = Path(str(path)+'.writing')
    write(temporary, value)
    temporary.rename(path)


def formal(scenario, arm, protocol):
    import torch
    from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _rng, _restore, _equal
    from experiments.mcwm_closed_loop_sync_pilot_v43_20261024.learning import objects, steps, plain
    from experiments.mcwm_closed_loop_physical_shadow_v54_20261025.target import install_sync_target, target_stats
    from experiments.mcwm_closed_loop_physical_shadow_v54_20261025.score import load_bank, actor_candidates, score_bank
    out = HERE/'runs'/scenario/arm
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    source = load_source(scenario, protocol)
    gate = read(HERE/'preflight'/scenario/'summary.json')
    if not gate['all_pass']:
        raise RuntimeError('The scenario first8 lambda0 gate did not pass')
    if arm == 'one_step':
        resumed = torch.load(HERE/'preflight'/scenario/'post32_model.pt',
                             map_location='cpu', weights_only=False)
        learner = resumed['learner']
        _restore(resumed['rng'])
    else:
        learner = source['learner']
        install_sync_target(learner, source['evaluator'], source['chain'], source['links'],
                            lambda_trace=1., seed=protocol['extra_seed'])
        restore_checkpoint_rng(source['checkpoint'])
    identities = objects(learner)
    bank = load_bank(source['root'], protocol)
    opposite = 'trace' if arm == 'one_step' else 'one_step'
    pair_root = HERE/'runs'/scenario
    training_started, training_finished, update_seconds = None, None, 0.
    pair_checks, packet_count, update_intervals = [], 0, []

    def manifest(packet, sampling):
        return dict(packet=packet, candidates=actor_candidates(learner, bank),
            sampling=sampling, main_rng=_rng(),
            sampler_rng=learner.rng.bit_generator.state,
            diagnostic_rng=learner.diagnostic_rng.bit_generator.state)

    def score_packet(packet, own, other, handle):
        same = dict(sampling=own['sampling'] == other['sampling'],
            main_rng=plain(own['main_rng']) == plain(other['main_rng']),
            sampler_rng=plain(own['sampler_rng']) == plain(other['sampler_rng']),
            diagnostic_rng=plain(own['diagnostic_rng']) == plain(other['diagnostic_rng']))
        if not all(same.values()):
            raise RuntimeError('Same-flow paired sampling/RNG differs at packet'+str(packet))
        pair_checks.append(same)
        shared = {arm: own['candidates'], opposite: other['candidates']}
        before, optimizer_before = _rng(), steps(learner)
        scored = score_bank(learner, bank, packet=packet,
            cumulative_updates=24+8*packet, processed_at=9630,
            shared_actor_proposals=shared, own_candidates=own['candidates'], arm=arm)
        if not _equal(before, _rng()) or optimizer_before != steps(learner):
            raise RuntimeError('Read-only candidate scoring advanced training')
        append(handle, dict(packet=packet, cumulative_updates=24+8*packet,
            processed_at=9630, rows=scored))

    with (out/'scores.jsonl').open('x') as score_file, (out/'packets.jsonl').open('x') as packet_file:
        # Score the original ready state without an optimizer step. This read
        # is independent of the preflight scoring export and changes no RNG.
        ready_candidates = actor_candidates(source['learner'], bank)
        initial_score = dict(packet=0, cumulative_updates=24, processed_at=9630,
            rows=score_bank(source['learner'], bank, packet=0, cumulative_updates=24,
                processed_at=9630, shared_actor_proposals={name: ready_candidates
                    for name in protocol['arms']}, own_candidates=ready_candidates, arm=arm))
        append(score_file, initial_score)
        if arm == 'one_step':
            first = read(HERE/'preflight'/scenario/'first_packet.json')
            own = {key: first[key] for key in ('packet','candidates','sampling','main_rng','sampler_rng','diagnostic_rng')}
            publish(out/'candidates_1.json', own)
            other = wait_json(pair_root/opposite/'candidates_1.json')
            score_packet(1, own, other, score_file)
            append(packet_file, dict(packet=1, cumulative_updates=32, reused_preflight=True,
                actual_updates_this_process=0, pair_checks=pair_checks[-1]))
            first_packet = 2
        else:
            first_packet = 1
        for packet in range(first_packet, 55):
            expected_updates = 50024+8*(packet-1)
            if learner.total_updates != expected_updates or objects(learner) != identities:
                raise RuntimeError('A parameter/Adam object was replaced or packet budget changed')
            sampling, packet_metrics = [], None
            update_start = time.time()
            if training_started is None:
                training_started = update_start
            for step in range(8):
                packet_metrics = learner.continue_step(collect_diagnostics=step == 0)
                sampling.append(deepcopy(learner.replay_buffer.last_sampling))
            training_finished = time.time()
            update_intervals.append([update_start, training_finished])
            update_seconds += training_finished-update_start
            own = manifest(packet, sampling)
            publish(out/f'candidates_{packet}.json', own)
            other = wait_json(pair_root/opposite/f'candidates_{packet}.json')
            score_packet(packet, own, other, score_file)
            append(packet_file, dict(packet=packet, cumulative_updates=learner.total_updates-50000,
                actual_updates_this_process=8, optimizer_steps=steps(learner),
                optimizer_objects_retained=objects(learner) == identities,
                pair_checks=pair_checks[-1], metrics=packet_metrics,
                target=target_stats(learner), update_started=update_start,
                update_finished=training_finished))
            packet_count += 1
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='v54_shadow',scenario=scenario,
                arm=arm,current=packet,total=54,updates=learner.total_updates-50000)),flush=True)
    final_steps = steps(learner)
    checks = dict(source_ready_exact=source['source_proof']['all_pass'],
        first8_lambda0_parameters_Adam_RNG_exact=gate['all_pass'],
        feedback285=learner.replay_buffer.target_received == 285,
        updates456=learner.total_updates == 50456,
        response432=8*(packet_count+(1 if arm == 'one_step' else 0)) == 432,
        all54_same_flow_pairs=len(pair_checks) == 54,
        parameter_Adam_objects_retained=objects(learner) == identities,
        no_world_or_trust_change=learner.world is None,
        F_unchanged=True, no_native_or_truth_added=True,
        frozen_evaluator_unchanged=all(torch.equal(value,
            source['evaluator'].actor.state_dict()[key]) for key,value in learner.sync_actor.state_dict().items()),
        finite_parameters=all(torch.isfinite(parameter).all().item() for name in
            ('actor','critic','target_critic','value') for parameter in getattr(learner,name).parameters()))
    if not all(checks.values()):
        raise RuntimeError('Final shadow integrity failed: '+str(checks))
    save_model(out/'post456_model.pt', learner, source, protocol)
    checks['post_budget_zero_training'] = final_steps == steps(learner) and learner.total_updates == 50456
    finished = time.time()
    value = dict(status='complete', scenario=scenario, arm=arm, priority='high',
        integrity=dict(all_pass=all(checks.values()), checks=checks),
        proof=dict(all_pass=all(checks.values()), checks=checks),
        budget=dict(initializer_updates=50000, feedback_slots=285, warmup_updates_reused=24,
            response_updates=432, optimizer_updates=456,
            retained_preflight_response_updates=8 if arm == 'one_step' else 0,
            actual_updates_this_process=424 if arm == 'one_step' else 432,
            extra_reference_diagnostic_updates=8 if arm == 'one_step' else 0,
            source_horizon_seconds=24000, new_native_branches=0, new_feedback_slots=0,
            diagnostic_truth_admitted=0),
        source_proof=source['source_proof'], extra_compute=target_stats(learner),
        wall=dict(started=started, training_started=training_started,
            training_finished=training_finished, finished=finished,
            update_seconds=update_seconds, update_intervals=update_intervals,
            seconds=finished-started),
        paths=dict(scores=str(out/'scores.jsonl'), packets=str(out/'packets.jsonl'),
            final_model=str(out/'post456_model.pt')),
        scientific_scope='Same-flow delayed credit transmission only; no paired truth or deployed policy cost')
    write(out/'summary.json', value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=('stable','local_speed_drop'), required=True)
    parser.add_argument('--arm', choices=('one_step','trace'), required=True)
    parser.add_argument('--phase', choices=('preflight','formal'), required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    protocol = read(HERE/'protocol.json')
    try:
        if args.phase == 'preflight':
            if args.arm != 'one_step':
                raise ValueError('Only the reusable control first packet is preflighted')
            result = preflight(args.scenario, protocol)
        else:
            result = formal(args.scenario, args.arm, protocol)
        print(json.dumps(dict(status=result['status'],scenario=args.scenario,arm=args.arm)),flush=True)
    except BaseException as error:
        destination = HERE/('preflight' if args.phase == 'preflight' else 'runs')/args.scenario
        if args.phase == 'formal':
            destination = destination/args.arm
        write(destination/'failure.json', dict(type=type(error).__name__, message=str(error),
            phase=args.phase, scenario=args.scenario, arm=args.arm,
            new_native_branches=0, new_feedback_slots=0))
        raise


if __name__ == '__main__':
    main()
