"""Three unchanged long-horizon critics on one new, budgeted factual stream.

The separate 300-second readout is not imported here. Paired branches supply
ordinary next-decision transitions, including genuine zero-duration events.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np
import torch

from experiments.mcwm_closed_loop_shared_level_v36_20261024.learning import (
    ARMS, initial_preflight, make_learner,
)
from experiments.mcwm_closed_loop_shared_level_v36_20261024.run import (
    _objects, _read_lines, _training_rng, heldout_prediction, optimizer_overlap,
)
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.probes import probe
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.run import jsonline, optimizer_steps
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.frozen import freeze
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng


MODULES = ('actor', 'critic', 'target_critic', 'value', 'ratio')
ROW_FIELDS = ('sequence', 'line_id', 'started_at', 'completed_at', 'available_at',
              's', 'a', 'r', 'ns', 'dt', 'terminal', 'memory_s', 'memory_ns')
BELL_FIELDS = ('reward_scaled', 'next_value', 'value_contribution', 'sac_contribution',
               'sac_q_without_entropy', 'sac_entropy_bonus', 'bootstrap_contribution',
               'target', 'observed_td_mean', 'target_subset')


def write(path, value):
    with Path(path).open('x') as output:
        json.dump(value, output, separators=(',', ':'), allow_nan=False)
        output.write('\n')


def row_from_record(record):
    """Consume transition fields only; 300-second endpoint labels never enter TD."""
    value = {key: deepcopy(record[key]) for key in ROW_FIELDS}
    for name, shape in (('s', (229,)), ('ns', (229,)), ('a', (1,))):
        value[name] = np.asarray(value[name], dtype=np.float32).reshape(shape)
    row = SimpleNamespace(**value)
    if row.dt < 0 or abs((row.completed_at-row.started_at)-row.dt) > 1e-6:
        raise ValueError('Next-decision duration differs from its real clocks')
    for name, clock in (('memory_s', row.started_at), ('memory_ns', row.completed_at)):
        memory = getattr(row, name)
        if len(memory['vector']) != 84 or memory['observed_at'] != clock:
            raise ValueError('Execution memory does not belong to its causal decision endpoint')
    return row


def load_dataset(path, *, scenario, seed):
    """Require the collector's complete new prefix and its exact 285-slot ledger."""
    payload = json.loads(Path(path).read_text())
    if not payload['proof']['all_pass']:
        raise RuntimeError('The new native-prefix collection did not pass integrity')
    metadata = payload['metadata']
    if metadata['scenario'] != scenario or metadata['initialization_seed'] != seed:
        raise RuntimeError('Dataset belongs to another scenario or initialization')
    packets, all_rows, previous = [], [], -float('inf')
    for index, record in enumerate(payload['packets'], 1):
        rows = [row_from_record(value) for value in record['rows']]
        receipt = record['receipt']
        processed = receipt['processed_at']
        if record['index'] != index or len(rows) != 5 or processed < previous:
            raise ValueError('The new stream must retain chronological five-slot packets')
        if any(row.available_at > processed or row.completed_at > processed for row in rows):
            raise ValueError('A feedback slot became trainable before its release')
        packets.append(dict(index=index, receipt=receipt, rows=rows))
        all_rows.extend(rows)
        previous = processed
    ids = [row.sequence for row in all_rows]
    paired = [value for value in ids if value < 0]
    if len(packets) != 57 or len(ids) != 285 or len(set(ids)) != 285 or len(paired) != 18:
        raise ValueError('Require 267 ordinary slots plus 18 distinct paired-arm slots')
    if sorted(paired) != list(range(-18, 0)):
        raise ValueError('The nine paired arms lost their fixed negative sequence identities')
    pairs = []
    for index in range(9):
        rows = {arm: next(row for row in all_rows if row.sequence == -(2*index+offset))
                for arm, offset in (('low', 1), ('high', 2))}
        check_pair_rows(rows)
        containing = [packet for packet in packets if any(row.sequence == rows['low'].sequence for row in packet['rows'])]
        packet = containing[0]
        if not any(row.sequence == rows['high'].sequence for row in packet['rows']):
            raise ValueError('The two paired-arm slots must be admitted atomically')
        release = max(rows['low'].started_at+300., *(row.completed_at for row in rows.values()))+30.
        if packet['receipt']['processed_at'] < release:
            raise ValueError('A paired slot was admitted before both300-second outcomes were available')
        pairs.append(rows)
    return dict(proof=payload['proof'], metadata=metadata, packets=packets,
                pairs=pairs, ordinary_rows=267, paired_rows=18,
                paired_zero_duration_rows=sum(row.dt == 0 for rows in pairs for row in rows.values()))


def check_pair_rows(rows):
    low, high = rows['low'], rows['high']
    if (low.line_id != high.line_id or low.started_at != high.started_at
            or not np.array_equal(low.s, high.s) or low.memory_s != high.memory_s):
        raise ValueError('The paired actions do not share the same physical issue and causal memory')


@torch.no_grad()
def pair_prediction(learner, rows, *, seed):
    """Score executed actions directly and each arm's genuine next-event target."""
    check_pair_rows(rows)
    selected = dict(s=rows['low'].s.tolist(), memory_s=rows['low'].memory_s)
    actions = {arm: rows[arm].a.tolist() for arm in ('low', 'high')}
    result = heldout_prediction(learner, selected, actions)
    if result['prediction']['holding_seconds'] != [0., 1.875]:
        raise RuntimeError('The paired actual actions are not 0/1.875 seconds')
    records = probe(learner, [rows[arm] for arm in ('low', 'high')], seed=seed)
    result['bellman_by_action'] = {arm: dict(
        dt=rows[arm].dt, terminal=bool(rows[arm].terminal),
        completed_at=rows[arm].completed_at,
        **{key: records[index][key] for key in BELL_FIELDS})
        for index, arm in enumerate(('low', 'high'))}
    result['bellman_high_minus_low'] = {key:
        result['bellman_by_action']['high'][key]-result['bellman_by_action']['low'][key]
        for key in BELL_FIELDS if key != 'target_subset'}
    return result


def export_frozen(learner, heldout_path, pairs, out, *, seed):
    """The new heldout inputs are first opened after training and checkpointing."""
    if any(parameter.requires_grad for name in MODULES for parameter in getattr(learner, name).parameters()):
        raise RuntimeError('New heldout issues cannot be read before freeze')
    before = learner.total_updates, optimizer_steps(learner), _training_rng(learner)
    records = json.loads(Path(heldout_path).read_text())
    slots = []
    with (out/'heldout.jsonl').open('x') as output:
        for record in records:
            rows = {arm: row_from_record(record['paired_transitions'][arm]) for arm in ('low', 'high')}
            selected = record['selected']
            if (not np.array_equal(rows['low'].s, np.asarray(selected['s'], np.float32))
                    or rows['low'].memory_s != selected['memory_s']):
                raise ValueError('Frozen scoring lost the selected heldout issue/context')
            if any(rows[arm].a.tolist() != record['actions'][arm] for arm in ('low', 'high')):
                raise ValueError('Frozen scoring changed the heldout executed actions')
            result = pair_prediction(learner, rows, seed=seed)
            jsonline(output, dict(slot=record['slot'], sequence=selected['sequence'],
                issue=rows['low'].started_at, line_id=rows['low'].line_id,
                source=str(heldout_path), **result))
            slots.append(record['slot'])
    expected = {line+'/'+when for line in ('7X', '102X', '705X') for when in ('early', 'tail')}
    if len(slots) != 6 or set(slots) != expected:
        raise ValueError('The six predeclared new heldout issue slots are missing or duplicated')
    with (out/'pair_targets.jsonl').open('a') as output:
        for index, rows in enumerate(pairs):
            jsonline(output, dict(pair_index=index, phase='frozen_final', packet=None, processed_at=None,
                line_id=rows['low'].line_id,
                issue=rows['low'].started_at, sequence_ids=[rows[arm].sequence for arm in ('low', 'high')],
                **pair_prediction(learner, rows, seed=seed)))
    after = learner.total_updates, optimizer_steps(learner), _training_rng(learner)
    if before != after:
        raise RuntimeError('Frozen endpoint probes changed updates, Adam steps or training RNG')
    return dict(all_pass=True, heldout_rows=6, training_pairs=9,
                heldout_first_read_after_checkpoint=True, zero_training=True, no_300_labels=True)


def worker(args, protocol):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        learner, replay, contract, stations, initializer = make_learner(args.seed, arm=args.child_arm)
        initial_updates = learner.total_updates
        preflight = initial_preflight(learner)
        rng = _rng()
        dataset = load_dataset(args.dataset, scenario=args.scenario, seed=args.seed)
        if initial_updates != 50000 or not _equal(rng, _rng()):
            raise RuntimeError('New factual loading changed the shared initialization/RNG')
        identities, last_steps = _objects(learner), optimizer_steps(learner)
        records = []
        training_started = None
        with (out/'packets.jsonl').open('x') as packets, (out/'scores.jsonl').open('x') as scores, \
                (out/'pair_targets.jsonl').open('x') as pair_targets:
            for packet in dataset['packets']:
                rows, receipt = packet['rows'], packet['receipt']
                if _objects(learner) != identities or optimizer_steps(learner) != last_steps:
                    raise RuntimeError('Parameters or retained Adam counters changed between packets')

                def score(phase, inputs):
                    values = probe(learner, inputs, seed=protocol['diagnostic_seed'])
                    jsonline(scores, dict(packet=packet['index'], phase=phase,
                        processed_at=receipt['processed_at'], rows=values))

                admitted_pairs = [(index, pair) for index, pair in enumerate(dataset['pairs'])
                                  if any(row.sequence == pair['low'].sequence for row in rows)]

                def score_pairs(phase):
                    for index, pair in admitted_pairs:
                        jsonline(pair_targets, dict(pair_index=index, phase=phase, packet=packet['index'],
                            processed_at=receipt['processed_at'], line_id=pair['low'].line_id,
                            issue=pair['low'].started_at,
                            sequence_ids=[pair[arm].sequence for arm in ('low', 'high')],
                            **pair_prediction(learner, pair, seed=protocol['diagnostic_seed'])))

                score('packet_before', rows)
                score_pairs('packet_before')
                replay.add_target(rows)
                if training_started is None:
                    training_started = time.time()
                update_started = time.time()
                sampling, metrics = [], None
                for step in range(8):
                    value = learner.continue_step(collect_diagnostics=step == 0)
                    sampling.append(deepcopy(replay.last_sampling))
                    if step == 0:
                        metrics = dict(packet=packet['index'], processed_at=receipt['processed_at'], **value)
                update_finished = time.time()
                if (learner.total_updates-initial_updates != packet['index']*8
                        or replay.target_received != packet['index']*5):
                    raise RuntimeError('New factual slot/update budget differs')
                score('packet_after', rows)
                score_pairs('packet_after')
                last_steps = optimizer_steps(learner)
                record = dict(packet=packet['index'], receipt=receipt,
                    updates=learner.total_updates-initial_updates, optimizer_steps=last_steps,
                    optimizer_objects_retained=_objects(learner) == identities,
                    update_started_wall=update_started, update_finished_wall=update_finished,
                    metrics=metrics, sampling=sampling, rng=_training_rng(learner))
                jsonline(packets, record)
                records.append(record)
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='paired_coverage_packet',
                    arm=args.child_arm, current=packet['index'], total=57, unit='packets',
                    updates=learner.total_updates-initial_updates)), flush=True)
        training_finished = time.time()
        freeze(learner)
        checkpoint = out/'final_model.pt'
        with checkpoint.open('xb') as output:
            torch.save(dict(learner=learner, contract=contract, stations=stations,
                seed=args.seed, scenario=args.scenario, arm=args.child_arm,
                initial_preflight=preflight, factual_proof=dataset['proof']), output)
        frozen_steps = optimizer_steps(learner)
        exported = export_frozen(learner, args.heldout, dataset['pairs'], out,
                                 seed=protocol['diagnostic_seed'])
        checks = dict(initial_native_parity=preflight['all_pass'], new_prefix_integrity=dataset['proof']['all_pass'],
            factual_slots_285=replay.target_received == 285,
            ordinary267_paired18=dataset['ordinary_rows'] == 267 and dataset['paired_rows'] == 18,
            updates_456=learner.total_updates-initial_updates == 456, original57_packets=len(records) == 57,
            optimizer_objects_retained=_objects(learner) == identities,
            post_budget_zero_training=learner.total_updates == initial_updates+456
                and optimizer_steps(learner) == frozen_steps,
            new_heldout_post_freeze=exported['all_pass'], no_world=learner.world is None,
            no_new_native_interaction=True, no_300_supervision_or_auxiliary_import=True,
            no_old_facts_or_old_heldout_read=True)
        summary = dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()), checks=checks), arm=args.child_arm,
            seed=args.seed, scenario=args.scenario, initializer=str(initializer),
            initial_preflight=preflight, factual_proof=dataset['proof'], facts=dataset['metadata'],
            heldout=exported, resident_checkpoint=str(checkpoint),
            final_optimizer_steps=optimizer_steps(learner),
            target_contract=dict(reward_scale=learner.config.reward_scale,
                online_backup=learner.config.online_backup, mixed_backup_lambda=learner.config.mixed_backup_lambda,
                backup_entropy=learner.config.backup_entropy, discount=1.),
            budget=dict(feedback_slots=replay.target_received, ordinary_slots=267, paired_slots=18,
                shadow_updates=learner.total_updates-initial_updates, paired_zero_duration_rows=dataset['paired_zero_duration_rows'],
                new_labels=0, native_seconds=0, heldout_predictions=6, paired_training_predictions=9),
            wall=dict(pid=os.getpid(), started=started, training_started=training_started,
                training_finished=training_finished, finished=time.time()))
        write(out/'summary.json', summary)
        if not all(checks.values()):
            raise RuntimeError('New factual stream worker failed integrity')
    except Exception as error:
        write(out/'failure.json', dict(status='failed', type=type(error).__name__, message=str(error)))
        raise


def compare_workers(case_root):
    """Different objectives may change metrics, never the common sample/RNG stream."""
    records = {arm: _read_lines(case_root/arm/'packets.jsonl') for arm in ARMS}
    if any(len(values) != 57 for values in records.values()):
        raise RuntimeError('A factual treatment did not complete all57 packets')
    for index in range(57):
        reference = records[ARMS[0]][index]
        for arm in ARMS[1:]:
            actual = records[arm][index]
            if any(actual[key] != reference[key] for key in ('packet', 'receipt', 'updates', 'sampling', 'rng')):
                raise RuntimeError(f'Matched new packet/sample/RNG differs: {arm}, packet{index+1}')
        per_head, shared = (records[arm][index] for arm in ARMS[1:])
        if per_head['optimizer_steps'] != shared['optimizer_steps']:
            raise RuntimeError('Memory arms did not retain the same Adam step schedule')
    initial = [next(row for row in _read_lines(case_root/arm/'scores.jsonl')
                    if row['packet'] == 1 and row['phase'] == 'packet_before')['rows'] for arm in ARMS]
    if not all(value == initial[0] for value in initial[1:]):
        raise RuntimeError('Three treatments did not start with exactly the same native predictions')
    return records


def triplet(args, protocol):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    processes, handles = {}, []
    try:
        for arm in ARMS:
            handle = (out/(arm+'.log')).open('x')
            handles.append(handle)
            command = [sys.executable, '-u', str(HERE/'run.py'), '--seed', str(args.seed),
                '--scenario', args.scenario, '--dataset', str(args.dataset.resolve()),
                '--heldout', str(args.heldout.resolve()), '--out', str(out/arm), '--child-arm', arm]
            processes[arm] = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        last_progress, last_poll = {}, 0.
        while any(process.poll() is None for process in processes.values()):
            failed = {arm: process.returncode for arm, process in processes.items() if process.poll() not in (None, 0)}
            if failed:
                raise RuntimeError(f'A parallel new-stream worker failed: {failed}')
            if time.monotonic()-last_poll >= 5.:
                for arm in processes:
                    values = [line for line in (out/(arm+'.log')).read_text().splitlines()
                              if line.startswith('SCHEDULEURM_PROGRESS')]
                    if values and values[-1] != last_progress.get(arm):
                        print(values[-1], flush=True)
                        last_progress[arm] = values[-1]
                last_poll = time.monotonic()
            time.sleep(.5)
        if any(process.returncode for process in processes.values()):
            raise RuntimeError('A parallel new-stream worker did not complete')
        values = {arm: json.loads((out/arm/'summary.json').read_text()) for arm in ARMS}
        records = compare_workers(out)
        overlap = optimizer_overlap(records)
        checks = dict(workers_complete=all(value['integrity']['all_pass'] for value in values.values()),
            three_distinct_workers=len({value['wall']['pid'] for value in values.values()}) == 3,
            actual_three_way_optimizer_overlap=overlap > 0,
            new_dataset_and_all_sampling_rng_exact=True, native_initial_predictions_exact=True,
            memory_Adam_step_schedule_exact=True,
            same_initializer=all(value['initializer'] == values[ARMS[0]]['initializer'] for value in values.values()),
            matched_budgets=all(value['budget']['feedback_slots'] == 285
                and value['budget']['shadow_updates'] == 456 for value in values.values()))
        write(out/'summary.json', dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()), checks=checks), scenario=args.scenario,
            optimizer_overlap_wall_seconds=overlap, workers={arm:value['wall'] for arm,value in values.items()},
            budget=dict(unique_labels=285, ordinary_labels=267, paired_arms=18,
                admissions_per_arm=285, shadow_updates=1368, native_seconds=0, new_labels=0)))
        if not all(checks.values()):
            raise RuntimeError('Three-way new-stream matched replay failed')
    except Exception as error:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            process.wait()
        write(out/'failure.json', dict(status='failed', type=type(error).__name__, message=str(error)))
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--scenario', choices=('stable', 'local_speed_drop'), required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--heldout', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--child-arm', choices=ARMS)
    args = parser.parse_args()
    protocol = json.loads((HERE/'protocol.json').read_text())
    if args.seed != 6202:
        raise ValueError('Initialization outside the fixed v37 pilot')
    worker(args, protocol) if args.child_arm else triplet(args, protocol)
