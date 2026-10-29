"""Natural replay of v43's recorded 24+432 one-step updates.

The source logged post-packet torch and private RNG, all eight replay draws,
and first-step diagnostics. It saved full learners only at post24/post456.
Those are the available exactness claims; no missing snapshot is invented and
no packet RNG is forced to its recorded end state.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.mcwm_closed_loop_shared_level_v36_20261024.learning import (
    make_learner, initial_preflight,
)
from experiments.mcwm_closed_loop_sync_pilot_v43_20261024.learning import plain, steps, objects
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import (
    _equal, _rng, _restore, _state,
)
from experiments.mcwm_closed_loop_trace_target_v44_20261024.probe import replay_parity


LINES, ARMS = ('7X', '102X', '705X'), ('low', 'high')


def read(path):
    return json.loads(Path(path).read_text())


def records(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def decode(record):
    value = deepcopy(record)
    for key in ('s', 'a', 'ns', 'raw_s', 'raw_ns'):
        if key in value and value[key] is not None:
            value[key] = np.asarray(value[key], dtype=np.float32)
    return SimpleNamespace(**value)


def load_source(root, *, scenario, seed):
    """Read resident factual records and the source's actual update receipts."""
    root = Path(root)
    pilot = read(root/'pilot_summary.json')
    learning = read(root/'learning/update_summary.json')
    response = read(root/'response_receipt.json')
    snapshot = read(root/'behavior_snapshot.json')
    packets = records(root/'learning/packets.jsonl')
    if not (pilot['status'] == 'complete' and pilot['integrity']['all_pass']
            and pilot['scenario'] == scenario and pilot['seed'] == seed
            and learning['integrity']['all_pass']
            and learning['budget']['received'] == 285
            and learning['budget']['actual_updates'] == 456
            and response['processed_at'] == 9630.
            and response['all270_admitted_before_updates']
            and len(packets) == 57):
        raise RuntimeError('The completed v43 factual source or original285/456 contract differs')
    warmup = [read(root/f'warmup_packet_{index}.json') for index in (1, 2, 3)]
    ordinary = [row for packet in warmup for row in packet['rows']]
    if len(ordinary) != 15 or any(len(packet['rows']) != 5 for packet in warmup):
        raise RuntimeError('The source requires three five-row warmup admissions')
    indexed = {pair['line_id']: pair for pair in response['pairs']}
    if set(indexed) != set(LINES):
        raise RuntimeError('The source lacks an originally declared line pair')
    chains = {}
    for line in LINES:
        root_rows = []
        for arm in ARMS:
            pair = indexed[line]
            chain = [value['transition'] for value in records(pair['branches'][arm]['events_path'])]
            if (len(chain) != 45 or chain[0]['sequence'] != pair['selected']['sequence']
                    or any(row['available_at'] > 9630. for row in chain)):
                raise RuntimeError('The source real45-event chain is incomplete, misaligned or late')
            for current, following in zip(chain, chain[1:]):
                if (current['terminal'] or current['completed_at'] != following['started_at']
                        or current['ns'] != following['s']
                        or current['memory_ns'] != following['memory_s']):
                    raise RuntimeError('The source real event chain or causal memory is discontinuous')
            for row in chain:
                if (len(row['memory_s']['vector']) != 84
                        or len(row['memory_ns']['vector']) != 84
                        or row['memory_s']['observed_at'] != row['started_at']
                        or row['memory_ns']['observed_at'] != row['completed_at']):
                    raise RuntimeError('A source context is not from its own causal endpoint')
            chains[(line, arm)] = chain
            root_rows.append(chain[0])
        low, high = root_rows
        if (low['s'] != high['s'] or low['memory_s'] != high['memory_s']
                or low['started_at'] != high['started_at'] or low['sequence'] != high['sequence']):
            raise RuntimeError('Paired source root issue states or memory differ')
    response_rows = [chains[(line, arm)][event]
                     for event in range(45) for line in LINES for arm in ARMS]
    all_rows = ordinary+response_rows
    ids = [row['feedback_id'] for row in all_rows]
    if (len(all_rows) != 285 or len(set(ids)) != 285
            or [row['feedback_id'] for row in response_rows] != response['feedback_ids']):
        raise RuntimeError('Source admission ordering or unique285-slot identity differs')
    schedule = []
    for index, logged in enumerate(packets, 1):
        slots = 5*index if index <= 3 else 285
        if (logged['packet'] != index or logged['updates'] != 8*index
                or logged['feedback_slots'] != slots or len(logged['sampling']) != 8
                or not logged['optimizer_objects_retained']):
            raise RuntimeError('Source packet index, optimizer retention or original8-step schedule differs')
        added = warmup[index-1]['rows'] if index <= 3 else response_rows if index == 4 else []
        clock = logged['receipt']['processed_at']
        if ((index >= 4 and clock != 9630.)
                or any(row['available_at'] > clock for row in added)):
            raise RuntimeError('Source feedback precedes physical release or response admission is not9630')
        schedule.append(dict(index=index, logged=logged, added=added,
                             rows=[decode(row) for row in added], processed_at=clock))
    return dict(root=root, pilot=pilot, learning=learning, response=response,
        snapshot=snapshot, schedule=schedule, all_rows=all_rows,
        checkpoint_paths={24: Path(snapshot['resident_checkpoint']),
                          456: root/'learning/post456_model.pt'})


def initial(seed):
    learner, replay, contract, stations, initializer = make_learner(seed, arm='shared_level')
    proof = initial_preflight(learner)
    if learner.total_updates != 50000 or replay.target_received or learner.world is not None:
        raise RuntimeError('Replay must begin at the original real-only50000 initializer')
    return learner, replay, contract, stations, initializer, proof


def certify_checkpoint(learner, source_path, expected_rows):
    """Compare actual saved tensors/Adam, without loading a source RNG into SGD."""
    live_rng = _rng()
    try:
        payload = torch.load(source_path, map_location='cpu', weights_only=False)
        reference = payload['learner']
        actual, expected = _state(learner), _state(reference)
        checks = {name: _equal(actual[name], expected[name]) for name in actual}
        checks['diagnostic_rng'] = _equal(learner.diagnostic_rng.bit_generator.state,
                                          reference.diagnostic_rng.bit_generator.state)
        checks['ratio_optimizer'] = _equal(learner.ratio.optimizer.state_dict(),
                                           reference.ratio.optimizer.state_dict())
        checks['replay_values_and_memory'] = replay_parity(learner, expected_rows)['all_pass']
        if not all(checks.values()):
            raise RuntimeError('M1 differs from original saved checkpoint: '+
                               str([key for key, value in checks.items() if not value]))
        return dict(updates=learner.total_updates-50000, source=str(source_path),
                    all_pass=True, checks=checks)
    finally:
        _restore(live_rng)


class ReplayEngine:
    """Keep one learner/Adam alive through all57 recorded eight-step packets."""
    def __init__(self, learner, replay, *, arm):
        self.learner, self.replay, self.arm = learner, replay, arm
        self.identities, self.last_steps = objects(learner), steps(learner)
        self.packet_count = 0
        self.admitted = []
        self.audit = dict(sampling_packets=0, sampling_steps=0, rng_packets=0,
                          optimizer_step_packets=0, first_step_metric_packets=0)
        self.target_calls = []

    def advance(self, packet):
        learner, replay, index = self.learner, self.replay, packet['index']
        if (index != self.packet_count+1 or objects(learner) != self.identities
                or steps(learner) != self.last_steps):
            raise RuntimeError('Packet order, optimizer object or retained Adam state changed')
        if index == 4 and (learner.total_updates != 50024 or replay.target_received != 15
                          or len(packet['rows']) != 270 or packet['processed_at'] != 9630.):
            raise RuntimeError('All270 response slots must enter before any response update')
        if packet['rows']:
            replay.add_target(packet['rows'])
            self.admitted.extend(packet['added'])
        slots = 5*index if index <= 3 else 285
        if replay.target_received != slots:
            raise RuntimeError('The original15+270 feedback-slot schedule changed')
        sampling, metrics, target_calls = [], None, []
        native_target = learner._target

        def capture_target(batch, *, backup):
            # This observes the existing update dispatch, without another
            # target call or draw. The source/history/fresh order is verified
            # against the exact current covariates and sampled source rows.
            size = len(batch['observations'])
            current = replay.last_covariates
            factual = torch.cat((current['observations'], current['memory_s']), dim=1)
            source_ids = replay.last_sampling['source']
            source_states = torch.as_tensor(replay.source['s'][source_ids],
                dtype=factual.dtype, device=learner.device)
            source_states = torch.cat((source_states,
                torch.zeros((len(source_states), factual.shape[1]-source_states.shape[1]),
                            dtype=factual.dtype, device=learner.device)), dim=1)
            combined = torch.cat((factual, source_states))
            if not torch.equal(batch['observations'], combined):
                raise RuntimeError('Actual target dispatch differs from the factual/source combined batch')
            target_calls.append(dict(step=step+1, backup=backup, batch_rows=size,
                partitions=dict(historical=len(replay.last_sampling['history']),
                    fresh=len(replay.last_sampling['recent']), source=len(source_ids)),
                combined_observation_order_exact=True,
                positive_estimator_used=backup in ('h2oplus', 'sac')))
            return native_target(batch, backup=backup)

        learner._target = capture_target
        try:
            for step in range(8):
                value = learner.continue_step(collect_diagnostics=step == 0)
                sampling.append(deepcopy(replay.last_sampling))
                if step == 0:
                    metrics = plain(value)
        finally:
            del learner._target
        if len(target_calls) != 8:
            raise RuntimeError('The recorded online update must dispatch one combined target per step')
        if learner.total_updates != 50000+8*index:
            raise RuntimeError('The original456 update budget changed')
        expected = packet['logged']
        actual_steps = steps(learner)
        rng = plain(dict(torch=torch.get_rng_state(), sampler=learner.rng.bit_generator.state,
                         diagnostic=learner.diagnostic_rng.bit_generator.state))
        checks = dict(sampling=sampling == expected['sampling'],
            torch_and_private_rng=rng == expected['rng'],
            optimizer_steps=actual_steps == expected['optimizer_steps'],
            optimizer_objects_retained=objects(learner) == self.identities)
        if self.arm == 'M1':
            checks['first_step_metrics'] = metrics == expected['metrics']
        if not all(checks.values()):
            failed = [name for name, value in checks.items() if not value]
            if not checks.get('first_step_metrics', True):
                failed.append('metric_keys='+str([key for key in metrics
                    if metrics[key] != expected['metrics'].get(key)]))
            raise RuntimeError(f'Natural {self.arm} replay differs at packet{index}: '+str(failed))
        self.packet_count, self.last_steps = index, actual_steps
        self.audit['sampling_packets'] += 1
        self.audit['sampling_steps'] += 8
        self.audit['rng_packets'] += 1
        self.audit['optimizer_step_packets'] += 1
        self.audit['first_step_metric_packets'] += int(self.arm == 'M1')
        self.target_calls.extend(target_calls)
        return dict(packet=index, receipt=expected['receipt'], updates=8*index,
            feedback_slots=replay.target_received, metrics=metrics, sampling=sampling,
            optimizer_steps=actual_steps, optimizer_objects_retained=True, rng=rng,
            source_checks=checks, actual_target_calls=target_calls)
