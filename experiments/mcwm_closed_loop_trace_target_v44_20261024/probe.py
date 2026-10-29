"""Read-only native targets and executed-action ratios from resident v43 saves.

The actual post24 checkpoint had only15 admitted rows.  Later rows are scored
there as a counterfactual frozen-target diagnostic, never inserted into replay.
Targets use a declared one-row CRN computation; original SGD target draws were
not logged and are not claimed to be reconstructed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.mcwm_closed_loop_multievent_credit_v38_20261024.frozen import (
    frozen_state, row_from_record, unchanged,
)
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.frozen import freeze
from experiments.mcwm_closed_loop_trace_recovery_v39r1_20261024.probe import audit_probe
from experiments.mcwm_closed_loop_sync_pilot_v43_20261024.credit import prepare_chain, score_chain
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import (
    _equal, _restore, _rng,
)
from .trace import decompose_trace


LINES = ('7X', '102X', '705X')
ARMS = ('low', 'high')
REPLAY_FIELDS = ('s', 'a', 'ns', 'r', 'dt', 'terminal')


def read(path):
    return json.loads(Path(path).read_text())


def jsonlines(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def optimizer_versions(learner):
    """Tensor identity/version proof also covers Adam moments, not only steps."""
    result = []
    for name in ('actor_optimizer', 'critic_optimizer', 'value_optimizer', 'alpha_optimizer'):
        optimizer = getattr(learner, name)
        for parameter, state in optimizer.state.items():
            for field, value in state.items():
                result.append((name, id(parameter), field,
                    (id(value), value._version) if isinstance(value, torch.Tensor) else value))
    return result


def replay_parity(learner, expected):
    """Compare resident admitted values directly; no target ingestion or hashes."""
    actual = list(learner.replay_buffer.recent)
    if len(actual) != len(expected) or learner.replay_buffer.target_received != len(expected):
        raise RuntimeError('Saved replay admission count differs from its original rows')
    for index, (saved, row) in enumerate(zip(actual, expected)):
        for field in REPLAY_FIELDS:
            if not np.array_equal(np.asarray(saved[field]), np.asarray(row[field])):
                raise RuntimeError(f'Saved replay differs at row{index}/{field}')
        for field in ('memory_s', 'memory_ns'):
            if not np.array_equal(np.asarray(saved[field]), np.asarray(row[field]['vector'], np.float32)):
                raise RuntimeError(f'Saved replay context differs at row{index}/{field}')
    return dict(all_pass=True, admitted_rows=len(actual), values_exact=True,
                causal_contexts_exact=True, replay_modified=False)


def _source(source_root):
    root = Path(source_root)
    pilot, receipt = read(root/'pilot_summary.json'), read(root/'response_receipt.json')
    updates, snapshot = read(root/'learning/update_summary.json'), read(root/'behavior_snapshot.json')
    budget = pilot['native']['budget']
    if not (pilot['status'] == 'complete' and pilot['integrity']['all_pass']
            and updates['integrity']['all_pass'] and updates['budget']['received'] == 285
            and updates['budget']['actual_updates'] == 456
            and receipt['processed_at'] == 9630. and receipt['all270_admitted_before_updates']
            and budget['feedback_slots'] == 285 and budget['optimizer_updates'] == 456):
        raise RuntimeError('The v43 source did not complete its original285/456 admission contract')
    ordinary = [row for index in (1, 2, 3) for row in read(root/f'warmup_packet_{index}.json')['rows']]
    indexed = {pair['line_id']: pair for pair in receipt['pairs']}
    if set(indexed) != set(LINES) or len(ordinary) != 15:
        raise RuntimeError('Original three-line pairs or15 warmup feedback rows are missing')
    chains = []
    for line in LINES:
        pair = indexed[line]
        root_records = []
        for arm in ARMS:
            branch = pair['branches'][arm]
            records = [value['transition'] for value in jsonlines(branch['events_path'])]
            if (len(records) != 45 or any(row['available_at'] > 9630. for row in records)
                    or records[0]['sequence'] != pair['selected']['sequence']):
                raise RuntimeError('A prescribed real45-event chain is short, late or misaligned')
            root_records.append(records[0])
            decoded = [row_from_record(row) for row in records]
            for current, following in zip(decoded, decoded[1:]):
                if (current.terminal or current.completed_at != following.started_at
                        or not np.array_equal(current.ns, following.s)
                        or current.memory_ns['observed_at'] != following.memory_s['observed_at']
                        or current.memory_ns['vector'] != following.memory_s['vector']):
                    raise RuntimeError('A real45-event chain has a noncausal state/memory link')
            chains.append(dict(line_id=line, arm=arm, records=records,
                selected=pair['selected'], branch=branch))
        low, high = root_records
        if (low['started_at'] != high['started_at'] or low['sequence'] != high['sequence']
                or not np.array_equal(low['s'], high['s'])
                or low['memory_s']['observed_at'] != high['memory_s']['observed_at']
                or low['memory_s']['vector'] != high['memory_s']['vector']):
            raise RuntimeError('Paired root issue states or causal memories differ')
    response = [chain['records'][event] for event in range(45) for chain in chains]
    all_rows = ordinary+response
    ids = [row['feedback_id'] for row in all_rows]
    if (len(ids) != 285 or len(set(ids)) != 285
            or [row['feedback_id'] for row in response] != receipt['feedback_ids']):
        raise RuntimeError('Original feedback ordering or unique285-slot identity changed')
    credit = {}
    with (root/'learning/credit_packets.jsonl').open() as handle:
        for line in handle:
            value = json.loads(line)
            if value['updates'] in (24, 456):
                credit[value['updates']] = {(item['line_id'], item['arm']): item for item in value['chains']}
    if set(credit) != {24, 456} or any(len(value) != 6 for value in credit.values()):
        raise RuntimeError('Original post24/post456 executed-ratio boundary logs are missing')
    return root, pilot, updates, snapshot, ordinary, all_rows, chains, credit


def _trace_row(learner, supplied, following, seed):
    row = row_from_record(supplied)
    result = audit_probe(learner, row,
        None if following is None else row_from_record(following), None, seed=seed)
    parts = result['parts']
    # audit_probe exercises the actual learner target after resetting the same
    # target samples and head subset. This is frozen native parity, not SGD replay.
    exact = np.float32(parts['reward_scaled'])+np.float32(parts['bootstrap_contribution'])
    if exact.view(np.uint32) != np.float32(result['native_target']).view(np.uint32):
        raise RuntimeError('Native float32 reward plus bootstrap assembly differs')
    if following is not None and not row.terminal and result['recorded_next_target_q'] is None:
        raise RuntimeError('An internal real event link has no legal matching-subset baseline: '+
                           str(result['baseline_missing_reason']))
    return dict(sequence=row.sequence, line_id=row.line_id, started_at=row.started_at,
        completed_at=row.completed_at, available_at=row.available_at, dt=row.dt,
        terminal=bool(row.terminal), gamma=1., executed_action=float(row.a[0]),
        holding_seconds=float(learner.action_transform.hold_seconds(torch.as_tensor(row.a))[0]),
        native_target=result['native_target'],
        **{field: parts[field] for field in ('reward_scaled', 'bootstrap_contribution',
            'value_contribution', 'sac_contribution', 'next_value', 'sac_q_without_entropy',
            'sac_entropy_bonus')},
        target_subset=result['target_subset'], recorded_next_target_q=result['recorded_next_target_q'],
        baseline_missing_reason=result['baseline_missing_reason'],
        subset_statistics=result['subset_statistics'],
        native_target_exact=result['native_target_equals_actual_learner_target'],
        causal_memory_s_observed_at=row.memory_s['observed_at'],
        causal_memory_ns_observed_at=row.memory_ns['observed_at'])


def _loaded(path):
    rng = _rng()
    try:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        learner = freeze(payload['learner'])
        if learner.world is not None:
            raise RuntimeError('The v43 diagnostic must retain its unchanged no-world/trust configuration')
        return learner, payload
    finally:
        _restore(rng)


@torch.no_grad()
def score_scenario(source_root, out=None, checkpoints=('actual_post24', 'post456'),
                   diagnostic_seeds=(336202, 336203, 336204)):
    """Return resident detail; caller alone writes small transferable summaries.

    ``source_root`` is v43 ``runs/{scenario}``. ``out`` is accepted for runner
    symmetry; this function performs no writes. For each checkpoint the first
    CRN seed audits all285 authorized rows; all three seeds score all six chains.
    The same seed is reset on every row/arm, so paired comparisons share actor
    samples and target-head subsets while retaining their actual successor states.
    """
    if not diagnostic_seeds or len(set(diagnostic_seeds)) != len(diagnostic_seeds):
        raise ValueError('Declared CRN seeds must be distinct and nonempty')
    if tuple(checkpoints) != ('actual_post24', 'post456'):
        raise ValueError('The predeclared actual post24 and post456 checkpoints are both required')
    torch.set_num_threads(1)
    global_rng = _rng()
    root, pilot, updates, snapshot, ordinary, all_rows, chains, saved_credit = _source(source_root)
    result = dict(source_budget=dict(initializer_updates=50000, feedback_slots=285,
        ordinary_slots=15, response_slots=270, updates=456, warmup_updates=24,
        response_updates=432, horizon_seconds=24000, original_run_complete=True),
        source_contract=dict(scenario=pilot['scenario'], initialization_seed=pilot['seed'],
            native_external_seed=pilot['native_external_seed'], F='unchanged_original_execution_transform',
            world_or_trust_changed=False, response_release_clock=9630.,
            roots=3, chains=6, events_per_chain=45, native_training_target_draws_replayed=False,
            target_contract='frozen native one-row CRN exact parity, not original SGD target sampling',
            label_truth_added=False, native_suffixes_added=0), checkpoints={})
    try:
        reference, reference_payload = _loaded(snapshot['resident_checkpoint'])
        if (reference.total_updates != 50024 or reference.replay_buffer.target_received != 15
                or reference_payload['scenario'] != pilot['scenario']
                or reference_payload['seed'] != pilot['seed']
                or reference_payload['total_updates'] != 50024
                or snapshot['total_updates'] != 50024):
            raise RuntimeError('The behavior snapshot is not the actual15-row post24 learner')
        reference_before, reference_rng = frozen_state(reference), _rng()
        reference_adam = optimizer_versions(reference)
        for chain in chains:
            chain['prepared'] = prepare_chain(reference, chain['records'], chain['selected'])
        for checkpoint in checkpoints:
            learner, payload = ((reference, reference_payload) if checkpoint == 'actual_post24'
                else _loaded(root/'learning/post456_model.pt'))
            update = 24 if checkpoint == 'actual_post24' else 456
            slots = 15 if checkpoint == 'actual_post24' else 285
            if (learner.total_updates != 50000+update or learner.replay_buffer.target_received != slots
                    or learner.config.mixed_backup_lambda != .1):
                raise RuntimeError('Frozen checkpoint budget or original mixed backup changed')
            if checkpoint == 'post456' and (payload['updates'] != 456 or
                    payload['source'] != 'new_v43_native_feedback'):
                raise RuntimeError('Post456 checkpoint does not belong to the completed v43 learner')
            before, rng = frozen_state(learner), _rng()
            adam = optimizer_versions(learner)
            parity = replay_parity(learner, all_rows[:slots])
            credit_by_chain = {}
            for chain in chains:
                saved = saved_credit[update][(chain['line_id'], chain['arm'])]
                prefixes = ({} if saved['first_reward_event'] is None else
                    {'first_reward': saved['first_reward_event']})
                actual = score_chain(learner, chain['prepared'], prefixes=prefixes)
                if not (actual['rows'] == saved['rows'] and actual['prefixes'] == saved['prefixes']):
                    raise RuntimeError('Actual F-executed ratios differ from original v43 boundary logs')
                credit_by_chain[(chain['line_id'], chain['arm'])] = actual
            seeds = {}
            for seed in diagnostic_seeds:
                scored = []
                for chain in chains:
                    records = chain['records']
                    rows = [_trace_row(learner, row, records[i+1] if i+1 < len(records) else None, seed)
                            for i, row in enumerate(records)]
                    credit = credit_by_chain[(chain['line_id'], chain['arm'])]
                    scored.append(dict(line_id=chain['line_id'], arm=chain['arm'], rows=rows,
                        c=[row['c'] for row in credit['rows']],
                        metadata=dict(root_intervention_excluded_from_ratio=True,
                            physical_span=records[-1]['completed_at']-records[0]['started_at'],
                            zero_duration_rows=sum(row['dt'] == 0. for row in records),
                            actual_hold=chain['branch']['execution']['native_applied_hold_seconds'],
                            next_real_event_only=True, truncated_after_real_events=45,
                            cutoff_is_terminal=False, feedback_available_before9630=True)))
                by_pair = {(chain['line_id'], chain['arm']): chain for chain in scored}
                for line in LINES:
                    low, high = (by_pair[(line, arm)]['rows'] for arm in ARMS)
                    if any(left['target_subset'] != right['target_subset']
                           for left, right in zip(low, high)):
                        raise RuntimeError('Paired one-row CRN target-head subsets differ')
                if seed == diagnostic_seeds[0]:
                    for chain in scored:
                        if not decompose_trace(chain['rows'], chain['c'], 0.)['lambda_zero_native_exact']:
                            raise RuntimeError('A285-slot response trace fails native lambda-zero parity')
                seeds[str(seed)] = dict(chains=scored)
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='v44_frozen_native_targets',
                    checkpoint=checkpoint, seed=seed, chains=6, rows=270)), flush=True)
            ordinary_scores = [_trace_row(learner, row, None, diagnostic_seeds[0]) for row in ordinary]
            if not all(decompose_trace([row], [], 0.)['lambda_zero_native_exact'] for row in ordinary_scores):
                raise RuntimeError('An ordinary285-slot row fails native lambda-zero parity')
            proof = unchanged(learner, before, rng)
            if adam != optimizer_versions(learner):
                raise RuntimeError('Read-only native target probe changed an Adam moment')
            result['checkpoints'][checkpoint] = dict(proof=dict(**proof,
                replay_parity=parity, actual_executed_ratio_replay_exact=True,
                pair_root_state_and_memory_exact=True, all_real_chain_links_causal=True,
                paired_CRN_target_head_subsets_exact=True,
                optimizer_tensor_versions_unchanged=True, locally_loaded_copy_frozen=True),
                all285_lambda0=dict(all_pass=True, rows=285, ordinary_rows=15, response_rows=270,
                    diagnostic_seed=diagnostic_seeds[0], native_target_exact=True,
                    float32_reward_bootstrap_assembly_exact=True,
                    all285_lambda_zero_trace_exact=True,
                    current_snapshot_admitted_rows=slots,
                    future_rows_scored_without_admission=285-slots,
                    original_SGD_sampling_reproduced=False),
                ordinary_rows_first_seed=ordinary_scores, seeds=seeds)
            if checkpoint == 'post456':
                del learner, payload
        reference_proof = unchanged(reference, reference_before, reference_rng)
        if reference_adam != optimizer_versions(reference):
            raise RuntimeError('The actual behavior reference Adam tensors changed')
        if not _equal(global_rng, _rng()):
            raise RuntimeError('Resident checkpoint load or frozen target scoring changed global RNG')
        result['integrity'] = dict(all_pass=True, original_source_audits_pass=True,
            reference=reference_proof, no_updates=True, no_new_feedback=True,
            no_simulator=True, global_rng_unchanged=True, checkpoints_downloaded=0)
    finally:
        _restore(global_rng)
    return result
