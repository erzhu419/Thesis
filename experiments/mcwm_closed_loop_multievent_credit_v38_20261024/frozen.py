"""Server-resident v37r1 models and read-only scores at real event endpoints."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np
import torch

from experiments.mcwm_closed_loop_paired_recovery_v37r1_20261024.bootstrap import formal
from experiments.mcwm_closed_loop_paired_recovery_v37r1_20261024.run import (
    ARMS, BELL_FIELDS, MODULES, pair_prediction, row_from_record,
)
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.learning import critic_states
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.probes import probe
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.run import optimizer_steps
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import (
    _equal, _restore, _rng,
)


def read(path):
    return json.loads(Path(path).read_text())


def jsonlines(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, separators=(',', ':'), allow_nan=False)
        handle.write('\n')


def frozen_state(learner):
    """Cheap in-process mutation evidence; no parameter digest or model copy."""
    parameters = [parameter for name in MODULES
                  for parameter in getattr(learner, name).parameters()]
    parameters.append(learner.log_alpha)
    if any(parameter.requires_grad or parameter.grad is not None for parameter in parameters):
        raise RuntimeError('The resident model must already be frozen before scoring')
    return dict(updates=learner.total_updates, feedback=learner.replay_buffer.target_received,
        online_updates=learner.online_critic_updates, steps=optimizer_steps(learner),
        parameters=[(id(parameter), parameter._version) for parameter in parameters],
        optimizers=[id(getattr(learner, name)) for name in
            ('actor_optimizer', 'critic_optimizer', 'value_optimizer', 'alpha_optimizer')],
        private={name: deepcopy(getattr(learner, name).bit_generator.state)
                 for name in ('rng', 'diagnostic_rng')})


def unchanged(learner, before, global_rng):
    checks = dict(frozen_parameters_and_versions=_equal(before['parameters'],
            frozen_state(learner)['parameters']),
        updates=learner.total_updates == before['updates'],
        feedback=learner.replay_buffer.target_received == before['feedback'],
        online_updates=learner.online_critic_updates == before['online_updates'],
        optimizer_steps=optimizer_steps(learner) == before['steps'],
        optimizer_objects=frozen_state(learner)['optimizers'] == before['optimizers'],
        global_rng=_equal(global_rng, _rng()),
        private_rng=all(_equal(value, getattr(learner, name).bit_generator.state)
                        for name, value in before['private'].items()))
    if not all(checks.values()):
        raise RuntimeError('Frozen scoring mutated '+str([key for key, value in checks.items() if not value]))
    return dict(all_pass=True, checks=checks, new_updates=0, new_feedback=0)


def verify_old(learner, inputs, saved, *, seed):
    """Repeat all four available v37r1 issue/CRN predictions once, without truth."""
    before, rng = frozen_state(learner), _rng()
    indexed = {value['slot']: value for value in saved}
    expected = {'7X/early', '102X/early', '705X/early', '705X/tail'}
    if (len(inputs) != 4 or len(saved) != 4 or set(indexed) != expected
            or {value['slot'] for value in inputs} != expected):
        raise RuntimeError('The source four-issue reproduction scope changed')
    for record in inputs:
        rows = {arm: row_from_record(record['paired_transitions'][arm]) for arm in ('low', 'high')}
        actual = pair_prediction(learner, rows, seed=seed)
        reference = indexed[record['slot']]
        if any(actual[key] != reference.get(key) for key in actual):
            changed = [key for key in actual if actual[key] != reference.get(key)]
            raise RuntimeError('Old frozen prediction differs at '+record['slot']+': '+str(changed))
    proof = unchanged(learner, before, rng)
    return dict(**proof, old_issues=4, old_predictions_exact=True,
        variants_and_components_exact=True, common_random_number_targets_exact=True,
        native_two_row_sampling_noise_exact=True, truth_read=False)


def load_frozen(scenario, model_root, source_root, diagnostic_seed=336202):
    """Load existing checkpoints on their resident server; never reconstruct SGD."""
    torch.set_num_threads(1)
    model_root, source_root = Path(model_root), Path(source_root)
    native = source_root/'runs'/'collection'/scenario
    inputs = read(native/'heldout_prediction_inputs.json')
    learners, proofs, contract, stations = {}, {}, None, None
    saved_rng = _rng()
    try:
        for arm in ARMS:
            root = model_root/scenario/arm
            summary = read(root/'summary.json')
            if (not summary['integrity']['all_pass']
                    or summary['budget']['feedback_slots'] != 285
                    or summary['budget']['shadow_updates'] != 456):
                raise RuntimeError('The source learner did not pass the original285/456 gate')
            payload = torch.load(root/'final_model.pt', map_location='cpu', weights_only=False)
            learner = payload['learner']
            if (payload['seed'] != 6202 or payload['scenario'] != scenario or payload['arm'] != arm
                    or not payload['factual_proof']['all_pass']
                    or learner.total_updates != 50456 or learner.replay_buffer.target_received != 285
                    or learner.world is not None):
                raise RuntimeError('Resident checkpoint identity or training budget changed')
            if contract is None:
                contract, stations = payload['contract'], payload['stations']
            elif payload['contract'] != contract or not np.array_equal(payload['stations'], stations):
                raise RuntimeError('The three frozen learners do not share the same environment contract')
            proofs[arm] = verify_old(learner, inputs, jsonlines(root/'heldout.jsonl'), seed=diagnostic_seed)
            learners[arm] = learner
        if not _equal(saved_rng, _rng()):
            raise RuntimeError('Resident checkpoint loading changed the native-prefix RNG')
    finally:
        _restore(saved_rng)
    return learners, contract, stations, dict(all_pass=True, scenario=scenario, arms=proofs,
        source_initializer_updates=50000, source_updates=456, source_feedback_slots=285,
        new_updates=0, new_feedback=0, checkpoints_loaded=len(learners), checkpoints_downloaded=0,
        old_heldout_truth_read=False, old_twelve_state_truth_read=False)


@torch.no_grad()
def score_chain(learner, rows, seed=336202):
    """Score each legal closed event with one-row CRN, including genuine dt=0."""
    before, rng = frozen_state(learner), _rng()
    results = []
    for supplied in rows:
        record = supplied.get('transition', supplied) if isinstance(supplied, dict) else supplied
        row = row_from_record(record) if isinstance(record, dict) else record
        parts = probe(learner, [row], seed=seed)[0]
        states = torch.as_tensor(row.s, dtype=torch.float32, device=learner.device).reshape(1, 229)
        memory = torch.as_tensor(row.memory_s['vector'], dtype=states.dtype,
                                 device=learner.device).reshape(1, 84)
        explicit = critic_states(learner, states, memory)
        action = torch.as_tensor(row.a, dtype=states.dtype, device=learner.device).reshape(1, 1)
        current = learner.critic(explicit, action)[:, 0]
        target = learner.target_critic(explicit, action)[:, 0]
        q_min, q_mean = float(current.min()), float(current.mean())
        target_q_min, target_q_mean = float(target.min()), float(target.mean())
        results.append(dict(sequence=row.sequence, line_id=row.line_id,
            started_at=row.started_at, completed_at=row.completed_at, dt=row.dt,
            terminal=bool(row.terminal), executed_action=float(action[0, 0]),
            holding_seconds=float(learner.action_transform.hold_seconds(action)[0, 0]),
            q_heads=current.tolist(), q_min=q_min, q_mean=q_mean,
            target_q_heads=target.tolist(), target_q_min=target_q_min, target_q_mean=target_q_mean,
            target_lag_min=target_q_min-q_min, target_lag_mean=target_q_mean-q_mean,
            td_min=parts['target']-q_min, td_mean=parts['target']-q_mean,
            **{key: parts[key] for key in BELL_FIELDS}))
    proof = unchanged(learner, before, rng)
    return dict(status='complete' if results else 'no_closed_transition', rows=results,
        integrity=proof, actual_event_endpoints_only=True, synthetic300_state=False,
        probe_contract='one_row_same_seed_common_random_numbers', seed=seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', required=True, choices=('stable', 'local_speed_drop'))
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--native-root', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=336202)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    learners, _, _, proof = load_frozen(args.scenario, args.model_root, args.source_root, args.seed)
    if args.verify_only:
        write(args.out, proof)
        return
    if args.native_root is None:
        parser.error('--native-root is required for event-chain scoring')
    args.out.mkdir(parents=True, exist_ok=False)
    suffixes = sorted(args.native_root.rglob('events.jsonl'))
    count = 0
    for suffix in suffixes:
        rows = [value for value in jsonlines(suffix) if value.get('transition') is not None]
        scored = {arm: score_chain(learner, rows, args.seed) for arm, learner in learners.items()}
        target = args.out/suffix.relative_to(args.native_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x') as output:
            for index, row in enumerate(rows):
                value = dict(row, scores={arm: result['rows'][index] for arm, result in scored.items()})
                json.dump(value, output, separators=(',', ':'), allow_nan=False)
                output.write('\n')
        count += sum(len(result['rows']) for result in scored.values())
    write(args.out/'summary.json', dict(status='complete', integrity=proof,
        suffix_logs=len(suffixes), model_scored_rows=count, new_updates=0,
        new_feedback=0, checkpoints_downloaded=0))


if __name__ == '__main__':
    main()
