"""Frozen issue action attribution and separate factual-chain target accounting."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    from experiments.mcwm_closed_loop_memory_truth_v34_20261024.frozen import plain
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(plain(value), handle, separators=(',', ':'), allow_nan=False)
        handle.write('\n')


def training_accounting(path):
    """Actual logged target sums, without inferring per-issue optimizer causes."""
    fields = ('trace_corrected_rows', 'correction_sum', 'future_reward_sum',
              'future_bootstrap_sum', 'future_minus_matching_q_sum',
              'decomposition_rounding_sum')
    entries = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(entries) != 54 or [r['packet'] for r in entries] != list(range(1, 55)):
        raise RuntimeError('Expected all54 original trace packet receipts')
    zero = {field: 0. for field in fields}
    points = [(0, zero), (27, entries[26]['target']), (54, entries[-1]['target'])]
    result = {}
    for label, left, right in [('all', points[0], points[2]),
                               ('first27', points[0], points[1]),
                               ('last27', points[1], points[2])]:
        delta = {field: right[1][field]-left[1][field] for field in fields}
        n = delta['trace_corrected_rows']
        result[label] = dict(packets=[left[0]+1, right[0]], corrected_sample_rows=n,
            mean_reward=delta['future_reward_sum']/n,
            mean_bootstrap=delta['future_bootstrap_sum']/n,
            mean_minus_matching_q=delta['future_minus_matching_q_sum']/n,
            mean_bootstrap_minus_q=(delta['future_bootstrap_sum']+delta['future_minus_matching_q_sum'])/n,
            mean_correction=delta['correction_sum']/n,
            mean_rounding=delta['decomposition_rounding_sum']/n)
    result['scope'] = 'Actual cumulative training statistics; no per-line or held-out-issue causal attribution is stored.'
    return result


def run(scenario):
    import numpy as np
    import torch
    from experiments.mcwm_closed_loop_physical_truth_v55_20261025.frozen import (
        build_frozen, score_frozen, executable_candidates, plain, optimizer_steps)
    from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import (
        _state, _equal, _rng)
    from experiments.mcwm_closed_loop_physical_shadow_v54_20261025.target import target_state
    from experiments.mcwm_closed_loop_action_attribution_v56_20261025.decomposition import (
        decompose_frozen, compare_frozen)
    from experiments.mcwm_closed_loop_action_attribution_v56_20261025.credit import compare_targets
    torch.set_num_threads(1)
    protocol = read(HERE/'protocol.json')
    out = HERE/'runs'/scenario
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    source = HERE.parent/protocol['state_version']/'runs'/scenario
    parent = read(source/'summary.json')
    if not (parent['status'] == 'complete' and parent['integrity']['all_pass']
            and parent['budget']['states'] == 4 and parent['budget']['suffixes'] == 8
            and parent['budget']['source_feedback_slots_per_model'] == 285
            and parent['budget']['source_updates_per_model'] == 456):
        raise RuntimeError('The complete immutable v55 parent evidence differs')
    learners, _, _, gate = build_frozen(scenario)
    before = {arm: deepcopy(_state(learner)) for arm, learner in learners.items()}
    private = {arm: deepcopy(target_state(learner)) for arm, learner in learners.items()}
    rng = _rng()
    proofs, details = [], []
    for slot in protocol['slots']:
        saved = read(source/'states'/slot.replace('/', '_')/'state_summary.json')
        selected = saved['selected']
        state, memory = selected['s'], selected['memory_s']
        actions = [saved['actions'][name][0] for name in ('low', 'high')]
        if not (slot == saved['slot'] and selected['line_id'] == slot.split('/')[0]
                and memory['observed_at'] <= selected['started_at']):
            raise RuntimeError('Saved issue identity or issue memory timing differs')
        for learner in learners.values():
            candidates = executable_candidates(learner.action_transform, state)
            if candidates != saved['candidates'] or not candidates['physically_distinct']:
                raise RuntimeError('Saved executable actions no longer match unchanged F')
        actual = score_frozen(learners, state, memory, actions)
        if plain(actual) != saved['prediction']:
            raise RuntimeError('The complete old v55 issue prediction differs: '+slot)
        decomposed, comparisons = {}, {}
        for critic in ('critic', 'target_critic'):
            decomposed[critic] = {arm: decompose_frozen(learner, state, memory, actions,
                critic_name=critic, reference=actual[arm] if critic == 'critic' else None)
                for arm, learner in learners.items()}
            comparisons[critic] = compare_frozen(decomposed[critic]['one_step'], decomposed[critic]['trace'])
        # Only after scoring: annotate the already collected truth. It is never fitted.
        truth = saved['comparison']['high_minus_low']['fixed300']
        if truth['status'] != 'complete':
            raise RuntimeError('Expected original complete300s diagnostic truth')
        detail = dict(slot=slot, issue=selected['started_at'], sequence=selected['sequence'],
            benefit300={field: -truth[field] for field in ('wait', 'ride', 'cost')},
            executed_actions=actions, holding_seconds=[0., 7.5], decomposition=decomposed,
            trace_minus_one_step=comparisons, physical=saved['comparison']['physical_separation'])
        details.append(detail)
        proofs.append(dict(slot=slot, full_old_prediction_exact=True, executable_F_exact=True,
            issue_memory_preserved=True, truth_not_fitted=True))
    chain_path = HERE.parent/protocol['chain_version']/'runs'/scenario/'continuous_feedback.jsonl'
    chain = [json.loads(line)['transition'] for line in chain_path.read_text().splitlines() if line.strip()]
    one, other = learners['one_step'], learners['trace']
    if len(chain) != 271 or not _equal(one.physical_batch, other.physical_batch):
        raise RuntimeError('The frozen arms do not share the admitted271-row factual chain')
    if not torch.equal(one.physical_coefficients, other.physical_coefficients):
        raise RuntimeError('Physical credit coefficients differ')
    for index, row in enumerate(chain):
        for field, name in [('s','observations'),('ns','next_observations')]:
            explicit = np.concatenate((np.asarray(row[field], np.float32),
                np.asarray(row['memory_'+field]['vector'], np.float32)))
            if not np.array_equal(explicit, one.physical_batch[name][index].numpy()):
                raise RuntimeError('The frozen physical state/memory chain differs')
        for field, name in [('a','actions'),('r','rewards'),('dt','delta_t_seconds'),('terminal','terminals')]:
            if not np.array_equal(np.asarray(row[field], np.float32).reshape(-1),
                one.physical_batch[name][index].numpy().reshape(-1)):
                raise RuntimeError('The frozen physical action/reward/time chain differs')
    credit = compare_targets(learners, chain, seeds=protocol['credit_draw_seeds'])
    history = training_accounting(HERE.parent/protocol['model_version']/'runs'/scenario/'trace'/'packets.jsonl')
    checks = dict(source_frozen_final_predictions_exact=gate['all_pass'],
        eight_old_issue_predictions_exact=len(proofs) == 4,
        model_parameters_and_Adam_unchanged=all(_equal(before[arm], _state(learner))
            for arm, learner in learners.items()),
        target_private_streams_counters_unchanged=all(_equal(private[arm], target_state(learner))
            for arm, learner in learners.items()),
        global_RNG_unchanged=_equal(rng, _rng()),
        original285_slots456_updates=all(learner.total_updates == 50456 and
            learner.replay_buffer.target_received == 285 for learner in learners.values()),
        no_world_or_trust_update=all(learner.world is None for learner in learners.values()),
        F_unchanged=True, no_training_or_native_or_truth_admission=True)
    result = dict(status='complete' if all(checks.values()) else 'integrity_failed',
        scenario=scenario, integrity=dict(all_pass=all(checks.values()), checks=checks),
        source_gate=gate, issue_proofs=proofs, issues=details, frozen_credit=credit,
        actual_training_target_accounting=history,
        budget=dict(source_feedback_slots=285, source_optimizer_updates=456,
            initializer_updates=50000, horizon_seconds=24000,
            new_training_labels=0, new_optimizer_updates=0, new_native_branches=0,
            truth_admitted=0, frozen_scored_issues=4),
        wall_seconds=time.time()-started)
    write(out/'details.json', result)
    # The full decomposition remains resident. Only compact results are collected.
    from experiments.mcwm_closed_loop_action_attribution_v56_20261025.analyze import compact_case
    write(out/'summary.json', compact_case(result))
    if not all(checks.values()):
        raise RuntimeError('Read-only attribution changed frozen model or RNG state')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=('stable', 'local_speed_drop'), required=True)
    args = parser.parse_args()
    try:
        value = run(args.scenario)
        print(json.dumps(dict(status=value['status'], scenario=args.scenario)))
    except BaseException as error:
        write(HERE/'runs'/args.scenario/'failure.json', dict(type=type(error).__name__,
            message=str(error), new_optimizer_updates=0, new_native_branches=0))
        raise
