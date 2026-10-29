"""Resident old-log stress of a prospective behavior and fixed feedback plan."""
from collections import Counter
import argparse
import json
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import numpy as np
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.learning import load_learner
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.frozen import freeze
from experiments.mcwm_closed_loop_multievent_credit_v38_20261024.frozen import frozen_state, unchanged
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _rng
from experiments.mcwm_closed_loop_trace_support_v39_20261024.context import behavior_context
from experiments.mcwm_closed_loop_trace_support_v39_20261024.trace import construct_trace
from experiments.mcwm_closed_loop_trace_design_v40_20261024.behavior import (
    actor_parameters, execution_measure, ratio_from_measures,
)
from experiments.mcwm_closed_loop_trace_design_v40_20261024.budget_plan import (
    build_plan, compact_plan, audit_real_branch,
)


def read(path): return json.loads(Path(path).read_text())


def lines(path):
    with Path(path).open() as handle:
        return [json.loads(row) for row in handle if row.strip()]


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, separators=(',', ':'), allow_nan=False); handle.write('\n')


def target_parameters(support):
    return dict(positive_probability=support['target_hold_probability'],
        zero_probability=support['target_zero_probability'],
        mean=support['target_positive_mean'], std=support['target_positive_std'])


def run(scenario, out, p):
    started = time.time(); out.mkdir(parents=True, exist_ok=False)
    v38 = HERE.parent/p['source_v38_directory']
    v39 = HERE.parent/p['source_v39_directory']
    recovered = HERE.parent/p['source_v39r1_directory']
    source_gate = read(recovered/'analysis/summary.json')
    if source_gate['status'] != 'complete' or not source_gate['integrity']['all_pass']:
        raise RuntimeError('Completed v39/v39r1 construction evidence required')
    reference, replay, contract, stations, initializer = load_learner(p['seed'], arm='fresh')
    freeze(reference)
    if reference.total_updates != 50000 or replay.target_received:
        raise RuntimeError('Prospective behavior must be the unchanged original initializer')
    before, rng = frozen_state(reference), _rng()
    plan = build_plan(); write(out/'budget_plan.json', compact_plan(plan))
    results = {name: [] for name in p['models']}; identities = Counter()
    for state in read(v38/'runs'/scenario/'native/summary.json')['states']:
        state_record = read(state['path']); slot = state_record['slot']
        native_root = Path(state['path']).parent
        cached = {}
        metadata = {}
        for model in p['models']:
            summary = read(recovered/'runs'/scenario/model/'summary.json')
            if not summary['integrity']['all_pass']:
                raise RuntimeError('Frozen source model evidence is incomplete')
            metadata[model] = {f"{r['slot']}/{r['arm']}": r for r in summary['heldout']}
        for arm in ('low', 'high'):
            branch = read(native_root/arm/'summary.json')
            relative = native_root.relative_to(v38/'runs'/scenario/'native')/arm/'events.jsonl'
            supplied = lines(v38/'runs'/scenario/'scored'/relative)
            rows = [r['transition'] for r in supplied[:p['chain_events']]]
            issue = rows[0]['started_at']
            # The prospective windows all allow at least600s before first packet.
            physical = audit_real_branch(rows, issue=issue, first_deadline=issue+600.)
            for model in p['models']:
                filename = slot.replace('/', '_')+'_'+arm+'.jsonl'
                path = recovered/'runs'/scenario/model/filename
                if not path.exists(): path = v39/'runs'/scenario/model/filename
                cache = lines(path)[:len(rows)]
                if len(cache) != len(rows) or any(
                    any(a[key] != b[key] for key in ('sequence', 'started_at', 'completed_at', 'dt'))
                    for a, b in zip(cache, rows)):
                    raise RuntimeError('Cached target evidence is not this physical chain')
                cached[model] = cache
            behavior_measures, contexts = [], []
            for i, row in enumerate(rows[1:], 1):
                context = behavior_context(row, branch['selected'], branch['execution'], contract, stations)
                if not context['known']:
                    raise RuntimeError('Actual atomic-batch input missing')
                parameters = actor_parameters(reference, context['state'])
                measure = execution_measure(reference.action_transform, context['state'], row['a'], parameters)
                identity = ratio_from_measures(measure, measure)
                identities['checked'] += 1
                identities['exact_unit'] += identity['c'] == 1. and identity.get('log_ratio') == 0.
                identities['reconstructed_inputs'] += context['reconstructed']
                behavior_measures.append(measure)
                contexts.append({k: v for k, v in context.items() if k != 'state'})
            for model in p['models']:
                cache = cached[model]; ratios = []
                for i, row in enumerate(rows[1:], 1):
                    support = cache[i]['support']
                    target = execution_measure(reference.action_transform, row['s'], row['a'],
                        target_parameters(support), geometry=support['geometry'])
                    ratios.append(ratio_from_measures(behavior_measures[i-1], target))
                native = [row['probe']['native_target'] for row in cache]
                baseline = [row['probe']['recorded_next_target_q'] for row in cache[:-1]]
                c = [r['c'] for r in ratios]
                zero = construct_trace(native, baseline, c, [True]*len(rows), 0., dt=[r['dt'] for r in rows])
                diagnostic = construct_trace(native, baseline, c, [True]*len(rows), 1., dt=[r['dt'] for r in rows])
                exact = np.array_equal(np.asarray(zero['targets'], np.float32).view(np.uint32),
                    np.asarray(native, np.float32).view(np.uint32))
                old = metadata[model][slot+'/'+arm]
                witness = old['first_reward_index0']
                in_capacity = witness is not None and witness < len(rows)
                mathematical = in_capacity and all(r['mathematical_point_support'] for r in ratios[:witness])
                log_weight = (witness*math.log(.9)+sum(r['log_c'] for r in ratios[:witness])
                    if mathematical and all(r.get('log_c') is not None for r in ratios[:witness]) else None)
                weight = diagnostic['reach_weights'][witness] if in_capacity else None
                result = dict(slot=slot, arm=arm, events=len(rows), physical=physical,
                    first_reward_index0=witness, reward_within_fixed45=in_capacity,
                    reward_mathematically_supported=bool(mathematical), reward_weight=weight,
                    reward_log10_weight=log_weight/math.log(10.) if log_weight is not None else None,
                    ideal_same_input_same_snapshot_reward_weight=.9**witness if in_capacity else None,
                    old_dirac_reward_weight=old['first_reward_diagnostic_weight'],
                    lambda_zero_bitwise_exact=exact,
                    root_native_target=native[0], root_design_target=diagnostic['targets'][0],
                    point_support_failures=sum(not r['mathematical_point_support'] for r in ratios),
                    global_map_support_certificates=sum(r['global_support_certificate'] for r in ratios),
                    different_execution_maps=sum(not r['same_execution_map'] for r in ratios),
                    coefficient_float64_underflow=sum(r.get('coefficient_underflow', False) for r in ratios),
                    atom_mass_underflow=sum(r['status'] == 'numerical_underflow' for r in ratios),
                    float32_link_zeros=sum(r['c'] > 0. and link['coefficient'] == 0.
                        for r, link in zip(ratios, diagnostic['links'])),
                    reasons=dict(Counter(r['reason'] for r in ratios)),
                    counterfactual_old_log=True, training_authorized=False,
                    heldout_feedback_admitted=0)
                write(out/model/(slot.replace('/', '_')+'_'+arm+'.json'), result)
                with (out/model/(slot.replace('/', '_')+'_'+arm+'.jsonl')).open('x') as handle:
                    for i, ratio in enumerate(ratios, 1):
                        json.dump(dict(event=i, sequence=rows[i]['sequence'], ratio=ratio,
                            behavior_context=contexts[i-1], counterfactual_only=True,
                            training_authorized=False), handle, separators=(',', ':'), allow_nan=False)
                        handle.write('\n')
                results[model].append(result)
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='design_shadow',
                scenario=scenario, slot=slot, arm=arm, completed=len(results[p['models'][0]]), total=12)), flush=True)
    frozen = unchanged(reference, before, rng)
    checks = dict(frozen_initializer=frozen['all_pass'],
        identity_all_exact=identities['checked'] == identities['exact_unit'],
        fixed_budget_arithmetic=plan['all_pass'],
        all_real45_closed=all(r['physical']['all_pass'] for values in results.values() for r in values),
        lambda_zero_all_exact=all(r['lambda_zero_bitwise_exact'] for values in results.values() for r in values),
        all_expected_old_logs=all(len(values) == 12 for values in results.values()))
    write(out/'summary.json', dict(status='complete', integrity=dict(all_pass=all(checks.values()), checks=checks),
        scenario=scenario, source_initializer=str(initializer), identities=dict(identities),
        models=results, frozen=frozen, budget=compact_plan(plan)['financing'],
        new_training_labels=0, new_updates=0, new_native_seconds=0,
        new_native_branches=0, old_heldout_feedback_admitted=0,
        reused_frozen_target_evidence=True, final_model_checkpoints_loaded=0,
        elapsed_seconds=time.time()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', required=True, choices=('stable', 'local_speed_drop'))
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    try: run(args.scenario, args.out, read(HERE/'protocol.json'))
    except BaseException as error:
        if not isinstance(error, SystemExit):
            write(args.out/'failure.json', dict(type=type(error).__name__, message=str(error),
                new_native_seconds=0, new_updates=0, new_training_labels=0))
        raise
