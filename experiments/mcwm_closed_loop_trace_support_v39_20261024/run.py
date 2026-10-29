"""Resident-only frozen trace construction, execution support and slot audit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_multievent_credit_v38_20261024.frozen import (
    load_frozen, frozen_state, unchanged, row_from_record,
)
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.learning import load_learner
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.frozen import freeze
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _rng
from experiments.mcwm_closed_loop_trace_support_v39_20261024.budget import build_budget_index, audit_rows
from experiments.mcwm_closed_loop_trace_support_v39_20261024.context import behavior_context
from experiments.mcwm_closed_loop_trace_support_v39_20261024.probe import audit_probe
from experiments.mcwm_closed_loop_trace_support_v39_20261024.support import support_at
from experiments.mcwm_closed_loop_trace_support_v39_20261024.trace import construct_trace
import numpy as np
import torch


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as output:
        json.dump(value, output, separators=(',', ':'), allow_nan=False); output.write('\n')


def jsonlines(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def contiguous(left, right):
    return (left['sequence']+1 == right['sequence'] and left['completed_at'] == right['started_at']
        and left['ns'] == right['s'] and left['raw_ns'] == right['raw_s']
        and left['memory_ns'] == right['memory_s'])


def budget_chains(index):
    ordinary = sorted((slot for slot in index['slots'] if slot['kind'] == 'ordinary'),
        key=lambda slot: slot['source_sequence'])
    groups = []
    for slot in ordinary:
        if not groups or not contiguous(groups[-1][-1]['row'], slot['row']):
            groups.append([])
        groups[-1].append(slot)
    values = []
    for number, group in enumerate(groups):
        values.append(dict(id='ordinary/'+str(number), rows=[slot['row'] for slot in group],
            origin=dict(scenario=index['scenario'], prefix_id=index['prefix_id'], kind='ordinary')))
    for slot in index['slots']:
        if slot['kind'] == 'paired':
            values.append(dict(id='paired/'+slot['branch_id']+'/'+slot['arm'], rows=[slot['row']],
                origin=dict(scenario=index['scenario'], prefix_id=index['prefix_id'], kind='paired',
                    branch_id=slot['branch_id'], arm=slot['arm'], source_sequence=slot['source_sequence'])))
    return values


def witness_index(rows, identity):
    seen = Counter()
    for i, supplied in enumerate(rows):
        row = supplied['transition']
        base = tuple(str(row[key]) for key in ('line_id', 'vehicle_id', 'trip_id', 'station_id'))
        number = seen[base]; seen[base] += 1
        if list(base)+[number] == identity:
            return i
    raise RuntimeError('The saved real reward witness is absent from its chain')


def evaluate_chain(learner, behavior, contract, stations, rows, origin, index, *, cached,
                   selected=None, execution=None, reward_identity=None, out, seed):
    raw = [row['transition'] if 'transition' in row else row for row in rows]
    membership = audit_rows(raw, origin, index)
    probes, supports, contexts = [], [], []
    for i, supplied in enumerate(rows):
        following = raw[i+1] if i+1 < len(rows) else None
        reference = supplied['scores'][learner.memory_arm] if cached else None
        probes.append(audit_probe(learner, raw[i], following, reference, seed=seed))
        if i:
            context = (behavior_context(raw[i], selected, execution, contract, stations) if cached
                else dict(state=np.asarray(raw[i]['s'], np.float32), known=True,
                          reconstructed=False, reason='original_factual_policy_input'))
            contexts.append({key: value for key, value in context.items() if key != 'state'})
            supports.append(support_at(behavior, learner, raw[i], behavior_state=context['state'],
                behavior_state_known=context['known']))
    targets = [value['native_target'] for value in probes]
    baseline = [value['recorded_next_target_q'] for value in probes[:-1]]
    coefficients = [value['c'] if value['status'] != 'unsupported' else None for value in supports]
    eligibility = [value['authorized'] for value in membership['rows']]
    dt = [row['dt'] for row in raw]
    zero = construct_trace(targets, baseline, coefficients, [True]*len(rows), 0., dt=dt)
    diagnostic = construct_trace(targets, baseline, coefficients, [True]*len(rows), 1., dt=dt)
    admitted = construct_trace(targets, baseline, coefficients, eligibility, 1., dt=dt)
    bits_exact = np.array_equal(np.asarray(zero['targets'], np.float32).view(np.uint32),
        np.asarray(targets, np.float32).view(np.uint32))
    cached_exact = all(value['parts_exact'] is True for value in probes) if cached else None
    subset = [value['subset_statistics'] for value in probes[:-1]]
    subset_valid = all(value is not None and value['baseline_uses_native_subset_path']
        and value['duplicate_actions_exact'] and value['reduction'] == 'min'
        and value['head_indices'] == probes[i]['target_subset'] for i, value in enumerate(subset))
    structure = [True]
    for value in supports:
        structure.append(structure[-1] and value['behavior_action_exact']
            and value['target_has_structural_support'])
    # Truth is joined only after the coefficients and both fixed traces exist.
    k = witness_index(rows, reward_identity) if reward_identity is not None else None
    with Path(out).open('x') as output:
        for i, row in enumerate(raw):
            json.dump(dict(sequence=row['sequence'], started_at=row['started_at'], completed_at=row['completed_at'],
                dt=row['dt'], probe=probes[i], support=supports[i-1] if i else None,
                behavior_context=contexts[i-1] if i else None, budget=membership['rows'][i],
                lambda_zero=zero['targets'][i], lambda_one_diagnostic=diagnostic['targets'][i],
                lambda_one_budget=admitted['targets'][i],
                diagnostic_reach_weight=diagnostic['reach_weights'][i],
                budget_reach_weight=admitted['reach_weights'][i]), output, separators=(',', ':'), allow_nan=False)
            output.write('\n')
    reasons = Counter(value['reason'] for value in supports)
    return dict(events=len(rows), zero_dt=sum(t == 0 for t in dt), elapsed=sum(dt),
        lambda_zero_bitwise_exact=bits_exact, cached_v38_parts_exact=cached_exact,
        all_native_targets_equal_actual_learner=all(p['native_target_equals_actual_learner_target'] for p in probes),
        subset_consistent=subset_valid, baseline_links=len(subset),
        full_head_min_would_differ=sum(p['full_head_min_differs'] for p in subset if p is not None),
        full_subset_path_nonexact=sum(not p['same_encoded_full_subset_path_exact'] for p in subset if p is not None),
        max_full_subset_path_ulp=max((p['same_encoded_full_subset_max_ulp'] for p in subset if p is not None), default=0),
        source_budget=membership['summary'], support_reasons=dict(reasons),
        behavior_unknown=sum(s['status'] == 'unsupported' for s in supports),
        behavior_context_reconstructed=sum(c['reconstructed'] for c in contexts),
        mass_underflow=sum(s.get('mass_underflow', False) for s in supports),
        diagnostic_first_constraint_stop=diagnostic['first_stop'], budget_first_stop=admitted['first_stop'],
        first_numerical_zero=next((i for i,w in enumerate(diagnostic['reach_weights']) if w == 0.), None),
        structural_prefix_events=sum(structure), numerical_prefix_events=sum(w > 0. for w in diagnostic['reach_weights']),
        diagnostic_root_target=diagnostic['targets'][0], budget_root_target=admitted['targets'][0],
        root_native_target=targets[0], first_reward_index0=k,
        first_reward_structurally_reached=structure[k] if k is not None else None,
        first_reward_diagnostic_weight=diagnostic['reach_weights'][k] if k is not None else None,
        first_reward_budget_weight=admitted['reach_weights'][k] if k is not None else None,
        real_event_endpoints_only=True, feedback_admitted=0, new_updates=0)


def worker(name, learner, behavior, contract, stations, index, groups, v38, scenario, out, seed):
    out.mkdir(parents=True, exist_ok=False)
    started = time.time(); before, behavior_before, rng = frozen_state(learner), frozen_state(behavior), _rng()
    heldout, admitted = [], []
    for state in read(v38/'runs'/scenario/'native/summary.json')['states']:
        record = read(state['path']); slot = record['slot']
        native_root = Path(state['path']).parent
        identity = read(v38/'analysis'/scenario/(slot.replace('/', '_')+'.json'))['analysis']['semantic_alignment']['first_differences']['r']['identity']
        for arm in ('low', 'high'):
            branch = read(native_root/arm/'summary.json')
            relative = native_root.relative_to(v38/'runs'/scenario/'native')/arm/'events.jsonl'
            rows = jsonlines(v38/'runs'/scenario/'scored'/relative)
            origin = dict(scenario=scenario, prefix_id=index['prefix_id'], kind='heldout', branch_id=slot, arm=arm)
            result = evaluate_chain(learner, behavior, contract, stations, rows, origin, index, cached=True,
                selected=branch['selected'], execution=branch['execution'], reward_identity=identity,
                out=out/(slot.replace('/', '_')+'_'+arm+'.jsonl'), seed=seed)
            result.update(slot=slot, arm=arm); heldout.append(result)
            if len(heldout) == 1:
                write(out/'preflight_gate.json', dict(reused_in_full_run=True,
                    lambda_zero_bitwise_exact=result['lambda_zero_bitwise_exact'],
                    native_exact=result['cached_v38_parts_exact'], subset_consistent=result['subset_consistent'],
                    new_updates=0, new_native_seconds=0))
                if not (result['lambda_zero_bitwise_exact'] and result['cached_v38_parts_exact'] and result['subset_consistent']):
                    raise RuntimeError('First existing-chain construction preflight failed')
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='trace_heldout',model=name,
                completed=len(heldout), total=12, slot=slot, arm=arm)), flush=True)
    for number, group in enumerate(groups):
        result = evaluate_chain(learner, behavior, contract, stations, group['rows'], group['origin'],
            index, cached=False, out=out/('budget_'+str(number)+'.jsonl'), seed=seed)
        result.update(id=group['id']); admitted.append(result)
    proof = unchanged(learner, before, rng); behavior_proof = unchanged(behavior, behavior_before, rng)
    checks = dict(model_frozen=proof['all_pass'], behavior_frozen=behavior_proof['all_pass'],
        all_old_chain_native_targets_exact=all(r['cached_v38_parts_exact'] for r in heldout),
        lambda_zero_all_exact=all(r['lambda_zero_bitwise_exact'] for r in heldout+admitted),
        all_real_native_targets=all(r['all_native_targets_equal_actual_learner'] for r in heldout+admitted),
        all_subset_links_consistent=all(r['subset_consistent'] for r in heldout+admitted),
        all_285_feedback_audited=sum(r['events'] for r in admitted) == 285,
        all_285_physical_sources_authorized=sum(r['source_budget']['authorized_rows'] for r in admitted) == 285,
        all_heldout_feedback_excluded=all(r['source_budget']['authorized_rows'] == 0 for r in heldout),
        all12_suffix_logs=len(heldout) == 12)
    write(out/'summary.json', dict(status='complete' if all(checks.values()) else 'integrity_failed',
        integrity=dict(all_pass=all(checks.values()), checks=checks), model=name, heldout=heldout,
        admitted=admitted, frozen=proof, behavior_frozen=behavior_proof,
        wall=dict(pid=os.getpid(), started=started, finished=time.time()),
        source_slots=285, source_updates=456, new_updates=0, new_feedback=0, new_native_seconds=0))
    return all(checks.values())


def run(args, protocol):
    out=args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    v37=HERE.parent/protocol['source_v37_directory']; v38=HERE.parent/protocol['source_v38_directory']
    source=v37/'runs/collection'/args.scenario
    started=time.time()
    try:
        if not read(v38/'runs'/args.scenario/'summary.json')['integrity']['all_pass']:
            raise RuntimeError('The source native-chain diagnostic is incomplete')
        learners, contract, stations, model_proof=load_frozen(args.scenario,
            HERE.parent/protocol['source_v37r1_directory']/'runs', v37, protocol['diagnostic_seed'])
        behavior,replay,behavior_contract,behavior_stations,_=load_learner(protocol['seed'],arm='fresh')
        freeze(behavior)
        if behavior.total_updates != 50000 or replay.target_received or behavior_contract != contract or not np.array_equal(stations,behavior_stations):
            raise RuntimeError('Original deterministic behavior initializer changed')
        old_pairs=read(source/'pair_truth.json')
        index=build_budget_index(read(source/'dataset.json'),pair_truth=old_pairs,prefix_id=str(source))
        budget=index['summary']
        if not (budget['complete_provenance'] and (budget['total_slots'],budget['ordinary_slots'],budget['paired_slots']) == (285,267,18)):
            raise RuntimeError('The source285-slot provenance is incomplete')
        groups=budget_chains(index)
        write(out/'input_gate.json',dict(model_proof=model_proof,budget=budget,
            ordinary_chains=sum(g['origin']['kind']=='ordinary' for g in groups),
            maximum_ordinary_events=max(len(g['rows']) for g in groups if g['origin']['kind']=='ordinary'),
            maximum_ordinary_seconds=max(sum(r['dt'] for r in g['rows']) for g in groups if g['origin']['kind']=='ordinary'),
            paired_chains=sum(g['origin']['kind']=='paired' for g in groups),
            paired_first_reward_differences=[p['outcomes']['high']['transition']['r']-p['outcomes']['low']['transition']['r'] for p in old_pairs if p['role']=='train'],
            new_feedback=0,new_updates=0,new_native_seconds=0))
        children={}
        sys.stdout.flush();sys.stderr.flush()
        for name, learner in learners.items():
            pid=os.fork()
            if pid == 0:
                log=os.open(out/(name+'.log'),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
                os.dup2(log,1);os.dup2(log,2);os.close(log)
                try:
                    ok=worker(name,learner,behavior,contract,stations,index,groups,v38,args.scenario,
                        out/name,protocol['diagnostic_seed'])
                    sys.stdout.flush();sys.stderr.flush();os._exit(0 if ok else 1)
                except BaseException as error:
                    write(out/name/'failure.json',dict(type=type(error).__name__,message=str(error),
                        new_updates=0,new_feedback=0,new_native_seconds=0))
                    import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
            children[name]=pid
        exitcodes={name:os.waitstatus_to_exitcode(os.waitpid(pid,0)[1]) for name,pid in children.items()}
        summaries={name:read(out/name/'summary.json') for name in children if (out/name/'summary.json').exists()}
        clocks=[s['wall'] for s in summaries.values()]
        overlap=min(c['finished'] for c in clocks)-max(c['started'] for c in clocks) if len(clocks)==3 else 0.
        all_pass=len(summaries)==3 and all(code==0 for code in exitcodes.values()) and all(s['integrity']['all_pass'] for s in summaries.values()) and overlap>0
        write(out/'summary.json',dict(status='complete' if all_pass else 'integrity_failed',
            integrity=dict(all_pass=all_pass), scenario=args.scenario, children=children, exitcodes=exitcodes,
            actual_three_model_overlap_seconds=max(0.,overlap), models=list(summaries),
            wall_seconds=time.time()-started,new_updates=0,new_feedback=0,new_native_seconds=0))
        return 0 if all_pass else 1
    except BaseException as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),
            new_updates=0,new_feedback=0,new_native_seconds=0));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario',required=True,choices=('stable','local_speed_drop'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    sys.exit(run(args,read(HERE/'protocol.json')))
