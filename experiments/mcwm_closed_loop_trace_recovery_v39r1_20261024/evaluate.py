"""Complete the one real failed v39 arm; reuse its completed traces exactly."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from experiments.mcwm_closed_loop_trace_support_v39_20261024.budget import audit_rows
from experiments.mcwm_closed_loop_trace_support_v39_20261024.context import behavior_context
from experiments.mcwm_closed_loop_trace_support_v39_20261024.support import support_at
from experiments.mcwm_closed_loop_trace_support_v39_20261024.trace import construct_trace
from experiments.mcwm_closed_loop_trace_recovery_v39r1_20261024.probe import audit_probe

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
                   selected=None, execution=None, reward_identity=None, out, seed, reused_events=None):
    raw = [row['transition'] if 'transition' in row else row for row in rows]
    membership = audit_rows(raw, origin, index)
    probes, supports, contexts = [], [], []
    if reused_events is not None:
        saved = jsonlines(reused_events)
        if len(saved) != len(rows): raise RuntimeError("Completed-chain row count changed")
        probes = [r["probe"] for r in saved]
        supports = [r["support"] for r in saved[1:]]
        contexts = [r["behavior_context"] for r in saved[1:]]
        if any(a != b["budget"] for a,b in zip(membership["rows"],saved)):
            raise RuntimeError("Completed-chain budget evidence changed")
    else:
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
        and (value['duplicate_actions_exact'] if reused_events is not None else
             value['duplicate_inputs_exact'] and value['repeated_forward_exact'] and value['baseline_slot'] == 0)
        and value['reduction'] == 'min' and value['head_indices'] == probes[i]['target_subset']
        for i, value in enumerate(subset))
    structure = [True]
    for value in supports:
        structure.append(structure[-1] and value['behavior_action_exact']
            and value['target_has_structural_support'])
    # Truth is joined only after the coefficients and both fixed traces exist.
    k = witness_index(rows, reward_identity) if reward_identity is not None else None
    if reused_events is None:
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
    else:
        for i,supplied in enumerate(saved):
            if (supplied['lambda_zero'] != zero['targets'][i]
                or supplied['lambda_one_diagnostic'] != diagnostic['targets'][i]
                or supplied['lambda_one_budget'] != admitted['targets'][i]
                or supplied['diagnostic_reach_weight'] != diagnostic['reach_weights'][i]
                or supplied['budget_reach_weight'] != admitted['reach_weights'][i]):
                raise RuntimeError('Completed-chain trace reconstruction changed')
        write(Path(out).with_suffix('.reuse.json'),dict(source=str(reused_events),rows=len(saved),
            original_trace_values_exact=True,new_model_evaluations=0))
    reasons = Counter(value['reason'] for value in supports)
    return dict(duplicate_slot_differences=sum(not p['duplicate_actions_exact'] for p in subset if p),
        maximum_duplicate_slot_ulp=max((p.get('duplicate_outputs_max_ulp',0) for p in subset if p),default=0),
        duplicate_slot_witness=next((p for p in subset if p and not p['duplicate_actions_exact']),None),
        reused_events=str(reused_events) if reused_events is not None else None,events=len(rows), zero_dt=sum(t == 0 for t in dt), elapsed=sum(dt),
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


