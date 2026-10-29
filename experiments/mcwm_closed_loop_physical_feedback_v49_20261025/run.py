"""Read only main-flow timing and behavior; no passenger outcomes or models."""
from pathlib import Path
import json
import math
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, separators=(',', ':'), allow_nan=False)
        stream.write('\n')


def projected_issues(path):
    rows = []
    with path.open() as source:
        for line in source:
            row = json.loads(line)
            clock = row['started_at']
            if clock < 9000.:
                continue
            if clock > 9600.:
                break
            # Outcome-bearing ledgers and state tensors are deliberately not forwarded.
            rows.append({k: row[k] for k in ('sequence', 'started_at', 'line_id', 'episode_id')})
    return rows


def projected_behavior(path, first, last):
    rows = []
    with path.open() as source:
        for line in source:
            row = json.loads(line)
            if row['sequence'] < first: continue
            if row['sequence'] > last: break
            rows.append(row)
    return rows


def main():
    from experiments.mcwm_closed_loop_physical_feedback_v49_20261025.plan import build_plan, precheck_issues
    from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
    from experiments.mcwm_closed_loop_trace_design_v40_20261024.behavior import execution_measure, ratio_from_measures
    import numpy as np
    import torch
    torch.set_num_threads(1)
    started, protocol = time.monotonic(), read(HERE/'protocol.json')
    plan = build_plan()
    write(HERE/'budget_plan.json', plan)
    contract = read(HERE.parent/'mcwm_closed_loop_formal_20261020/inputs/history_metadata.json')['contract']
    transform = ExecutionAction(contract)
    cases = {}
    for case in protocol['scenarios']:
        root = HERE.parent/protocol['source_version']/'runs'/case
        pilot = read(root/'pilot_summary.json')
        if not (pilot['status'] == 'complete' and pilot['integrity']['all_pass']
                and pilot['native']['integrity']['checks']['complete_prefix_trace']):
            raise RuntimeError('Source main trace is not a completed, fully recorded real native day')
        issues = projected_issues(root/'prefix_trace.jsonl')
        if len(issues) < 2: raise RuntimeError('No main-flow design window')
        behavior = projected_behavior(root/'behavior_prefix.jsonl', issues[0]['sequence'], issues[-1]['sequence'])
        result = precheck_issues(issues, behavior)
        lookup = {row['sequence']: row for row in behavior}
        active = issues[:270]
        exact, supported, atoms, continuous, errors = 0, 0, 0, 0, []
        for row in active:
            context = lookup[row['sequence']]['behavior_s']
            state, proposal, action = context['execution_input'], context['proposal'], context['executed_action']
            actual = transform.numpy(state, proposal)
            exact += bool(np.array_equal(actual, np.asarray(action, np.float32)))
            measure = execution_measure(transform, state, action, context['parameters'])
            self_ratio = ratio_from_measures(measure, measure)
            good = self_ratio['status'] == 'supported' and self_ratio['c'] == 1.
            supported += good
            atoms += measure['measure_kind'] == 'atom'
            continuous += measure['measure_kind'] == 'continuous'
            if not good: errors.append(dict(sequence=row['sequence'], measure=measure['measure_kind'], status=self_ratio['status']))
        result['recorded_behavior_execution_audit'] = dict(rows=len(active), native_F_exact=exact,
            supported_self_ratios=supported, atom_actions=atoms, continuous_actions=continuous,
            unresolved=errors, full_snapshot_weights_rebuilt=False,
            scope='recorded actual mu parameters versus themselves; not learned pi drift or new native evidence')
        if exact != len(active) or supported != len(active):
            raise RuntimeError('Recorded full stochastic behavior or native F self-audit failed')
        # Extra timing records are not admitted, rewarded, scored or joined into training.
        roots = {line: next((i for i,r in enumerate(issues[:270]) if r['line_id'] == line
                 and 9000. <= r['started_at'] < 9300.), None) for line in protocol['lines']}
        required = {}
        for line, index in roots.items():
            end = None if index is None else next((i for i,r in enumerate(issues)
                if i > index and r['started_at'] >= issues[index]['started_at']+300.), None)
            required[line] = None if end is None else dict(response_slots_from_fixed_start=end,
                clock=issues[end]['started_at'], extra_slots_above270=max(0,end-270))
        result['timing_only_minimum_slots_for_all_roots'] = dict(by_line=required,
            response_slots=max((v['response_slots_from_fixed_start'] for v in required.values() if v),default=None),
            root_selection_fixed=True, extra_rows_not_training_authorized=True)
        write(HERE/(case+'_detail.json'), dict(projected_issues=issues, result=result))
        cases[case] = result
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(case=case, capacity=result['all_pass'])),flush=True)
    value = dict(status='complete', priority_requested='high',
        integrity=dict(all_pass=True, source_days_complete=True, native_F_and_self_RN_exact=True,
            input_projection_has_no_passenger_outcomes=True, no_heldout_or_cross_stream_join=True),
        plan=plan,cases=cases,
        source_contract=dict(initializer_seed=6202,initializer_updates=50000,feedback_slots=285,
            optimizer_updates=456,target_horizon_seconds=24000,F_unchanged=True,trust_unchanged=True),
        diagnostic_budget=dict(new_training_updates=0,new_feedback_slots=0,new_native_prefixes=0,
            new_native_branches=0,new_native_seconds=0,new_target_draws=0,checkpoints_loaded=0),
        elapsed_seconds=time.monotonic()-started)
    encoded=json.dumps(value,separators=(',',':'),allow_nan=False)+'\n'
    if len(encoded.encode())>protocol['summary_max_bytes']:
        raise RuntimeError('Reduce resident summary before any download')
    with (HERE/'summary.json').open('x') as stream: stream.write(encoded)
    print(json.dumps(dict(status='complete',bytes=len(encoded.encode()),seconds=value['elapsed_seconds'])),flush=True)


if __name__=='__main__': main()
