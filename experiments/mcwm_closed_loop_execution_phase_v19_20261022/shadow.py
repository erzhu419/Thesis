"""Read existing main traces and compare execution maps on factual states only."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction

OBSERVABLES = ('forward_headway_seconds', 'backward_headway_seconds',
    'eligible_stranded_passengers', 'onboard_passengers', 'bus_capacity',
    'eligible_arrival_rate_per_second', 'base_dwell_seconds',
    'forward_bus_present', 'backward_bus_present')
STATISTICS = ('n', 'reserve_eligible', 'reserve_positive', 'reserve_changed',
    'floor_condition', 'floor_changed', 'holding_different', 'holding_difference_ge1s',
    'absolute_difference_mean', 'absolute_difference_p50', 'absolute_difference_p90',
    'absolute_difference_max', 'native_holding_mean', 'tail_holding_mean',
    'recorded_holding_mean', 'controlled_sequences', 'recorded_native_mismatch')
EPS = 1.e-4


@torch.no_grad()
def mapping_diagnostics(states, proposals, contract):
    """Use the original F; explain which of its two adjustments changed holding."""
    transform = ExecutionAction(contract)
    state = torch.as_tensor(np.asarray(states, dtype=np.float32))
    proposal = torch.as_tensor(np.asarray(proposals, dtype=np.float32).reshape(-1, 1))
    native_action = transform(state, proposal)
    native = transform.hold_seconds(native_action).reshape(-1)
    parent = transform.hold_seconds(proposal).reshape(-1)
    tail = .125*parent
    current = torch.round(state[:, 2]*15).long().clamp(0, 15)
    event = state[:, 5:].reshape(-1, 16, 14)[torch.arange(len(state)), current]
    forward, backward = event[:, 11] > .5, event[:, 12] > .5
    forward_error = (event[:, 3]-1.)*event[:, 5]*transform.headway_scale
    backward_error = (event[:, 4]-1.)*event[:, 6]*transform.headway_scale
    reserve = (.5*(backward_error*backward-forward_error*forward)).clamp(0., transform.max_hold)
    reserve_eligible = (state[:, 0] < .75) & (parent > 0.)
    pre_floor = torch.where(reserve_eligible, torch.minimum(parent, torch.maximum(tail, reserve)), tail)
    floor = ((state[:, 0] >= .15) & (state[:, 0] < .75) & forward & ~backward
        & (event[:, 9] > 0.) & (event[:, 8] <= .25))
    return {key: value.cpu().numpy() for key, value in dict(native_action=native_action.reshape(-1),
        native=native, tail=tail, reserve=reserve, reserve_eligible=reserve_eligible,
        reserve_positive=reserve > 0., reserve_changed=abs(pre_floor-tail) > EPS,
        floor_condition=floor, floor_changed=abs(native-pre_floor) > EPS,
        difference=abs(native-tail)).items()}


def summarize(rows):
    """Rows contain native/tail/recorded holding, five conditions, override and mismatch."""
    if not rows:
        return [0]*len(STATISTICS)
    data = np.asarray(rows, dtype=float)
    difference = abs(data[:, 0]-data[:, 1])
    return [len(rows), *data[:, 3:8].sum(axis=0).astype(int).tolist(),
        int((difference > EPS).sum()), int((difference >= 1.).sum()),
        float(difference.mean()), float(np.quantile(difference, .5)),
        float(np.quantile(difference, .9)), float(difference.max()),
        *data[:, :3].mean(axis=0).tolist(), int(data[:, 8].sum()), int(data[:, 9].sum())]


def phase(progress):
    return 'pre_floor' if progress < .15 else 'reserve_floor' if progress < .75 else 'tail'


def window_name(at, protocol):
    for window in protocol['training_windows']+protocol['heldout_windows']:
        if window['start'] <= at < window['end']:
            return window['name']
    return 'outside_windows'


def compact(value):
    if isinstance(value, float):
        return float(format(value, '.8g'))
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [compact(item) for item in value]
    return value


def compact_export(value):
    """Encode repeated categorical keys/field names; retain every scientific value."""
    result = dict(value)
    day_keys = [[row['seed'], row['scenario']] for row in value['days']]
    line_keys = list(dict.fromkeys(row['key'][2] for row in value['groups']))
    window_keys = list(dict.fromkeys(row['key'][3] for row in value['groups']))
    phase_keys = ['pre_floor', 'reserve_floor', 'tail']
    role_keys = ['train', 'heldout']
    anchor_fields = ['sequence', 'at', 'phase', 'actor_updates', 'observables',
        'source_features', 'canonical', 'mapping', 'source_exact', 'neighborhood']
    day_fields = ['trace_issues', 'selected_anchors', 'matched_anchors']
    budget_fields = list(value['days'][0]['original_budget'])
    result['schema'] = dict(value['schema'], compact_layout=dict(
        day_keys=day_keys, line_keys=line_keys, window_keys=window_keys,
        phase_keys=phase_keys, role_keys=role_keys, anchor_fields=anchor_fields,
        day_fields=day_fields, budget_fields=budget_fields,
        groups='[day_index,line_index,window_index,phase_index,statistics]',
        anchors='[day_index,line_index,role_index,window_index,...anchor_fields]; missing rows remain explicit dictionaries',
        coverage='[line_index,role_index,window_index,n,ranges]',
        days='[...day_fields,budget_values]; day_keys supplies seed/scenario'))
    def day_index(key):
        return day_keys.index(key[:2])
    result['groups'] = [[day_index(row['key']), line_keys.index(row['key'][2]),
        window_keys.index(row['key'][3]), phase_keys.index(row['key'][4]), row['statistics']]
        for row in value['groups']]
    anchors = []
    for row in value['anchors']:
        if 'status' in row:
            anchors.append(row)
            continue
        values = [row[field] for field in anchor_fields]
        values[2] = phase_keys.index(values[2])
        anchors.append([day_index(row['key']), line_keys.index(row['key'][2]),
            role_keys.index(row['key'][3]), window_keys.index(row['key'][4]), *values])
    result['anchors'] = anchors
    result['coverage'] = [[line_keys.index(row['key'][0]), role_keys.index(row['key'][1]),
        window_keys.index(row['key'][2]), row['n'], row['ranges']] for row in value['coverage']]
    result['days'] = [[*[row[field] for field in day_fields],
        [row['original_budget'][field] for field in budget_fields]] for row in value['days']]
    def integral_numbers(item):
        if isinstance(item, float) and item.is_integer():
            return int(item)
        if isinstance(item, dict):
            return {key: integral_numbers(x) for key, x in item.items()}
        if isinstance(item, list):
            return [integral_numbers(x) for x in item]
        return item
    return integral_numbers(result)


def decode_summary(value):
    """Restore the readable structure for analysis and exact export tests."""
    result = dict(value)
    schema = dict(value['schema'])
    layout = schema.pop('compact_layout')
    result['schema'] = schema
    days, lines, windows = (layout[key] for key in ('day_keys', 'line_keys', 'window_keys'))
    phases, roles = (layout[key] for key in ('phase_keys', 'role_keys'))
    result['groups'] = [dict(key=days[row[0]]+[lines[row[1]], windows[row[2]], phases[row[3]]],
        statistics=row[4]) for row in value['groups']]
    anchors = []
    for row in value['anchors']:
        if isinstance(row, dict):
            anchors.append(row)
            continue
        item = dict(zip(layout['anchor_fields'], row[4:]))
        item['phase'] = phases[item['phase']]
        item['key'] = days[row[0]]+[lines[row[1]], roles[row[2]], windows[row[3]]]
        anchors.append(item)
    result['anchors'] = anchors
    result['coverage'] = [dict(key=[lines[row[0]], roles[row[1]], windows[row[2]]], n=row[3], ranges=row[4])
        for row in value['coverage']]
    result['days'] = [dict(seed=day[0], scenario=day[1],
        **dict(zip(layout['day_fields'], row[:-1])), original_budget=dict(zip(layout['budget_fields'], row[-1])))
        for day, row in zip(days, value['days'])]
    return result


def analyze(source):
    from experiments.mcwm_closed_loop_v1.stream import EI
    source = Path(source)
    protocol = json.loads((source/'protocol.json').read_text())
    metadata = json.loads((source/protocol['reference_inputs']/'history_metadata.json').read_text())
    contract = metadata['contract']
    transform = ExecutionAction(contract)
    groups, ranges, anchor_rows, days, all_rows = defaultdict(list), defaultdict(list), [], [], []
    checks = dict(source_days_pass=True, anchors_found_exact=True, noncontrolled_actions_exact=True,
        tail_maps_equal=True, clock_unchanged=True)
    mismatch_count = 0
    mismatch_max = 0.
    for seed in protocol['training_seeds']:
        for scene in protocol['scenarios']:
            directory = source/'runs'/scene/f'seed{seed}'
            day = json.loads((directory/'summary.json').read_text())
            journal = json.loads((directory/'acquisitions.json').read_text())['records']
            checks['source_days_pass'] &= day.get('status') == 'complete' and day['integrity']['all_pass']
            by_sequence = {row['selected']['sequence']: row for row in journal}
            controlled = {row['selected']['sequence'] for row in journal if row['role'] == 'train'}
            nearby = {row['id']: [] for row in journal}
            anchor_values = {}
            row_count = 0

            def consume(batch):
                nonlocal mismatch_count, mismatch_max, row_count
                if not batch:
                    return
                states = np.asarray([row['s'] for row in batch], np.float32)
                before = states.copy()
                proposal = np.asarray([row['proposed_a'] for row in batch], np.float32)
                result = mapping_diagnostics(states, proposal, contract)
                checks['clock_unchanged'] &= np.array_equal(states, before)
                recorded = transform.hold_seconds(torch.as_tensor(np.asarray([row['a'] for row in batch], np.float32))).numpy().reshape(-1)
                for index, row in enumerate(batch):
                    row_count += 1
                    at, sequence = float(row['started_at']), row['sequence']
                    known = sequence in controlled
                    discrepancy = abs(float(result['native'][index])-float(recorded[index]))
                    mismatch = discrepancy > EPS
                    if mismatch and not known:
                        mismatch_count += 1
                        mismatch_max = max(mismatch_max, discrepancy)
                        checks['noncontrolled_actions_exact'] = False
                    if states[index, 0] >= .75 and result['difference'][index] > EPS:
                        checks['tail_maps_equal'] = False
                    values = [float(result[key][index]) for key in ('native', 'tail')]+[float(recorded[index])]
                    values += [int(result[key][index]) for key in ('reserve_eligible', 'reserve_positive',
                        'reserve_changed', 'floor_condition', 'floor_changed')]+[int(known), int(mismatch)]
                    key = (seed, scene, row['line_id'], window_name(at, protocol), phase(states[index, 0]))
                    groups[key].append(values)
                    all_rows.append(values)
                    for anchor in journal:
                        started = anchor['selected']['started_at']
                        if started <= at < started+300.:
                            nearby[anchor['id']].append(values)
                    if sequence in by_sequence:
                        anchor = by_sequence[sequence]
                        selected = anchor['selected']
                        exact = (at == selected['started_at'] and all(row[name] == selected[name]
                            for name in ('s', 'raw_s', 'event')) and row['a'] == anchor['nominal_a'])
                        checks['anchors_found_exact'] &= exact
                        observed = [float(row['event'][EI[name]]) for name in OBSERVABLES]
                        ranges[(row['line_id'], anchor['role'], anchor['window_name'])].append(observed)
                        actor = anchor['prefix']['counters']['policy']
                        current = int(round(states[index, 2]*15))
                        slot = states[index, 5:].reshape(16, 14)[current]
                        anchor_values[anchor['id']] = dict(key=[seed, scene, row['line_id'], anchor['role'], anchor['window_name']],
                            sequence=sequence, at=at, phase=phase(states[index, 0]),
                            actor_updates=[actor, actor-anchor['prefix']['initial_updates']],
                            observables=observed, source_features=anchor['features'],
                            canonical=[float(states[index, 0]), float(slot[3]), float(slot[4]), float(slot[8]), float(slot[9])],
                            mapping=[float(result[key][index]) for key in ('native', 'tail', 'reserve')]+[float(recorded[index])],
                            source_exact=exact)
            batch = []
            with (directory/'prefix_trace.jsonl').open() as trace:
                for line in trace:
                    batch.append(json.loads(line))
                    if len(batch) == 512:
                        consume(batch)
                        batch = []
                consume(batch)
            checks['anchors_found_exact'] &= len(anchor_values) == len(journal)
            for anchor in journal:
                item = anchor_values.get(anchor['id'])
                if item is None:
                    anchor_rows.append(dict(key=[seed, scene, anchor['line_id'], anchor['role'], anchor['window_name']], status='missing_trace_anchor'))
                    continue
                item.update(neighborhood=summarize(nearby[anchor['id']]))
                anchor_rows.append(item)
            days.append(dict(seed=seed, scenario=scene, trace_issues=row_count,
                selected_anchors=len(journal), matched_anchors=len(anchor_values),
                original_budget={key:day['budget'][key] for key in ('target_labels', 'packets', 'ordinary_labels',
                    'paired_action_labels', 'training_pairs', 'native_seconds', 'policy_updates', 'trust_updates', 'world_updates')}))
    fixed = [row for row in anchor_rows if row['key'][:2] == [6201, 'stable']
        and row['key'][3:] == ['train', 'late']]
    checks['fixed_three_anchors_present'] = {row['key'][2] for row in fixed} == set(protocol['target_lines'])
    coverage = [dict(key=list(key), n=len(values), ranges=np.stack([np.min(values, axis=0), np.max(values, axis=0)], axis=1).tolist())
        for key, values in sorted(ranges.items())]
    return compact(dict(status='complete' if all(checks.values()) else 'integrity_failed',
        integrity=dict(all_pass=all(checks.values()), checks=checks,
            noncontrolled_mismatches=mismatch_count, maximum_noncontrolled_holding_error=mismatch_max),
        schema=dict(group_key_order=['seed', 'scenario', 'line_id', 'window', 'phase'],
            anchor_key_order=['seed', 'scenario', 'line_id', 'role', 'window'], statistics_order=STATISTICS,
            observable_order=OBSERVABLES, canonical_order=['horizon_progress', 'forward_ratio', 'backward_ratio', 'load_ratio', 'demand_ratio'],
            actor_update_order=['total_policy_updates', 'target_policy_updates'],
            mapping_order=['native_holding', 'tail_holding', 'reserve_holding', 'recorded_holding'],
            coverage_key_order=['line_id', 'role', 'window'], coverage_range_order=['minimum', 'maximum']),
        days=days, total=summarize(all_rows), groups=[dict(key=list(key), statistics=summarize(rows)) for key, rows in sorted(groups.items())],
        anchors=anchor_rows, fixed_anchor_keys=[row['key'] for row in fixed], coverage=coverage,
        budget=dict(additional_target_labels=0, additional_native_seconds=0, additional_training_updates=0),
        interpretation='Same factual main-trace states and actor proposals; native F is recomputed, not read from controlled recorded actions. Neighborhoods include [issue,issue+300) factual controls across all lines. No counterfactual300-second effect or future service identification. Reserve/floor condition counts are separate from actual holding changes.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    value = compact_export(analyze(args.source))
    payload = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'
    if len(payload.encode()) > 80000:
        raise RuntimeError('Reduce resident shadow summary before collection: '+str(len(payload.encode())))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload)
    print(json.dumps(dict(status=value['status'], integrity=value['integrity'], bytes=len(payload.encode()),
        trace_issues=sum(day[0] for day in value['days']), anchors=len(value['anchors']))))


if __name__ == '__main__':
    main()
