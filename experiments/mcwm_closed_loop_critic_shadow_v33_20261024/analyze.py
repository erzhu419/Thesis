"""Resident paired analysis, gated by exact zero replay and actual parallelism."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


METRICS = ['actor_holding_delta', 'target_delta', 'value_part_delta', 'sac_part_delta',
           'advantage7_5_delta', 'advantage15_delta', 'direct_advantage7_5', 'direct_advantage15']
GROUP_COLUMNS = ['n', 'nonalias', 'actor_holding_changed', 'fixed_ranking_changed', 'metrics']
BANK_COLUMNS = ['line_id', 'window', 'sequence', 'holding_seconds', 'actor_holding',
                'advantage7_5', 'advantage15', 'masked_memory_advantage']


def readlines(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def interval_overlap(left, right):
    i=j=0; total=0.
    while i<len(left) and j<len(right):
        total += max(0., min(left[i][1],right[j][1])-max(left[i][0],right[j][0]))
        if left[i][1]<right[j][1]: i+=1
        else: j+=1
    return total


def stats(values):
    if not values:
        return None
    return [float(format(v,'.7g')) for v in (sum(values)/len(values),
        sum(abs(x) for x in values)/len(values),max(abs(x) for x in values))]


def aggregate(pairs, *, compact=False):
    result=dict(n=len(pairs), nonalias=[0,0], actor_holding_changed=0, fixed_ranking_changed=0)
    metrics=[[] for _ in METRICS]
    for left,right in pairs:
        result['actor_holding_changed'] += left['actor_holding_seconds']!=right['actor_holding_seconds']
        # A changed chosen index is meaningful only if it changes actual hold.
        result['fixed_ranking_changed'] += left['holding_seconds'][left['best_fixed_action']] != \
            right['holding_seconds'][right['best_fixed_action']]
        for i,key in enumerate(('actor_holding_seconds','target','value_contribution','sac_contribution')):
            metrics[i].append(right[key]-left[key])
        for candidate in (1,2):
            if left['executed_actions'][candidate] != left['executed_actions'][0]:
                result['nonalias'][candidate-1] += 1
                metrics[candidate+3].append(right['advantage_min'][candidate]-left['advantage_min'][candidate])
                metrics[candidate+5].append(right['advantage_min'][candidate]-right['zero_context_advantage_min'][candidate])
    result['metrics']=[stats(v) for v in (metrics[:4] if compact else metrics)]
    return result


def packed_group(pairs, *, compact=False):
    value=aggregate(pairs,compact=compact)
    return [value[column] for column in GROUP_COLUMNS]


def analyze(root):
    p=json.loads((root/'protocol.json').read_text())
    result=dict(status='complete', integrity=dict(all_pass=True), cases=[],
        columns=dict(metric_order=METRICS, stats=['signed_mean','mean_absolute','max_absolute'],
            metric_direction='memory minus zero_memory; direct is memory weights with m minus same weights with zero m',
            candidates=['proposal0','proposal7.5','proposal15','factual_executed'],
            by_line_metric_order=METRICS[:4], group_columns=GROUP_COLUMNS, bank_columns=BANK_COLUMNS),
        budget=p['compute_budget'], limitations=['Development states only, no saved tail contexts or 300s truth.',
            'No real controller was deployed; holding differences are shadow proposals after the unchanged F.'])
    for scenario in p['scenarios']:
        cell=root/'runs'/'seed6202'/scenario
        parent=json.loads((cell/'summary.json').read_text())
        summaries={a:json.loads((cell/a/'summary.json').read_text()) for a in p['arms']}
        packets={a:readlines(cell/a/'packets.jsonl') for a in p['arms']}
        left,right=packets['zero_memory'],packets['memory']
        checks=dict(parent_pass=parent['integrity']['all_pass'],
            workers_pass=all(v['integrity']['all_pass'] for v in summaries.values()),
            exactly57_packets=len(left)==len(right)==57,
            common_receipts=[r['receipt'] for r in left]==[r['receipt'] for r in right],
            common_sampling=[r['sampling'] for r in left]==[r['sampling'] for r in right],
            common_rng=[r['rng'] for r in left]==[r['rng'] for r in right],
            optimizers_retained=all(r['optimizer_objects_retained'] for v in packets.values() for r in v),
            no_extra_updates=all(v['budget']['shadow_updates']==456 for v in summaries.values()),
            zero57_exact=summaries['zero_memory']['integrity']['checks']['zero_reference_updates_exact'],
            zero_full_actor_trace=summaries['zero_memory']['integrity']['checks']['zero_full_actor_trace_exact'])
        overlap=interval_overlap([(r['update_started_wall'],r['update_finished_wall']) for r in left],
                                 [(r['update_started_wall'],r['update_finished_wall']) for r in right])
        checks['optimizer_intervals_overlap']=overlap>0
        # No scientific comparison is emitted before the zero replay gates.
        if not all(checks.values()):
            raise RuntimeError(f'{scenario} failed gates: {[k for k,v in checks.items() if not v]}')
        scores={}
        for arm in p['arms']:
            scores[arm]={}
            for chunk in readlines(cell/arm/'scores.jsonl'):
                for row in chunk['rows']:
                    key=(chunk['packet'],chunk['phase'],row['sequence'])
                    if key in scores[arm]: raise RuntimeError('Duplicate paired probe')
                    scores[arm][key]=row
        if scores['zero_memory'].keys()!=scores['memory'].keys():
            raise RuntimeError('The two learners did not score the same saved contexts')
        phase_pairs=defaultdict(list); line_pairs=defaultdict(list); windows=defaultdict(list); last=[]
        for key,l in scores['zero_memory'].items():
            r=scores['memory'][key]
            for field in ('line_id','issue','context_observed_at','holding_seconds','executed_actions','reward_scaled'):
                if l[field]!=r[field]: raise RuntimeError('Common factual/candidate input differs: '+field)
            phase_pairs[key[1]].append((l,r))
            if key[1]=='packet_after':
                line_pairs[l['line_id']].append((l,r))
                if l['line_id'] in p['focus_lines']:
                    windows[l['line_id']+'/'+l['window']].append((l,r))
            if key[0]==57 and key[1]=='bank_after':
                last.append(dict(line_id=l['line_id'],window=l['window'],sequence=l['sequence'],
                    holding_seconds=l['holding_seconds'][:3],
                    actor_holding=[l['actor_holding_seconds'],r['actor_holding_seconds']],
                    advantage7_5=[l['advantage_min'][1],r['advantage_min'][1]],
                    advantage15=[l['advantage_min'][2],r['advantage_min'][2]],
                    masked_memory_advantage=[r['zero_context_advantage_min'][1],r['zero_context_advantage_min'][2]]))
        checks['common_candidates_and_rewards']=True
        checks['no_tail_bank']=all(row['window']!='tail' for row in last)
        case=dict(scenario=scenario,integrity=dict(all_pass=all(checks.values()),checks=checks),
            worker_parallel_overlap_seconds=parent['training_overlap_wall_seconds'],
            optimizer_overlap_seconds=overlap,zero_actor_issues_checked=summaries['zero_memory']['zero_actor_issues_checked'],
            budget=parent['budget'],by_phase={k:aggregate(v) for k,v in phase_pairs.items()},
            by_line={k:packed_group(v,compact=True) for k,v in sorted(line_pairs.items())},
            focus_line_windows={k:packed_group(v) for k,v in sorted(windows.items())},
            final_seen_bank=[[row[column] for column in BANK_COLUMNS] for row in last])
        result['cases'].append(case)
    result['integrity']['all_pass']=all(c['integrity']['all_pass'] for c in result['cases'])
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    try:
        result=analyze(args.root)
        text=json.dumps(result,separators=(',',':'),allow_nan=False)
        if len(text.encode())+1>20000:raise RuntimeError('Shadow summary exceeds20000byte transfer bound')
        with args.out.open('x') as handle: handle.write(text+'\n')
    except Exception as error:
        with (args.root/'shadow_summary_failure.json').open('x') as handle:
            json.dump(dict(type=type(error).__name__,message=str(error)),handle)
        raise
