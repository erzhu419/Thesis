"""Frozen target precheck on resident real chains; no model, learner or simulator."""
from pathlib import Path
import json
import math
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as handle:
        json.dump(value,handle,separators=(',',':'),allow_nan=False)
        handle.write('\n')


def stats(values):
    import numpy as np
    x=np.asarray(values,dtype=float)
    return [float(x.mean()),float(x.std(ddof=1)) if len(x)>1 else 0.,float(x.min()),float(x.max())]


def values(pair):
    p=pair['components']
    return dict(target=pair['trace_difference'],native=pair['native_difference'],reward=p['reward'],
        bootstrap_minus_Q=pair['bootstrap_minus_matching_Q'],cutoff_bootstrap=pair['terminal_bootstrap_difference'],
        intermediate_bootstrap_minus_Q=pair['bootstrap_minus_matching_Q']-pair['terminal_bootstrap_difference'],
        rounding=p['native_rounding']+p['recurrence_rounding'],
        common_weight_reward=pair['reward_value_vs_weight']['common_weight_value_difference'],
        changed_weight_reward=pair['reward_value_vs_weight']['common_value_weight_difference'])


def long_timing(protocol):
    result={}
    for scenario in protocol['scenarios']:
        root=HERE.parent/protocol['long_timing_source']
        saved=read(root/(scenario+'_detail.json'))
        issues=saved['projected_issues'][:271]
        entries=[]
        for line,source in saved['result']['root_coverage'].items():
            index=source['root_index']
            times=[r['started_at'] for r in issues[index:270]]
            weights=[math.exp(-(t-times[0])/300.) for t in times]
            end_index=source['first_real_300_endpoint_index']
            before=None if end_index is None else issues[end_index-1]['started_at']
            entries.append(dict(line=line,root_clock=times[0],end_clock=issues[-1]['started_at'],
                physical_span=issues[-1]['started_at']-times[0],actual_300_endpoint=source['first_real_300_endpoint_clock'],
                old_c1_weight_at300=source['ideal_c1_mixed_weight_to_300'],
                physical_c1_weight_at300=None if before is None else math.exp(-(before-times[0])/300.),
                old_c1_cutoff_weight=.9**(len(times)-1),physical_c1_cutoff_weight=weights[-1],
                c1_sum_reach=math.fsum(weights),
                c1_uniform_Q_reference_gain=.9-.1*math.fsum(weights[1:]),
                learned_pi_c_available=False,real_reward_target_available=False))
        result[scenario]=dict(capacity_pass=saved['result']['all_pass'],rows=entries,
            scope='actual main issue clocks and c1 upper bound only; no missing TD/Q/memory imputed')
    return result


def main():
    import numpy as np
    from experiments.mcwm_closed_loop_trace_target_v44_20261024.probe import _source
    from experiments.mcwm_closed_loop_trace_target_v44_20261024.trace import decompose_trace
    from experiments.mcwm_closed_loop_trace_target_v44_20261024.summary import paired_decomposition,compact
    from experiments.mcwm_closed_loop_physical_credit_v50_20261025.credit import (
        physical_trace,probe_reward_band,probe_bootstrap_shift,probe_uniform_q_shift,
    )
    p,started=read(HERE/'protocol.json'),time.monotonic()
    detail=HERE/'detail';detail.mkdir(exist_ok=False)
    cases={};count=dict(native_lambda0_rows=0,mean8_lambda0_rows=0,old_full_trace_exact=0)
    functions=dict(event=decompose_trace,physical=physical_trace)
    for case in p['scenarios']:
        data=HERE.parent/p['native_source']/'runs'/case
        _,pilot,updates,snapshot,ordinary,all_rows,chains,credit=_source(data)
        indexed={(c['line_id'],c['arm']):c for c in chains}
        native=read(HERE.parent/p['native_target_arrays']/'runs'/case/'probe_detail.json')
        if not native['integrity']['all_pass']: raise RuntimeError('Native source target audit failed')
        checkpoints={}
        for checkpoint in p['checkpoints']:
            original=native['checkpoints'][checkpoint]
            bank={(c['line_id'],c['arm']):c for c in original['seeds']['336202']['chains']}
            for chain in bank.values():
                z=physical_trace(chain['rows'],chain['c'],0.)
                if not z['lambda_zero_native_exact']: raise RuntimeError('Original native lambda0 changed')
                count['native_lambda0_rows']+=45
            for row in original['ordinary_rows_first_seed']:
                if not physical_trace([row],[],0.)['lambda_zero_native_exact']: raise RuntimeError('Ordinary lambda0 changed')
                count['native_lambda0_rows']+=1
            meanroot=HERE.parent/p['mean8_arrays']/'runs'/case/checkpoint
            for scored in read(meanroot/'detail/ordinary_first_block.json'):
                if not physical_trace([scored['m8']],[],0.)['lambda_zero_native_exact']: raise RuntimeError('M8 ordinary lambda0 changed')
                count['mean8_lambda0_rows']+=1
            lines={}
            for line in p['lines']:
                blocks=[];structure={}
                for block in range(p['blocks']):
                    old=read(meanroot/'detail'/f'{line}_block{block:02d}.json')
                    traces={name:{} for name in functions};probes={name:{} for name in functions}
                    for arm in ('low','high'):
                        rows=old['raw_scores'][arm]['mean8'];cached=bank[(line,arm)]
                        real=indexed[(line,arm)]['records'];c=cached['c']
                        if len(rows)!=45 or any(a['sequence']!=b['sequence'] or a['started_at']!=b['started_at']
                            or a['completed_at']!=b['completed_at'] or a['dt']!=b['dt']
                            or a['reward_scaled']!=d['reward_scaled'] for a,b,d in zip(rows,real,cached['rows'])):
                            raise RuntimeError('M8 arrays differ from original real chain')
                        logged=credit[24 if checkpoint=='actual_post24' else 456][(line,arm)]
                        if c!=[r['c'] for r in logged['rows']]: raise RuntimeError('Original F ratios changed')
                        if not physical_trace(rows,c,0.)['lambda_zero_native_exact']: raise RuntimeError('M8 lambda0 changed')
                        count['mean8_lambda0_rows']+=45
                        for name,fn in functions.items():
                            t=fn(rows,c,1.);traces[name][arm]=t
                            if abs(t['root']['reconstruction_residual'])>1e-8:
                                raise RuntimeError('Root contribution accounting does not close')
                            probes[name][arm]=dict(reward=probe_reward_band(rows,c,fn),
                                tail=probe_bootstrap_shift(rows,c,fn),qshift=probe_uniform_q_shift(rows,c,fn))
                        if traces['event'][arm]['root']['trace_target']!=old['decomposition'][arm]['mean8']['trace_target']:
                            raise RuntimeError('Old complete trace changed')
                        count['old_full_trace_exact']+=1
                        if block==0:
                            structure[arm]=dict(span=rows[-1]['completed_at']-rows[0]['started_at'],
                                zero_duration=sum(r['dt']==0 for r in rows),
                                c_product=math.prod(c),
                                physical_clock_retention=math.exp(-math.fsum(r['dt'] for r in rows[:-1])/300.),
                                event_cutoff_weight=traces['event'][arm]['reach_weights'][-1],
                                physical_cutoff_weight=traces['physical'][arm]['reach_weights'][-1],
                                event_sum_reach=math.fsum(traces['event'][arm]['reach_weights']),
                                physical_sum_reach=traces['physical'][arm]['gain']['sum_reach'],
                                event_uniform_Q_gain=.9-.1*math.fsum(traces['event'][arm]['reach_weights'][1:]),
                                physical_uniform_Q_gain=traces['physical'][arm]['gain']['uniform_q_offset_gain'])
                    pairs={name:paired_decomposition(byarm['low'],byarm['high']) for name,byarm in traces.items()}
                    blocks.append(dict(block=block,pairs=pairs,probes=probes))
                write(detail/f'{case}_{checkpoint}_{line}.json',dict(structure=structure,blocks=blocks))
                summaries={}
                for name in functions:
                    byblock=[values(b['pairs'][name]) for b in blocks]
                    part={key:stats([row[key] for row in byblock]) for key in byblock[0]}
                    probe_summary={}
                    for arm in ('low','high'):
                        samples=[b['probes'][name][arm] for b in blocks]
                        probe_summary[arm]=[
                            samples[0]['reward']['selected_count'],samples[0]['reward']['selected_nonzero_rewards'],
                            stats([v['reward']['root_change'] for v in samples]),samples[0]['reward']['algebraic_change'],
                            sum(v['reward']['root_change']!=0 for v in samples),
                            stats([v['tail']['perturbations'][0]['root_change'] for v in samples]),
                            stats([v['tail']['perturbations'][1]['root_change'] for v in samples]),
                            stats([v['qshift']['perturbations'][0]['root_change'] for v in samples]),
                            stats([v['qshift']['perturbations'][1]['root_change'] for v in samples])]
                    summaries[name]=dict(pair_components=part,probes=probe_summary,
                        trace_positive_blocks=sum(v['target']>0 for v in byblock),
                        reward_band_action_difference_change=stats([b['probes'][name]['high']['reward']['root_change']-
                            b['probes'][name]['low']['reward']['root_change'] for b in blocks]))
                lines[line]=dict(structure=structure,estimators=summaries)
            checkpoints[checkpoint]=lines
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(case=case,checkpoint=checkpoint,status='complete')),flush=True)
        cases[case]=checkpoints
    result=dict(status='complete',integrity=dict(all_pass=True,source285_456_and_real_state_memory_links=True,
        native_and_M8_lambda0_bit_exact=True,old_M8_full_traces_exact=True,root_component_closure=True,
        no_models_loaded_or_updated=True,no_synthetic_state_or_heldout_join=True),
        counts=count,cases=cases,long_timing=long_timing(p),
        schema=dict(stats=['mean','sample_std','min','max'],difference='high1.875_minus_low0; scaled target units',
            probe_columns=['reward_selected_rows','reward_selected_nonzero','reward_root_change_stats',
                'reward_algebraic_change','reward_native_nonzero_blocks','tail_negative_stats','tail_positive_stats',
                'Q_negative_stats','Q_positive_stats'],
            pair_components='reward + intermediate_bootstrap_minus_Q + cutoff_bootstrap + rounding',
            pair_scope='event chain root-target differences, not matched-clock or300s welfare',
            reward_probe='whole genuine reward rows completed in[30,45)s, zeroed only for numerical sensitivity',
            uniform_Q_probe='scalar reference Q offset, V/c/actor fixed; not network training or a convergence proof'),
        source_contract=dict(initializer_seed=6202,initializer_updates=50000,feedback_slots=285,
            optimizer_updates=456,horizon_seconds=24000,F_unchanged=True,trust_unchanged=True),
        budget=dict(new_training_updates=0,new_feedback_slots=0,new_native_prefixes=0,new_native_branches=0,
            new_native_seconds=0,new_target_draws=0,checkpoints_loaded=0),elapsed_seconds=time.monotonic()-started)
    write(HERE/'resident_summary.json',result)
    encoded=json.dumps(compact(result),separators=(',',':'),allow_nan=False)+'\n'
    if len(encoded.encode())>p['summary_max_bytes']: raise RuntimeError('Reduce resident summary before downloading')
    with (HERE/'summary.json').open('x') as handle: handle.write(encoded)
    print(json.dumps(dict(status='complete',bytes=len(encoded.encode()),seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__': main()
