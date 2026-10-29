"""Frozen bounded-credit reduction over resident arrays; no learner or simulator."""
from pathlib import Path
import math
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from experiments.mcwm_closed_loop_physical_credit_v50_20261025.run import read,write,stats,values


def compact_summary(result):
    from experiments.mcwm_closed_loop_trace_target_v44_20261024.summary import compact
    result=compact(result)
    for checkpoints in result['cases'].values():
        for lines in checkpoints.values():
            for row in lines.values():
                for key in ('intermediate_bootstrap_minus_Q','common_weight_reward','changed_weight_reward'):
                    row['pair_components'].pop(key)
                for arm in ('low','high'):
                    for key,value in row['rounding'][arm].items():
                        row['rounding'][arm][key]=[value[0],max(abs(value[2]),abs(value[3]))]
                    probes=row['probes'][arm]
                    probes['root_ulp']=[probes['root_ulp'][2],probes['root_ulp'][3]]
                    probes['tail_negative']=probes['tail_negative'][0]
                    probes['Q_negative']=probes['Q_negative'][0]
    keep=('line','root_clock','physical_span','actual_300_endpoint','bounded_c1_future_mass',
          'bounded_c1_uniform_Q_gain','bounded_c1_cutoff_weight','bounded_c1_weight_at300')
    for case in result['long_timing'].values():
        case['rows']=[{key:row[key] for key in keep} for row in case['rows']]
    result['schema'].update(rounding_summary=['mean','maximum_absolute'],
        root_ulp=['min','max'],negative_Q_and_tail_probes='mean only; positive probe keeps full stats')
    return result


def long_timing(protocol):
    from experiments.mcwm_closed_loop_physical_credit_v50_20261025.run import long_timing as previous
    from experiments.mcwm_closed_loop_bounded_credit_v51_20261025.credit import ETA
    result=previous(protocol)
    for case in result.values():
        for row in case['rows']:
            future=ETA*(row['c1_sum_reach']-1.)
            row.update(bounded_c1_future_mass=future,bounded_c1_uniform_Q_gain=.9-.1*future,
                bounded_c1_cutoff_weight=ETA*row['physical_c1_cutoff_weight'],
                bounded_c1_weight_at300=None if row['physical_c1_weight_at300'] is None else ETA*row['physical_c1_weight_at300'])
    return result


def main():
    import numpy as np
    from experiments.mcwm_closed_loop_trace_target_v44_20261024.probe import _source
    from experiments.mcwm_closed_loop_trace_target_v44_20261024.trace import decompose_trace
    from experiments.mcwm_closed_loop_trace_target_v44_20261024.summary import paired_decomposition
    from experiments.mcwm_closed_loop_bounded_credit_v51_20261025.credit import (
        bounded_trace,probe_reward_band,probe_bootstrap_shift,probe_uniform_q_shift,ETA,
    )
    protocol,started=read(HERE/'protocol.json'),time.monotonic()
    if protocol['future_residual_scale']!='9/269' or protocol['response_capacity']!=270:
        raise RuntimeError('Declared single candidate differs from implementation')
    detail=HERE/'detail';detail.mkdir(exist_ok=False)
    count=dict(native_lambda0_rows=0,mean8_lambda0_rows=0,old_full_trace_exact=0,
        old_physical_pairs_exact=0,bounded_roots=0)
    cases={}
    for case in protocol['scenarios']:
        _,pilot,updates,snapshot,ordinary,all_rows,chains,credit=_source(
            HERE.parent/protocol['native_source']/'runs'/case)
        indexed={(chain['line_id'],chain['arm']):chain for chain in chains}
        native=read(HERE.parent/protocol['native_target_arrays']/'runs'/case/'probe_detail.json')
        if not native['integrity']['all_pass']: raise RuntimeError('Native source target audit failed')
        checkpoints={}
        for checkpoint in protocol['checkpoints']:
            original=native['checkpoints'][checkpoint]
            bank={(chain['line_id'],chain['arm']):chain for chain in original['seeds']['336202']['chains']}
            for rows,c in [(chain['rows'],chain['c']) for chain in bank.values()]+[
                ([row],[]) for row in original['ordinary_rows_first_seed']]:
                if not bounded_trace(rows,c,0.)['lambda_zero_native_exact']: raise RuntimeError('Native lambda0 changed')
                count['native_lambda0_rows']+=len(rows)
            meanroot=HERE.parent/protocol['mean8_arrays']/'runs'/case/checkpoint
            for scored in read(meanroot/'detail/ordinary_first_block.json'):
                if not bounded_trace([scored['m8']],[],0.)['lambda_zero_native_exact']:
                    raise RuntimeError('M8 ordinary lambda0 changed')
                count['mean8_lambda0_rows']+=1
            lines={}
            for line in protocol['lines']:
                reference=read(HERE.parent/protocol['physical_reference']/'detail'/f'{case}_{checkpoint}_{line}.json')
                blocks=[];structure={}
                for block in range(protocol['blocks']):
                    saved=read(meanroot/'detail'/f'{line}_block{block:02d}.json')
                    traces={};probes={};physical_roots={}
                    for arm in ('low','high'):
                        rows=saved['raw_scores'][arm]['mean8'];cached=bank[(line,arm)]
                        real=indexed[(line,arm)]['records'];c=cached['c']
                        if len(rows)!=45 or any(a['sequence']!=b['sequence'] or a['started_at']!=b['started_at']
                            or a['completed_at']!=b['completed_at'] or a['dt']!=b['dt']
                            or a['reward_scaled']!=d['reward_scaled'] for a,b,d in zip(rows,real,cached['rows'])):
                            raise RuntimeError('Resident M8 arrays differ from the real factual chain')
                        logged=credit[24 if checkpoint=='actual_post24' else 456][(line,arm)]
                        if c!=[row['c'] for row in logged['rows']]: raise RuntimeError('F correction ratios changed')
                        if not bounded_trace(rows,c,0.)['lambda_zero_native_exact']: raise RuntimeError('M8 lambda0 changed')
                        count['mean8_lambda0_rows']+=len(rows)
                        old=decompose_trace(rows,c,1.)
                        if old['root']['trace_target']!=saved['decomposition'][arm]['mean8']['trace_target']:
                            raise RuntimeError('Old complete trace changed')
                        count['old_full_trace_exact']+=1
                        trace=bounded_trace(rows,c,1.);traces[arm]=trace
                        physical_roots[arm]=trace['per_event'][0]['physical_target']
                        if abs(trace['root']['reconstruction_residual'])>1e-8:
                            raise RuntimeError('Bounded root decomposition does not close')
                        if trace['gain']['ideal_future_sum_reach']>9.+1e-12:
                            raise RuntimeError('Fixed-capacity residual mass bound failed')
                        count['bounded_roots']+=1
                        probe=dict(reward=probe_reward_band(rows,c,bounded_trace),
                            tail=probe_bootstrap_shift(rows,c,bounded_trace),
                            qshift=probe_uniform_q_shift(rows,c,bounded_trace))
                        probe['root_ulp']=abs(float(np.spacing(np.float32(trace['root']['trace_target']))))
                        probes[arm]=probe
                        if block==0:
                            structure[arm]=dict(span=rows[-1]['completed_at']-rows[0]['started_at'],
                                zero_duration=sum(row['dt']==0 for row in rows),c_product=math.prod(c),
                                cutoff_weight=trace['reach_weights'][-1],
                                future_mass=trace['gain']['future_sum_reach'],
                                ideal_future_mass=trace['gain']['ideal_future_sum_reach'],
                                Q_level_gain=trace['gain']['uniform_q_offset_gain'],
                                mass_float32_rounding=trace['gain']['float32_mass_rounding'])
                    if physical_roots['high']-physical_roots['low']!=reference['blocks'][block]['pairs']['physical']['trace_difference']:
                        raise RuntimeError('v50 complete physical target pair changed')
                    count['old_physical_pairs_exact']+=1
                    blocks.append(dict(block=block,pair=paired_decomposition(traces['low'],traces['high']),
                        probes=probes,rounding={arm:traces[arm]['rounding_breakdown'] for arm in traces}))
                write(detail/f'{case}_{checkpoint}_{line}.json',dict(structure=structure,blocks=blocks))
                byblock=[values(block['pair']) for block in blocks]
                parts={key:stats([row[key] for row in byblock]) for key in byblock[0]}
                probe_summary={};rounding={}
                for arm in ('low','high'):
                    samples=[block['probes'][arm] for block in blocks]
                    probe_summary[arm]=dict(
                        reward_selected_rows=samples[0]['reward']['selected_count'],
                        reward_selected_nonzero=samples[0]['reward']['selected_nonzero_rewards'],
                        reward_root_change=stats([sample['reward']['root_change'] for sample in samples]),
                        reward_algebraic_change=samples[0]['reward']['algebraic_change'],
                        reward_nonzero_blocks=sum(sample['reward']['root_change']!=0 for sample in samples),
                        reward_algebraic_in_root_ulps=stats([abs(sample['reward']['algebraic_change'])/sample['root_ulp'] for sample in samples]),
                        root_ulp=stats([sample['root_ulp'] for sample in samples]),
                        tail_negative=stats([sample['tail']['perturbations'][0]['root_change'] for sample in samples]),
                        tail_positive=stats([sample['tail']['perturbations'][1]['root_change'] for sample in samples]),
                        Q_negative=stats([sample['qshift']['perturbations'][0]['root_change'] for sample in samples]),
                        Q_positive=stats([sample['qshift']['perturbations'][1]['root_change'] for sample in samples]))
                    rounding[arm]={key:stats([block['rounding'][arm][key] for block in blocks])
                        for key in ('scaled_physical_recurrence','combined_recurrence')}
                    rounding[arm]['root_blend']=stats([block['rounding'][arm]['root_blend']['total'] for block in blocks])
                lines[line]=dict(structure=structure,pair_components=parts,probes=probe_summary,rounding=rounding,
                    trace_positive_blocks=sum(row['target']>0 for row in byblock),
                    reward_band_action_difference_change=stats([block['probes']['high']['reward']['root_change']-
                        block['probes']['low']['reward']['root_change'] for block in blocks]))
            checkpoints[checkpoint]=lines
            print('SCHEDULEURM_PROGRESS '+str(dict(case=case,checkpoint=checkpoint,status='complete')),flush=True)
        cases[case]=checkpoints
    result=dict(status='complete',integrity=dict(all_pass=True,source285_456_and_real_state_memory_links=True,
        native_and_M8_lambda0_bit_exact=True,old_M8_full_traces_exact=True,old_physical_pairs_exact=True,
        root_component_closure=True,fixed_capacity_residual_mass_bound=True,
        no_models_loaded_or_updated=True,no_synthetic_state_or_heldout_join=True),counts=count,
        cases=cases,long_timing=long_timing(protocol),
        candidate=dict(capacity=270,mass_bound=9,eta=ETA,eta_float32=float(np.float32(ETA))),
        schema=dict(stats=['mean','sample_std','min','max'],difference='high1.875_minus_low0; scaled target units',
            pair_scope='event chain root-target differences, not matched-clock or300s welfare',
            reward_probe='whole actual rewards completed in[30,45)s; frozen numerical sensitivity only',
            Q_probe='nonterminal scalar synchronous Q offset, V/actor/c fixed; not network training',
            repeated_blocks='16 scoring blocks per same physical state; not independent physical samples',
            rounding='scaled v50 internal recurrence and final root blend kept separately'),
        source_contract=dict(initializer_seed=6202,initializer_updates=50000,feedback_slots=285,
            optimizer_updates=456,horizon_seconds=24000,F_unchanged=True,trust_unchanged=True),
        budget=dict(new_training_updates=0,new_feedback_slots=0,new_native_prefixes=0,new_native_branches=0,
            new_native_seconds=0,new_target_draws=0,checkpoints_loaded=0),elapsed_seconds=time.monotonic()-started)
    write(HERE/'resident_summary.json',result)
    import json
    encoded=json.dumps(compact_summary(result),separators=(',',':'),allow_nan=False)+'\n'
    if len(encoded.encode())>protocol['summary_max_bytes']:
        raise RuntimeError('Reduce resident summary before downloading')
    with (HERE/'summary.json').open('x') as handle: handle.write(encoded)
    print(json.dumps(dict(status='complete',bytes=len(encoded.encode()),seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__': main()
