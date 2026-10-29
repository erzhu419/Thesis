"""Summarize construction, execution support and budget gates separately."""
import argparse
from collections import Counter
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
MODELS=('zero_memory','per_head_level','shared_level')
COLUMNS=('slot','arm','events','first_reward_index0','structural_reach','numeric_weight',
    'budget_weight','first_stop_event','first_stop_reason','behavior_unknown','context_reconstructed',
    'numerical_atom_underflow','root_trace_delta')


def read(path): return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as output:
        json.dump(value,output,separators=(',',':'),allow_nan=False);output.write('\n')


def summarize_model(value):
    heldout,admitted=value['heldout'],value['admitted']
    source=Counter()
    for chain in admitted:
        kind='ordinary' if chain['id'].startswith('ordinary/') else 'paired'
        source[kind+'_chains']+=1;source[kind+'_rows']+=chain['events']
        source[kind+'_max_events']=max(source[kind+'_max_events'],chain['events'])
        source[kind+'_multievent_chains']+=chain['events']>1
        source[kind+'_root_targets_changed']+=chain['diagnostic_root_target']!=chain['root_native_target']
    support=Counter()
    for chain in heldout:
        support.update(chain['support_reasons'])
    rows=[]
    for chain in heldout:
        stop=chain['diagnostic_first_constraint_stop'] or {}
        rows.append([chain['slot'],chain['arm'],chain['events'],chain['first_reward_index0'],
            chain['first_reward_structurally_reached'],chain['first_reward_diagnostic_weight'],
            chain['first_reward_budget_weight'],stop.get('event'),stop.get('reason'),
            chain['behavior_unknown'],chain['behavior_context_reconstructed'],chain['mass_underflow'],
            chain['diagnostic_root_target']-chain['root_native_target']])
    all_chains=heldout+admitted
    return dict(integrity=value['integrity'],source_slots=value['source_slots'],source_updates=value['source_updates'],
        native_target_rows=sum(c['events'] for c in all_chains),
        old_v38_rows=sum(c['events'] for c in heldout),
        lambda_zero_bitwise_exact=all(c['lambda_zero_bitwise_exact'] for c in all_chains),
        old_v38_target_parts_exact=all(c['cached_v38_parts_exact'] for c in heldout),
        subset_links=sum(c['baseline_links'] for c in all_chains),
        full_head_min_would_differ=sum(c['full_head_min_would_differ'] for c in all_chains),
        full_subset_path_nonexact=sum(c['full_subset_path_nonexact'] for c in all_chains),
        max_full_subset_path_ulp=max(c['max_full_subset_path_ulp'] for c in all_chains),
        zero_dt=sum(c['zero_dt'] for c in heldout),support_reasons=dict(support),
        heldout_structural_reward_reach=sum(c['first_reward_structurally_reached'] is True for c in heldout),
        heldout_numerical_reward_reach=sum(c['first_reward_diagnostic_weight']>0 for c in heldout),
        heldout_budget_reward_reach=sum(c['first_reward_budget_weight']>0 for c in heldout),
        heldout_authorized_rows=sum(c['source_budget']['authorized_rows'] for c in heldout),
        budget_rows=sum(c['source_budget']['authorized_rows'] for c in admitted),
        original_budget_construction=dict(source),heldout=rows,
        new_updates=value['new_updates'],new_feedback=value['new_feedback'])


def analyze(runs,out):
    out.mkdir(parents=True,exist_ok=False)
    cases={}
    for scenario in ('stable','local_speed_drop'):
        root=runs/scenario
        if not (root/'summary.json').exists():
            failure=read(root/'failure.json') if (root/'failure.json').exists() else None
            cases[scenario]=dict(status='failed',failure=failure,models={});continue
        parent=read(root/'summary.json');gate=read(root/'input_gate.json')
        models={name:summarize_model(read(root/name/'summary.json')) for name in MODELS if (root/name/'summary.json').exists()}
        cases[scenario]=dict(status=parent['status'],integrity=parent['integrity'],
            old_models_exact=gate['model_proof']['all_pass'],source285_provenance=gate['budget']['complete_provenance'],
            original_budget=dict(total=gate['budget']['total_slots'],ordinary=gate['budget']['ordinary_slots'],
                paired=gate['budget']['paired_slots'],ordinary_chains=gate['ordinary_chains'],
                maximum_ordinary_events=gate['maximum_ordinary_events'],maximum_ordinary_seconds=gate['maximum_ordinary_seconds'],
                paired_chains=gate['paired_chains'],
                maximum_paired_events=max((m['original_budget_construction']['paired_max_events'] for m in models.values()),default=None),
                paired_immediate_reward_differences=gate['paired_first_reward_differences'],
                finite300_aggregate_already_authorized=True,later_paired_td_chain_authorized=False),
            actual_model_overlap_seconds=parent['actual_three_model_overlap_seconds'],models=models,
            new_updates=parent['new_updates'],new_feedback=parent['new_feedback'],new_native_seconds=parent['new_native_seconds'])
    complete=all(case.get('integrity',{}).get('all_pass') for case in cases.values())
    construction=complete and all(m['lambda_zero_bitwise_exact'] for c in cases.values() for m in c['models'].values())
    paired_chains_available=complete and all(c['original_budget']['maximum_paired_events']>1 for c in cases.values())
    diagnostic_feedback_authorized=complete and all(m['heldout_budget_reward_reach']==12 for c in cases.values() for m in c['models'].values())
    response_support=complete and all(m['heldout_numerical_reward_reach']==12 for c in cases.values() for m in c['models'].values())
    summary=dict(status='complete' if complete else 'partial',integrity=dict(all_pass=complete),
        priority='high',schema=dict(heldout_columns=list(COLUMNS),
            reach_weight='coefficient of that real reward in the lambda1 diagnostic at issue; not a measured SGD gradient',
            lambda1='fixed construction/support upper-bound diagnostic, not tuned or trained'),cases=cases,
        gates=dict(construction_integrity=construction,
            all_heldout_reward_execution_support=complete and all(m['heldout_structural_reward_reach']==12 for c in cases.values() for m in c['models'].values()),
            all_heldout_reward_numerical_reach=complete and all(m['heldout_numerical_reward_reach']==12 for c in cases.values() for m in c['models'].values()),
            heldout_feedback_can_enter_training=diagnostic_feedback_authorized,
            original_training_pairs_have_budgeted_response_event_chain=paired_chains_available),
        decision=dict(training_ready=construction and response_support and paired_chains_available,
            new_training=False,new_native_branches=False,
            next_gate='Review execution truncation and original-budget contiguous-chain coverage. Heldout traces cannot authorize training; original paired feedback contains one TD per arm only.'),
        compute_budget=dict(new_native_prefixes=0,new_native_suffixes=0,new_native_seconds=0,
            new_training_labels=0,new_optimizer_updates=0),
        limitations='No training, causal policy improvement, or mixed-operator convergence is certified. Positive actor-clamp atoms and zero-probability continuous targets are distinct. Existing finite300 training aggregates do not authorize intermediate TD labels; no300-state is fabricated.')
    encoded=json.dumps(summary,separators=(',',':'),allow_nan=False)+'\n'
    if len(encoded.encode())>24000: raise RuntimeError('Resident summary exceeds24000bytes')
    (out/'summary.json').write_text(encoded)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    try: print(json.dumps(dict(status=analyze(args.runs,args.out)['status'])))
    except Exception as error:
        args.out.mkdir(parents=True,exist_ok=True)
        write(args.out/'failure.json',dict(type=type(error).__name__,message=str(error)));raise
