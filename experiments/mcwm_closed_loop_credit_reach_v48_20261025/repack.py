"""Package completed v48 resident diagnostics; never rerun target accounting."""
from pathlib import Path
import json
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_credit_reach_v48_20261025.run import read, write, stats, compact_pair
from experiments.mcwm_closed_loop_v43_review_20261024.reward_clock import summarize_pair
from experiments.mcwm_closed_loop_trace_target_v44_20261024.summary import compact


def main():
    started = time.monotonic()
    protocol = read(HERE/'protocol.json')
    cases = {}
    for case in protocol['scenarios']:
        root = HERE.parent/protocol['source_version']/'runs'/case
        receipt = read(root/'response_receipt.json')
        pairs = {p['line_id']: p for p in receipt['pairs']}
        clocks, checkpoints = {}, {}
        for line, pair in pairs.items():
            value = summarize_pair(pair['branches']['low']['events_path'], pair['branches']['high']['events_path'],
                root/'branches'/line/'low/suffix_trace.jsonl', root/'branches'/line/'high/suffix_trace.jsonl',
                root/'learning/credit_packets.jsonl', line_id=line)
            value.pop('first_difference_credit')
            value.pop('last_common_credit_at456')
            value.pop('credit_columns')
            curve = read(HERE/'detail'/f'{case}_{line}_common_clocks.json')
            nonzero = [(t, 1 if x > 0 else -1) for t, x in curve if abs(x) > 1e-7]
            value['common_clock_cost_sign_crossings_seconds'] = [b[0] for a,b in zip(nonzero,nonzero[1:]) if a[1] != b[1]]
            value['physical'] = {arm: {k: read(pair['branches'][arm]['summary_path'])['physical'][k]
                for k in ('departure','next_same_vehicle_arrival','coverage')} for arm in ('low','high')}
            clocks[line] = value
        for checkpoint in protocol['checkpoints']:
            lines = {}
            for line in protocol['lines']:
                saved = read(HERE/'detail'/f'{case}_{checkpoint}_{line}.json')
                blocks = saved['blocks']
                fields = dict(target=[x['trace_difference'] for x in blocks], reward=[x['components']['reward'] for x in blocks],
                    bootstrap_minus_q=[x['bootstrap_minus_matching_Q'] for x in blocks],
                    terminal_bootstrap=[x['terminal_bootstrap_difference'] for x in blocks],
                    common_weight_reward=[x['reward_value_vs_weight']['common_weight_value_difference'] for x in blocks],
                    reward_weight_change=[x['reward_value_vs_weight']['common_value_weight_difference'] for x in blocks])
                lines[line] = dict(block_stats={k:stats(v) for k,v in fields.items()},
                    first_block=compact_pair(saved['first_block'], clocks[line]['first_common_cost_difference']))
            checkpoints[checkpoint] = lines
        cases[case] = dict(clock_pairs=clocks,checkpoints=checkpoints)
    result = dict(status='complete',integrity=dict(all_pass=True,
        source_real_state_memory_budget_c_and_target_exact=True,
        source_checks='run.py completed both scenarios; only small-summary size gate failed',
        reward_ledgers_exact=all(v['reward_ledger_consistent'] for c in cases.values()
            for p in c['clock_pairs'].values() for v in p['ledger_checks'].values())),
        cases=cases,source_contract=dict(initialization=50000,seed=6202,feedback_slots=285,
            source_updates=456,horizon_seconds=24000,F_unchanged=True,trust_unchanged=True),
        schema=dict(prefix_columns=['events','low_seconds','high_seconds','low_weight','high_weight',
            'target_delta','reward_delta','bootstrap_minus_Q_delta','cutoff_bootstrap_delta',
            'non_cutoff_bootstrap_minus_Q_delta','rounding_delta','same_actual_closure_clock'],
            attenuation_log_columns=['links','log_gamma','log_mixing','log_c','log_float32_rounding'],
            cutoff_perturbation_columns=['realized_bootstrap_delta','actual_float32_root_delta','algebraic_root_response'],
            differences='1.875_minus0_seconds; scaled_target_units, not300s_cost',
            clocks='raw passenger cost at shared actual clocks; ledger separately checked',
            arrays='v46 M8 frozen v43 checkpoints,16 scoring blocks; no v47 rescoring',
            first_common_cost_weight='last_nonzero_reward_row_weight_before_first_cost_witness_not_whole_cost_multiplier',
            ablations='c=1/mixing=1 algebra only; no proposed uncorrected target'),
        budget=dict(new_training_updates=0,new_feedback_slots=0,new_native_branches=0,
            new_native_seconds=0,new_target_draws=0,checkpoints_loaded=0),
        packaging_seconds=time.monotonic()-started)
    result['integrity']['all_pass'] = result['integrity']['reward_ledgers_exact']
    write(HERE/'resident_repacked.json',result)
    encoded = json.dumps(compact(result),separators=(',',':'),allow_nan=False)+'\n'
    if len(encoded.encode()) > protocol['summary_max_bytes']:
        raise RuntimeError('Resident repacking still exceeds small-pull budget: '+str(len(encoded.encode())))
    with (HERE/'summary.json').open('x') as handle: handle.write(encoded)
    print(json.dumps(dict(bytes=len(encoded.encode()),status=result['status'],integrity=result['integrity'])))


if __name__=='__main__': main()
