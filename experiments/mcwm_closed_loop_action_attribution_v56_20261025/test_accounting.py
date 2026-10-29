"""Cumulative receipts must be differenced before comparing training stages."""
import json

from experiments.mcwm_closed_loop_action_attribution_v56_20261025.run import training_accounting


def test_cumulative_stage_accounting_does_not_double_count(tmp_path):
    values=[]
    for packet in range(1,55):
        n=2*min(packet,27)+4*max(0,packet-27)
        values.append(dict(packet=packet,target=dict(trace_corrected_rows=n,
            correction_sum=3*n,future_reward_sum=-n,future_bootstrap_sum=-100*n,
            future_minus_matching_q_sum=104*n,decomposition_rounding_sum=0.)))
    path=tmp_path/'packets.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in values))
    result=training_accounting(path)
    assert result['all']['corrected_sample_rows']==162
    assert result['first27']['corrected_sample_rows']==54
    assert result['last27']['corrected_sample_rows']==108
    assert result['all']['mean_reward']==-1
    assert result['last27']['mean_bootstrap_minus_q']==4
    assert result['last27']['mean_correction']==3
