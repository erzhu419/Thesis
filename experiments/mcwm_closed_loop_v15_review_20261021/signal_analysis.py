"""Read-only v15 contrast analysis on the nine distinct physical prefixes."""
from pathlib import Path
import json
import math

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'mcwm_closed_loop_training_pairs_v15_20261021/training_pair_summary.json'


def rms(values):
    return math.sqrt(sum(x*x for x in values)/len(values)) if values else None


def stats(rows, endpoint, models):
    values = {model: [] for model in models}
    common = []
    for row in rows:
        effects = dict(zip(models, row['effects'][endpoint]))
        for model, value in effects.items():
            if value is not None:
                values[model].append(value)
        if all(value is not None for value in effects.values()):
            common.append((row['state_id'], effects))
    def summarize(items):
        n = len(items)
        return dict(n=n, gain_mae=sum(abs(x[1]) for x in items)/n if n else None,
            gain_rmse=rms([x[1] for x in items]), direction_correct=sum(x[2] for x in items),
            selected_action_correct=sum(x[3] for x in items), ties=sum(x[0] == 0 for x in items),
            regret=sum(x[4] for x in items) if n else None)
    return dict(design_n=len(rows), missing_truth=sum(row['truth'][endpoint]['gain'] is None for row in rows),
        models={model:summarize(items) for model,items in values.items()},
        common_state_ids=[state_id for state_id,_ in common],
        common_metrics={model:summarize([effects[model] for _,effects in common]) for model in models})


def main():
    v = json.loads(SOURCE.read_text())
    models = v['schema']['endpoint_model_order']
    global_models = v['schema']['global_model_order']
    duplicates = v['duplicate_prefixes']
    assert all(group['prediction_and_truth_exact'] for group in duplicates)
    excluded = {state for group in duplicates for state in group['state_ids'][1:]}
    pairs = [row for row in v['pairs'] if row['state_id'] not in excluded]
    assert len(pairs) == v['state_counts']['unique_physical_prefixes'] == 9
    assert all(row['integrity']['all_pass'] for row in pairs)
    rows = []
    low, high, center, differential = [], [], [], []
    for pair in pairs:
        truth = pair['truth']['fixed300']
        delta = truth['low_minus_high']
        assert delta[0] == 0. and delta[3] == 2*delta[1]+delta[2]
        a,b = pair['background_residuals']['fixed300']
        assert a[0] == b[0] and a[1] == b[1]
        assert a[3]-b[3] == truth['gain']
        low.append(a[3]); high.append(b[3]); center.append((a[3]+b[3])/2); differential.append(a[3]-b[3])
        rows.append(dict(state_id=pair['state_id'], line=pair['target_line'], scene=pair['scenario'],
            window=pair['window_name'], global_duration_group=pair['duration'],
            gain=truth['gain'], wait_gain=delta[1], weighted_wait_gain=2*delta[1], ride_gain=delta[2],
            low_cost=truth['branches']['low'][3], high_cost=truth['branches']['high'][3],
            predictions={model:value for model,value in zip(models,pair['effects']['fixed300'])},
            background_center_residual=(a[3]+b[3])/2,
            source_frozen_nominal_cost_change=pair['source_vs_frozen_nominal']['endpoints']['fixed300']['frozen_minus_source'][3]))
    low_mean,high_mean=sum(low)/len(low),sum(high)/len(high)
    correlation=sum((a-low_mean)*(b-high_mean) for a,b in zip(low,high))/math.sqrt(
        sum((a-low_mean)**2 for a in low)*sum((b-high_mean)**2 for b in high))
    cohorts=[]
    for name, selected in (
        ('pre_shock_shared', [p for p in pairs if p['window_name']=='pre_shock']),
        ('stable_post', [p for p in pairs if p['scenario']=='stable' and p['window_name']=='post_shock']),
        ('shock_post', [p for p in pairs if p['scenario']=='local_speed_drop' and p['window_name']=='post_shock']),
        *[(line,[p for p in pairs if p['target_line']==line]) for line in ('7X','102X','705X')]):
        cohorts.append(dict(name=name,state_ids=[p['state_id'] for p in selected],fixed300=stats(selected,'fixed300',models)))
    duration={}
    for name in sorted({p['duration'] for p in pairs}):
        subset=[p for p in pairs if p['duration']==name]
        duration[name]=dict(state_ids=[p['state_id'] for p in subset],global_next=stats(subset,'global_next',global_models),
            raw_model_support={model:sum(p['raw_global_effects'][i] is not None for p in subset) for i,model in enumerate(global_models)},
            fixed300_gains={p['state_id']:p['truth']['fixed300']['gain'] for p in subset})
    output=dict(source_bytes=SOURCE.stat().st_size,units='weighted passenger seconds: cost=2wait+ride',
        gain_sign='low cost minus high cost: positive chooses7.5 seconds; ties choose0',
        effect_order=v['schema']['effect_order'],unique_states=9,excluded_duplicate_states=sorted(excluded),
        states=rows,overall={endpoint:stats(pairs,endpoint,global_models if endpoint=='global_next' else models)
                            for endpoint in ('global_next','same_vehicle_next','fixed300')},cohorts=cohorts,duration_groups=duration,
        common_mode=dict(low_high_background_residual_correlation=correlation,
            pair_center_cost_residual_rms=rms(center),true_gain_rms=rms(differential),
            true_gain_mean_absolute=sum(abs(x) for x in differential)/len(differential),
            subtraction_reconstruction_max_error=max(abs(d-p['truth']['fixed300']['gain']) for d,p in zip(differential,pairs)),
            interpretation='Same-state baseline and saved background cancel exactly in the fixed300 paired contrast; this does not remove cross-seed simulation variability.'),
        decision=dict(budgeted_difference_learning='small_development_candidate_is_supported', independent_confirmation=False,
            reasons=['Paired targets expose nonzero small action differences while cancelling dominant common level error.',
                'Nine unique states and nine action columns are saturated; full-interaction fitting is not independent evidence.',
                '102X changes sign in shock post-period; route-constant model is an insufficient sole candidate.',
                'Use a new jointly budgeted training prefix, count each pair as two slots, and validate on additional native heldout states. Current diagnostic labels remain unfitted.']))
    (HERE/'signal_metrics.json').write_text(json.dumps(output,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
    print(json.dumps(dict(bytes=(HERE/'signal_metrics.json').stat().st_size,
        unique_prefixes=9, common_mode=output['common_mode'], fixed300=output['overall']['fixed300']),ensure_ascii=False))


if __name__=='__main__':
    main()
