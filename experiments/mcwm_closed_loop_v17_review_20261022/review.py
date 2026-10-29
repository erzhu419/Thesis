"""Derive readable v17 comparisons from the small immutable review export."""
from collections import Counter
from itertools import product
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    value=json.loads((HERE/'review_summary.json').read_text())
    schema=value['schema'];models=schema['model_order'];fields=schema['effect_order']
    def decode(row):
        row=dict(row)
        row['effects']={model:dict(zip(fields,items)) if items else None for model,items in zip(models,row['effects'])}
        row['truth']={name:dict(zip(schema['pairs']['truth_fields'],items)) for name,items in zip(schema['pairs']['truth_endpoint_order'],row['truth'])}
        return row
    training=[decode(row) for row in value['training_pairs']]
    heldout=[decode(row) for row in value['heldout_pairs']]
    selected=[row for row in heldout if row['selection_status']=='selected' and row['integrity']['all_pass']]
    def metrics(rows):
        result={}
        for model in models:
            effects=[row['effects'][model] for row in rows if row['effects'][model] is not None]
            result[model]=dict(n=len(effects),direction_correct=sum(e['direction_correct'] for e in effects),
                action_correct=sum(e['action_correct'] for e in effects),regret=sum(e['regret'] for e in effects),
                cost_gain_mae=sum(abs(e['cost_signed_error']) for e in effects)/len(effects) if effects else None)
        return result
    groups=[]
    for keys in (('scenario',),('line_id',),('window_name',),('seed',),('scenario','line_id'),
                 ('scenario','window_name'),('line_id','window_name'),('scenario','line_id','window_name')):
        for cohort in sorted({tuple(row[key] for key in keys) for row in selected}):
            rows=[r for r in selected if tuple(r[key] for key in keys)==cohort]
            groups.append(dict(group=dict(zip(keys,cohort)),metrics=metrics(rows),
                line_label_age_mean=sum(r['paired_label_age']['line_admitted_age_seconds'] for r in rows)/len(rows),
                shared_label_age_mean=sum(r['paired_label_age']['shared_admitted_age_seconds'] for r in rows)/len(rows)))
    flips=[]
    cases=[]
    for row in selected:
        gain=row['truth']['fixed300']['gain']
        case=dict(seed=row['seed'],scenario=row['scenario'],line_id=row['line_id'],window_name=row['window_name'],
            at=row['selected']['started_at'],duration=row['duration'],features=row['features'],
            true_gain=dict(zip(schema['metric_order'],gain)),label_age=row['paired_label_age'],effects=row['effects'])
        cases.append(case)
        route=row['effects']['route']['predicted_cost_gain'];shared=row['effects']['shared']['predicted_cost_gain']
        if (route>0)==(shared>0):continue
        day=next(d for d in value['days'] if d['seed']==row['seed'] and d['scenario']==row['scenario'])
        coef=day['readouts']['shared']['coefficients']
        intercept=coef[row['line_id']+'_intercept']['cost']
        slope=coef['forward_minus_backward_headway_deviation']['cost']
        flips.append(dict(seed=row['seed'],scenario=row['scenario'],line_id=row['line_id'],window_name=row['window_name'],
            true_gain=gain[3],route_prediction=route,shared_prediction=shared,
            intercept_change=intercept-route,headway_contribution=slope*row['features'][-1],
            regret_change=row['effects']['shared']['regret']-row['effects']['route']['regret']))
    holding={}
    for model in ('route','shared'):
        choices=[r for r in selected if r['effects'][model]['predicted_cost_gain']>0]
        holding[model]=dict(n=len(choices),beneficial=sum(r['truth']['fixed300']['gain'][3]>0 for r in choices),
            harmful=sum(r['truth']['fixed300']['gain'][3]<0 for r in choices),
            captured_gain=sum(max(0,r['truth']['fixed300']['gain'][3]) for r in choices),
            holding_loss=sum(max(0,-r['truth']['fixed300']['gain'][3]) for r in choices))
    latest=[]
    for row in selected:
        pair=max((r for r in training if r['seed']==row['seed'] and r['scenario']==row['scenario']
                  and r['line_id']==row['line_id']),key=lambda r:r['admission']['processed_at'])
        prediction=pair['truth']['fixed300']['gain'][3];truth=row['truth']['fixed300']['gain'][3]
        latest.append(dict(correct=(prediction>0)==(truth>0),regret=max(0,-truth) if prediction>0 else max(0,truth)))
    dedup={json.dumps([r['selected'],r['features'],r['prefix']],sort_keys=True) for r in training}
    result=dict(overall=metrics(selected),groups=groups,cases=cases,shared_action_changes=flips,holding=holding,
        training_design_pairs=len(training),unique_training_physical_prefixes=len(dedup),
        training_duration_counts=dict(Counter(r['duration'] for r in training)),
        heldout_duration_counts=dict(Counter(r['duration'] for r in selected)),
        unsupported={m:sum(r['effects'][m] is None for r in selected) for m in models},
        latest_pair_sign_diagnostic=dict(n=len(latest),correct=sum(r['correct'] for r in latest),regret=sum(r['regret'] for r in latest),
            role='Posthoc descriptive diagnostic only; not a new evaluated candidate.'),
        decision='No independent confirmation of current shared model; retain route baseline partial effects and diagnose response sign changes.')
    path=HERE/'analysis.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(bytes=path.stat().st_size,overall=result['overall'],flips=flips,holding=holding,
        training_prefixes=len(dedup),training_durations=result['training_duration_counts'],heldout_durations=result['heldout_duration_counts']),ensure_ascii=False))


if __name__=='__main__':main()
