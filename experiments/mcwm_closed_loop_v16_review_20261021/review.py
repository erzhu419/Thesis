"""Decode the existing small summary; keep source results unchanged."""
from collections import Counter
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'mcwm_closed_loop_budgeted_difference_v16_20261021/learning_summary.json'


def main():
    value = json.loads(SOURCE.read_text())
    schema = value['schema']
    models = schema['model_order']
    endpoint_order = schema['pairs']['truth_endpoint_order']
    effect_order = schema['effect_order']
    def decode(row):
        result = dict(row)
        result['truth'] = {name: dict(zip(schema['pairs']['truth_fields'], fields))
                           for name, fields in zip(endpoint_order, row['truth'])}
        result['effects'] = {name: dict(zip(effect_order, fields)) if fields else None
                             for name, fields in zip(models, row['effects'])}
        return result
    training = [decode(row) for row in value['training_pairs']]
    heldout = [decode(row) for row in value['heldout_pairs']]
    selected = [row for row in heldout if row['selection_status']=='selected']
    def metrics(rows):
        out = {}
        for model in models:
            effects = [row['effects'][model] for row in rows if row['effects'][model] is not None]
            n = len(effects)
            out[model] = dict(n=n, direction_correct=sum(e['direction_correct'] for e in effects),
                action_correct=sum(e['action_correct'] for e in effects),
                regret_total=sum(e['regret'] for e in effects) if n else None,
                cost_gain_mae=sum(abs(e['cost_signed_error']) for e in effects)/n if n else None,
                wait_gain_mae=sum(abs(e['wait_signed_error']) for e in effects)/n if n else None,
                ride_gain_mae=sum(abs(e['ride_signed_error']) for e in effects)/n if n else None)
        return out
    groups = []
    for scene in ('stable','local_speed_drop'):
        for line in ('7X','102X','705X'):
            rows = [r for r in heldout if r['scenario']==scene and r['line_id']==line]
            actual = [r for r in rows if r['selection_status']=='selected']
            groups.append(dict(scenario=scene,line=line,design_states=len(rows),
                               selected_states=len(actual),metrics=metrics(actual)))
    cases = []
    for row in selected:
        endpoint = row['truth']['fixed300']
        cases.append(dict(seed=row['seed'],scenario=row['scenario'],line=row['line_id'],
            window=row['window_name'],at=row['selected']['started_at'],duration=row['duration'],
            true_gain=dict(zip(schema['metric_order'],endpoint['gain'])),
            low_cost=endpoint['low'][3],high_cost=endpoint['high'][3],
            models={model:dict(**row['effects'][model],
                chosen_hold_seconds=7.5 if row['effects'][model]['predicted_cost_gain']>0 else 0.)
                for model in models}))
    clocks=[]
    for day in value['days']:
        same_day=lambda r:r['seed']==day['seed'] and r['scenario']==day['scenario']
        train=[r for r in training if same_day(r)]
        test=[r for r in selected if same_day(r)]
        last=max(r['admission']['processed_at'] for r in train)
        clocks.append(dict(seed=day['seed'],scenario=day['scenario'],
            last_pair_issue=max(r['selected']['started_at'] for r in train),
            last_pair_admitted=last,first_heldout=min(r['selected']['started_at'] for r in test),
            gap_after_last_pair=min(r['selected']['started_at'] for r in test)-last))
    physical={json.dumps([r['selected'],r['features'],r['prefix']],sort_keys=True) for r in training}
    def signal(rows):
        gains=[r['truth']['fixed300']['gain'] for r in rows]
        return {key:sum(abs(g[i]) for g in gains)/len(gains)
                for i,key in enumerate(schema['metric_order'])}
    result=dict(source=str(SOURCE),source_bytes=SOURCE.stat().st_size,
        decision='Do not advance this candidate to independent confirmation.',
        selected_states=len(selected),missing_states=len(heldout)-len(selected),
        unsupported_on_selected={model:sum(r['effects'][model] is None for r in selected) for model in models},
        overall=metrics(selected),strata=groups,cases=cases,freshness=clocks,
        training_design_pairs=len(training),unique_training_physical_prefixes=len(physical),
        training_durations=dict(Counter(r['duration'] for r in training)),
        heldout_durations={name:dict(n=sum(r['duration']==name for r in selected),
            metrics=metrics([r for r in selected if r['duration']==name]))
            for name in sorted({r['duration'] for r in selected})},
        signal_absolute_mean=dict(training=signal(training),heldout=signal(selected)),
        last_pair_only_same_actions=all(
            (next(r for r in training if r['seed']==row['seed'] and r['scenario']==row['scenario']
                  and r['line_id']==row['line_id'] and r['window_name']=='post2')['truth']['fixed300']['gain'][3]>0)
            ==(row['effects']['route']['predicted_cost_gain']>0) for row in selected))
    path=HERE/'analysis.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(bytes=path.stat().st_size,overall=result['overall'],
        strata=groups,freshness=clocks,signal=result['signal_absolute_mean'],
        durations=result['heldout_durations'],last_pair_only_same_actions=result['last_pair_only_same_actions']),ensure_ascii=False))


if __name__=='__main__':main()
