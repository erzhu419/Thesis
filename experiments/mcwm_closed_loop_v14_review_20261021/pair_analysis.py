"""Reduce the local v14 summary without changing any experiment artifact."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'mcwm_closed_loop_assignment_innovation_v14_20261021'/'innovation_summary.json'
v=json.loads(SOURCE.read_text())
MODELS=('baseline','separated','innovation','innovation_line')
ENDS=('global_next','same_vehicle_next','fixed300')

def metrics(pairs, endpoint, model, *, raw=False):
    valid=[];missing_truth=0;unsupported=0
    for pair in pairs:
        truth=pair['truth'][endpoint]['true_gain']
        row=(pair['raw_global_effects'] if raw else pair['effects'][endpoint]).get(model)
        if truth is None: missing_truth+=1
        elif row is None: unsupported+=1
        elif pair['integrity']['all_pass']: valid.append((row,truth))
    n=len(valid)
    return dict(n=n,unsupported=unsupported,missing_truth=missing_truth,
        mae=sum(abs(row[1]) for row,_ in valid)/n if n else None,
        mse=sum(row[1]**2 for row,_ in valid)/n if n else None,
        direction=sum(row[2] for row,_ in valid), ties=sum(row[0]==0 for row,_ in valid),
        action_correct=sum(row[3]==0 for row,_ in valid),
        regret=sum(row[3] for row,_ in valid))

def groups(pairs):
    return {m:metrics(pairs,'fixed300',m) for m in MODELS}

scene_line=[]
for scene in ('stable','local_speed_drop'):
    for line in ('all','7X','102X','705X'):
        pairs=[p for p in v['pairs'] if p['scenario']==scene and (line=='all' or p['line']==line)]
        scene_line.append(dict(scene=scene,line=line,models=groups(pairs)))

phase_tables=[]
for scene in ('stable','local_speed_drop'):
    for line in ('all','7X','102X','705X'):
        rows=[r for r in v['strata'] if r['role']=='heldout' and r['scenario']==scene and (line=='all' or r['line']==line)]
        endpoints={}
        for endpoint in ENDS:
            model_order=v['schema']['phase_tables']['global_model_order' if endpoint=='global_next' else 'other_endpoint_model_order']
            index=1 if endpoint=='global_next' else 3
            results={}
            for model in MODELS:
                pos=model_order.index(model)
                errors=[row['endpoints'][endpoint][2][pos] for row in rows]
                n=sum(item[0] for item in errors)
                results[model]=dict(available=sum(row['endpoints'][endpoint][1][pos] for row in rows),
                    matched_n=n,mae=sum(item[0]*item[1][index] for item in errors if item[0])/n if n else None,
                    mse=sum(item[0]*item[2][index] for item in errors if item[0])/n if n else None)
            endpoints[endpoint]=dict(rows=sum(r['n'] for r in rows), missing_truth=sum(r['endpoints'][endpoint][0] for r in rows),models=results)
        phase_tables.append(dict(scene=scene,line=line,endpoints=endpoints))

states=[]
for p in v['pairs']:
    states.append(dict(scene=p['scenario'],line=p['line'],index=p['anchor_index'],duration=p['duration'],
        true_gain=p['truth']['fixed300']['true_gain'],
        predicted_gain={m:p['effects']['fixed300'][m][0] if p['effects']['fixed300'][m] else None for m in MODELS}))

missing_global=[]
for p in v['pairs']:
    missing=[m for m in MODELS if p['effects']['global_next'].get(m) is None]
    if missing:missing_global.append(dict(scene=p['scenario'],line=p['line'],state=p['anchor_index'],models=missing))

changes={m:dict(correct_change_direction=sum(r['endpoints']['fixed300']['models'][m]['change_direction_correct'] for r in v['within_line_state_changes']),
                   predicted_flips=sum(r['endpoints']['fixed300']['models'][m]['predicted_direction_flip'] for r in v['within_line_state_changes']))
         for m in MODELS}
result=dict(source_bytes=SOURCE.stat().st_size,
    gain_sign=v['schema']['gain_sign'],
    overall={endpoint:{m:metrics(v['pairs'],endpoint,m) for m in MODELS} for endpoint in ENDS},
    by_scene_line_300=scene_line,
    by_state_index_300={str(index):groups([p for p in v['pairs'] if p['anchor_index']==index]) for index in (0,1)},
    state_gains=states,
    heldout_cost_errors=phase_tables,
    global_support_missing=missing_global,
    duration_groups={name:dict(n=sum(p['duration']==name for p in v['pairs']),
         decoded={m:metrics([p for p in v['pairs'] if p['duration']==name],'global_next',m) for m in MODELS},
         raw={m:metrics([p for p in v['pairs'] if p['duration']==name],'global_next',m,raw=True) for m in MODELS})
         for name in ('positive_dt','zero_dt','mixed_dt','missing_global')},
    changes=dict(true_flips=sum(r['endpoints']['fixed300']['true_direction_flip'] for r in v['within_line_state_changes']),models=changes),
    fixed_weight_ablation=v['fixed_weight_ablation'],
    missing=v['missing'],
    decision=dict(independent_confirmation=False,
        reason='The same12 development states show stable300s regret690 versus v13 separated130 and physical baseline7. Shock gains improve in full9, but the joint stable no-harm gate fails.',
        next='Before more single-action regression or independent confirmation, design training-period same-prefix low/high paired labels with separate evaluation states and explicit label/native-branch budget.'))
for name, entry in result['duration_groups'].items():
    selected=[p for p in v['pairs'] if p['duration']==name]
    for raw in (False,True):
        matched=[p for p in selected if all((p['raw_global_effects'] if raw else p['effects']['global_next']).get(m) is not None for m in MODELS)]
        entry['matched_raw' if raw else 'matched_decoded']={m:metrics(matched,'global_next',m,raw=raw) for m in MODELS}
(HERE/'pair_metrics.json').write_text(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n')
for row in scene_line:print(row['scene'],row['line'], {m:(d['direction'],d['regret'],round(d['mae'],1)) for m,d in row['models'].items()})
print('heldout totals')
for row in phase_tables:
    if row['line']=='all':print(row)
print('bytes', (HERE/'pair_metrics.json').stat().st_size)
