"""Read only v14 compact output; attribute fixed-fit direct cold RHS contributions."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / 'mcwm_closed_loop_assignment_innovation_v14_20261021'
v = json.loads((ROOT / 'innovation_summary.json').read_text())
lines = ['7X', '102X', '705X']
state_order = [name.removeprefix('holding_x_') for name in v['schema']['innovation']['action_feature_order'][3:]]
cells = {(c['seed'], c['scenario']): c for c in v['cells']}
rows, checks, day = [], [], []


def stats(items, key):
    selected = [r for r in items if r[key] is not None]
    n = len(selected)
    return dict(n=n, mae=sum(abs(r[key] - r['truth']) for r in selected)/n if n else None,
        direction_correct=sum((r[key] > 0)-(r[key] < 0) == (r['truth'] > 0)-(r['truth'] < 0) for r in selected),
        regret=sum(max(0., -r['truth']) if r[key] > 0 else max(0., r['truth']) for r in selected),
        positive=sum(r[key] > 0 for r in selected))


for c in v['cells']:
    d = dict(seed=c['seed'], scenario=c['scenario'], assigned=c['assignment_audit'], models={})
    for model in ('innovation', 'innovation_line'):
        audits = c[model+'_model']['innovation_diagnostics']
        d['models'][model] = {endpoint: {key: a[key] for key in ('first_rows', 'second_rows', 'first_coverage',
            'cold_start_first_rows', 'snapshot_scope_valid', 'action_design', 'innovation_error_rmse',
            'score_error_rmse', 'cold_warm_beta_reconstruction_error')} for endpoint, a in audits.items()}
    day.append(d)

for pair in v['pairs']:
    c = cells[(pair['seed'], pair['scenario'])]
    f = [-2.*int(pair['line'] == line) for line in lines] + [-2.*pair['action_diagnostics']['state_values'][name] for name in state_order]
    for model in ('innovation', 'innovation_line'):
        for endpoint in ('global_next', 'same_vehicle_next', 'fixed300'):
            effect = pair['effects'][endpoint][model]
            a = c[model+'_model']['innovation_diagnostics'][endpoint]
            diag = dict(zip(v['schema']['action_diagnostic_order'], pair['action_diagnostics']['models'][model][endpoint]))
            cold_raw = sum(x*y for x,y in zip(f, a['cold_start_action_cost_coefficient']))
            coef = [row[3] for row in a['action_coefficient_physical']]
            reconstruction = diag['baseline'] + sum(x*y for x,y in zip(f, coef)) if diag['baseline'] is not None else None
            # Global zero-duration correction removes all learned response terms.
            raw_effect = pair['raw_global_effects'][model] if endpoint == 'global_next' else effect
            cold = 0. if pair['duration'] == 'zero_dt' and endpoint == 'global_next' else (cold_raw if effect is not None else None)
            delta = reconstruction - raw_effect[0] if reconstruction is not None and raw_effect is not None else None
            if delta is not None: checks.append(abs(delta))
            rows.append(dict(seed=pair['seed'], scenario=pair['scenario'], line=pair['line'],
                state=pair['anchor_index'], duration=pair['duration'], model=model, endpoint=endpoint,
                truth=pair['truth'][endpoint]['true_gain'], predicted=effect[0] if effect is not None else None,
                cold_direct_raw=cold_raw if raw_effect is not None else None, cold_direct=cold,
                without_direct_cold=effect[0]-cold if effect is not None else None,
                full_from_summary_coef_reconstruction_error=delta))

summary=[]
for endpoint in ('global_next','same_vehicle_next','fixed300'):
 for model in ('innovation','innovation_line'):
  for scene in ('all','stable','local_speed_drop'):
   subset=[r for r in rows if r['endpoint']==endpoint and r['model']==model and (scene=='all' or r['scenario']==scene)]
   supported=[r for r in subset if r['predicted'] is not None]
   summary.append(dict(endpoint=endpoint,model=model,scenario=scene,
    full=stats(subset,'predicted'), without_direct_cold=stats(subset,'without_direct_cold'),
    cold_mean_abs=sum(abs(r['cold_direct']) for r in supported)/len(supported) if supported else None,
    cold_max_abs=max((abs(r['cold_direct']) for r in supported),default=None),
    changed_choices=sum((r['predicted']>0)!=(r['without_direct_cold']>0) for r in supported)))

output=dict(source=str(ROOT/'innovation_summary.json'),schema=dict(state_order=state_order, gain='low minus high; positive chooses high; ties low'),
    methodology='Cold direct RHS term uses the same final precision (A^TA+ridge I)^-1 times A_cold^T e_cold. Removing it is a fixed-fit decomposition, not refitting without cold rows. Later warm residuals may also depend on background models trained using that first label.',
    max_compact_coefficient_reconstruction_error=max(checks), original_cold_warm_beta_reconstruction_error_max=max(
        a['cold_warm_beta_reconstruction_error'] for c in v['cells'] for key in ('innovation_model','innovation_line_model')
        for a in c[key]['innovation_diagnostics'].values()),
    days=day, summary=summary, row_order=list(rows[0]), rows=[[r[k] for k in rows[0]] for r in rows])
(HERE/'cold_start_metrics.json').write_text(json.dumps(output,separators=(',',':'),allow_nan=False)+'\n')
print(json.dumps(dict(bytes=(HERE/'cold_start_metrics.json').stat().st_size,
 max_reconstruction_error=output['max_compact_coefficient_reconstruction_error'],
 fixed300=[r for r in summary if r['endpoint']=='fixed300'],
 fixed300_rows=[{k:r[k] for k in ('scenario','line','state','model','truth','predicted','cold_direct','without_direct_cold')}
   for r in rows if r['endpoint']=='fixed300']),indent=2))
