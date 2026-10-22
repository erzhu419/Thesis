"""Read only old native pairs and export observable states and endpoint accounting."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import shlex
import sys

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'mcwm_closed_loop_fresh_difference_v17_20261021'
sys.path.insert(0,str(HERE))
import infra


def resident(root):
    import sys
    sys.path.insert(0,str(root.parents[1]/'experiments/sumo_eval/runtime/H2Oplus'))
    from bus_h2o.mdp_contract_v7 import RAW_EVENT_STATE_INDEX as EI
    p=json.loads((root/'protocol.json').read_text())
    issue_fields=['station_index','forward_headway_seconds','backward_headway_seconds',
        'target_forward_headway_seconds','target_backward_headway_seconds',
        'eligible_stranded_passengers','onboard_passengers','eligible_arrival_rate_per_second',
        'base_dwell_seconds','forward_bus_present','backward_bus_present','bus_capacity']
    endpoint_fields=['dt','wait','ride','waiting_change','onboard_change','global_events']
    endpoints=['global_next','same_vehicle_next','fixed300']
    rows=[];complete=missing=0;max_cost_error=max_clock_error=0.
    for seed in p['training_seeds']:
        for scene in p['scenarios']:
            day=root/'runs'/scene/('seed%d'%seed)
            summary=json.loads((day/'summary.json').read_text())
            journal=json.loads((day/'acquisitions.json').read_text())['records']
            for source in journal:
                selected=source['selected'];event=selected['event'];state=selected['s']
                slot=round(state[2]*15);station_progress=state[5+14*slot+2]
                issue=[event[EI[name]] for name in issue_fields]
                issue.extend([state[0],station_progress,selected['issue_ledger']['waiting_passengers'],
                              selected['issue_ledger']['in_vehicle_passengers']])
                branches=[];endpoint_meta=[]
                for arm in ('low','high'):
                    path=day/'training_branches'/str(source['index'])/arm if source['role']=='train' else root/'pairs'/scene/('seed%d'%seed)/source['line_id']/source['window_name']/arm
                    branch=json.loads((path/'summary.json').read_text())
                    if not branch['integrity']['all_pass']:raise RuntimeError('Old native evidence failed: '+str(path))
                    items=[];metadata=[]
                    for name in endpoints:
                        endpoint=branch['truth'][name]
                        if endpoint['status']!='complete':
                            missing+=1;items.append(None);metadata.append(dict(status=endpoint['status'],reason=endpoint.get('reason')));continue
                        complete+=1
                        max_cost_error=max(max_cost_error,abs(endpoint['cost']-2*endpoint['wait']-endpoint['ride']))
                        max_clock_error=max(max_clock_error,abs(endpoint['completed_at']-selected['started_at']-endpoint['dt']))
                        items.append([endpoint.get(key) for key in endpoint_fields])
                        metadata.append({key:endpoint[key] for key in ('next_vehicle_id','next_station_id','terminal','zero_duration','event_detected_at','event_delivered_at','delivery_lag_seconds') if key in endpoint})
                    branches.append(items);endpoint_meta.append(metadata)
                row=dict(key=[seed,scene,source['line_id'],source['role'],source['window_name']],
                    sequence=selected['sequence'],at=selected['started_at'],issue=issue,
                    features=source['features'],endpoint=branches)
                # Same-car event metadata preserve timing/service identity;
                # global vehicle IDs are available resident and not needed here.
                row['same_vehicle_meta']=[[metadata[1].get('next_station_id'),metadata[1].get('delivery_lag_seconds')]
                                          for metadata in endpoint_meta]
                rows.append(row)
    def compact(value):
        if isinstance(value,float):return int(value) if value.is_integer() else float(format(value,'.8g'))
        if isinstance(value,dict):return {k:compact(v) for k,v in value.items()}
        if isinstance(value,list):return [compact(v) for v in value]
        return value
    return compact(dict(source='Read-only v17 acquisitions and native branch summaries',
        issue_fields=issue_fields+['horizon_progress','station_progress','system_waiting_passengers','system_in_vehicle_passengers'],
        endpoint_fields=endpoint_fields,endpoint_order=endpoints,action_order=['low','high'],
        key_fields=['seed','scenario','line_id','role','window_name'],same_vehicle_meta_fields=['next_station_id','delivery_lag_seconds'],
        reconstruction=dict(cost='2*wait+ride',completed_at='issue.at+endpoint.dt',
            cost_error_max=max_cost_error,clock_error_max=max_clock_error,
            endpoint_complete=complete,endpoint_missing=missing),rows=rows))


def main():
    code='import json\nfrom pathlib import Path\n'+inspect.getsource(resident)+'''
value=resident(Path(__ROOT__))
payload=json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\\n'
if len(payload.encode())>30000:raise RuntimeError('Reduce old-pair observable summary before collection')
print(payload,end='')
'''.replace('__ROOT__',repr(infra.remote(SOURCE)))
    result=infra.scheduler.run_on('node001',shlex.join([infra.PYTHON,'-c',code]),timeout=90,check=False)
    if result[0]:raise RuntimeError(result[2][-1000:])
    value=json.loads(result[1])
    path=HERE/'existing_diagnostic.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(result[1])
    print(json.dumps(dict(bytes=path.stat().st_size,rows=len(value['rows']),reconstruction=value['reconstruction']),ensure_ascii=False))


def analyze_local():
    """Accounting at arm-specific native clocks; no same-clock causal claim."""
    data=json.loads((HERE/'existing_diagnostic.json').read_text())
    previous=json.loads((HERE.parent/'mcwm_closed_loop_v17_review_20261022/review_summary.json').read_text())
    records=[]
    for row in data['rows']:
        issue=dict(zip(data['issue_fields'],row['issue']))
        endpoint={name:{arm:row['endpoint'][a][e] for a,arm in enumerate(('low','high'))}
                  for e,name in enumerate(data['endpoint_order'])}
        def gains(name):
            lo,hi=endpoint[name]['low'],endpoint[name]['high']
            return dict(dt=lo[0]-hi[0],wait=lo[1]-hi[1],ride=lo[2]-hi[2],
                        cost=2*(lo[1]-hi[1])+lo[2]-hi[2],
                        waiting_stock=lo[3]-hi[3],onboard_stock=lo[4]-hi[4],
                        global_events=[lo[5],hi[5]])
        same=gains('same_vehicle_next');fixed=gains('fixed300')
        initial_wait=issue['system_waiting_passengers'];initial_ride=issue['system_in_vehicle_passengers']
        adjusted=dict(wait=same['wait']-initial_wait*same['dt'],
                      ride=same['ride']-initial_ride*same['dt'])
        adjusted['cost']=2*adjusted['wait']+adjusted['ride']
        after={key:fixed[key]-adjusted[key] for key in ('wait','ride','cost')}
        records.append(dict(key=row['key'],at=row['at'],issue=issue,features=row['features'],
            same_vehicle=same,same_vehicle_durations=[endpoint['same_vehicle_next'][arm][0] for arm in ('low','high')],
            fixed300=fixed,issue_rate_adjusted_same_vehicle=adjusted,after_same_vehicle_remainder=after))
    selected=[row for row in records if row['key'][3]=='heldout']
    by_key={tuple(row['key']):row for row in records}
    changes=[]
    for old in previous['heldout_pairs']:
        route,shared=old['effects'][1],old['effects'][2]
        if route is None or shared is None or (route[2]>0)==(shared[2]>0):continue
        key=[old['seed'],old['scenario'],old['line_id'],'heldout',old['window_name']]
        row=dict(by_key[tuple(key)])
        day=next(d for d in previous['days'] if d['seed']==old['seed'] and d['scenario']==old['scenario'])
        coeff=day['readouts']['shared']['coefficients'];h=row['features'][-1]
        decomposition={}
        for i,target in enumerate(('wait','ride')):
            intercept=coeff[old['line_id']+'_intercept'][target]
            term=coeff['forward_minus_backward_headway_deviation'][target]*h
            decomposition[target]=dict(route=route[i],intercept_refit=intercept-route[i],
                headway=term,shared=shared[i],reconstruction_error=intercept+term-shared[i])
        decomposition['cost']=dict(route=route[2],shared=shared[2],
            intercept_refit=2*decomposition['wait']['intercept_refit']+decomposition['ride']['intercept_refit'],
            headway=2*decomposition['wait']['headway']+decomposition['ride']['headway'])
        row['prediction_decomposition']=decomposition
        row['regret_change']=shared[8]-route[8]
        changes.append(row)
    crossings=[]
    for first in selected:
        if first['key'][-1]!='heldout1':continue
        second=by_key[tuple(first['key'][:-1]+['heldout2'])]
        a,b=first['fixed300']['cost'],second['fixed300']['cost']
        crossings.append(dict(key=first['key'][:3],cost_gain=[a,b],sign_flip=(a>0)!=(b>0),
            onboard=[r['issue']['onboard_passengers'] for r in (first,second)],
            wait_gain=[r['fixed300']['wait'] for r in (first,second)],
            ride_gain=[r['fixed300']['ride'] for r in (first,second)],
            events=[r['fixed300']['global_events'] for r in (first,second)],
            stock_changes=[{k:r['fixed300'][k] for k in ('waiting_stock','onboard_stock')} for r in (first,second)]))
    support={}
    for line in ('7X','102X','705X'):
        groups={role:[r for r in records if r['key'][2]==line and r['key'][3]==role] for role in ('train','heldout')}
        low=min(r['issue']['onboard_passengers'] for r in groups['train']);high=max(r['issue']['onboard_passengers'] for r in groups['train'])
        support[line]=dict(train_onboard_range=[low,high],heldout_onboard_range=[min(r['issue']['onboard_passengers'] for r in groups['heldout']),max(r['issue']['onboard_passengers'] for r in groups['heldout'])],
            heldout_outside_line_training_range=sum(not low<=r['issue']['onboard_passengers']<=high for r in groups['heldout']))
    value=dict(schema='v18_existing_pair_accounting',cross_zero_cases=changes,cross_window=crossings,
        cross_window_sign_flips=sum(row['sign_flip'] for row in crossings),cross_window_pairs=len(crossings),
        load_support=support,arrival_rate_phase={role:dict(n=sum(r['key'][3]==role for r in records),
            zero=sum(r['key'][3]==role and r['issue']['eligible_arrival_rate_per_second']==0 for r in records),
            queue_zero=sum(r['key'][3]==role and r['issue']['eligible_stranded_passengers']==0 for r in records)) for role in ('train','heldout')},
        same_vehicle_extra_duration_range=[min(-r['same_vehicle']['dt'] for r in records),max(-r['same_vehicle']['dt'] for r in records)],
        accounting='For each arm subtract the common issue ledger rate times its own native same-car duration; remainder completes exact300 seconds. Adjusted same-car plus remainder equals fixed300 difference.',
        causal_limit='Arm-specific same-car endpoints occur at different clocks. This is exact accounting, not a matched-clock early-versus-late causal attribution. Stocks/events show associations; full matched-clock service trajectories were not stored.',
        feature_limit='Diagnostic-station scheduled eligible arrival rate is positive at all36 train issues and zero at all24 heldout issues; this does not imply no passengers or no service network-wide.')
    path=HERE/'existing_mechanism.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(bytes=path.stat().st_size,cross_window_sign_flips=value['cross_window_sign_flips'],load_support=support,arrival_phase=value['arrival_rate_phase'],duration_range=value['same_vehicle_extra_duration_range']),ensure_ascii=False))


if __name__=='__main__':
    if '--analyze' in sys.argv:analyze_local()
    else:main()
