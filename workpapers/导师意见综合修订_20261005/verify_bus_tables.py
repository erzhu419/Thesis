"""Compare saved Word table cells with the frozen experiment exports."""
from pathlib import Path
import json,re,statistics as S,math
from zipfile import ZipFile
from lxml import etree as E

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def read(p):return json.loads(Path(p).read_text())
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
with ZipFile(ROOT/'output/排版完成_导师意见复核_20261005.docx') as z:doc=E.fromstring(z.read('word/document.xml'))
tables={};caption=None
for el in doc.find('w:body',ns):
    tag=E.QName(el).localname
    if tag=='p':
        style=el.find('w:pPr/w:pStyle',ns)
        if style is not None and style.get('{'+ns['w']+'}val')=='tablecaption':caption=''.join(el.xpath('.//w:t/text()',namespaces=ns)).split()[0]
    elif tag=='tbl' and caption:
        tables[caption]=[[''.join(c.xpath('.//w:t/text()|.//m:t/text()',namespaces=ns)) for c in r.findall('w:tc',ns)] for r in el.findall('w:tr',ns)]
        caption=None
checks=[]
def check(table,r,c,expected,source):
    cell=tables[table][r][c].replace('−','-').replace(',','')
    nums=re.findall(r'[-+]?\d+(?:\.\d+)?',cell)
    expected=expected if isinstance(expected,(tuple,list)) else [expected]
    ok=len(nums)==len(expected) and all(abs(float(n)-e)<=.50001*10**(-len(n.split('.')[1]) if '.' in n else 0) for n,e in zip(nums,expected))
    checks.append(dict(table=table,row=r,col=c,displayed=tables[table][r][c],expected=expected,ok=ok,source=source))

# Original full-episode offline/online experiment's generated tables.
base=ROOT.parent/'offline-sumo/paper/generated'
methods=['BC','CQL','RE-SAC','OnlineSAC','WSRL','RLPD-0.50','WSRL-E10','RLPD-E10']
for fname,column,col in [('table_main_methods_best.tex',2,1),('table_main_methods_final.tex',2,2),('table_operational_metrics.tex',1,1),('table_operational_metrics.tex',3,2)]:
    lines=(base/fname).read_text().splitlines()
    for i,name in enumerate(methods,1):
        line=next(s for s in lines if s.startswith(name+' &'))
        values=[float(n) for n in re.findall(r'-?\d+(?:\.\d+)?',line.split('&')[column].replace(',',''))]
        check('表5-2' if fname=='table_operational_metrics.tex' else '表5-1',i,col,values,str(base/fname))

f=ROOT.parent/'sumo-rl/H2Oplus/reproducibility/h2oplus_v20_formal_confirmation_results/formal_confirmation_analysis_v1.json'
formal=read(f)['formal_confirmation_scientific_result'];means=formal['algorithm_means']
for c,key in enumerate([formal['direct_behavior_parent'],formal['candidate']],1):
    for r,metric in enumerate(['primary','completion','unfinished_passengers'],1):check('表5-3',r,c,means[key][metric],str(f))
for i,scale in enumerate(['0.75','1.0','1.25'],1):
    d=formal['candidate_vs_direct_parent_by_demand_scale'][scale]
    for c,key in enumerate(['primary_change_seconds','completion_change'],1):check('表5-4',i,c,d[key],str(f))
arms=[formal['candidate'],formal['direct_behavior_parent'],'h2oplus-v14-terminal-quarter','h2oplus','sac-online','h2oplus-v10-delayed-actor','h2oplus-v11-actor-hold','h2oplus-v12-actor-ramp','h2oplus-v13-proximal-strong']
for i,a in enumerate(arms,1):
    for c,k in enumerate(['primary','completion'],1):check('表5-5',i,c,means[a][k],str(f))

f=ROOT/'experiments/ratio_split_v2/summary.json';data=read(f);mc=data['mc']
for i,a in enumerate(['source_only','offline_only','mixed_nomodel'],1):
    for c,d in enumerate(['demand1','demand1.25'],1):check('表5-9',i,c,mc['results'][d]['means'][a]['cost']['mean'],str(f))
for i,a in enumerate(['mixed_nomodel','fixed_direct','fixed_residual','calibrated_selection'],1):
    for c,d in enumerate(['demand1','demand1.25'],1):check('表6-6',i,c,mc['results'][d]['means'][a]['cost']['mean'],str(f))
    check('表6-6',i,3,S.mean(r['training_wall_seconds'] for r in mc['costs'] if r['arm']==a)/60,str(f))
keys=['joint_auc','state_action_auc','weight_mean','raw_ess_fraction','ess_fraction','high_clip_fraction']
for i,r in enumerate(data['ratio']['cells'],1):
    for c,k in enumerate(keys,1):check('表5-12',i,c,r[k]*(100 if c==6 else 1),str(f))
for c,k in enumerate(keys,1):check('表5-12',6,c,S.mean(r[k] for r in data['ratio']['cells'])*(100 if c==6 else 1),str(f))

f=ROOT/'experiments/mcwm_alignment_v2/summary.json';d=read(f)['results']
for i,(scene,demand) in enumerate([(s,v) for s in ['calibrated','joint25','joint50'] for v in ['1.0','1.25']],1):
    r=d[scene][demand]
    values=[r['h2o_mean'],r['mcwm_mean'],100*(r['h2o_mean']-r['mcwm_mean'])/r['h2o_mean'],100*S.mean(s['mcwm']['completion']-s['h2o_matched']['completion'] for s in r['by_seed'])]
    for c,v in enumerate(values,2):check('表6-7',i,c,v,str(f))

f=ROOT/'experiments/review_round3/learning_summary.json';d=read(f)['mcwm']
for i,(scene,demand) in enumerate([(s,v) for s in ['calibrated','joint25','joint50'] for v in [1.,1.25]],1):
    rows={r['arm']:r for r in d['groups'] if r['scenario']==scene and r['demand']==demand}
    for c,a in enumerate(['h2o_matched','scaled_nomodel','mcwm'],2):check('表6-8',i,c,rows[a]['time']['mean'],str(f))
    check('表6-8',i,5,rows['mcwm']['completion']['mean']-rows['h2o_matched']['completion']['mean'],str(f))
for c,a in enumerate(['h2o_matched','scaled_nomodel','mcwm'],1):
    rows=[r for r in d['training'] if r['arm']==a]
    for i,k in enumerate(['source','policy','refresh','model_batch','trust','diagnostic','other'],1):check('表6-9',i,c,S.mean(r['exclusive_seconds'][k] for r in rows),str(f))
    for i,k in enumerate(['minutes','gpu_mb','cpu_actor_ms'],8):check('表6-9',i,c,S.mean(r[k] for r in rows),str(f))

f=ROOT/'experiments/mcwm_usage_v3/analysis.json';d=read(f)['cells']
arms=['h2o_matched','scaled_nomodel','mcwm','normalized','fixed_half','source_repair','offline_only']
for i,a in enumerate(arms,1):
    for c,(s,v) in enumerate([(s,v) for s in ['calibrated','joint50','joint75'] for v in [1.,1.25]],1):
        rows=[r for r in d if r['arm']==a and r['scenario']==s and r['demand_scale']==v]
        assert len(rows)==1
        check('表6-10',i,c,rows[0]['cost'],str(f))
rules=read(ROOT/'experiments/mcwm_usage_v3/rule_analysis.json')['baselines']
for i,(s,v) in enumerate([(s,v) for s in ['calibrated','joint50','joint75'] for v in [1.,1.25]],1):
    rule=next(r for r in rules if r['method']=='headway' and r['demand_scale']==v)
    h=next(r for r in d if r['arm']=='h2o_matched' and r['scenario']==s and r['demand_scale']==v)
    m=next(r for r in d if r['arm']=='normalized' and r['scenario']==s and r['demand_scale']==v)
    for c,num in enumerate([rule['cost'],h['cost'],m['cost'],100*(rule['cost']-m['cost'])/rule['cost'],m['completion_percent']-rule['completion_percent']],2):check('表6-11',i,c,num,str(f)+' + rule_analysis.json')

f=ROOT/'experiments/mcwm_seed_confirmation_v13/completed_analysis.json';d=read(f)
for i,a in enumerate(['h2o_matched','fixed0250','fixed0300','shared_gated'],1):
    for c,s in enumerate(['calibrated','joint50','joint75'],1):
        r=next(r for r in d['all_seed_metrics'] if r['arm']==a and r['scenario']==s)['metrics']['cost_seconds']
        check('表6-12',i,c,[r['mean'],r['sd']],str(f))
for i,s in enumerate(['calibrated','joint50','joint75'],1):
    r=next(r for r in d['comparisons'] if r['arm']=='fixed0250' and r['reference']=='h2o_matched' and r['scenario']==s)
    check('表6-13',i,1,r['cost_delta_seconds']['mean'],str(f));check('表6-13',i,2,r['cost_delta_seconds']['ci95'],str(f));check('表6-13',i,3,[r['better_seeds'],20],str(f))
    r=next(r for r in d['comparisons'] if r['arm']=='fixed0250' and r['reference']=='daganzo_duet' and r['scenario']==s)
    check('表6-13',i,4,r['cost_delta_seconds']['mean'],str(f));check('表6-13',i,5,[r['better_seeds'],20],str(f))
    for c,a in enumerate(['h2o_matched','fixed0250','fixed0300','shared_gated'],1):
        r=next(r for r in d['selected_metrics'] if r['arm']==a and r['scenario']==s)
        check('表6-14',i,c,r['test']['cost_seconds'],str(f))
f=ROOT/'experiments/mcwm_context_last_v14/completed_analysis.json';d=read(f)
for i,a in enumerate(['shared_gated','hysteresis0300','gradient_gated'],1):
    for c,s in enumerate(['calibrated','joint50','joint75'],1):
        r=next(r for r in d['comparisons'] if r['arm']==a and r['reference']=='fixed0250' and r['scenario']==s)
        check('表6-15',i,c,r['cost_delta_seconds']['mean'],str(f))

f=ROOT/'manuscript/figures/source_data/h2o_service_decomposition.json';d=read(f)['summaries']
for i,s in enumerate(['0.75','1','1.25','all'],1):
    for c,k in enumerate(['wait','in_vehicle','generalized'],1):check('表5-6',i,c,d[s]['difference'][k],str(f))
f=ROOT/'experiments/sumo_eval/attribution_v1/summary.json';d=read(f)['comparisons']
for i,a in enumerate(['offline','source','zero','headway','welfare'],1):
    for c,l in enumerate(['no_floor','late20','early60'],1):
        r=next(r for r in d if r['name']==f'hybrid-minus-{a}/{l}' and r['demand_scale']=='all_equal_weight')['metrics']
        check('表5-7',i,c,r['cost_seconds']['mean_difference'],str(f))
        if l=='early60':check('表5-7',i,4,r['completion_percentage_points']['mean_difference'],str(f))
f=ROOT/'experiments/review_round3/completed_operational_summary.json';d=read(f)['guard']['rows']
for i,a in enumerate(['hybrid','offline','source','zero'],1):
    r=next(r for r in d if r['method']==a)['groups']['all_departed']
    for c,k in enumerate(['wait_seconds','ride_seconds','generalized_seconds','completed_only_with_guard','completed_only_without_guard'],1):check('表5-10',i,c,r[k],str(f))
f=ROOT/'experiments/review_round3/historical_cost_summary.json';d=read(f)['target_visits']
groups=[[k for k in d if any(k.startswith('h2oplus_v'+str(v)+'_') for v in range(8,15))],
        [k for k in d if any(k.startswith('h2oplus_v'+str(v)+'_') for v in range(15,20))],
        ['h2oplus_v20_full_control_results'],['h2oplus_v20_formal_confirmation_results'],list(d)]
for i,group in enumerate(groups,1):
    for c,k in enumerate(['retained_episodes','measured_simulated_hours','measured_evaluation_hours'],1):check('表5-11',i,c,sum(d[g][k] for g in group),str(f))

f=ROOT/'experiments/mcwm_headway_holding_v19/analysis_summary.json';mc=read(f)['aggregates']
base=read(ROOT/'experiments/mcwm_action_matrix_v18/analysis_summary.json')['aggregates']
completion=read(HERE/'v19_completion_remote.json')
for i,(s,l,a) in enumerate([(s,l,a) for s in ['calibrated','joint75'] for l in ['guarded','direct'] for a in ['h2oplus','mcwm']],1):
    rows=[r for r in (mc if a=='mcwm' else base) if r['scene']==s and r['layer']==l and (a=='mcwm' or (r.get('arm')=='h2oplus' and r.get('objective')=='headway'))]
    assert len(rows)==2,(a,s,l,len(rows),base[0])
    for c,k in enumerate(['cost','reward','plan','imbalance'],2):check('表6-16',i,c,S.mean(r[k] for r in rows),str(f)+' + v18 baseline')
    rows=[r for r in completion if r['scene']==s and r['layer']==l and r['arm']==a]
    check('表6-16',i,6,100*S.mean(r['completion'] for r in rows),str(HERE/'v19_completion_remote.json'))

report={'checked_cells':len(checks),'tables':sorted(set(r['table'] for r in checks)),'mismatches':[r for r in checks if not r['ok']],'checks':checks}
(HERE/'复核_公交表格数值.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='checks'},ensure_ascii=False,indent=2))
