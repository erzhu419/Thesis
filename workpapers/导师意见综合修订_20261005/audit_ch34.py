from pathlib import Path
import json,csv,re,statistics as S, math,zipfile,xml.etree.ElementTree as E
D=Path(__file__).resolve().parent; R=D.parents[1]
def read(p):return json.loads(Path(p).read_text())
# Read the saved manuscript again, independent of any previous extraction.
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
with zipfile.ZipFile(R/'output/排版完成_导师意见复核_20261005.docx') as z:root=E.fromstring(z.read('word/document.xml'))
def tx(e):return ''.join(e.itertext()) if False else ''.join(t.text or '' for t in e.iter() if t.tag in {f'{{{ns["w"]}}}t',f'{{{ns["m"]}}}t'})
tables={};cap=None
for e in root.find('w:body',ns):
 if e.tag.endswith('}p'):
  m=re.match(r'表([34]-\d+)\s',tx(e))
  if m:cap=m[1]
 elif e.tag.endswith('}tbl') and cap:
  tables[cap]=[[tx(c) for c in row.findall('w:tc',ns)] for row in e.findall('w:tr',ns)];cap=None
checks=[];raw=[]
def check(t,r,c,v,src):
 txt=tables[t][r][c];numbers=re.findall(r'[-+]?\d+(?:\.\d+)?',txt.replace('−','-').replace(',',''))
 vals=v if isinstance(v,list) else [v]
 ok=len(numbers)==len(vals) and all(abs(float(a)-b)<=.500001*10**(-len(a.split('.')[1]) if '.' in a else 0) for a,b in zip(numbers,vals))
 checks.append(dict(table=t,row=r,column=c,shown=txt,expected=vals,ok=ok,source=str(src)))
def verify(label,a,b,src,tol=1e-7):raw.append(dict(label=label,ok=abs(a-b)<=tol,actual=a,expected=b,source=str(src)))
F=R/'manuscript/figures/source_data'
f=F/'resac_archive_evaluation.json';data=read(f)
for i,x in enumerate(data['rows'],1):
 for j,v in enumerate([x['archive_train']/1000,[x['extension_train'][k]/1000 for k in ['mean','sd']],x['archive_operational']['reward']/1000,[x['extension_operational']['reward'][k]/1000 for k in ['mean','sd']]],1):check('3-1',i,j,v,f)
 for k in ['extension_train','extension_operational']:
  v=x[k] if k=='extension_train' else x[k]['reward']
  verify('3-1 '+x['variant']+' '+k+' sample sd',S.stdev(v['values']),v['sd'],f)
# Historical rule table: exact reported source, no retained per-run raw file found.
f=R.parent/'RE-SAC/paper/paper.tex'
for i,vs in enumerate([[3.76,3.62],[10.93,12.19],[127.1,59.2],[13.4,.7]],1):
 for j,v in enumerate(vs,1):check('3-2',i,j,v,f)
# Independent recorded diagnostic completions in the experiment protocol.
f=R.parent/'RE-SAC/jax_experiments/experiments/UNCERTAINTY_PROTOCOL.md'
for i,vs in enumerate([[-.379127,-.343582,-10.742344,-.431821],[-.840540,.149828,-5.004307,-5.343469],[.252354,.727809,.880217,9.937654]],1):
 for j,v in enumerate(vs,1):check('3-3',i,j,v,f)
f=F/'bus_supplement_summary.json';sup=read(f)
for i,x in enumerate(sup['representation']['means'],1):
 for j,v in enumerate([x['reward']/1000,x['cost'],x['completion_percent']],2):check('3-4',i,j,v,f)
f=F/'reward_ablation_summary.json';d=read(f)
for i,(s,a) in enumerate([(s,a) for s in ['normal','fast_switch'] for a in ['ridge','exp60','exp60_hold','exp1_hold']],1):
 x=next(x for x in d['means'] if x['scene']==s and x['method']==a)
 for j,k in enumerate(['common_return_K','imbalance_seconds','hold_seconds','cost_seconds'],2):check('3-5',i,j,x[k],f)
f=R/'experiments/enhancement_v1/summary.json';en=read(f)
for t,arms,metrics in [('3-6',['learned_full','masked_full','onehot_full'],['cost','headway_imbalance']),('3-7',['learned_full','critic_off','actor_off','both_off'],['reward','headway_imbalance'])]:
 for i,(s,a) in enumerate([(s,a) for s in ['normal','demand_surge','fast_switch'] for a in arms],1):
  for j,k in enumerate(metrics,2):check(t,i,j,en['legacy']['results'][s]['means'][a][k]['mean']/(1000 if k=='reward' else 1),f)
for i,a in enumerate(['nau_bc0','nau_bc1','mlp_bc0','mlp_bc1'],1):
 for j,s in enumerate(['normal','single5','repeated5'],1):check('4-9',i,j,en['cs']['results'][s]['means'][a]['reward']['mean']/1000,f)
f=R/'experiments/review_round3/completed_operational_summary.json'
for i,x in enumerate(read(f)['reward']['rows'],1):
 for j,k in enumerate(['front_plan_absolute_seconds','back_plan_absolute_seconds','trip_duration_seconds','generalized_seconds'],2):check('3-8',i,j,x['metrics'][k],f)
f=R/'experiments/review_round3/learning_summary.json';ens=read(f)['ensemble']
for i,(s,k) in enumerate([(s,k) for s in ['normal','fast_switch'] for k in [2,5,10,20]],1):
 x=next(x for x in ens['groups'] if x['scene']==s and x['size']==k)
 for j,key in enumerate(['front','back','time'],2):check('3-9',i,j,x[key]['mean'],f)
f=R/'experiments/bus_legacy_dual_v1/summary.json';dual=read(f)['seed_means']
for i,(o,s) in enumerate([(o,s) for o in ['headway','passenger'] for s in ['normal','demand_surge','fast_switch']],1):
 k='plan_error_seconds' if o=='headway' else 'cost_seconds'
 vals=[S.mean(x[k] for x in dual if x['kind']=='resac' and x['mode']=='holding' and x['objective']==o and x['scene']==s and x['arm']==a) for a in ['ensemble_sac_control','resac']]
 for j,v in enumerate(vals+[100*(1-vals[1]/vals[0])],3):check('3-10',i,j,v,f)
f=F/'csbapr_reported_tables.csv';old=list(csv.DictReader(f.open()))
methods=['bapr-pristine','bapr','csbapr-no-nau','csbapr','csbapr-tanh','csbapr-kan']
for table,src,metrics in [('4-2','1',['ID total return mean','ID total return sd']),('4-3','2',[f'm={m} return median' for m in [1,2,5,10,20,50]]),('4-4','3',[f'burst {m}x return median' for m in [5,10,20,50]]),('4-5','4',[m+' return median' for m in ['commuter_day','square_wave_20x','escalating_10x']])]:
 for i,a in enumerate(methods,1):
  for j,k in enumerate(metrics,1):check(table,i,j,float(next(x for x in old if x['table']==src and x['method']==a and x['metric']==k)['value']),f)
f=F/'bus_supplement_summary.json'
for i,s in enumerate(['normal','congestion_severe','demand_surge','slow_switch','fast_switch'],1):
 for j,a in enumerate(['no_hold','headway','resac','bapr'],1):check('4-6',i,j,next(x for x in sup['legacy']['means'] if x['scene']==s and x['method']==a)['cost'],f)
for i,a in enumerate(['bapr-pristine','csbapr-no-nau','csbapr'],1):
 for j,k in enumerate(['excess_wait_passenger_hours','final_excess_waiting'],1):check('4-7',i,j,S.mean(x[k] for x in sup['recovery']['rows'] if x['method']==a)/(10000 if j==1 else 1),f)
f=F/'csbapr_repeated_summary.json';rep=read(f)
for i,s in enumerate(['normal','single_2x','repeated_2x','single_5x','repeated_5x'],1):
 for j,(k,a) in enumerate([(k,a) for k in ['reward_gain_K','waiting_change_percent'] for a in ['csbapr-no-nau','csbapr']],1):check('4-8',i,j,next(x for x in rep['summary'] if x['scene']==s and x['method']==a)[k],f)
f=R/'experiments/bus_alignment_followup_v5/summary.json';v5=read(f)['cs_results']
for i,s in enumerate(['normal','pulse5','repeated5','static5','static10','burst50'],1):
 def vs(o,a,k):return S.mean(x[k] for x in v5 if x['family']=='cs_recipe' and x['mode']=='holding' and x['objective']==o and x['scene']==s and x['arm']==a)
 p=[-vs('headway',a,'recorded_reward_per_decision') for a in ['pristine_mlp','stabilized_mlp']]
 vals=p+[100*(1-p[1]/p[0]),100*(vs('headway','stabilized_mlp','plan_mae')/vs('headway','pristine_mlp','plan_mae')-1)]
 for j,v in enumerate(vals,2):check('4-10',i,j,v,f)
 check('4-11',i,1,100*(vs('passenger','stabilized_mlp','cost')/vs('passenger','pristine_mlp','cost')-1),f)
# Independently reaggregate retained per-evaluation JSONs, not manuscript exports.
for family,pattern in [('legacy','experiments/enhancement_v1/runs/legacy/*/results/eval_*.json'),('cs','experiments/enhancement_v1/runs/cs/*/results/eval_*.json')]:
 records=[(p,read(p)) for p in R.glob(pattern)]
 for s,sd in en[family]['results'].items():
  for a,ad in sd['means'].items():
   for metric,summary in ad.items():
    if metric not in ['reward','cost','headway_imbalance','mean_hold']:continue
    for seed,expected in summary['by_training_seed'].items():
     selected=[(p,x) for p,x in records if x['scene']==s and x['arm']==a and str(x['training_seed'])==seed]
     if not selected:continue
     if family=='cs' and metric!='reward':continue
     def value(x):
      if metric=='reward':return x['reward']
      if metric=='cost':return (2*x['wait_passenger_seconds']+x['ride_passenger_seconds'])/x['generated_passengers']
      num='imbalance_sum' if metric=='headway_imbalance' else 'hold_sum'
      return sum(w[num] for w in x['windows'])/sum(w['decisions'] for w in x['windows'])
     verify(f'{family} {s} {a} {seed} {metric}',S.mean(value(x) for p,x in selected),expected,pattern)
# The repetition table uses per-evaluation paired percentage, then seed average.
base=R/'experiments/csbapr_bus/runs/repeated_v1'
for x in rep['training_seed_effects']:
 vals=[]
 for ev in [17301,17302,17303]:
  a=x['method'];seed=x['training_seed'];s=x['scene'];y=read(base/f'{a}_seed{seed}/{a}_seed{seed}_{s}_eval{ev}.json');b=read(base/f'bapr-pristine_seed{seed}/bapr-pristine_seed{seed}_{s}_eval{ev}.json')
  vals.append([(y['reward']-b['reward'])/1000,100*(y['wait_passenger_seconds']/b['wait_passenger_seconds']-1)])
 for i,k in enumerate(['reward_gain_K','waiting_change_percent']):verify(f'4-8 {s} {a} {seed} {k}',S.mean(z[i] for z in vals),x[k],str(base))
# Independent paired t intervals (df=4) for the factorial study.
ci_count=0
for family in ['legacy','cs']:
 for s,v in en[family]['results'].items():
  for contrast,metrics in v.get('paired_differences',{}).items():
   for metric,stats in metrics.items():
    ys=list(stats['by_training_seed'].values());mu=S.mean(ys);half=2.7764451051977987*S.stdev(ys)/math.sqrt(len(ys))
    verify(f'CI {family}/{s}/{contrast}/{metric} lower',mu-half,stats['ci95'][0],R/'experiments/enhancement_v1/summary.json',1e-3);ci_count+=1
# Other Chapter 3/4 primary tables: recompute directly from retained evaluation records.
for name,table,summary_key in [('representation_v1','3-4','representation'),('paired_v1','4-6','legacy')]:
 base=R/'experiments/legacy_bus/runs'/name
 records=[]
 for p in base.rglob('*.json'):
  if p.stat().st_size>1000000:continue
  x=read(p)
  if isinstance(x,dict) and all(k in x for k in ['scene','generated_passengers','reward','method']):records.append(x)
 for x in sup[summary_key]['means']:
  rs=[r for r in records if r['method']==x['method'] and r['scene']==x['scene']]
  for k in ['reward','cost','completion_percent']:
   if k not in x:continue
   def val(r):return r['reward'] if k=='reward' else ((2*r['wait_passenger_seconds']+r['ride_passenger_seconds'])/r['generated_passengers'] if k=='cost' else 100*r['arrived_passengers']/r['generated_passengers'])
   verify(f'{table} raw {x["method"]} {x["scene"]} {k}',S.mean(val(r) for r in rs),x[k],base)
base=R/'experiments/legacy_bus/runs/reward_v1'
records=[read(p) for p in base.glob('*/results/*_seed1500?.json')]
for x in read(F/'reward_ablation_summary.json')['means']:
 rs=[r for r in records if r['method']==x['method'] and r['scene']==x['scene']]
 for k in ['common_return_K','cost_seconds','imbalance_seconds','hold_seconds']:
  def val(r):
   if k=='common_return_K':return r['reward']/1000
   if k=='cost_seconds':return (2*r['wait_passenger_seconds']+r['ride_passenger_seconds'])/r['generated_passengers']
   return sum(w['imbalance_sum' if k=='imbalance_seconds' else 'hold_sum'] for w in r['windows'])/sum(w['decisions'] for w in r['windows'])
  verify(f'3-5 raw {x["method"]} {x["scene"]} {k}',S.mean(val(r) for r in rs),x[k],base)
base=R/'experiments/review_round3/runs/ensemble'
records=[read(p) for p in base.glob('*/results/eval*.json')]
for x in ens['cells']:
 rs=[r for r in records if r['scene']==x['scene'] and r['training_seed']==x['seed'] and r['ensemble_size']==x['size']]
 for k in ['time','front','back']:
  def val(r):return r['generalized_seconds_per_passenger'] if k=='time' else r['operational'][k+'_plan_absolute_error_seconds']['mean']
  verify(f'3-9 raw K{x["size"]} seed{x["seed"]} {x["scene"]} {k}',S.mean(val(r) for r in rs),x[k],base)
base=R.parent/'BAPR/reproducibility/v32';b=read(base/'analysis.json');pairs=[]
for seed in b['training_seeds']:
 a=read(base/f'audits/seed_{seed}/audit.json')['switching_holdout']
 pair=[S.mean(e[k]['return_mean'] for e in a.values()) for k in ['resac_b0_8p4m','canonical_v5_map_compensation']];pairs.append(pair)
for i,k in enumerate(['resac_b0_8p4m','canonical_v5_map_compensation']):verify('BAPR v32 raw '+k,S.mean(x[i] for x in pairs),b['switching_return_means'][k],base)
ys=[y-x for x,y in pairs];comp=b['comparisons']['causal_vs_equal_budget_resac'];verify('BAPR v32 paired mean',S.mean(ys),comp['paired_mean'],base)
verify('BAPR v32 paired t lower',S.mean(ys)-2.2621571628540993*S.stdev(ys)/math.sqrt(10),comp['ci95_low'],base,1e-5)

report={'manuscript':str(R/'output/排版完成_导师意见复核_20261005.docx'),'table_cells_checked':len(checks),'tables_checked':list(tables),'numeric_tables_checked':sorted(set(x['table'] for x in checks)),'numeric_mismatches':[x for x in checks if not x['ok']],'raw_reaggregation_checks':len(raw),'raw_mismatches':[x for x in raw if not x['ok']],'table_checks':checks,'raw_checks':raw}
(D/'复核_第34章.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['table_checks','raw_checks']},ensure_ascii=False,indent=2))
