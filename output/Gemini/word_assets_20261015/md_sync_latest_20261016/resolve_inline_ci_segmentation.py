"""Resolve the observed split confidence-interval objects in one native data table."""
from pathlib import Path
import json,importlib.util
T=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('settled_math',T.parent/'exact_library_v2_20261016/prepare_math_verification.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
import native_helpers as n
p=T/'qa_math_latest_native.json';r=json.loads(p.read_text());source=n.core(n.read(T/'word_sync_after_native.docx'));positions=json.loads((T/'math_inline_native_positions.json').read_text());word=list(n.read(Path(r['final'])).find(h.W+'body'))
i=734;a=source[i];b=word[positions[str(i)]];ac=list(a.iter(h.W+'tc'));bc=list(b.iter(h.W+'tc'));assert len(ac)==len(bc)
obj=lambda v:json.loads(json.dumps(v,ensure_ascii=False))
seg=[]
for j,(x,y)in enumerate(zip(ac,bc)):
 xm=list(x.iter(h.M+'oMath'));ym=list(y.iter(h.M+'oMath'))
 assert obj(h.content(xm))==obj(h.content(ym)),(j,n.text(x),n.text(y))
 if len(xm)!=len(ym):seg.append({'source_block':i,'table_cell':j,'source_math_objects':len(xm),'native_math_objects':len(ym),'source_math_text':[h.text(m)for m in xm],'native_math_text':[h.text(m)for m in ym],'canonical_aggregate_identical':True})
remaining=[d for d in r['differences']if d.get('source_block')!=i]
resolved=sum(d['kind']=='inline_math'for d in r['differences']if d.get('source_block')==i)
r['inline_passed']+=resolved;r['differences']=remaining;r['equivalent_inline_segmentation']=seg;r['native_inline_math_object_count']=sum(len(list(word[j].iter(h.M+'oMath')))for j in positions.values());r['correct']=r['display_passed']==376 and r['inline_passed']==r['inline_expected'] and not remaining and r['whitespace_only_formula_preserved'];p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items()if k!='differences'},ensure_ascii=False))
