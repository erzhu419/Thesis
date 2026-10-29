"""Continue only inline checks after replacing the pre-equation body-index map."""
from pathlib import Path
import importlib.util,json,bisect
T=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('settled_math',T.parent/'exact_library_v2_20261016/prepare_math_verification.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
import native_helpers as n
p=T/'qa_math_latest_native.json';report=json.loads(p.read_text());final=Path(report['final']);source=n.core(n.read(T/'word_sync_after_native.docx'));word=list(n.read(final).find(h.W+'body'))
lookup={}
for i,e in enumerate(word):lookup.setdefault((e.tag,n.norm(n.text(e))),[]).append(i)
cursor=0;mapping={};diffs=[d for d in report['differences']if not d['kind'].startswith('inline')];passed=0;total=0;blocks=0
obj=lambda c:json.loads(json.dumps(c,ensure_ascii=False))
for i,e in enumerate(source):
 candidates=lookup.get((e.tag,n.norm(n.text(e))),[]);j=bisect.bisect_left(candidates,cursor)
 if j==len(candidates):diffs.append({'kind':'inline_source_anchor_missing','source_block':i,'text':n.text(e)[:150]});continue
 pos=candidates[j];mapping[i]=pos;cursor=pos+1
 left=list(e.iter(h.M+'oMath'))
 if not left:continue
 blocks+=1;total+=len(left);right=list(word[pos].iter(h.M+'oMath'))
 if len(left)!=len(right):diffs.append({'kind':'inline_math_count','source_block':i,'expected':len(left),'actual':len(right)})
 for j,(a,b)in enumerate(zip(left,right),1):
  ca=obj(h.content([a]));cb=obj(h.content([b]))
  if ca==cb:passed+=1
  else:diffs.append({'kind':'inline_math','source_block':i,'within_block':j,'source_text':h.text(a),'actual_text':h.text(b),'expected_canonical':ca,'actual_canonical':cb})
report.update({'inline_source_blocks_checked':blocks,'inline_expected':total,'inline_passed':passed,'inline_source_flow_anchors_matched':len(mapping),'inline_alignment':'按真实Word段落/表格文字单调对齐；不复用插入显示式表之前的body索引','differences':diffs,'correct':report['display_passed']==376 and passed==total and not diffs and report['whitespace_only_formula_preserved']})
p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');(T/'math_inline_native_positions.json').write_text(json.dumps(mapping,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items()if k!='differences'},ensure_ascii=False));print('DIFFERENCES',json.dumps(diffs,ensure_ascii=False)[:14000])
