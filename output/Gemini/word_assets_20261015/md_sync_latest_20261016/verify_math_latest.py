"""Read-only display/inline inventory and final native mathematical-content comparison."""
from pathlib import Path
from zipfile import ZipFile
import sys,importlib.util,json,re,difflib
from lxml import etree as E
import pypandoc
T=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('settled_math',T.parent/'exact_library_v2_20261016/prepare_math_verification.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
import native_helpers as n
obj=lambda x:json.loads(json.dumps(x,ensure_ascii=False))
blanknorm=lambda s:re.sub(r'\s+','',s)

def inline_tex(path):
 ast=json.loads(pypandoc.convert_text(path.read_text(),'json',format='markdown+tex_math_dollars-smart'))
 def walk(x):
  if isinstance(x,dict):
   if x.get('t')=='Math' and x['c'][0]['t']=='InlineMath':yield x['c'][1]
   else:
    for v in x.values():yield from walk(v)
  elif isinstance(x,list):
   for v in x:yield from walk(v)
 return list(walk(ast))

def donor_rows(path):
 with ZipFile(path)as z:r=E.fromstring(z.read('word/document.xml'))
 by={};current=None
 for p in r.find(h.W+'body'):
  m=re.fullmatch(r'EXPECT_SLOT_(\d+)',h.text(p))
  if m:current=int(m[1]);by[current]=[]
  elif current is not None:by[current].extend(list(p.iter(h.M+'oMath')))
 return by

old_md=h.md_entries(T/'上次MD.md');new_md=h.md_entries(T/'最新MD.md');old_rows=h.word_rows(T/'修改前.docx')
assert len(old_md)==len(new_md)==len(old_rows)==376
assert [e['label']for e in old_md]==[e['label']for e in new_md]
changes=json.loads((T/'changed_math_inventory.json').read_text());donor=donor_rows(T/'math_changed_donor.docx');assert len(changes)==len(donor)==96
prepared=[]
for e in changes:
 i=e['ordinal'];assert e['tex']==new_md[i-1]['tex']
 content=obj(h.content(donor[i],e['tex']));previous=obj(h.content(old_rows[i-1]['maths'],old_md[i-1]['tex']))
 prepared.append({**e,'whitespace_only':blanknorm(e['old_tex'])==blanknorm(e['tex']),'expected_content':content,'old_content_equivalent':content==previous})
old_inline=inline_tex(T/'上次MD.md');new_inline=inline_tex(T/'最新MD.md');sm=difflib.SequenceMatcher(a=[blanknorm(t)for t in old_inline],b=[blanknorm(t)for t in new_inline],autojunk=False)
inline_changes=[];blank=[]
for mode,a,b,c,d in sm.get_opcodes():
 if mode=='equal':
  blank.extend([{'old_ordinal':i+1,'new_ordinal':j+1,'old':old_inline[i],'new':new_inline[j]}for i,j in zip(range(a,b),range(c,d))if old_inline[i]!=new_inline[j]])
 else:inline_changes.append({'kind':mode,'old_range':[a,b],'new_range':[c,d],'old':old_inline[a:b],'new':new_inline[c:d]})
prep={'display_count':376,'display_changed':96,'display_whitespace_only':sum(e['whitespace_only']for e in prepared),'display_old_content_equivalent_labels':[e['label']for e in prepared if e['old_content_equivalent']],'equivalence_scope':'canonical数学内容等价；不以此覆盖最新MD对mathrm等字体/对齐格式的明确变化。','inline_old':len(old_inline),'inline_new':len(new_inline),'inline_changed_groups':len(inline_changes),'inline_old_affected':sum(e['old_range'][1]-e['old_range'][0]for e in inline_changes),'inline_new_affected':sum(e['new_range'][1]-e['new_range'][0]for e in inline_changes),'inline_whitespace_only':blank,'donors_reused':['math_changed_donor.docx','word_sync_after_native.docx'],'conversion_repeated':False}
(T/'math_latest_preparation.json').write_text(json.dumps(prep,ensure_ascii=False,indent=2)+'\n')
(T/'math_latest_display_expected.json').write_text(json.dumps(prepared,ensure_ascii=False,indent=2)+'\n')
(T/'math_latest_inline_changes.json').write_text(json.dumps(inline_changes,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(prep,ensure_ascii=False))
if len(sys.argv)<2:sys.exit(0)
final=Path(sys.argv[1]);actual=h.word_rows(final);expected={e['ordinal']:e for e in prepared};diffs=[];passed=0
if len(actual)!=376:diffs.append({'kind':'display_count','expected':376,'actual':len(actual)})
for i,(old,new,orig,row)in enumerate(zip(old_md,new_md,old_rows,actual),1):
 if row['label']!=new['label']:diffs.append({'kind':'tag','ordinal':i,'expected':new['label'],'actual':row['label']})
 left=expected[i]['expected_content'] if i in expected else obj(h.content(orig['maths'],new['tex']))
 right=obj(h.content(row['maths'],new['tex']))
 if left==right:passed+=1
 else:diffs.append({'kind':'display_math','ordinal':i,'label':new['label'],'tex':new['tex'],'expected_canonical':left,'actual_canonical':right})
source_core=n.core(n.read(T/'word_sync_after_native.docx'));positions=json.loads((T/'word_sync_final_flow_positions.json').read_text());word=list(n.read(final).find(h.W+'body'));inline_passed=0;inline_expected=0;blocks=0
for i,e in enumerate(source_core):
 source=list(e.iter(h.M+'oMath'))
 if not source:continue
 blocks+=1;inline_expected+=len(source);pos=positions[str(i)];row=word[pos]
 if n.norm(n.text(e))!=n.norm(n.text(row)):diffs.append({'kind':'inline_block_anchor','source_block':i,'word_block':pos});continue
 actual_inline=list(row.iter(h.M+'oMath'))
 if len(source)!=len(actual_inline):diffs.append({'kind':'inline_math_count','source_block':i,'expected':len(source),'actual':len(actual_inline)})
 for j,(a,b)in enumerate(zip(source,actual_inline),1):
  left=obj(h.content([a]));right=obj(h.content([b]))
  if left==right:inline_passed+=1
  else:diffs.append({'kind':'inline_math','source_block':i,'within_block':j,'source_text':h.text(a),'actual_text':h.text(b),'expected_canonical':left,'actual_canonical':right})
blank_formula='R(0)=0,R(1)=1'
a=[m for r in old_rows for m in r['maths']if h.clean(h.text(m))==blank_formula]
# Its native objects lie in prose, outside numbered formula rows.
orig_root=n.read(T/'修改前.docx');a=[m for m in orig_root.iter(h.M+'oMath')if h.clean(h.text(m))==blank_formula];final_root=n.read(final);b=[m for m in final_root.iter(h.M+'oMath')if h.clean(h.text(m))==blank_formula]
blank_preserved=len(a)==len(b)==1 and n.mathkey(a[0])==n.mathkey(b[0])
report={'final':str(final),'all_display_math_matches':passed==376 and len(actual)==376,'display_donor_checks':96,'unchanged_display_checks':280,'display_passed':passed,'inline_source_blocks_checked':blocks,'inline_expected':inline_expected,'inline_passed':inline_passed,'whitespace_only_formula_preserved':blank_preserved,'A3_C9':[{'label':r['label'],'oMath_count':len(r['maths'])}for r in actual if r['label']in('A-3','C-9')],'correct':passed==376 and inline_passed==inline_expected and not diffs and blank_preserved,'differences':diffs}
(T/'qa_math_latest_native.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='differences'},ensure_ascii=False));print('DIFFERENCES',json.dumps(diffs,ensure_ascii=False)[:12000])
