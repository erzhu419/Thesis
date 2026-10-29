"""Prepare changed-display OMML; final comparison is a separate explicit invocation."""
from pathlib import Path
from zipfile import ZipFile
from copy import deepcopy
import ast,json,re,sys,unicodedata
from lxml import etree as E
import pypandoc
ROOT=Path('/home/erzhu419/mine_code/Thesis/output/Gemini');T=ROOT/'word_assets_20261015/exact_library_v2_20261016'
OLD=T/'修改前.md';NEW=ROOT/'重构论文_替换库二次修订_20261016.md';BASEWORD=ROOT/'重构论文_替换库修订_20261015.docx'
HELPER=ROOT/'word_assets_20261015/appendix_completion_20261016/check_math_roundtrip.py'
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}';M='{http://schemas.openxmlformats.org/officeDocument/2006/math}'
NS={'w':W[1:-1],'m':M[1:-1]}
# Reuse only the settled canonical functions; do not execute the helper's old workflow.
names={'text','clean','local','prop','sequence','scripts','canon'}
mod=ast.parse(HELPER.read_text());defs=[n for n in mod.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
exec(compile(ast.Module(body=defs,type_ignores=[]),str(HELPER),'exec'),globals())

def md_entries(path):
 s=path.read_text();out=[]
 for i,m in enumerate(re.finditer(r'\$\$(.*?)\$\$',s,re.S),1):
  tex=m[1].strip();tags=re.findall(r'\\tag\{([^}]+)\}',tex)
  assert len(tags)==1,(i,tags)
  out.append({'ordinal':i,'label':tags[0],'tex':tex,'source_line':s.count('\n',0,m.start())+1})
 return out

def word_rows(path):
 with ZipFile(path)as z:r=E.fromstring(z.read('word/document.xml'))
 rows=[]
 for bi,block in enumerate(r.find(W+'body')):
  if block.tag!=W+'tbl':continue
  for ri,row in enumerate(block.findall(W+'tr')):
   cells=row.findall(W+'tc')
   if len(cells)<2:continue
   label=re.fullmatch(r'\s*[（(]([1-7A-F]-\d+[a-z]*)[）)]\s*',text(cells[-1]))
   if not label:continue
   maths=[m for cell in cells[:-1]for m in cell.iter(M+'oMath')]
   rows.append({'ordinal':len(rows)+1,'label':label[1],'body_index':bi,'table_row':ri,'maths':maths})
 return rows

def content(nodes,tex=''):
 vals=[]
 layout=bool(re.search(r'\\begin\{(?:aligned|gathered|split|align\*?)\}',tex))
 for node in nodes:
  node=deepcopy(node)
  # Pandoc emits isolated TeX alignment markers; Word represents these with m:aln.
  # The marker is layout only inside eqArr. An ampersand elsewhere stays mathematical text.
  for eq in node.iter(M+'eqArr'):
   for t in eq.iter(M+'t'):
    if t.text=='&':t.text=''
  c=canon(node)
  # Only an outer alignment matrix is layout. Nested mathematical matrices remain intact.
  if layout and c and c[0]=='matrix':c=sequence([cell for row in c[1]for cell in row])
  vals.append(c)
 return sequence(vals)

def prepared():
 old,new=md_entries(OLD),md_entries(NEW);assert len(old)==len(new)==376
 changes=[i for i,(a,b)in enumerate(zip(old,new))if a['tex']!=b['tex']]
 extra_labels={'A-3','C-9','B-43','B-44','B-51','B-52','D-4d','D-4e','D-42','D-43','E-1','E-2','E-32','E-33','E-41','E-42'}
 selected=sorted(set(changes)|{i for i,a in enumerate(old)if a['label']in extra_labels})
 entries=[]
 for i in selected:
  entries.append({**new[i],'old_label':old[i]['label'],'changed':i in changes})
 batch=[]
 for e in entries:
  tex=re.sub(r'\\tag\{[^}]+\}','',e['tex']).strip()
  batch.extend([f'EXPECT_SLOT_{e["ordinal"]:04d}',f'$$\n{tex}\n$$'])
 batch='\n\n'.join(batch)+'\n';(T/'math_expected_selected.md').write_text(batch)
 donor=T/'math_expected_selected.docx';pypandoc.convert_text(batch,'docx',format='markdown+tex_math_dollars',outputfile=str(donor))
 with ZipFile(donor)as z:r=E.fromstring(z.read('word/document.xml'))
 by={};current=None
 for p in r.find(W+'body'):
  mark=re.fullmatch(r'EXPECT_SLOT_(\d+)',text(p))
  if mark:current=int(mark[1]);by[current]=[]
  elif current is not None:by[current].extend(list(p.iter(M+'oMath')))
 for e in entries:
  maths=by.get(e['ordinal'],[]);assert maths,(e['ordinal'],e['label'])
  e['donor_oMath_count']=len(maths);e['expected_content']=content(maths,e['tex'])
 (T/'math_expected_inventory.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2)+'\n')
 rows=word_rows(BASEWORD);assert len(rows)==376,len(rows)
 assert [r['label']for r in rows]==[e['label']for e in old]
 tables={}
 for r in rows:tables.setdefault(r['body_index'],[]).append(r['label'])
 info={'source_md':str(NEW),'baseline_md':str(OLD),'baseline_word':str(BASEWORD),'all_display_slots':376,'changed_display_slots':len(changes),'expected_donor_slots':len(entries),'extra_control_slots':[e['old_label']for e in entries if not e['changed']],'tag_differences':[{'ordinal':i+1,'old':a['label'],'new':b['label']}for i,(a,b)in enumerate(zip(old,new))if a['label']!=b['label']],'multi_label_tables':[ls for ls in tables.values()if len(ls)>1],'multi_expression_rows':[{'label':r['label'],'math_count':len(r['maths'])}for r in rows if len(r['maths'])>1],'final_word_compared':False}
 (T/'math_verification_preparation.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n');print(json.dumps(info,ensure_ascii=False))

if __name__=='__main__':prepared()
