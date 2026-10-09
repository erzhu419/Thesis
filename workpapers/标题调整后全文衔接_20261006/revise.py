from pathlib import Path
from copy import deepcopy
from lxml import etree as E
from collections import Counter
import zipfile,json,runpy,difflib
ROOT=Path('/home/erzhu419/mine_code/Thesis'); WORK=Path(__file__).parent
SRC=WORK/'修改前.docx';OUT=ROOT/'排版完成_标题调整后全文衔接_20261006.docx'
N={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}; W='{'+N['w']+'}'
plan=runpy.run_path(str(WORK/'plan.py'));z=zipfile.ZipFile(SRC);r=E.fromstring(z.read('word/document.xml'));body=r.find('w:body',N);orig=list(body)
def txt(p):return ''.join(p.xpath('.//w:t/text()',namespaces=N))
def replace_span(p,start,end,new):
 spans=[];pos=0
 for t in p.xpath('.//w:t',namespaces=N):
  s=t.text or '';spans.append((t,pos,pos+len(s)));pos+=len(s)
 if start==end:
  for t,a,b in spans:
   if a<=start<=b and b>a:
    s=t.text or '';t.text=s[:start-a]+new+s[start-a:]
    t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');return
  raise AssertionError(('insertion',start))
 hits=[(t,a,b) for t,a,b in spans if b>start and a<end];assert hits
 for j,(t,a,b) in enumerate(hits):
  s=t.text or '';t.text=s[:max(0,start-a)]+(new if j==0 else '')+s[min(len(s),end-a):]
  t.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
changes=[]
for i,pairs in plan['EDITS'].items():
 p=orig[i];before=txt(p)
 for old,new in pairs:
  cur=txt(p)
  if old is None:
   assert not p.xpath('.//m:oMath|.//w:instrText|.//w:drawing',namespaces=N)
   updated=new
  else:
   assert cur.count(old)==1,(i,old,cur)
   updated=cur.replace(old,new,1)
  for op,a,b,c,d in reversed(difflib.SequenceMatcher(None,cur,updated,autojunk=False).get_opcodes()):
   if op!='equal':replace_span(p,a,b,updated[c:d])
  assert txt(p)==updated
 changes.append({'action':'正文改写','index':i,'before':before,'after':txt(p)})
for i,new in plan['INSERT_BEFORE'].items():
 target=orig[i];assert target.tag==W+'p'
 p=E.Element(W+'p');pp=target.find(W+'pPr')
 if pp is not None:p.append(deepcopy(pp))
 assert not p.xpath('.//w:sectPr',namespaces=N)
 run=E.SubElement(p,W+'r');rp=target.find('w:r/w:rPr',N)
 if rp is not None:run.append(deepcopy(rp))
 E.SubElement(run,W+'t').text=new
 body.insert(body.index(target),p)
 changes.append({'action':'补充衔接','index':i,'after':new})
for i in plan['REMOVE']:
 p=orig[i]
 assert not p.xpath('.//m:oMath|.//w:instrText|.//w:drawing|.//w:sectPr|.//w:bookmarkStart|.//w:bookmarkEnd|.//w:commentRangeStart|.//w:commentRangeEnd',namespaces=N)
 changes.append({'action':'融入相邻叙述','index':i,'before':txt(p)})
 body.remove(p)
for i,ids in plan['MOVE_BEFORE'].items():
 target=orig[i]
 for j in ids:
  p=orig[j];body.remove(p);body.insert(body.index(target),p)
 changes.append({'action':'调整内容归属','indexes':ids,'before_index':i})
# Untouched nodes must remain identical, including all existing formatting.
oldroot=E.fromstring(z.read('word/document.xml'));oldbody=oldroot.find('w:body',N)
for i,p in enumerate(orig):
 if i not in plan['EDITS'] and i not in plan['REMOVE']:assert E.tostring(p)==E.tostring(oldbody[i]),i
assert Counter(E.tostring(p) for p in r.xpath('//m:oMath',namespaces=N))==Counter(E.tostring(p) for p in oldroot.xpath('//m:oMath',namespaces=N))
with zipfile.ZipFile(OUT,'w') as out:
 for info in z.infolist():out.writestr(info,E.tostring(r,xml_declaration=True,encoding='UTF-8',standalone=True) if info.filename=='word/document.xml' else z.read(info.filename))
(WORK/'修改记录.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2))
(WORK/'目录更新前.docx').write_bytes(OUT.read_bytes())
print(json.dumps({'output':str(OUT),'edits':len(plan['EDITS']),'bridges':len(plan['INSERT_BEFORE']),'merged_fragments':len(plan['REMOVE']),'moved_groups':len(plan['MOVE_BEFORE'])},ensure_ascii=False))
