from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from copy import deepcopy
import json
A=Path(__file__).resolve().parent;out=A.parents[1]/'调整完格式的_公式终检_20261019.docx';source=A/'source_snapshot.docx'
N={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'};W='{'+N['w']+'}'
with ZipFile(out) as z:
 r=E.fromstring(z.read('word/document.xml'));maths=r.xpath('.//m:oMath',namespaces=N);added=[]
 for i,m in enumerate(maths,1):
  for ri,run in enumerate(m.xpath('.//m:r',namespaces=N)):
   if run.find('w:rPr/w:noProof',N) is not None:continue
   rpr=run.find('w:rPr',N);assert rpr is not None
   rpr.append(E.Element(W+'noProof'));added.append({'math_id':i,'run_index_0based':ri,'text':''.join(run.xpath('./m:t/text()',namespaces=N))})
 assert sorted(set(t['math_id'] for t in added))==list(range(675,682))
 data=E.tostring(r,encoding='UTF-8',xml_declaration=True,standalone=True)
 temp=out.with_name(out.stem+'_writing.docx')
 with ZipFile(temp,'w') as d:
  for info in z.infolist():d.writestr(info,data if info.filename=='word/document.xml' else z.read(info.filename))
temp.replace(out)
with ZipFile(source) as b,ZipFile(out) as a:
 assert b.namelist()==a.namelist()
 changed=[n for n in b.namelist() if b.read(n)!=a.read(n)];assert changed==['word/document.xml']
 rb=E.fromstring(b.read('word/document.xml'));ra=E.fromstring(a.read('word/document.xml'));bm=rb.xpath('.//m:oMath',namespaces=N);am=ra.xpath('.//m:oMath',namespaces=N)
 assert len(bm)==len(am)==4825
 diff=[i+1 for i,(x,y) in enumerate(zip(bm,am)) if E.tostring(x)!=E.tostring(y)];assert diff==[675,676,677,678,679,680,681,2241]
 for item in added:
  run=am[item['math_id']-1].xpath('.//m:r',namespaces=N)[item['run_index_0based']];no=run.find('w:rPr/w:noProof',N);no.getparent().remove(no)
 ts=am[2240].xpath('.//m:t',namespaces=N);t=next(t for t in ts if t.text==',0,+');t.text=',+'
 assert E.tostring(rb,method='c14n')==E.tostring(ra,method='c14n')
 result={'source':str(source),'output':str(out),'math_objects':4825,'changed_math_ids':diff,'proofing_runs_added':len(added),'proofing_math_ids':list(range(675,682)),'content_changed_math_ids':[2241],'changed_zip_parts':changed,'chinese_text_manual_line_breaks_and_other_format_exact':True,'all_other_parts_exact':True,'proofing_added':added}
(A/'change_scope_qa.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='proofing_added'},ensure_ascii=False))
