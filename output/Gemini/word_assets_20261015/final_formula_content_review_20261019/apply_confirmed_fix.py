from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from copy import deepcopy
import json
A=Path(__file__).resolve().parent
source=A/'source_snapshot.docx';out=A.parents[1]/'调整完格式的_公式终检_20261019.docx'
N={'m':'http://schemas.openxmlformats.org/officeDocument/2006/math','w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with ZipFile(source) as z:
 old=E.fromstring(z.read('word/document.xml'));new=deepcopy(old)
 math=new.xpath('.//m:oMath',namespaces=N);target=math[2240]
 ts=target.xpath('.//m:t',namespaces=N)
 matches=[t for t in ts if t.text==',+'];assert len(matches)==1
 matches[0].text=',0,+'
 assert ''.join(target.xpath('.//m:t/text()',namespaces=N))=='Gz∈{-1,0,+1}6×6'
 doc=E.tostring(new,encoding='UTF-8',xml_declaration=True,standalone=True)
 assert not out.exists(),out
 with ZipFile(out,'w') as dest:
  for info in z.infolist():dest.writestr(info,doc if info.filename=='word/document.xml' else z.read(info.filename))
# Specific verification: only the two added math characters may differ.
with ZipFile(source) as before,ZipFile(out) as after:
 assert before.namelist()==after.namelist()
 changed=[n for n in before.namelist() if before.read(n)!=after.read(n)];assert changed==['word/document.xml'],changed
 b=E.fromstring(before.read('word/document.xml'));a=E.fromstring(after.read('word/document.xml'))
 bm=b.xpath('.//m:oMath',namespaces=N);am=a.xpath('.//m:oMath',namespaces=N);assert len(bm)==len(am)==4825
 diffs=[i+1 for i,(x,y) in enumerate(zip(bm,am)) if E.tostring(x)!=E.tostring(y)];assert diffs==[2241],diffs
 tgt=am[2240];t=[t for t in tgt.xpath('.//m:t',namespaces=N) if t.text==',0,+'];assert len(t)==1;t[0].text=',+'
 assert E.tostring(b,method='c14n')==E.tostring(a,method='c14n')
 result={'source':str(source),'output':str(out),'math_objects':len(am),'changed_math_ids':diffs,'changed_zip_parts':changed,'all_other_xml_content_structure_format_and_parts_exact':True,'before':'G_z∈{−1,+1}^{6×6}','after':'G_z∈{−1,0,+1}^{6×6}','reason':'现有对角矩阵、自逆、对角元素±1条件要求非对角元为0。','chinese_text_and_manual_line_breaks_unchanged':True}
(A/'change_scope_qa.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False))
