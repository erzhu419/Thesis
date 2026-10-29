from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
import json
A=Path(__file__).resolve().parent;O=A.parents[1]/'调整完格式的_公式终检_20261019.docx';S=A/'source_snapshot.docx';N={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'};W='{'+N['w']+'}'
with ZipFile(O) as z:
 r=E.fromstring(z.read('word/document.xml'));ms=r.xpath('.//m:oMath',namespaces=N)
 for i in (2991,2993):
  tc=next(x for x in ms[i-1].iterancestors() if x.tag==W+'tc');pr=tc.find('w:tcPr',N);assert pr.find('w:tcMar',N) is None
  mar=E.SubElement(pr,W+'tcMar')
  for side in ('left','right'):
   e=E.SubElement(mar,W+side);e.set(W+'w','0');e.set(W+'type','dxa')
  # tcMar precedes vAlign in the OOXML cell-property sequence.
  pr.remove(mar);pos=next((j for j,e in enumerate(pr) if e.tag in (W+'textDirection',W+'tcFitText',W+'vAlign',W+'hideMark')),len(pr));pr.insert(pos,mar)
 data=E.tostring(r,encoding='UTF-8',xml_declaration=True,standalone=True);tmp=O.with_name(O.stem+'_writing.docx')
 with ZipFile(tmp,'w') as d:
  for info in z.infolist():d.writestr(info,data if info.filename=='word/document.xml' else z.read(info.filename))
tmp.replace(O)
with ZipFile(S) as b,ZipFile(O) as a:
 assert b.namelist()==a.namelist()
 parts=[n for n in b.namelist() if b.read(n)!=a.read(n)];assert parts==['word/document.xml']
 rb=E.fromstring(b.read('word/document.xml'));ra=E.fromstring(a.read('word/document.xml'));bm=rb.xpath('.//m:oMath',namespaces=N);am=ra.xpath('.//m:oMath',namespaces=N)
 assert len(bm)==len(am)==4825
 q=json.loads((A/'change_scope_qa.json').read_text())
 for item in q['proofing_added']:
  run=am[item['math_id']-1].xpath('.//m:r',namespaces=N)[item['run_index_0based']];n=run.find('w:rPr/w:noProof',N);n.getparent().remove(n)
 next(t for t in am[2240].xpath('.//m:t',namespaces=N) if t.text==',0,+').text=',+'
 for i in (2991,2993):
  tc=next(x for x in am[i-1].iterancestors() if x.tag==W+'tc');mar=tc.find('w:tcPr/w:tcMar',N);assert [e.get(W+'w') for e in mar]==['0','0'];mar.getparent().remove(mar)
 assert E.tostring(rb,method='c14n')==E.tostring(ra,method='c14n')
 q.update({'table_cells_with_reduced_margins_math_ids':[2991,2993],'manual_line_breaks_equation_paragraphs_table_width_font_size_unchanged':True,'all_changes_exhaustively_reversed_and_original_xml_identical':True})
 (A/'change_scope_qa.json').write_text(json.dumps(q,ensure_ascii=False,indent=2));print('PASS: only the G matrix 0 insertion, 63 proofing flags, and 2 cell margin overrides differ.')
